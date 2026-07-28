/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME (Adreno Frame Motion Engine) — EGL Layer for GPU Frame Generation
 *
 * Architecture:
 * ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌─────────┐
 * │  Game    │────>│ AFME Layer│────>│ Adreno   │────>│ Display │
 * │ (60fps)  │     │ eglSwap   │     │ GPU      │     │ (120fps)│
 * └──────────┘     │ intercept │     │ AFME HW  │     └─────────┘
 *                  └───────────┘     └──────────┘
 *
 * Two frame-generation methods, selected by persist.sys.afme.method:
 *
 *   method=0  EXTRAPOLATE — glExtrapolateTex2DQCOM(prev, curr, out, factor).
 *             The driver's black-box frame extrapolator. One call, lowest CPU
 *             cost, no shaders. Quality is whatever the firmware does.
 *
 *   method=1  MOTION — glTexEstimateMotionQCOM(prevLuma, currLuma, blockMV)
 *             followed by our own warp shader. The HW motion estimator gives a
 *             block-granularity motion field; we forward-project `curr` along
 *             it. Costs two extra full-screen passes but the motion field is
 *             inspectable, so we can clamp wild vectors and refuse to warp
 *             disoccluded regions — which the black-box path cannot do.
 *
 * Both methods keep the same presentation ORDER: the real frame goes out first,
 * then the synthetic one. Neither delays the real frame, so neither adds input
 * latency. (True interpolation — holding frame N to blend it with N+1 — would;
 * that is the Vulkan layer's MobFGSR path, which has the frame budget for it.)
 *
 * Flow per frame:
 * 1. Game renders frame N to default FBO
 * 2. AFME layer intercepts eglSwapBuffers
 * 3. Blit default FBO → currTex
 * 4. If a previous frame exists:
 *    a. Present the REAL frame N via the real eglSwapBuffers (low latency)
 *    b. Synthesize: extrapolate, or estimate-motion + warp
 *    c. Blit the synthetic frame to the default FBO and present it
 * 5. Save current as previous
 *
 * Control:
 *   persist.sys.afme.enable     = 1/0       enable/disable
 *   persist.sys.afme.multiplier = 2/3/4     frame multiplier
 *   persist.sys.afme.method     = 0/1       0=extrapolate, 1=motion estimation
 *   persist.sys.afme.factor     = float     phase override (0/empty = auto)
 *   persist.sys.afme.display_hz = int       panel rate, for the headroom clamp
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <GLES2/gl2ext.h>   // GL_MOTION_ESTIMATION_SEARCH_BLOCK_{X,Y}_QCOM

#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <utility>

#include <android/log.h>
#include <cutils/properties.h>

// ─── Android EGL Layer types (not in standard EGL headers) ──────────────────
// Defined by Android's GLES layer loading system.
// See: frameworks/native/opengl/libs/EGL/GLES_layers.md
typedef void* EGLFuncPointer;
typedef void* (*PFNEGLGETNEXTLAYERPROCADDRESSPROC)(void*, const char*);

#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Older gl2ext.h revisions predate QCOM_motion_estimation. The values are
// stable ABI (see external/angle/include/GLES2/gl2ext.h).
#ifndef GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM 0x8C90
#endif
#ifndef GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM 0x8C91
#endif

// ─── GL function pointer typedefs (resolved at runtime via eglGetProcAddress) ─
// GLES 3.0 functions — NOT available at link time for vendor EGL layers
typedef void (*PFNGLGENFRAMEBUFFERSPROC)(GLsizei, GLuint*);
typedef void (*PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei, const GLuint*);
typedef void (*PFNGLBINDFRAMEBUFFERPROC)(GLenum, GLuint);
typedef void (*PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef void (*PFNGLBLITFRAMEBUFFERPROC)(GLint, GLint, GLint, GLint,
                                          GLint, GLint, GLint, GLint,
                                          GLbitfield, GLenum);
typedef GLenum (*PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum);
typedef void (*PFNGLGENTEXTURESPROC)(GLsizei, GLuint*);
typedef void (*PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint*);
typedef void (*PFNGLBINDTEXTUREPROC)(GLenum, GLuint);
typedef void (*PFNGLTEXIMAGE2DPROC_)(GLenum, GLint, GLint, GLsizei, GLsizei,
                                      GLint, GLenum, GLenum, const void*);
typedef void (*PFNGLTEXSTORAGE2DPROC_)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
typedef void (*PFNGLTEXPARAMETERIPROC_)(GLenum, GLenum, GLint);
typedef void (*PFNGLACTIVETEXTUREPROC_)(GLenum);
typedef void (*PFNGLFINISHPROC)(void);
typedef void (*PFNGLFLUSHPROC)(void);
typedef void (*PFNGLGETINTEGERVPROC_)(GLenum, GLint*);
typedef void (*PFNGLGETBOOLEANVPROC_)(GLenum, GLboolean*);
typedef GLboolean (*PFNGLISENABLEDPROC_)(GLenum);
typedef void (*PFNGLENABLEPROC_)(GLenum);
typedef void (*PFNGLDISABLEPROC_)(GLenum);
typedef void (*PFNGLVIEWPORTPROC_)(GLint, GLint, GLsizei, GLsizei);
typedef void (*PFNGLCOLORMASKPROC_)(GLboolean, GLboolean, GLboolean, GLboolean);
typedef void (*PFNGLDEPTHMASKPROC_)(GLboolean);
typedef void (*PFNGLDRAWARRAYSPROC_)(GLenum, GLint, GLsizei);
typedef GLenum (*PFNGLGETERRORPROC_)(void);

// Shader / program objects (for the motion-estimation warp)
typedef GLuint (*PFNGLCREATESHADERPROC_)(GLenum);
typedef void (*PFNGLSHADERSOURCEPROC_)(GLuint, GLsizei, const GLchar* const*, const GLint*);
typedef void (*PFNGLCOMPILESHADERPROC_)(GLuint);
typedef void (*PFNGLGETSHADERIVPROC_)(GLuint, GLenum, GLint*);
typedef void (*PFNGLGETSHADERINFOLOGPROC_)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void (*PFNGLDELETESHADERPROC_)(GLuint);
typedef GLuint (*PFNGLCREATEPROGRAMPROC_)(void);
typedef void (*PFNGLATTACHSHADERPROC_)(GLuint, GLuint);
typedef void (*PFNGLLINKPROGRAMPROC_)(GLuint);
typedef void (*PFNGLGETPROGRAMIVPROC_)(GLuint, GLenum, GLint*);
typedef void (*PFNGLGETPROGRAMINFOLOGPROC_)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void (*PFNGLDELETEPROGRAMPROC_)(GLuint);
typedef void (*PFNGLUSEPROGRAMPROC_)(GLuint);
typedef GLint (*PFNGLGETUNIFORMLOCATIONPROC_)(GLuint, const GLchar*);
typedef void (*PFNGLUNIFORM1IPROC_)(GLint, GLint);
typedef void (*PFNGLUNIFORM1FPROC_)(GLint, GLfloat);
typedef void (*PFNGLUNIFORM2FPROC_)(GLint, GLfloat, GLfloat);
typedef void (*PFNGLGENVERTEXARRAYSPROC_)(GLsizei, GLuint*);
typedef void (*PFNGLBINDVERTEXARRAYPROC_)(GLuint);
typedef void (*PFNGLDELETEVERTEXARRAYSPROC_)(GLsizei, const GLuint*);

// QCOM extensions
typedef void (*PFNGLEXTRAPOLATETEX2DQCOMPROC)(GLuint, GLuint, GLuint, GLfloat);
typedef void (*PFNGLTEXESTIMATEMOTIONQCOMPROC)(GLuint, GLuint, GLuint);
typedef void (*PFNGLSHADINGRATEQCOMPROC)(GLenum);

// GL_QCOM_shading_rate — ABI values match external/angle include/GLES2/gl2ext.h
// (not guaranteed to exist in vendor GLES headers).
#ifndef GL_SHADING_RATE_1X1_PIXELS_QCOM
#define GL_SHADING_RATE_1X1_PIXELS_QCOM 0x96A6
#endif
#ifndef GL_SHADING_RATE_2X2_PIXELS_QCOM
#define GL_SHADING_RATE_2X2_PIXELS_QCOM 0x96A9
#endif

// EGL_ANDROID_presentation_time: stamp a buffer's latch time
typedef EGLBoolean (*PFNEGLPRESENTATIONTIMEANDROIDPROC)(EGLDisplay, EGLSurface, EGLnsecsANDROID);

// ─── Frame-generation method ────────────────────────────────────────────────
enum FGMethod {
    FG_EXTRAPOLATE = 0,  // glExtrapolateTex2DQCOM — driver black box
    FG_MOTION      = 1,  // glTexEstimateMotionQCOM + our warp shader
};

// ─── Layer state ────────────────────────────────────────────────────────────
namespace {

// Resolved GL function pointers
struct GLFuncs {
    PFNGLGENFRAMEBUFFERSPROC     GenFramebuffers = nullptr;
    PFNGLDELETEFRAMEBUFFERSPROC  DeleteFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC     BindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = nullptr;
    PFNGLBLITFRAMEBUFFERPROC     BlitFramebuffer = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus = nullptr;
    PFNGLGENTEXTURESPROC         GenTextures = nullptr;
    PFNGLDELETETEXTURESPROC      DeleteTextures = nullptr;
    PFNGLBINDTEXTUREPROC         BindTexture = nullptr;
    PFNGLTEXIMAGE2DPROC_         TexImage2D = nullptr;
    PFNGLTEXSTORAGE2DPROC_       TexStorage2D = nullptr;
    PFNGLTEXPARAMETERIPROC_      TexParameteri = nullptr;
    PFNGLACTIVETEXTUREPROC_      ActiveTexture = nullptr;
    PFNGLFINISHPROC              Finish = nullptr;
    PFNGLFLUSHPROC               Flush = nullptr;
    PFNGLGETINTEGERVPROC_        GetIntegerv = nullptr;
    PFNGLGETBOOLEANVPROC_        GetBooleanv = nullptr;
    PFNGLISENABLEDPROC_          IsEnabled = nullptr;
    PFNGLENABLEPROC_             Enable = nullptr;
    PFNGLDISABLEPROC_            Disable = nullptr;
    PFNGLVIEWPORTPROC_           Viewport = nullptr;
    PFNGLCOLORMASKPROC_          ColorMask = nullptr;
    PFNGLDEPTHMASKPROC_          DepthMask = nullptr;
    PFNGLDRAWARRAYSPROC_         DrawArrays = nullptr;
    PFNGLGETERRORPROC_           GetError = nullptr;

    PFNGLCREATESHADERPROC_       CreateShader = nullptr;
    PFNGLSHADERSOURCEPROC_       ShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC_      CompileShader = nullptr;
    PFNGLGETSHADERIVPROC_        GetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC_   GetShaderInfoLog = nullptr;
    PFNGLDELETESHADERPROC_       DeleteShader = nullptr;
    PFNGLCREATEPROGRAMPROC_      CreateProgram = nullptr;
    PFNGLATTACHSHADERPROC_       AttachShader = nullptr;
    PFNGLLINKPROGRAMPROC_        LinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC_       GetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC_  GetProgramInfoLog = nullptr;
    PFNGLDELETEPROGRAMPROC_      DeleteProgram = nullptr;
    PFNGLUSEPROGRAMPROC_         UseProgram = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC_ GetUniformLocation = nullptr;
    PFNGLUNIFORM1IPROC_          Uniform1i = nullptr;
    PFNGLUNIFORM1FPROC_          Uniform1f = nullptr;
    PFNGLUNIFORM2FPROC_          Uniform2f = nullptr;
    PFNGLGENVERTEXARRAYSPROC_    GenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAYPROC_    BindVertexArray = nullptr;
    PFNGLDELETEVERTEXARRAYSPROC_ DeleteVertexArrays = nullptr;

    PFNGLEXTRAPOLATETEX2DQCOMPROC ExtrapolateTex2D = nullptr;
    PFNGLTEXESTIMATEMOTIONQCOMPROC EstimateMotion = nullptr;
    PFNGLSHADINGRATEQCOMPROC ShadingRate = nullptr;
    PFNEGLPRESENTATIONTIMEANDROIDPROC PresentationTime = nullptr;
    bool resolved = false;
};

GLFuncs sGL;

// EGL layer function map
std::unordered_map<std::string, EGLFuncPointer> sFuncMap;
std::mutex sMapMutex;

// Layer initialization state
void* sLayerId = nullptr;
PFNEGLGETNEXTLAYERPROCADDRESSPROC sGetNextLayerProcAddress = nullptr;

// ─── Shaders for the motion-estimation warp ─────────────────────────────────
//
// Both passes are full-screen draws with no vertex buffer: the vertex shader
// synthesizes a covering triangle from gl_VertexID, so we never touch the
// game's array buffers or attribute state.

static const char* kFullscreenVertSrc = R"(#version 300 es
out vec2 vUV;
void main() {
    // ids 0,1,2 → (-1,-1), (3,-1), (-1,3): one triangle covering the viewport
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
})";

// RGB → luminance (R8). glTexEstimateMotionQCOM takes single-channel inputs.
static const char* kLumaFragSrc = R"(#version 300 es
precision mediump float;
uniform mediump sampler2D uSrc;
in vec2 vUV;
out vec4 outColor;
void main() {
    vec3 c = texture(uSrc, vUV).rgb;
    outColor = vec4(0.299 * c.r + 0.587 * c.g + 0.114 * c.b, 0.0, 0.0, 1.0);
})";

// Forward-project `curr` along the estimated motion field.
//
// The HW motion estimator gives, per block, mv = pos_curr - pos_prev in PIXELS,
// indexed at the CURRENT position (same convention the Vulkan layer's MobFGSR
// reproject/warp shaders assume). Content presently at p is therefore heading
// for p + mv per frame interval, so the pixel that will occupy p at time
// curr+factor is currently at p - mv*factor: a backward fetch with a scaled MV.
//
// uMV is sampled with LINEAR filtering, which bilinearly upsamples the
// block-resolution field to per-pixel for free — no separate upsample pass.
static const char* kWarpFragSrc = R"(#version 300 es
precision highp float;
uniform mediump sampler2D uCurr;  // current color, values in [0,1]
// highp is load-bearing: GLSL ES 3.0 defaults sampler2D to lowp in fragment
// shaders, and this texture holds motion in PIXELS (hundreds), which lowp's
// ~±2 range would flatten to nothing. The Vulkan layer's MobFGSR shaders get
// away with mediump because they store motion in UV space instead.
uniform highp sampler2D uMV;      // block motion field, RGBA16F, .xy = pixels
uniform vec2  uRenderSize;
uniform float uFactor;          // how far past `curr` to project (0..1)
uniform float uMaxMV;           // reject implausible vectors, in pixels
in  vec2 vUV;
out vec4 outColor;
void main() {
    vec2 mvPx = texture(uMV, vUV).xy;

    // Guard 1 — magnitude. A scene cut or an ME miss produces vectors far
    // larger than any real object motion; warping by those smears the whole
    // frame. Above the threshold we distrust the vector entirely rather than
    // clamping it, since a clamped bogus vector is still bogus.
    float len = length(mvPx);
    if (len > uMaxMV) mvPx = vec2(0.0);

    vec2 src = vUV - (mvPx / uRenderSize) * uFactor;

    // Guard 2 — disocclusion at the frame border. Sampling outside the frame
    // would stretch edge pixels inward; keep the real pixel there instead.
    vec2 cl = clamp(src, vec2(0.0), vec2(1.0));
    float outside = any(notEqual(cl, src)) ? 1.0 : 0.0;

    outColor = mix(texture(uCurr, cl), texture(uCurr, vUV), outside);
})";

// ─── Per-surface AFME state ─────────────────────────────────────────────────
struct AFMEState {
    GLuint prevTex = 0;
    GLuint currTex = 0;
    GLuint synthTex = 0;
    GLuint readFBO = 0;
    GLuint drawFBO = 0;
    GLint width = 0;
    GLint height = 0;
    bool initialized = false;
    bool hasPrevFrame = false;
    uint32_t frameCount = 0;
    bool extensionsAvailable = false;

    // Frame-interval EMA for the adaptive generation clamp
    int64_t lastSwapNs = 0;
    float emaFrameMs = 0.0f;

    // Synthetic-frame spacing (see sSpacingEnabled)
    int64_t realSwapNs = 0;      // when this frame's real swap was issued
    float spacingScale = 0.0f;   // 0 = present immediately, 1 = full spacing
    float bestRealFps = 0.0f;    // reference rate the governor protects

    // ── Motion-estimation resources (method=1 only) ──
    bool motionReady = false;      // ME path fully initialized
    bool motionAttempted = false;  // don't retry setup every frame on failure
    GLuint prevLumaTex = 0;
    GLuint currLumaTex = 0;
    GLuint mvBlockTex = 0;
    GLuint lumaFBO = 0;            // renders into {prev,curr}LumaTex
    GLuint genFBO = 0;             // renders into synthTex
    GLuint vao = 0;
    GLuint lumaProg = 0;
    GLuint warpProg = 0;
    GLint lumaSrcLoc = -1;
    GLint warpCurrLoc = -1, warpMVLoc = -1;
    GLint warpSizeLoc = -1, warpFactorLoc = -1, warpMaxMVLoc = -1;
    GLint blockX = 16, blockY = 16;
    bool hasLumaHistory = false;   // prevLumaTex holds a real previous frame

    // ── Stats for the AFME-STATS channel GameSpace parses ──
    // Per-surface, never file-scope statics: a surface recreated at a new size
    // must not inherit the old baseline (the Vulkan layer hit exactly that and
    // reported "real=2147483647 total=-2").
    int64_t statsBaseNs = 0;
    uint32_t statsBaseReal = 0;
    uint32_t statsBaseGen = 0;
    uint32_t realFrames = 0;
    uint32_t genFrames = 0;
};

std::unordered_map<EGLSurface, AFMEState> sStates;
std::mutex sStateMutex;

std::atomic<bool> sEnabled{false};
std::atomic<int> sMultiplier{2};
std::atomic<float> sFactorOverride{0.0f};  // 0 = auto
std::atomic<int> sDisplayHz{120};          // panel rate from GameStateDispatcher
std::atomic<int> sMethod{FG_EXTRAPOLATE};

// Synthetic-frame spacing. Swapping the synthetic frame straight after the real
// one puts it in the very next vsync slot, so a 30fps game on a 120Hz panel
// shows the real frame for 8ms and the synthetic one for 25ms — 60 presents a
// second that still read as 30fps with a strobe. Spacing the synthetic swap
// across the game's frame interval is what actually makes generation visible.
std::atomic<bool> sSpacingEnabled{true};

// eglPresentationTimeANDROID stamping of synthetic frames. Default OFF: on the
// bp4a SurfaceFlinger build shipping on onyx, delayed buffers were measured to
// be DROPPED rather than held (the VK layer's VkPresentTimesInfoGOOGLE path
// lost >50% of synth frames), so our own spacing sleep is the working
// mechanism. Kept behind persist.sys.afme.pacing for future SF releases.
std::atomic<bool> sPacingEnabled{false};

// Apply GL_QCOM_shading_rate to our own full-screen fragment passes (luma,
// warp). 2x2 VRS quarters the fragment cost of generation; it does NOT touch
// the game's rendering — we leave 1X1 behind and the driver resets it per
// framebuffer anyway. Compute dispatches are unaffected by VRS.
std::atomic<bool> sVrsFgEnabled{true};

// Set the fragment shading rate for our passes when supported.
void setVrsRate(GLenum rate) {
    if (sGL.ShadingRate && sVrsFgEnabled.load(std::memory_order_relaxed)) {
        sGL.ShadingRate(rate);
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

void checkProperties() {
    char value[PROPERTY_VALUE_MAX];

    property_get("persist.sys.afme.enable", value, "0");
    sEnabled.store(value[0] == '1');

    // Read multiplier (2x, 3x, 4x) — compute factor from this
    property_get("persist.sys.afme.multiplier", value, "2");
    int mult = atoi(value);
    if (mult < 2) mult = 2;
    if (mult > 4) mult = 4;
    sMultiplier.store(mult);

    // Frame-generation method. Unknown values fall back to extrapolation,
    // which needs no shaders and so can never fail to initialize.
    property_get("persist.sys.afme.method", value, "0");
    int method = atoi(value);
    sMethod.store((method == FG_MOTION) ? FG_MOTION : FG_EXTRAPOLATE);

    // User-configurable factor override from GameSpace
    property_get("persist.sys.afme.factor", value, "");
    float f = strtof(value, nullptr);
    sFactorOverride.store((f > 0.0f && f <= 2.0f) ? f : 0.0f);

    // Panel refresh rate — clamp generation so we never queue more frames
    // than the display can show (BufferQueue back-pressure would throttle
    // the game itself, e.g. 60fps + 2x on a 60Hz-locked panel → 30fps).
    property_get("persist.sys.afme.display_hz", value, "120");
    int hz = atoi(value);
    if (hz >= 30 && hz <= 240) sDisplayHz.store(hz);

    property_get("persist.sys.afme.spacing", value, "1");
    sSpacingEnabled.store(value[0] == '1');

    property_get("persist.sys.afme.pacing", value, "0");
    sPacingEnabled.store(value[0] == '1');

    property_get("persist.sys.afme.vrs_fg", value, "1");
    sVrsFgEnabled.store(value[0] == '1');
}

// Hold a synthetic frame until its slot inside the game's frame interval.
// Scaled by the per-surface governor, so if the sleep costs the game real
// frames it decays to a no-op by itself.
void spaceSyntheticFrame(AFMEState& state, int index, int numGenFrames) {
    if (!sSpacingEnabled.load(std::memory_order_relaxed)) return;
    if (state.spacingScale <= 0.01f || state.emaFrameMs <= 0.0f) return;
    if (state.realSwapNs <= 0) return;

    int64_t intervalNs = (int64_t)(state.emaFrameMs * 1e6f);
    int64_t slotNs = intervalNs / (int64_t)(numGenFrames + 1);
    int64_t offsetNs = (int64_t)((float)(slotNs * (int64_t)(index + 1))
                                 * state.spacingScale);

    struct timespec tsNow;
    clock_gettime(CLOCK_MONOTONIC, &tsNow);
    int64_t curNs = (int64_t)tsNow.tv_sec * 1000000000LL + tsNow.tv_nsec;
    int64_t sleepNs = (state.realSwapNs + offsetNs) - curNs;

    // Below ~0.3ms the sleep costs more than it buys; above 50ms the interval
    // estimate is wrong and we must not hang the game thread on it.
    if (sleepNs > 300000LL && sleepNs < 50000000LL) {
        struct timespec req = { (time_t)(sleepNs / 1000000000LL),
                                (long)(sleepNs % 1000000000LL) };
        nanosleep(&req, nullptr);
    }
}

// Regulate the spacing scale on the real frame rate — the quantity spacing
// must not damage. Free when the game is capped below the panel (its own
// limiter just sleeps less); costly when it is work-bound. A single interval
// measurement cannot tell those apart, so let the loop find out.
void updateSpacingGovernor(AFMEState& state) {
    if (state.emaFrameMs <= 0.0f) return;
    float realFps = 1000.0f / state.emaFrameMs;
    state.bestRealFps = (realFps > state.bestRealFps)
                            ? realFps : state.bestRealFps * 0.999f;
    if (state.bestRealFps > 1.0f && realFps < state.bestRealFps * 0.94f) {
        state.spacingScale -= 0.08f;   // back off fast: we are costing frames
    } else {
        state.spacingScale += 0.02f;   // creep up: 50 frames to full
    }
    if (state.spacingScale < 0.0f) state.spacingScale = 0.0f;
    if (state.spacingScale > 1.0f) state.spacingScale = 1.0f;
}

void resolveGLFunctions() {
    if (sGL.resolved) return;

    #define RESOLVE(field, type, name) \
        sGL.field = (type)eglGetProcAddress(name)

    // Framebuffers / textures — the extrapolation path needs only these
    RESOLVE(GenFramebuffers, PFNGLGENFRAMEBUFFERSPROC, "glGenFramebuffers");
    RESOLVE(DeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSPROC, "glDeleteFramebuffers");
    RESOLVE(BindFramebuffer, PFNGLBINDFRAMEBUFFERPROC, "glBindFramebuffer");
    RESOLVE(FramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC, "glFramebufferTexture2D");
    RESOLVE(BlitFramebuffer, PFNGLBLITFRAMEBUFFERPROC, "glBlitFramebuffer");
    RESOLVE(CheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC, "glCheckFramebufferStatus");
    RESOLVE(GenTextures, PFNGLGENTEXTURESPROC, "glGenTextures");
    RESOLVE(DeleteTextures, PFNGLDELETETEXTURESPROC, "glDeleteTextures");
    RESOLVE(BindTexture, PFNGLBINDTEXTUREPROC, "glBindTexture");
    RESOLVE(TexImage2D, PFNGLTEXIMAGE2DPROC_, "glTexImage2D");
    RESOLVE(TexStorage2D, PFNGLTEXSTORAGE2DPROC_, "glTexStorage2D");
    RESOLVE(TexParameteri, PFNGLTEXPARAMETERIPROC_, "glTexParameteri");
    RESOLVE(ActiveTexture, PFNGLACTIVETEXTUREPROC_, "glActiveTexture");
    RESOLVE(Finish, PFNGLFINISHPROC, "glFinish");
    RESOLVE(Flush, PFNGLFLUSHPROC, "glFlush");
    RESOLVE(GetIntegerv, PFNGLGETINTEGERVPROC_, "glGetIntegerv");
    RESOLVE(GetBooleanv, PFNGLGETBOOLEANVPROC_, "glGetBooleanv");
    RESOLVE(IsEnabled, PFNGLISENABLEDPROC_, "glIsEnabled");
    RESOLVE(Enable, PFNGLENABLEPROC_, "glEnable");
    RESOLVE(Disable, PFNGLDISABLEPROC_, "glDisable");
    RESOLVE(Viewport, PFNGLVIEWPORTPROC_, "glViewport");
    RESOLVE(ColorMask, PFNGLCOLORMASKPROC_, "glColorMask");
    RESOLVE(DepthMask, PFNGLDEPTHMASKPROC_, "glDepthMask");
    RESOLVE(DrawArrays, PFNGLDRAWARRAYSPROC_, "glDrawArrays");
    RESOLVE(GetError, PFNGLGETERRORPROC_, "glGetError");

    // Shader objects — only the motion-estimation path needs these
    RESOLVE(CreateShader, PFNGLCREATESHADERPROC_, "glCreateShader");
    RESOLVE(ShaderSource, PFNGLSHADERSOURCEPROC_, "glShaderSource");
    RESOLVE(CompileShader, PFNGLCOMPILESHADERPROC_, "glCompileShader");
    RESOLVE(GetShaderiv, PFNGLGETSHADERIVPROC_, "glGetShaderiv");
    RESOLVE(GetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC_, "glGetShaderInfoLog");
    RESOLVE(DeleteShader, PFNGLDELETESHADERPROC_, "glDeleteShader");
    RESOLVE(CreateProgram, PFNGLCREATEPROGRAMPROC_, "glCreateProgram");
    RESOLVE(AttachShader, PFNGLATTACHSHADERPROC_, "glAttachShader");
    RESOLVE(LinkProgram, PFNGLLINKPROGRAMPROC_, "glLinkProgram");
    RESOLVE(GetProgramiv, PFNGLGETPROGRAMIVPROC_, "glGetProgramiv");
    RESOLVE(GetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC_, "glGetProgramInfoLog");
    RESOLVE(DeleteProgram, PFNGLDELETEPROGRAMPROC_, "glDeleteProgram");
    RESOLVE(UseProgram, PFNGLUSEPROGRAMPROC_, "glUseProgram");
    RESOLVE(GetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC_, "glGetUniformLocation");
    RESOLVE(Uniform1i, PFNGLUNIFORM1IPROC_, "glUniform1i");
    RESOLVE(Uniform1f, PFNGLUNIFORM1FPROC_, "glUniform1f");
    RESOLVE(Uniform2f, PFNGLUNIFORM2FPROC_, "glUniform2f");
    RESOLVE(GenVertexArrays, PFNGLGENVERTEXARRAYSPROC_, "glGenVertexArrays");
    RESOLVE(BindVertexArray, PFNGLBINDVERTEXARRAYPROC_, "glBindVertexArray");
    RESOLVE(DeleteVertexArrays, PFNGLDELETEVERTEXARRAYSPROC_, "glDeleteVertexArrays");

    // QCOM extensions
    RESOLVE(ExtrapolateTex2D, PFNGLEXTRAPOLATETEX2DQCOMPROC, "glExtrapolateTex2DQCOM");
    RESOLVE(EstimateMotion, PFNGLTEXESTIMATEMOTIONQCOMPROC, "glTexEstimateMotionQCOM");
    RESOLVE(ShadingRate, PFNGLSHADINGRATEQCOMPROC, "glShadingRateQCOM");
    // EGL_ANDROID_presentation_time — used to place synthetic frames on
    // exact vsync slots.
    sGL.PresentationTime = (PFNEGLPRESENTATIONTIMEANDROIDPROC)
            eglGetProcAddress("eglPresentationTimeANDROID");

    #undef RESOLVE

    sGL.resolved = true;

    if (sGL.ExtrapolateTex2D) {
        ALOGI("AFME: GL_QCOM_frame_extrapolation resolved (%p)",
              sGL.ExtrapolateTex2D);
    } else {
        ALOGW("AFME: GL_QCOM_frame_extrapolation NOT available");
    }

    if (sGL.EstimateMotion) {
        ALOGI("AFME: GL_QCOM_motion_estimation resolved (%p)", sGL.EstimateMotion);
    } else {
        ALOGW("AFME: GL_QCOM_motion_estimation NOT available — method=1 unusable");
    }

    if (sGL.BlitFramebuffer) {
        ALOGI("AFME: glBlitFramebuffer resolved (%p)", sGL.BlitFramebuffer);
    } else {
        ALOGW("AFME: glBlitFramebuffer NOT available — cannot operate");
    }
}

// ─── GL state isolation ─────────────────────────────────────────────────────
//
// We run our passes inside the GAME's context, so every binding we touch must
// be put back exactly as we found it. Anything we miss corrupts the next draw
// the game issues — which shows up as flicker or geometry loss that looks
// nothing like a frame-generation bug.
struct GLStateGuard {
    GLint program = 0;
    GLint drawFBO = 0, readFBO = 0;
    GLint vao = 0;
    GLint activeTex = GL_TEXTURE0;
    GLint tex0 = 0, tex1 = 0;
    GLint viewport[4] = {0, 0, 0, 0};
    GLboolean colorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
    GLboolean depthMask = GL_TRUE;
    GLboolean depthTest = GL_FALSE, scissor = GL_FALSE, stencil = GL_FALSE;
    GLboolean cull = GL_FALSE, blend = GL_FALSE;
    bool active = false;

    // Every getter and setter we rely on must exist before we start rewriting
    // the game's pipeline state; a half-applied guard is worse than none.
    static bool supported() {
        return sGL.GetIntegerv && sGL.GetBooleanv && sGL.IsEnabled &&
               sGL.Enable && sGL.Disable && sGL.ColorMask && sGL.DepthMask &&
               sGL.Viewport && sGL.ActiveTexture && sGL.BindTexture &&
               sGL.BindVertexArray && sGL.UseProgram && sGL.BindFramebuffer;
    }

    void save() {
        active = supported();
        if (!active) return;
        sGL.GetIntegerv(GL_CURRENT_PROGRAM, &program);
        sGL.GetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFBO);
        sGL.GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFBO);
        sGL.GetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        sGL.GetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
        sGL.GetIntegerv(GL_VIEWPORT, viewport);
        sGL.GetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        sGL.GetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
        depthTest = sGL.IsEnabled(GL_DEPTH_TEST);
        scissor   = sGL.IsEnabled(GL_SCISSOR_TEST);
        stencil   = sGL.IsEnabled(GL_STENCIL_TEST);
        cull      = sGL.IsEnabled(GL_CULL_FACE);
        blend     = sGL.IsEnabled(GL_BLEND);

        // Texture bindings are per-unit; we only ever use units 0 and 1.
        sGL.ActiveTexture(GL_TEXTURE0);
        sGL.GetIntegerv(GL_TEXTURE_BINDING_2D, &tex0);
        sGL.ActiveTexture(GL_TEXTURE1);
        sGL.GetIntegerv(GL_TEXTURE_BINDING_2D, &tex1);
    }

    // Put the pipeline into the neutral configuration our passes assume:
    // no depth/stencil/cull/blend/scissor, full color write, no depth write.
    void neutralize() {
        if (!active) return;
        if (depthTest) sGL.Disable(GL_DEPTH_TEST);
        if (scissor)   sGL.Disable(GL_SCISSOR_TEST);
        if (stencil)   sGL.Disable(GL_STENCIL_TEST);
        if (cull)      sGL.Disable(GL_CULL_FACE);
        if (blend)     sGL.Disable(GL_BLEND);
        sGL.ColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        sGL.DepthMask(GL_FALSE);
    }

    void restore() {
        if (!active) return;
        sGL.ActiveTexture(GL_TEXTURE1);
        sGL.BindTexture(GL_TEXTURE_2D, (GLuint)tex1);
        sGL.ActiveTexture(GL_TEXTURE0);
        sGL.BindTexture(GL_TEXTURE_2D, (GLuint)tex0);
        sGL.ActiveTexture((GLenum)activeTex);

        sGL.BindVertexArray((GLuint)vao);
        sGL.UseProgram((GLuint)program);
        sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)drawFBO);
        sGL.BindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)readFBO);
        sGL.Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        sGL.ColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        sGL.DepthMask(depthMask);

        if (depthTest) sGL.Enable(GL_DEPTH_TEST);
        if (scissor)   sGL.Enable(GL_SCISSOR_TEST);
        if (stencil)   sGL.Enable(GL_STENCIL_TEST);
        if (cull)      sGL.Enable(GL_CULL_FACE);
        if (blend)     sGL.Enable(GL_BLEND);
    }
};

// Clear any error flags our own passes raised. glGetError is sticky, so an
// error we leave behind would surface in the game's next check and be blamed on
// the game's own draw. Logged rather than silently swallowed so our bugs stay
// visible.
void drainGLErrors(AFMEState& state, const char* where) {
    if (!sGL.GetError) return;
    GLenum err;
    int drained = 0;
    while ((err = sGL.GetError()) != GL_NO_ERROR) {
        if (drained == 0 && (state.frameCount % 300) == 0) {
            ALOGW("AFME: GL error 0x%x in %s", err, where);
        }
        if (++drained > 8) break;  // don't spin if the context is lost
    }
}

// ─── Texture / program creation ─────────────────────────────────────────────

GLuint createTexture(GLint width, GLint height) {
    GLuint tex;
    sGL.GenTextures(1, &tex);
    sGL.BindTexture(GL_TEXTURE_2D, tex);
    sGL.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    sGL.BindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

GLuint createStorageTexture(GLenum internalFormat, GLint w, GLint h, GLint filter) {
    GLuint tex = 0;
    sGL.GenTextures(1, &tex);
    sGL.BindTexture(GL_TEXTURE_2D, tex);
    sGL.TexStorage2D(GL_TEXTURE_2D, 1, internalFormat, w, h);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    sGL.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    sGL.BindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

GLuint compileShader(GLenum type, const char* src, const char* label) {
    GLuint sh = sGL.CreateShader(type);
    if (!sh) return 0;
    sGL.ShaderSource(sh, 1, &src, nullptr);
    sGL.CompileShader(sh);
    GLint ok = GL_FALSE;
    sGL.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512] = {0};
        sGL.GetShaderInfoLog(sh, sizeof(log) - 1, nullptr, log);
        ALOGE("AFME: %s shader compile failed: %s", label, log);
        sGL.DeleteShader(sh);
        return 0;
    }
    return sh;
}

GLuint buildProgram(const char* vertSrc, const char* fragSrc, const char* label) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc, label);
    if (!vs) return 0;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc, label);
    if (!fs) { sGL.DeleteShader(vs); return 0; }

    GLuint prog = sGL.CreateProgram();
    if (!prog) { sGL.DeleteShader(vs); sGL.DeleteShader(fs); return 0; }
    sGL.AttachShader(prog, vs);
    sGL.AttachShader(prog, fs);
    sGL.LinkProgram(prog);

    // Shaders are reference-counted by the program; drop our references now so
    // they are freed with it.
    sGL.DeleteShader(vs);
    sGL.DeleteShader(fs);

    GLint linked = GL_FALSE;
    sGL.GetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512] = {0};
        sGL.GetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
        ALOGE("AFME: %s program link failed: %s", label, log);
        sGL.DeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ─── Motion-estimation path setup ───────────────────────────────────────────

bool haveMotionFuncs() {
    return sGL.EstimateMotion && sGL.TexStorage2D && sGL.CreateProgram &&
           sGL.GenVertexArrays && sGL.DrawArrays && sGL.Uniform2f &&
           sGL.GetUniformLocation && sGL.ActiveTexture && sGL.IsEnabled &&
           sGL.GetBooleanv && sGL.ColorMask && sGL.DepthMask && sGL.Viewport;
}

// Must run with the game's context current and a GLStateGuard in effect.
bool initMotionEstimation(AFMEState& state) {
    state.motionAttempted = true;

    if (!haveMotionFuncs()) {
        ALOGW("AFME: motion estimation unavailable (missing entrypoints) — "
              "falling back to extrapolation");
        return false;
    }

    // Block granularity is a driver property, not a guess. If the driver does
    // not know these enums it raises GL_INVALID_ENUM and leaves the values
    // alone, so clear the flag and treat the defaults as unverified.
    if (sGL.GetError) while (sGL.GetError() != GL_NO_ERROR) {}
    sGL.GetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &state.blockX);
    sGL.GetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &state.blockY);
    if (sGL.GetError && sGL.GetError() != GL_NO_ERROR) {
        ALOGW("AFME: driver rejected the ME block-size query — "
              "GL_QCOM_motion_estimation not really present");
        while (sGL.GetError() != GL_NO_ERROR) {}
        return false;
    }
    if (state.blockX <= 0 || state.blockY <= 0) {
        ALOGW("AFME: bogus ME block size %dx%d — falling back to extrapolation",
              state.blockX, state.blockY);
        return false;
    }

    GLint mvW = state.width / state.blockX;
    GLint mvH = state.height / state.blockY;
    if (mvW < 1 || mvH < 1) {
        ALOGW("AFME: surface %dx%d smaller than ME block %dx%d",
              state.width, state.height, state.blockX, state.blockY);
        return false;
    }

    state.lumaProg = buildProgram(kFullscreenVertSrc, kLumaFragSrc, "luma");
    state.warpProg = buildProgram(kFullscreenVertSrc, kWarpFragSrc, "warp");
    if (!state.lumaProg || !state.warpProg) return false;

    state.lumaSrcLoc    = sGL.GetUniformLocation(state.lumaProg, "uSrc");
    state.warpCurrLoc   = sGL.GetUniformLocation(state.warpProg, "uCurr");
    state.warpMVLoc     = sGL.GetUniformLocation(state.warpProg, "uMV");
    state.warpSizeLoc   = sGL.GetUniformLocation(state.warpProg, "uRenderSize");
    state.warpFactorLoc = sGL.GetUniformLocation(state.warpProg, "uFactor");
    state.warpMaxMVLoc  = sGL.GetUniformLocation(state.warpProg, "uMaxMV");

    state.prevLumaTex = createStorageTexture(GL_R8, state.width, state.height, GL_NEAREST);
    state.currLumaTex = createStorageTexture(GL_R8, state.width, state.height, GL_NEAREST);
    // LINEAR: the warp shader relies on bilinear filtering to upsample the
    // block field to per-pixel, so there is no separate upsample pass.
    state.mvBlockTex  = createStorageTexture(GL_RGBA16F, mvW, mvH, GL_LINEAR);

    sGL.GenFramebuffers(1, &state.lumaFBO);
    sGL.GenFramebuffers(1, &state.genFBO);
    sGL.GenVertexArrays(1, &state.vao);

    if (!state.prevLumaTex || !state.currLumaTex || !state.mvBlockTex ||
        !state.lumaFBO || !state.genFBO || !state.vao) {
        ALOGE("AFME: motion-estimation resource allocation failed");
        return false;
    }

    // Validate the luma FBO once rather than trusting it every frame.
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, state.lumaFBO);
    sGL.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, state.currLumaTex, 0);
    GLenum fbStatus = sGL.CheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    if (fbStatus != GL_FRAMEBUFFER_COMPLETE) {
        ALOGE("AFME: luma FBO incomplete (0x%x) — falling back to extrapolation",
              fbStatus);
        return false;
    }

    state.hasLumaHistory = false;
    state.motionReady = true;
    ALOGI("AFME: motion estimation ready (%dx%d, MV block %dx%d → %dx%d)",
          state.width, state.height, state.blockX, state.blockY, mvW, mvH);
    return true;
}

// Full-screen pass: render `prog` into `targetTex` at w×h.
void runFullscreenPass(AFMEState& state, GLuint fbo, GLuint targetTex,
                       GLuint prog, GLint w, GLint h) {
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    sGL.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, targetTex, 0);
    sGL.Viewport(0, 0, w, h);
    sGL.UseProgram(prog);
    sGL.BindVertexArray(state.vao);
    sGL.DrawArrays(GL_TRIANGLES, 0, 3);
}

// Convert currTex → currLumaTex. Once per real frame.
void updateLuma(AFMEState& state) {
    sGL.ActiveTexture(GL_TEXTURE0);
    sGL.BindTexture(GL_TEXTURE_2D, state.currTex);
    sGL.UseProgram(state.lumaProg);
    if (state.lumaSrcLoc >= 0) sGL.Uniform1i(state.lumaSrcLoc, 0);
    setVrsRate(GL_SHADING_RATE_2X2_PIXELS_QCOM);  // luma is downsampled input
    runFullscreenPass(state, state.lumaFBO, state.currLumaTex,
                      state.lumaProg, state.width, state.height);
    setVrsRate(GL_SHADING_RATE_1X1_PIXELS_QCOM);

    // Drop lumaFBO as the draw target before anyone samples currLumaTex.
    // Sampling a texture still attached to the *bound* framebuffer is a
    // feedback loop and undefined — and glTexEstimateMotionQCOM reads exactly
    // this texture on the very next call.
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

// Warp currTex by factor along the motion field → synthTex.
void warpFrame(AFMEState& state, float factor) {
    sGL.ActiveTexture(GL_TEXTURE0);
    sGL.BindTexture(GL_TEXTURE_2D, state.currTex);
    sGL.ActiveTexture(GL_TEXTURE1);
    sGL.BindTexture(GL_TEXTURE_2D, state.mvBlockTex);

    sGL.UseProgram(state.warpProg);
    if (state.warpCurrLoc >= 0)   sGL.Uniform1i(state.warpCurrLoc, 0);
    if (state.warpMVLoc >= 0)     sGL.Uniform1i(state.warpMVLoc, 1);
    if (state.warpSizeLoc >= 0)   sGL.Uniform2f(state.warpSizeLoc,
                                                (float)state.width,
                                                (float)state.height);
    if (state.warpFactorLoc >= 0) sGL.Uniform1f(state.warpFactorLoc, factor);
    // 12% of the diagonal: comfortably above real object motion at 60fps,
    // below the frame-wide displacement a scene cut produces.
    if (state.warpMaxMVLoc >= 0) {
        float diag = std::sqrt((float)state.width * (float)state.width +
                               (float)state.height * (float)state.height);
        sGL.Uniform1f(state.warpMaxMVLoc, diag * 0.12f);
    }

    runFullscreenPass(state, state.genFBO, state.synthTex,
                      state.warpProg, state.width, state.height);
}

void initState(AFMEState& state, EGLDisplay dpy, EGLSurface surface) {
    eglQuerySurface(dpy, surface, EGL_WIDTH, &state.width);
    eglQuerySurface(dpy, surface, EGL_HEIGHT, &state.height);

    if (state.width <= 0 || state.height <= 0) {
        ALOGW("AFME: Invalid surface dimensions %dx%d", state.width, state.height);
        return;
    }

    resolveGLFunctions();

    // Check both AFME extension AND basic GL functions
    state.extensionsAvailable = (sGL.ExtrapolateTex2D != nullptr
                                 && sGL.BlitFramebuffer != nullptr);

    if (!state.extensionsAvailable) {
        ALOGW("AFME: Required functions not available, passthrough mode");
        state.initialized = true;
        return;
    }

    state.prevTex = createTexture(state.width, state.height);
    state.currTex = createTexture(state.width, state.height);
    state.synthTex = createTexture(state.width, state.height);

    sGL.GenFramebuffers(1, &state.readFBO);
    sGL.GenFramebuffers(1, &state.drawFBO);

    state.initialized = true;
    state.hasPrevFrame = false;
    state.frameCount = 0;

    ALOGI("AFME: Initialized %dx%d (prev=%u curr=%u synth=%u)",
          state.width, state.height, state.prevTex, state.currTex, state.synthTex);
}

void captureFramebuffer(AFMEState& state, GLuint targetTex) {
    sGL.BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, state.drawFBO);
    sGL.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, targetTex, 0);
    sGL.BlitFramebuffer(0, 0, state.width, state.height,
                        0, 0, state.width, state.height,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    sGL.BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void blitTextureToFramebuffer(AFMEState& state, GLuint srcTex) {
    sGL.BindFramebuffer(GL_READ_FRAMEBUFFER, state.readFBO);
    sGL.FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, srcTex, 0);
    sGL.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    sGL.BlitFramebuffer(0, 0, state.width, state.height,
                        0, 0, state.width, state.height,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    sGL.BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void cleanupState(AFMEState& state) {
    if (sGL.DeleteTextures) {
        if (state.prevTex) { sGL.DeleteTextures(1, &state.prevTex); state.prevTex = 0; }
        if (state.currTex) { sGL.DeleteTextures(1, &state.currTex); state.currTex = 0; }
        if (state.synthTex) { sGL.DeleteTextures(1, &state.synthTex); state.synthTex = 0; }
        if (state.prevLumaTex) { sGL.DeleteTextures(1, &state.prevLumaTex); state.prevLumaTex = 0; }
        if (state.currLumaTex) { sGL.DeleteTextures(1, &state.currLumaTex); state.currLumaTex = 0; }
        if (state.mvBlockTex) { sGL.DeleteTextures(1, &state.mvBlockTex); state.mvBlockTex = 0; }
    }
    if (sGL.DeleteFramebuffers) {
        if (state.readFBO) { sGL.DeleteFramebuffers(1, &state.readFBO); state.readFBO = 0; }
        if (state.drawFBO) { sGL.DeleteFramebuffers(1, &state.drawFBO); state.drawFBO = 0; }
        if (state.lumaFBO) { sGL.DeleteFramebuffers(1, &state.lumaFBO); state.lumaFBO = 0; }
        if (state.genFBO) { sGL.DeleteFramebuffers(1, &state.genFBO); state.genFBO = 0; }
    }
    if (sGL.DeleteVertexArrays && state.vao) {
        sGL.DeleteVertexArrays(1, &state.vao); state.vao = 0;
    }
    if (sGL.DeleteProgram) {
        if (state.lumaProg) { sGL.DeleteProgram(state.lumaProg); state.lumaProg = 0; }
        if (state.warpProg) { sGL.DeleteProgram(state.warpProg); state.warpProg = 0; }
    }
    state.initialized = false;
    state.hasPrevFrame = false;
    state.motionReady = false;
    state.motionAttempted = false;
    state.hasLumaHistory = false;
}

// Publish real/generated FPS once per second.
//
// property_set() from a game process can never work — platform sepolicy has
// `neverallow all_untrusted_apps property_type:property_service set` — so the
// overlay reads these log lines instead. GameSpace (system_app, READ_LOGS)
// greps "AFME-STATS real=N gen=N total=N" off logcat at AFME:I.
void publishStats(AFMEState& state, int64_t nowNs) {
    if (state.statsBaseNs == 0) {
        state.statsBaseNs = nowNs;
        state.statsBaseReal = state.realFrames;
        state.statsBaseGen = state.genFrames;
        return;
    }
    double elapsed = (double)(nowNs - state.statsBaseNs) / 1e9;
    if (elapsed < 1.0) return;

    uint32_t realDelta = state.realFrames - state.statsBaseReal;
    uint32_t genDelta = state.genFrames - state.statsBaseGen;
    int realFps = (int)((double)realDelta / elapsed + 0.5);
    int genFps = (int)((double)genDelta / elapsed + 0.5);

    ALOGI("AFME-STATS real=%d gen=%d total=%d", realFps, genFps, realFps + genFps);

    state.statsBaseNs = nowNs;
    state.statsBaseReal = state.realFrames;
    state.statsBaseGen = state.genFrames;
}

// ─── Hooked EGL functions ───────────────────────────────────────────────────

EGLBoolean EGLAPIENTRY afme_eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    EGLFuncPointer realSwap;
    {
        std::lock_guard<std::mutex> lock(sMapMutex);
        realSwap = sFuncMap["eglSwapBuffers"];
    }

    typedef EGLBoolean (*PFNEGLSWAPBUFFERSPROC)(EGLDisplay, EGLSurface);
    auto nextSwap = reinterpret_cast<PFNEGLSWAPBUFFERSPROC>(realSwap);

    // Check every 64 frames
    static uint32_t sCheckCounter = 0;
    if ((sCheckCounter++ & 0x3F) == 0) {
        checkProperties();
    }

    if (!sEnabled.load(std::memory_order_relaxed)) {
        return nextSwap(dpy, surface);
    }

    AFMEState* state;
    {
        std::lock_guard<std::mutex> lock(sStateMutex);
        state = &sStates[surface];
    }

    if (!state->initialized) {
        initState(*state, dpy, surface);
    }

    if (!state->extensionsAvailable) {
        return nextSwap(dpy, surface);
    }

    // Check for surface resize
    GLint curW, curH;
    eglQuerySurface(dpy, surface, EGL_WIDTH, &curW);
    eglQuerySurface(dpy, surface, EGL_HEIGHT, &curH);
    if (curW != state->width || curH != state->height) {
        ALOGI("AFME: Surface resized %dx%d -> %dx%d",
              state->width, state->height, curW, curH);
        cleanupState(*state);
        initState(*state, dpy, surface);
    }

    // ═══ AFME Frame Generation Pipeline ═══

    // Adaptive generation clamp: measure the game's swap interval and never
    // queue more frames than the panel has vsync slots for. Without this,
    // BufferQueue back-pressure throttles the game itself (60fps + 2x on a
    // 60Hz-locked panel → 30fps real).
    struct timespec tsNow;
    clock_gettime(CLOCK_MONOTONIC, &tsNow);
    int64_t nowNs = (int64_t)tsNow.tv_sec * 1000000000LL + tsNow.tv_nsec;
    if (state->lastSwapNs > 0) {
        float ms = (float)(nowNs - state->lastSwapNs) / 1e6f;
        if (ms > 0.05f && ms < 200.0f) {
            state->emaFrameMs = (state->emaFrameMs <= 0.0f)
                                    ? ms : state->emaFrameMs * 0.9f + ms * 0.1f;
        }
    }
    state->lastSwapNs = nowNs;
    updateSpacingGovernor(*state);

    // Every swap the game makes is a real frame, whether or not we end up
    // generating from it. Counting it only on the generating path would report
    // real=0 for the whole time the panel has no headroom.
    state->realFrames++;

    int mult = sMultiplier.load(std::memory_order_relaxed);
    int numGenFrames = mult - 1;  // 2x→1, 3x→2, 4x→3
    if (state->emaFrameMs > 0.0f) {
        float gameFps = 1000.0f / state->emaFrameMs;
        int slots = (int)(((float)sDisplayHz.load(std::memory_order_relaxed)
                           * 1.02f) / gameFps);
        int maxGen = slots - 1;
        if (maxGen < 0) maxGen = 0;
        if (numGenFrames > maxGen) numGenFrames = maxGen;
    }

    if (numGenFrames == 0) {
        // No panel headroom: stay out of the way entirely.
        state->hasPrevFrame = false;
        state->hasLumaHistory = false;
        publishStats(*state, nowNs);
        return nextSwap(dpy, surface);
    }

    const int method = sMethod.load(std::memory_order_relaxed);

    // Everything below rewrites GL state the game owns. Save it once here and
    // restore it once before handing control back: the game cannot issue a draw
    // while we are inside its eglSwapBuffers, so there is no need to round-trip
    // the state around each present — and a save is ~13 glGet* calls, which
    // some drivers resolve by syncing with the GPU.
    GLStateGuard guard;
    guard.save();
    guard.neutralize();

    // Lazily bring up the motion-estimation path the first time it is asked
    // for. If it cannot be built we degrade to extrapolation rather than
    // dropping frame generation altogether.
    bool useMotion = false;
    if (method == FG_MOTION) {
        if (!state->motionReady && !state->motionAttempted) {
            initMotionEstimation(*state);
        }
        useMotion = state->motionReady;
    }

    // Step 1: Capture current frame
    captureFramebuffer(*state, state->currTex);

    // Step 2: For the motion path, derive this frame's luminance while the
    // real frame is still the newest thing we have. Runs once per real frame
    // regardless of multiplier, so 3x/4x do not pay for extra estimation.
    if (useMotion) {
        updateLuma(*state);
        if (state->hasLumaHistory) {
            sGL.EstimateMotion(state->prevLumaTex, state->currLumaTex,
                               state->mvBlockTex);
        }
    }

    // Step 3: Present the REAL frame first (low latency)
    EGLBoolean result = nextSwap(dpy, surface);
    if (result != EGL_TRUE) {
        guard.restore();
        return result;
    }
    {
        // Anchor for spacing the synthetic frames that follow.
        struct timespec tsReal;
        clock_gettime(CLOCK_MONOTONIC, &tsReal);
        state->realSwapNs = (int64_t)tsReal.tv_sec * 1000000000LL + tsReal.tv_nsec;
    }

    // Step 4: If we have a previous frame, synthesize and present.
    // The motion path additionally needs a luma history to have an MV field.
    const bool canGenerate = state->hasPrevFrame &&
                             (!useMotion || state->hasLumaHistory);
    if (canGenerate) {
        float userFactor = sFactorOverride.load(std::memory_order_relaxed);  // 0 = auto

        for (int i = 0; i < numGenFrames; i++) {
            // Phase within the interval is (i+1)/(numGenFrames+1) of the
            // frame interval, NOT (i+1)/mult: the adaptive clamp above may
            // have reduced generation below mult-1, and /mult then places
            // synth frames at the wrong temporal phase (e.g. 1 gen at 4x
            // produced factor=0.25 — quarter-phase motion in the half-slot).
            float autoFactor = (float)(i + 1) / (float)(numGenFrames + 1);
            float factor = (userFactor > 0.0f) ? autoFactor * userFactor : autoFactor;

            if (useMotion) {
                // Estimated motion field + our warp: artifact guards apply.
                warpFrame(*state, factor);
            } else {
                // GPU frame extrapolation: prev + curr → synthetic future frame
                sGL.ExtrapolateTex2D(state->prevTex, state->currTex,
                                     state->synthTex, factor);
            }

            // Blit synthetic frame to backbuffer and present.
            // No glFlush needed here: in the same GLES context, commands are
            // serialized by the GPU. The blit reads completed generation work.
            // eglSwapBuffers does an implicit flush before presenting.
            blitTextureToFramebuffer(*state, state->synthTex);
            spaceSyntheticFrame(*state, i, numGenFrames);
            if (sPacingEnabled.load(std::memory_order_relaxed) &&
                    sGL.PresentationTime && state->emaFrameMs > 0.0f) {
                // Ask SurfaceFlinger to latch THIS buffer at its temporal
                // slot rather than the next vsync (standard presentation-time placement).
                struct timespec tsP;
                clock_gettime(CLOCK_MONOTONIC, &tsP);
                int64_t nowPn = (int64_t)tsP.tv_sec * 1000000000LL + tsP.tv_nsec;
                int64_t offsetNs = (int64_t)(state->emaFrameMs * 1e6f *
                                             (float)(i + 1) / (float)(numGenFrames + 1))
                                   - 3000000LL;
                if (offsetNs < 0) offsetNs = 0;
                sGL.PresentationTime(dpy, surface, nowPn + offsetNs);
            }
            result = nextSwap(dpy, surface);
            state->genFrames++;
        }
    }

    drainGLErrors(*state, useMotion ? "motion path" : "extrapolate path");
    guard.restore();

    // Step 5: Rotate history (no data copy)
    GLuint tmp = state->prevTex;
    state->prevTex = state->currTex;
    state->currTex = tmp;
    state->hasPrevFrame = true;

    if (useMotion) {
        std::swap(state->prevLumaTex, state->currLumaTex);
        state->hasLumaHistory = true;
    }

    state->frameCount++;
    publishStats(*state, nowNs);

    if ((state->frameCount % 300) == 0) {
        ALOGI("AFME: %u frames generated (%dx mode, method=%s) [%dx%d]",
              state->frameCount, sMultiplier.load(),
              useMotion ? "motion" : "extrapolate",
              state->width, state->height);
    }

    return result;
}

EGLBoolean EGLAPIENTRY afme_eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
    {
        std::lock_guard<std::mutex> lock(sStateMutex);
        auto it = sStates.find(surface);
        if (it != sStates.end()) {
            cleanupState(it->second);
            sStates.erase(it);
            ALOGI("AFME: Cleaned up state for surface %p", surface);
        }
    }

    EGLFuncPointer realDestroy;
    {
        std::lock_guard<std::mutex> lock(sMapMutex);
        realDestroy = sFuncMap["eglDestroySurface"];
    }
    typedef EGLBoolean (*PFNEGLDESTROYSURF)(EGLDisplay, EGLSurface);
    return reinterpret_cast<PFNEGLDESTROYSURF>(realDestroy)(dpy, surface);
}

EGLBoolean EGLAPIENTRY afme_eglSwapInterval(EGLDisplay dpy, EGLint interval) {
    EGLFuncPointer realSwapInterval;
    {
        std::lock_guard<std::mutex> lock(sMapMutex);
        realSwapInterval = sFuncMap["eglSwapInterval"];
    }
    typedef EGLBoolean (*PFNEGLSWAPINTERVAL)(EGLDisplay, EGLint);
    auto next = reinterpret_cast<PFNEGLSWAPINTERVAL>(realSwapInterval);

    if (sEnabled.load(std::memory_order_relaxed) && interval > 0) {
        ALOGD("AFME: Overriding swap interval %d -> 0 for frame gen", interval);
        return next(dpy, 0);
    }
    return next(dpy, interval);
}

// ─── EGL Layer Interface ────────────────────────────────────────────────────

EGLFuncPointer eglGPA(const char* funcName) {
    #define GETPROCADDR(func) if(!strcmp(funcName, #func)) { \
        return (EGLFuncPointer)afme_##func; }

    GETPROCADDR(eglSwapBuffers);
    GETPROCADDR(eglDestroySurface);
    GETPROCADDR(eglSwapInterval);

    #undef GETPROCADDR
    return nullptr;
}

void glesLayer_InitializeLayer(
        void* layer_id,
        PFNEGLGETNEXTLAYERPROCADDRESSPROC get_next_layer_proc_address) {
    sLayerId = layer_id;
    sGetNextLayerProcAddress = get_next_layer_proc_address;
    checkProperties();
    ALOGI("AFME: Layer initialized (enabled=%d multiplier=%d method=%d)",
          sEnabled.load(), sMultiplier.load(), sMethod.load());
}

EGLFuncPointer glesLayer_GetLayerProcAddress(
        const char* funcName, EGLFuncPointer next) {
    EGLFuncPointer entry = eglGPA(funcName);
    if (entry != nullptr) {
        std::lock_guard<std::mutex> lock(sMapMutex);
        sFuncMap[std::string(funcName)] = next;
        return entry;
    }
    return next;
}

} // anonymous namespace

// ─── Exported symbols ───────────────────────────────────────────────────────
extern "C" {

__attribute__((visibility("default")))
EGLAPI void AndroidGLESLayer_Initialize(
        void* layer_id,
        PFNEGLGETNEXTLAYERPROCADDRESSPROC get_next_layer_proc_address) {
    return (void)glesLayer_InitializeLayer(layer_id, get_next_layer_proc_address);
}

__attribute__((visibility("default")))
EGLAPI void* AndroidGLESLayer_GetProcAddress(
        const char* funcName, EGLFuncPointer next) {
    return (void*)glesLayer_GetLayerProcAddress(funcName, next);
}

} // extern "C"
