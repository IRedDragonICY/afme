/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME (Adreno Frame Motion Engine) — EGL Layer for GPU Frame Generation
 *
 * Architecture:
 * ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌─────────┐
 * │  Game    │────>│ AFME Layer│────>│ Adreno   │────>│ Display │
 * │ (30fps) │     │ eglSwap   │     │ GPU      │     │ (60fps) │
 * └──────────┘     │ intercept │     │ AFME HW  │     └─────────┘
 *                  └───────────┘     └──────────┘
 *
 * Flow per frame:
 * 1. Game renders frame N to default FBO
 * 2. AFME layer intercepts eglSwapBuffers
 * 3. Copy default FBO → current texture
 * 4. If previous frame exists:
 *    a. glExtrapolateTex2DQCOM(prev, curr, synthetic, factor)
 *    b. Present real frame N via real eglSwapBuffers
 *    c. Blit synthetic frame to default FBO
 *    d. Present synthetic frame via real eglSwapBuffers
 * 5. Save current as previous
 *
 * Control:
 *   persist.sys.afme.enable = 1/0          (enable/disable)
 *   persist.sys.afme.multiplier = 2/3/4    (frame multiplier)
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <ctime>

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

// ─── GL function pointer typedefs (resolved at runtime via eglGetProcAddress) ─
// GLES 3.0 functions — NOT available at link time for vendor EGL layers
typedef void (*PFNGLGENFRAMEBUFFERSPROC)(GLsizei, GLuint*);
typedef void (*PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei, const GLuint*);
typedef void (*PFNGLBINDFRAMEBUFFERPROC)(GLenum, GLuint);
typedef void (*PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef void (*PFNGLBLITFRAMEBUFFERPROC)(GLint, GLint, GLint, GLint,
                                          GLint, GLint, GLint, GLint,
                                          GLbitfield, GLenum);
typedef void (*PFNGLGENTEXTURESPROC)(GLsizei, GLuint*);
typedef void (*PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint*);
typedef void (*PFNGLBINDTEXTUREPROC)(GLenum, GLuint);
typedef void (*PFNGLTEXIMAGE2DPROC_)(GLenum, GLint, GLint, GLsizei, GLsizei,
                                      GLint, GLenum, GLenum, const void*);
typedef void (*PFNGLTEXPARAMETERIPROC_)(GLenum, GLenum, GLint);
typedef void (*PFNGLFINISHPROC)(void);
typedef void (*PFNGLFLUSHPROC)(void);

// QCOM extensions
typedef void (*PFNGLEXTRAPOLATETEX2DQCOMPROC)(GLuint, GLuint, GLuint, GLfloat);
typedef void (*PFNGLTEXESTIMATEMOTIONQCOMPROC)(GLuint, GLuint, GLuint);

// ─── Layer state ────────────────────────────────────────────────────────────
namespace {

// Resolved GL function pointers
struct GLFuncs {
    PFNGLGENFRAMEBUFFERSPROC     GenFramebuffers = nullptr;
    PFNGLDELETEFRAMEBUFFERSPROC  DeleteFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC     BindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = nullptr;
    PFNGLBLITFRAMEBUFFERPROC     BlitFramebuffer = nullptr;
    PFNGLGENTEXTURESPROC         GenTextures = nullptr;
    PFNGLDELETETEXTURESPROC      DeleteTextures = nullptr;
    PFNGLBINDTEXTUREPROC         BindTexture = nullptr;
    PFNGLTEXIMAGE2DPROC_         TexImage2D = nullptr;
    PFNGLTEXPARAMETERIPROC_      TexParameteri = nullptr;
    PFNGLFINISHPROC              Finish = nullptr;
    PFNGLFLUSHPROC               Flush = nullptr;
    PFNGLEXTRAPOLATETEX2DQCOMPROC ExtrapolateTex2D = nullptr;
    PFNGLTEXESTIMATEMOTIONQCOMPROC EstimateMotion = nullptr;
    bool resolved = false;
};

GLFuncs sGL;

// EGL layer function map
std::unordered_map<std::string, EGLFuncPointer> sFuncMap;
std::mutex sMapMutex;

// Layer initialization state
void* sLayerId = nullptr;
PFNEGLGETNEXTLAYERPROCADDRESSPROC sGetNextLayerProcAddress = nullptr;

// Per-surface AFME state
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
};

std::unordered_map<EGLSurface, AFMEState> sStates;
std::mutex sStateMutex;

std::atomic<bool> sEnabled{false};
std::atomic<int> sMultiplier{2};
std::atomic<float> sFactorOverride{0.0f};  // 0 = auto
std::atomic<int> sDisplayHz{120};          // panel rate from GameStateDispatcher

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
}

void resolveGLFunctions() {
    if (sGL.resolved) return;

    // Resolve all GL functions via eglGetProcAddress
    sGL.GenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)
        eglGetProcAddress("glGenFramebuffers");
    sGL.DeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)
        eglGetProcAddress("glDeleteFramebuffers");
    sGL.BindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)
        eglGetProcAddress("glBindFramebuffer");
    sGL.FramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)
        eglGetProcAddress("glFramebufferTexture2D");
    sGL.BlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC)
        eglGetProcAddress("glBlitFramebuffer");
    sGL.GenTextures = (PFNGLGENTEXTURESPROC)
        eglGetProcAddress("glGenTextures");
    sGL.DeleteTextures = (PFNGLDELETETEXTURESPROC)
        eglGetProcAddress("glDeleteTextures");
    sGL.BindTexture = (PFNGLBINDTEXTUREPROC)
        eglGetProcAddress("glBindTexture");
    sGL.TexImage2D = (PFNGLTEXIMAGE2DPROC_)
        eglGetProcAddress("glTexImage2D");
    sGL.TexParameteri = (PFNGLTEXPARAMETERIPROC_)
        eglGetProcAddress("glTexParameteri");
    sGL.Finish = (PFNGLFINISHPROC)
        eglGetProcAddress("glFinish");
    sGL.Flush = (PFNGLFLUSHPROC)
        eglGetProcAddress("glFlush");

    // QCOM extensions
    sGL.ExtrapolateTex2D = (PFNGLEXTRAPOLATETEX2DQCOMPROC)
        eglGetProcAddress("glExtrapolateTex2DQCOM");
    sGL.EstimateMotion = (PFNGLTEXESTIMATEMOTIONQCOMPROC)
        eglGetProcAddress("glTexEstimateMotionQCOM");

    sGL.resolved = true;

    if (sGL.ExtrapolateTex2D) {
        ALOGI("AFME: GL_QCOM_frame_extrapolation resolved (%p)",
              sGL.ExtrapolateTex2D);
    } else {
        ALOGW("AFME: GL_QCOM_frame_extrapolation NOT available");
    }

    if (sGL.BlitFramebuffer) {
        ALOGI("AFME: glBlitFramebuffer resolved (%p)", sGL.BlitFramebuffer);
    } else {
        ALOGW("AFME: glBlitFramebuffer NOT available — cannot operate");
    }
}

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
    }
    if (sGL.DeleteFramebuffers) {
        if (state.readFBO) { sGL.DeleteFramebuffers(1, &state.readFBO); state.readFBO = 0; }
        if (state.drawFBO) { sGL.DeleteFramebuffers(1, &state.drawFBO); state.drawFBO = 0; }
    }
    state.initialized = false;
    state.hasPrevFrame = false;
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

    // Check every 60 frames
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
        return nextSwap(dpy, surface);
    }

    // Step 1: Capture current frame
    captureFramebuffer(*state, state->currTex);

    // Step 2: Present the REAL frame first (low latency)
    EGLBoolean result = nextSwap(dpy, surface);
    if (result != EGL_TRUE) {
        return result;
    }

    // Step 3: If we have previous frame, generate and present synthetic
    if (state->hasPrevFrame) {
        float userFactor = sFactorOverride.load(std::memory_order_relaxed);  // 0 = auto

        for (int i = 0; i < numGenFrames; i++) {
            // Auto: (i+1)/mult | User override: autoFactor * userFactor
            float autoFactor = (float)(i + 1) / (float)mult;
            float factor = (userFactor > 0.0f) ? autoFactor * userFactor : autoFactor;

            // GPU frame extrapolation: prev + curr → synthetic future frame
            sGL.ExtrapolateTex2D(state->prevTex, state->currTex,
                                 state->synthTex, factor);

            // Blit synthetic frame to backbuffer and present
            // No glFlush needed here: in same GLES context, commands are
            // serialized by the GPU. The blit reads completed extrapolation.
            // eglSwapBuffers does an implicit flush before presenting.
            blitTextureToFramebuffer(*state, state->synthTex);
            result = nextSwap(dpy, surface);
        }
    }

    // Step 4: Swap textures (no data copy)
    GLuint tmp = state->prevTex;
    state->prevTex = state->currTex;
    state->currTex = tmp;
    state->hasPrevFrame = true;
    state->frameCount++;

    if ((state->frameCount % 300) == 0) {
        ALOGI("AFME: %u frames generated (%dx mode) [%dx%d]",
              state->frameCount, sMultiplier.load(), state->width, state->height);
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
    ALOGI("AFME: Layer initialized (enabled=%d multiplier=%d)",
          sEnabled.load(), sMultiplier.load());
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
