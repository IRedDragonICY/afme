/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME color filter — real-time grading and post effects inside the game's
 * own present.
 *
 * ── Where this runs in the pipeline ────────────────────────────────────────
 *
 * AFME never writes the real frame; it only reads it as generation input. A
 * filter applied where AFME already writes would therefore alternate
 * filtered/unfiltered at panel rate — a 120Hz strobe. So the filter writes back
 * into the presented image.
 *
 *   present image ─copy─> stage ─[A]─> curr ─[B]─> present ─copy─> present image
 *                                       └────────> frame generation source
 *                                                        │
 *                          synth frames ─[B]─> present ───┘
 *
 * ── The two stages, and why the split is load-bearing ──────────────────────
 *
 * STAGE A runs BEFORE generation, so its result IS the generation source and
 * synthetic frames inherit it for free — one pass per REAL frame at any
 * multiplier. Everything in stage A is either pointwise or a deterministic
 * function of frame content, which is what makes that safe: such transforms
 * preserve the luma gradients block motion estimation keys on, and anything
 * derived from the content (a bloom halo, a sharpened edge) correctly warps
 * along with the content it came from.
 *
 *   grade · HDR toning · sharpen · clarity · bloom
 *
 * STAGE B runs AFTER generation, on every present. It holds exactly the
 * effects that are fixed in SCREEN space or random per frame — the two things
 * that must never reach motion estimation:
 *
 *   vignette · film grain · letterbox
 *
 * Film grain is per-frame noise; fed to the estimator it reads as motion and
 * produces generation artifacts. A vignette or letterbox baked in pre-generation
 * gets warped by the motion field and smears with the camera instead of staying
 * pinned to the screen. Stage B is pointwise in screen space, so it costs ~10
 * ALU and needs no neighbourhood — cheap enough to run per present.
 *
 * When no stage B effect is set the stage is skipped entirely and the pipeline
 * is exactly as it was: one pass, no extra buffers.
 *
 * ── Color space ────────────────────────────────────────────────────────────
 *
 * The grade runs on the values as sampled, which for an sRGB swapchain means
 * display-encoded (gamma) space, NOT linear. Deliberate: it is what ReShade and
 * NVIDIA Freestyle do, it is what the sliders feel like users expect, and it
 * avoids a double-applied transfer function. Whether the EGLImage import hands
 * back encoded or linear values has NOT been verified on device (see the
 * grey-ramp test in COLOR_FILTER_PLAN.md); working in the sampled space is
 * correct either way.
 *
 * The one exception is HDR toning, which linearizes explicitly before the
 * tonemap and re-encodes after, because a tonemap curve applied to already-
 * encoded values is meaningless.
 */
#ifndef AFME_FILTER_H
#define AFME_FILTER_H

#include <GLES3/gl3.h>

#include <cstdint>

namespace afme {

// ─── Parameters ─────────────────────────────────────────────────────────────

/**
 * One complete look. Defaults are the identity — a freshly constructed
 * FilterParams changes nothing, and isIdentity() reports it so the caller can
 * skip everything.
 */
struct FilterParams {
    // ── Stage A: tone ──
    float exposure   = 0.0f;   // stops, applied as exp2()
    float brightness = 0.0f;
    float contrast   = 1.0f;   // around 0.5 pivot
    float gamma      = 1.0f;
    float black      = 0.0f;
    float white      = 1.0f;
    float shadows    = 0.0f;   // -1..1, luma-masked lift
    float highlights = 0.0f;   // -1..1, luma-masked gain

    // ── Stage A: color ──
    float saturation  = 1.0f;
    float vibrance    = 0.0f;
    float temperature = 0.0f;  // -1 cool .. +1 warm
    float tint        = 0.0f;  // -1 green .. +1 magenta
    float hue         = 0.0f;  // radians about the grey axis
    float mono        = 0.0f;
    float sepia       = 0.0f;
    float intensity   = 1.0f;  // master wet/dry against the original

    // ── Stage A: detail and range ──
    float sharpen         = 0.0f;  // 0..1, contrast-adaptive
    float clarity         = 0.0f;  // -1..1, local contrast (wide unsharp mask)
    float bloom           = 0.0f;  // 0..1 intensity
    float bloomThreshold  = 0.75f; // luminance where bloom starts
    float hdrToning       = 0.0f;  // 0..1 blend toward an ACES-style tonemap

    // ── Stage B: screen space ──
    float vignette  = 0.0f;   // 0..1 strength
    float grain     = 0.0f;   // 0..1 amount
    float letterbox = 0.0f;   // 0..0.25, fraction of height masked each side

    // ── Accessibility / utility ──
    int   cbMode     = 0;      // 0 off, 1 protan, 2 deutan, 3 tritan
    float cbStrength = 0.0f;
    float split      = -1.0f;  // <0 off; compare divider in 0..1

    /** True when applying this would be a no-op. */
    bool isIdentity() const;

    /** Any effect that must run AFTER generation. */
    bool hasStageB() const;

    /** Any stage A effect that needs the downsampled scene chain. */
    bool needsMips() const;
};

/**
 * Read the filter properties. Grouped rather than one-per-value because
 * PROPERTY_VALUE_MAX is 92 bytes and persist.* properties get no exemption:
 *
 *   persist.sys.afme.filter        "0"/"1"
 *   persist.sys.afme.filter.tone   exposure,brightness,contrast,gamma,
 *                                  black,white,shadows,highlights
 *   persist.sys.afme.filter.color  saturation,vibrance,temperature,tint,
 *                                  hue,mono,sepia,intensity
 *   persist.sys.afme.filter.fx     cbMode,cbStrength,split,sharpen,clarity,
 *                                  bloom,bloomThreshold,hdrToning
 *   persist.sys.afme.filter.fx2    vignette,grain,letterbox
 *
 * A torn read across these costs at most one frame of mixed settings. Missing
 * or malformed fields keep their default.
 */
void pollFilterProps();

/** Master switch — persist.sys.afme.filter. */
bool filterEnabled();

/**
 * True while GameSpace has its filter panel open (persist.sys.afme.filter.live).
 *
 * The layers normally poll properties every 64 presents, which is up to ~1.5s —
 * far too laggy to drag a slider against. When set, the caller should re-read
 * the filter properties EVERY present. The property reads are shared-memory
 * lookups costing ~1us total, affordable for the seconds a user spends in the
 * panel, and it reverts to the cheap path when the panel closes.
 *
 * Reads a cached atomic, so calling it every present is free.
 */
bool filterLive();

/** Latest parsed parameters. Safe to call every present. */
FilterParams filterParams();

// ─── GL dispatch ────────────────────────────────────────────────────────────

/**
 * The GL entry points the filter needs.
 *
 * Passed in rather than called directly because the two layers obtain GL
 * differently: the Vulkan layer links libGLESv3, while the GLES layer resolves
 * everything through eglGetProcAddress and links only libEGL. Direct calls here
 * would be an undefined symbol in the latter under -Wl,--no-undefined.
 */
struct FilterGL {
    GLuint (*CreateShader)(GLenum) = nullptr;
    void (*ShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*) = nullptr;
    void (*CompileShader)(GLuint) = nullptr;
    void (*GetShaderiv)(GLuint, GLenum, GLint*) = nullptr;
    void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = nullptr;
    void (*DeleteShader)(GLuint) = nullptr;

    GLuint (*CreateProgram)() = nullptr;
    void (*AttachShader)(GLuint, GLuint) = nullptr;
    void (*LinkProgram)(GLuint) = nullptr;
    void (*GetProgramiv)(GLuint, GLenum, GLint*) = nullptr;
    void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*) = nullptr;
    void (*DeleteProgram)(GLuint) = nullptr;
    void (*UseProgram)(GLuint) = nullptr;

    GLint (*GetUniformLocation)(GLuint, const GLchar*) = nullptr;
    void (*Uniform1i)(GLint, GLint) = nullptr;
    void (*Uniform4f)(GLint, GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;

    void (*GenFramebuffers)(GLsizei, GLuint*) = nullptr;
    void (*DeleteFramebuffers)(GLsizei, const GLuint*) = nullptr;
    void (*BindFramebuffer)(GLenum, GLuint) = nullptr;
    void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;

    void (*GenVertexArrays)(GLsizei, GLuint*) = nullptr;
    void (*DeleteVertexArrays)(GLsizei, const GLuint*) = nullptr;
    void (*BindVertexArray)(GLuint) = nullptr;

    void (*GenTextures)(GLsizei, GLuint*) = nullptr;
    void (*DeleteTextures)(GLsizei, const GLuint*) = nullptr;
    void (*TexStorage2D)(GLenum, GLsizei, GLenum, GLsizei, GLsizei) = nullptr;
    void (*GenerateMipmap)(GLenum) = nullptr;

    void (*ActiveTexture)(GLenum) = nullptr;
    void (*BindTexture)(GLenum, GLuint) = nullptr;
    void (*TexParameteri)(GLenum, GLenum, GLint) = nullptr;
    void (*Viewport)(GLint, GLint, GLsizei, GLsizei) = nullptr;
    void (*DrawArrays)(GLenum, GLint, GLsizei) = nullptr;
    void (*Disable)(GLenum) = nullptr;

    /** Every pointer above is non-null. */
    bool complete() const;
};

// ─── The passes ─────────────────────────────────────────────────────────────

/**
 * Compiled filter programs plus their scratch buffers. One instance per
 * presentation target, built lazily the first time the filter is wanted, so a
 * session that never enables it pays nothing.
 *
 * All methods require the owning GL context to be current and neutral pipeline
 * state (no depth, blend, scissor or stencil) — in the GLES layer that is
 * GLStateGuard; the Vulkan layer's private context is always neutral.
 */
class Filter {
public:
    /** Compile and link. Safe to call repeatedly; only the first does work. */
    bool init(const FilterGL& gl);

    /** Release GL objects. Requires the same context that init() used. */
    void destroy();

    bool ready() const { return progA_ != 0; }

    /** True once init() has failed, so callers stop retrying every frame. */
    bool failed() const { return failed_; }

    /**
     * Stage A: grade + HDR toning + sharpen + clarity + bloom.
     *
     * @p srcTex → @p dstTex, both @p w by @p h. Builds and consumes the
     * downsampled scene chain internally when clarity or bloom is active.
     */
    void applyStageA(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                     const FilterParams& p);

    /**
     * Stage B: vignette + grain + letterbox, @p srcTex → @p dstTex.
     *
     * Call once per PRESENT — on the real frame and on each synthetic frame —
     * so screen-space effects stay pinned to the screen instead of being warped
     * by the motion field. @p frameIdx animates the grain.
     */
    void applyStageB(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                     const FilterParams& p, uint64_t frameIdx);

private:
    bool buildMips(uint32_t w, uint32_t h);
    void runPass(GLuint prog, GLuint dstTex, uint32_t w, uint32_t h);

    const FilterGL* gl_ = nullptr;
    bool failed_ = false;

    GLuint progA_ = 0;      // grade + detail
    GLuint progB_ = 0;      // screen-space post
    GLuint progDown_ = 0;   // half-res scene copy (mip source)
    GLuint progBright_ = 0; // bright-pass for bloom

    GLuint fbo_ = 0;
    GLuint vao_ = 0;

    // Downsampled chains, allocated only when clarity or bloom is used.
    GLuint sceneTex_ = 0;   // half res + mips — clarity's blur reference
    GLuint bloomTex_ = 0;   // quarter res + mips — bloom source
    uint32_t mipW_ = 0, mipH_ = 0;

    GLint aSrc_ = -1, aScene_ = -1, aBloom_ = -1;
    GLint aTone1_ = -1, aTone2_ = -1, aColor1_ = -1, aColor2_ = -1;
    GLint aDetail_ = -1, aMisc_ = -1;
    GLint bSrc_ = -1, bParams_ = -1;
    GLint downSrc_ = -1, brightSrc_ = -1, brightParams_ = -1;
};

}  // namespace afme

#endif  // AFME_FILTER_H
