/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME color filter — real-time color grading inside the game's own present.
 *
 * ── Where this runs in the pipeline ────────────────────────────────────────
 *
 * AFME never writes the real frame; it only reads it as generation input. A
 * filter applied where AFME already writes would therefore alternate
 * filtered/unfiltered at panel rate — a 120Hz strobe. So the filter writes back
 * into the presented image, and it does so BEFORE generation:
 *
 *     present image ──copy──> stage ──filter──> curr ──copy──> present image
 *                                                 └──> frame generation source
 *
 * Grading before generation means synthetic frames inherit the grade for free:
 * the cost is one full-screen pass per REAL frame at any multiplier, not one
 * per present.
 *
 * ── Why only pointwise operations ──────────────────────────────────────────
 *
 * Every operation here is a pure function of a single input color. That is what
 * makes it safe to bake into the frame-generation source: monotone per-pixel
 * transforms preserve the luma gradients block motion estimation keys on.
 *
 * Spatially-varying and stochastic effects (film grain, vignette, bloom,
 * sharpen) are deliberately NOT here. Grain would be read as motion and produce
 * generation artifacts; a screen-space vignette baked in pre-generation gets
 * warped by the motion field and smears with the camera. Those need a second
 * post-generation stage — see COLOR_FILTER_PLAN.md phase 3.
 *
 * ── Color space ────────────────────────────────────────────────────────────
 *
 * The grade runs on the values as sampled, which for an sRGB swapchain means
 * display-encoded (gamma) space, NOT linear. This is deliberate for v1: it is
 * what ReShade and NVIDIA Freestyle do, it is what the sliders feel like users
 * expect, and it avoids a double-applied transfer function. Whether the EGLImage
 * import hands back encoded or linear values has NOT been verified on device
 * (see the grey-ramp test in COLOR_FILTER_PLAN.md); working in the sampled space
 * is correct either way, which is why v1 does it.
 */
#ifndef AFME_FILTER_H
#define AFME_FILTER_H

#include <GLES3/gl3.h>

#include <cstdint>

namespace afme {

// ─── Parameters ─────────────────────────────────────────────────────────────

/**
 * One complete grade. Defaults are the identity — a freshly constructed
 * FilterParams changes nothing, and isIdentity() reports it so the caller can
 * skip the pass entirely.
 */
struct FilterParams {
    // Tone
    float exposure   = 0.0f;   // stops, applied as exp2()
    float brightness = 0.0f;   // additive offset
    float contrast   = 1.0f;   // around 0.5 pivot
    float gamma      = 1.0f;
    float black      = 0.0f;   // input black point
    float white      = 1.0f;   // input white point
    float shadows    = 0.0f;   // -1..1, luma-masked lift
    float highlights = 0.0f;   // -1..1, luma-masked gain

    // Color
    float saturation  = 1.0f;
    float vibrance    = 0.0f;   // saturation weighted by inverse existing sat
    float temperature = 0.0f;   // -1 cool .. +1 warm
    float tint        = 0.0f;   // -1 green .. +1 magenta
    float hue         = 0.0f;   // radians, rotation about the grey axis
    float mono        = 0.0f;   // 0..1 blend to luminance
    float sepia       = 0.0f;   // 0..1 blend to sepia

    // Accessibility
    int   cbMode     = 0;       // 0 off, 1 protan, 2 deutan, 3 tritan
    float cbStrength = 0.0f;    // 0..1

    // Global
    float intensity = 1.0f;     // master wet/dry against the original
    float split     = -1.0f;    // <0 off; else compare divider in 0..1 (left
                                // of the divider stays original)

    /** True when applying this would be a no-op, so the pass can be skipped. */
    bool isIdentity() const;
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
 *   persist.sys.afme.filter.fx     cbMode,cbStrength,split
 *
 * A torn read across two of these costs at most one frame of mixed settings,
 * which is cosmetic. Missing or malformed fields keep their default.
 */
void pollFilterProps();

/** Master switch — persist.sys.afme.filter. */
bool filterEnabled();

/**
 * True while GameSpace has its filter panel open (persist.sys.afme.filter.live).
 *
 * The layers normally poll properties every 64 presents, which is up to ~1.5s —
 * far too laggy to drag a slider against. When this is set the caller should
 * re-read the filter properties EVERY present instead. Four property_get calls
 * are shared-memory reads costing ~1us total, so this is affordable for the
 * seconds a user spends in the panel, and it reverts to the cheap path as soon
 * as the panel closes.
 *
 * Reads a cached atomic, so calling it every present is free.
 */
bool filterLive();

/** Latest parsed parameters. Safe to call every present. */
FilterParams filterParams();

// ─── GL dispatch ────────────────────────────────────────────────────────────

/**
 * The GL entry points the filter pass needs.
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

    void (*ActiveTexture)(GLenum) = nullptr;
    void (*BindTexture)(GLenum, GLuint) = nullptr;
    void (*TexParameteri)(GLenum, GLenum, GLint) = nullptr;
    void (*Viewport)(GLint, GLint, GLsizei, GLsizei) = nullptr;
    void (*DrawArrays)(GLenum, GLint, GLsizei) = nullptr;
    void (*Disable)(GLenum) = nullptr;

    /** Every pointer above is non-null. */
    bool complete() const;
};

// ─── The pass ───────────────────────────────────────────────────────────────

/**
 * One compiled filter program plus its FBO. One instance per presentation
 * target, built lazily the first time the filter is actually wanted so a
 * session that never enables it pays nothing.
 *
 * All methods require the owning GL context to be current.
 */
class Filter {
public:
    /** Compile and link. Safe to call repeatedly; only the first does work. */
    bool init(const FilterGL& gl);

    /** Release GL objects. Requires the same context that init() used. */
    void destroy();

    bool ready() const { return program_ != 0; }

    /** True once init() has failed, so callers stop retrying every frame. */
    bool failed() const { return failed_; }

    /**
     * Grade @p srcTex into @p dstTex. Both must be @p w by @p h.
     *
     * The caller is responsible for neutral pipeline state (no depth, blend,
     * scissor or stencil) — in the GLES layer that is GLStateGuard, in the
     * Vulkan layer the private context is always neutral.
     */
    void apply(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
               const FilterParams& p);

private:
    const FilterGL* gl_ = nullptr;
    GLuint program_ = 0;
    GLuint fbo_     = 0;
    GLuint vao_     = 0;
    bool   failed_  = false;

    GLint uSrc_ = -1, uTone1_ = -1, uTone2_ = -1;
    GLint uColor1_ = -1, uColor2_ = -1, uMisc_ = -1;
};

}  // namespace afme

#endif  // AFME_FILTER_H
