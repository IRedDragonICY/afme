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

/** Filter kinds, ordinals shared with GameSpace's FilterKind enum. */
enum FilterKind {
    kExposure    = 0,
    kColor       = 1,
    kDetails     = 2,
    kLevels      = 3,
    kBlackWhite  = 4,
    kSepia       = 5,
    kColorBlind  = 6,
    // Splitscreen belongs to stage A even though it is screen-space: it
    // compares against the UNFILTERED frame, and only stage A still has it.
    kSplitscreen = 7,
    kVignette    = 8,   // true screen space from here down
    kFilmGrain   = 9,
    kLetterbox   = 10,
    kKindCount   = 11,
};

/** Screen-space kinds must run AFTER generation — see the header comment. */
inline bool isScreenSpace(int kind) { return kind >= kVignette; }

/** One filter in the stack. */
struct FilterNode {
    int   kind = -1;
    float p[6] = {0, 0, 0, 0, 0, 0};
};

/**
 * An ordered stack, evaluated in sequence.
 *
 * Order is real: Color-then-Details and Details-then-Color produce different
 * pictures, exactly as they do in Freestyle. The one imposed constraint is that
 * screen-space nodes are pulled out into stage B regardless of where the user
 * put them, because they cannot survive frame generation; relative order within
 * each stage is preserved.
 */
struct FilterStack {
    static constexpr int kMaxNodes = 8;

    FilterNode nodes[kMaxNodes];
    int count = 0;

    /** Nothing to apply at all. */
    bool empty() const { return count == 0; }

    /** Any node that has to run after generation. */
    bool hasScreenSpace() const;

    /** Any stage-A node needing the downsampled scene chain (clarity/bloom). */
    bool needsMips() const;
};

/**
 * Read the filter properties.
 *
 * One property per stack slot, because PROPERTY_VALUE_MAX is 92 bytes and
 * persist.* gets no exemption — a whole stack would never fit in one:
 *
 *   persist.sys.afme.filter        "0"/"1"
 *   persist.sys.afme.filter.n      node count
 *   persist.sys.afme.filter.s0..s7 "kind,p0,p1,p2,p3,p4,p5"
 *
 * A torn read across slots costs at most one frame of mixed settings.
 */
void pollFilterProps();

/** Master switch — persist.sys.afme.filter. */
bool filterEnabled();

/**
 * True while GameSpace has its filter panel open (persist.sys.afme.filter.live).
 *
 * The layers normally poll properties every 64 presents, which is up to ~1.5s —
 * far too laggy to drag a slider against. When set, the caller should re-read
 * the filter properties EVERY present. The reads are shared-memory lookups
 * costing ~1us, affordable for the seconds a user spends in the panel, and it
 * reverts to the cheap path when the panel closes.
 *
 * Reads a cached atomic, so calling it every present is free.
 */
bool filterLive();

/** Latest parsed stack. Safe to call every present. */
FilterStack filterStack();

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
    void (*Uniform1f)(GLint, GLfloat) = nullptr;

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
     * Stage A: every non-screen-space node, in stack order.
     *
     * @p srcTex → @p dstTex, both @p w by @p h. Builds and consumes the
     * downsampled scene chain internally when clarity or bloom is active.
     */
    void applyStageA(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                     const FilterStack& s);

    /**
     * Stage B: every screen-space node, in stack order, @p srcTex → @p dstTex.
     *
     * Call once per PRESENT — on the real frame and on each synthetic frame —
     * so screen-space effects stay pinned to the screen instead of being warped
     * by the motion field. @p frameIdx animates the grain.
     */
    void applyStageB(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                     const FilterStack& s, uint64_t frameIdx);

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

    GLint aSrc_ = -1, aScene_ = -1, aBloom_ = -1, aCount_ = -1;
    GLint aKind_[FilterStack::kMaxNodes] = {};
    GLint aP0_[FilterStack::kMaxNodes] = {};
    GLint aP1_[FilterStack::kMaxNodes] = {};
    GLint bSrc_ = -1, bCount_ = -1, bTime_ = -1;
    GLint bKind_[FilterStack::kMaxNodes] = {};
    GLint bP0_[FilterStack::kMaxNodes] = {};
    GLint downSrc_ = -1, brightSrc_ = -1, brightParams_ = -1;
};

}  // namespace afme

#endif  // AFME_FILTER_H
