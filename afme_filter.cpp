/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME color filter — see afme_filter.h.
 */
#include "afme_filter.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>

#include <android/log.h>
#include <cutils/properties.h>

#ifndef LOG_TAG
#define LOG_TAG "AFME"
#endif
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace afme {
namespace {

std::mutex        gFilterLock;
FilterStack       gStack;
std::atomic<bool> gFilterOn{false};
std::atomic<bool> gFilterLive{false};

/**
 * Parse up to @p n comma-separated floats from a property.
 *
 * Fields left empty or missing keep whatever @p out already holds, so a shorter
 * property (an older GameSpace writing fewer values) degrades to defaults for
 * the tail rather than zeroing it.
 */
void parseCsv(const char* name, float* out, int n) {
    char value[PROPERTY_VALUE_MAX];
    property_get(name, value, "");
    if (value[0] == '\0') return;

    const char* p = value;
    for (int i = 0; i < n && p && *p; i++) {
        char* endp = nullptr;
        const float v = strtof(p, &endp);
        if (endp != p) out[i] = v;
        p = strchr(p, ',');
        if (p) p++;
    }
}

// ─── Shared shader pieces ───────────────────────────────────────────────────

// Fullscreen triangle from gl_VertexID — no VBO, so we never touch the game's
// array buffer or attribute state.
const char* kVertSrc = R"(#version 300 es
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
})";

// ── Half-resolution scene copy. Mipmapped afterwards; mip 3-4 is clarity's
//    blur reference, and mip 0 feeds the bloom bright-pass.
const char* kDownSrc = R"(#version 300 es
precision mediump float;
uniform sampler2D uSrc;
in vec2 vUV;
out vec4 outColor;
void main() { outColor = vec4(texture(uSrc, vUV).rgb, 1.0); })";

// ── Bloom bright-pass with a soft knee, so a highlight fading through the
//    threshold ramps in instead of popping.
const char* kBrightSrc = R"(#version 300 es
precision mediump float;
uniform sampler2D uSrc;
uniform vec4 uParams;   // threshold, knee, unused, unused
in vec2 vUV;
out vec4 outColor;
void main() {
    vec3 c = texture(uSrc, vUV).rgb;
    float l = max(c.r, max(c.g, c.b));
    float knee = max(uParams.y, 1e-3);
    float w = clamp((l - uParams.x) / knee, 0.0, 1.0);
    outColor = vec4(c * w * w, 1.0);
})";

// ── Stage B: the screen-space nodes, in stack order. Pointwise, no
//    neighbourhood — which is why running it on every present is affordable.
const char* kStageBSrc = R"(#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform int  uCount;
uniform int  uKind[8];
uniform vec4 uP0[8];
uniform float uTime;
in vec2 vUV;
out vec4 outColor;

// Cheap hash — no texture fetch, and decorrelated enough per frame that the
// grain does not visibly tile or crawl.
float hash(vec2 p, float t) {
    p = fract(p * vec2(443.897, 441.423) + t);
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}

void main() {
    vec3 c = texture(uSrc, vUV).rgb;

    for (int i = 0; i < 8; i++) {
        if (i >= uCount) break;
        int k = uKind[i];
        vec4 a = uP0[i];

        if (k == 8) {           // vignette
            // Aspect-agnostic falloff: a circular one on a 20:9 panel would
            // crush the sides long before it touched the top.
            vec2 d = (vUV - 0.5) * 2.0;
            c *= mix(1.0, clamp(1.0 - dot(d, d) * 0.5, 0.0, 1.0), a.x);
        } else if (k == 9) {    // film grain
            float n = hash(vUV, uTime) - 0.5;
            // Weighted by (1-luma) so it sits in shadows and mid-tones the way
            // film does, instead of speckling blown highlights.
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
            c += n * a.x * 0.25 * (1.0 - l * 0.8);
        } else if (k == 10) {   // letterbox
            if (vUV.y < a.x || vUV.y > 1.0 - a.x) c = vec3(0.0);
        }
    }

    outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
})";

// ── Stage A: every non-screen-space node, evaluated IN STACK ORDER.
//
// The loop bound is constant and the branch selector is a uniform, so every
// invocation in a draw walks the same path — no divergence, and a stack of two
// filters costs two filters, not eleven.
const char* kStageASrc = R"(#version 300 es
precision highp float;

uniform mediump sampler2D uSrc;
uniform mediump sampler2D uScene;   // half-res + mips (clarity)
uniform mediump sampler2D uBloom;   // quarter-res + mips (bloom)

uniform int  uCount;
uniform int  uKind[8];
uniform vec4 uP0[8];   // params 0..3
uniform vec4 uP1[8];   // params 4..5

in  vec2 vUV;
out vec4 outColor;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
float luma(vec3 c) { return dot(c, kLuma); }

vec3 hueRotate(vec3 c, float a) {
    const vec3 k = vec3(0.57735027);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

vec3 colorBlind(vec3 c, int mode, float strength) {
    if (mode == 0 || strength <= 0.0) return c;
    mat3 sim;
    if (mode == 1) {
        sim = mat3(0.567, 0.433, 0.000, 0.558, 0.442, 0.000, 0.000, 0.242, 0.758);
    } else if (mode == 2) {
        sim = mat3(0.625, 0.375, 0.000, 0.700, 0.300, 0.000, 0.000, 0.300, 0.700);
    } else {
        sim = mat3(0.950, 0.050, 0.000, 0.000, 0.433, 0.567, 0.000, 0.475, 0.525);
    }
    vec3 seen = c * sim;
    vec3 err  = c - seen;
    vec3 fix  = vec3(0.0, err.r * 0.7 + err.g, err.r * 0.7 + err.b);
    return clamp(c + fix * strength, 0.0, 1.0);
}

// ACES filmic approximation (Narkowicz). Its shoulder is what makes HDR toning
// read as "more range" rather than just "darker".
vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    vec3 orig = texture(uSrc, vUV).rgb;
    vec3 c = orig;

    for (int i = 0; i < 8; i++) {
        if (i >= uCount) break;
        int k = uKind[i];
        vec4 a = uP0[i];
        vec4 b = uP1[i];

        if (k == 0) {
            // ── Exposure: exposure, contrast, highlights, shadows, gamma ──
            c *= exp2(a.x);
            c = pow(max(c, 0.0), vec3(1.0 / max(b.x, 1e-3)));
            c = (c - 0.5) * a.y + 0.5;
            float l = luma(clamp(c, 0.0, 1.0));
            c *= 1.0 + a.w * (1.0 - smoothstep(0.0, 0.5, l))
                     + a.z * smoothstep(0.5, 1.0, l);
        } else if (k == 1) {
            // ── Color: temperature, tint, vibrance, saturation, hue ──
            c *= vec3(1.0 + 0.30 * a.x, 1.0 - 0.15 * a.y, 1.0 - 0.30 * a.x);
            if (abs(b.x) > 1e-4) c = hueRotate(c, b.x);
            float lc = luma(c);
            // Vibrance before saturation: it must see the pre-saturation spread,
            // or a high saturation setting starves it of its target pixels.
            if (abs(a.z) > 1e-4) {
                float mx = max(c.r, max(c.g, c.b));
                float mn = min(c.r, min(c.g, c.b));
                c = mix(vec3(lc), c, 1.0 + a.z * (1.0 - (mx - mn)));
            }
            c = mix(vec3(lc), c, a.w);
        } else if (k == 2) {
            // ── Details: sharpen, clarity, HDR toning, bloom, threshold ──
            if (a.x > 1e-4) {
                // Contrast-adaptive (CAS style) on the SOURCE neighbourhood:
                // grading nine taps would cost nine times the ALU, and sharpen
                // commutes with a monotone grade closely enough to be invisible.
                vec2 ts = 1.0 / vec2(textureSize(uSrc, 0));
                vec3 t0 = texture(uSrc, vUV + vec2(0.0, -ts.y)).rgb;
                vec3 t1 = texture(uSrc, vUV + vec2(-ts.x, 0.0)).rgb;
                vec3 t2 = texture(uSrc, vUV + vec2( ts.x, 0.0)).rgb;
                vec3 t3 = texture(uSrc, vUV + vec2(0.0,  ts.y)).rgb;
                vec3 mn = min(c, min(min(t0, t1), min(t2, t3)));
                vec3 mx = max(c, max(max(t0, t1), max(t2, t3)));
                // Sharpen less where the local range is already wide — this is
                // what stops CAS ringing on high-contrast edges.
                vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, 1e-4), 0.0, 1.0));
                vec3 w = amp * (-0.125 * mix(1.0, 0.2, a.x));
                c = clamp((c + (t0 + t1 + t2 + t3) * w) / (1.0 + 4.0 * w), 0.0, 1.0);
            }
            if (abs(a.y) > 1e-4) {
                c = clamp(c + (c - textureLod(uScene, vUV, 4.0).rgb) * a.y,
                          0.0, 1.0);
            }
            if (a.w > 1e-4) {
                vec3 bl = textureLod(uBloom, vUV, 1.0).rgb * 0.40
                        + textureLod(uBloom, vUV, 2.0).rgb * 0.30
                        + textureLod(uBloom, vUV, 3.0).rgb * 0.20
                        + textureLod(uBloom, vUV, 4.0).rgb * 0.10;
                c += bl * a.w;
            }
            if (a.z > 1e-4) {
                // The only step that leaves the sampled space: a tonemap curve
                // on already-encoded values is meaningless, so linearize first.
                vec3 lin = pow(max(c, 0.0), vec3(2.2));
                c = mix(c, pow(acesTonemap(lin * 1.8), vec3(1.0 / 2.2)), a.z);
            }
        } else if (k == 3) {
            // ── Levels: black point, white point, brightness ──
            c = (c - a.x) / max(a.y - a.x, 1e-3);
            c += a.z;
        } else if (k == 4) {
            c = mix(c, vec3(luma(c)), a.x);           // black & white
        } else if (k == 5) {
            c = mix(c, luma(c) * vec3(1.07, 0.87, 0.66), a.x);   // sepia
        } else if (k == 6) {
            c = colorBlind(c, int(a.x + 0.5), a.y);
        } else if (k == 7) {
            // Splitscreen compares against the untouched frame, which is why it
            // lives here and not with the other screen-space effects.
            if (vUV.x < a.x) c = orig;
        }

        c = clamp(c, 0.0, 1.0);
    }

    outColor = vec4(c, 1.0);
})";

}  // namespace

// ─── Parameters ─────────────────────────────────────────────────────────────

bool FilterStack::hasScreenSpace() const {
    for (int i = 0; i < count; i++)
        if (isScreenSpace(nodes[i].kind)) return true;
    return false;
}

bool FilterStack::needsMips() const {
    for (int i = 0; i < count; i++) {
        if (nodes[i].kind != kDetails) continue;
        // p1 = clarity, p3 = bloom; both read the downsampled chain.
        if (fabsf(nodes[i].p[1]) > 1e-4f || nodes[i].p[3] > 1e-4f) return true;
    }
    return false;
}

void pollFilterProps() {
    char value[PROPERTY_VALUE_MAX];
    property_get("persist.sys.afme.filter", value, "0");
    gFilterOn.store(value[0] == '1');
    property_get("persist.sys.afme.filter.live", value, "0");
    gFilterLive.store(value[0] == '1');
    if (!gFilterOn.load()) return;

    property_get("persist.sys.afme.filter.n", value, "0");
    int n = atoi(value);
    if (n < 0) n = 0;
    if (n > FilterStack::kMaxNodes) n = FilterStack::kMaxNodes;

    FilterStack st;
    for (int i = 0; i < n; i++) {
        char name[64];
        snprintf(name, sizeof(name), "persist.sys.afme.filter.s%d", i);

        float f[7] = { -1.0f, 0, 0, 0, 0, 0, 0 };
        parseCsv(name, f, 7);

        const int kind = (int)(f[0] + 0.5f);
        // A slot the app has not written yet reads as -1 and is skipped rather
        // than defaulting to kind 0, which would flash an Exposure filter the
        // user never added.
        if (kind < 0 || kind >= kKindCount) continue;

        FilterNode& node = st.nodes[st.count];
        node.kind = kind;
        for (int j = 0; j < 6; j++) node.p[j] = f[j + 1];
        st.count++;
    }

    std::lock_guard<std::mutex> lock(gFilterLock);
    gStack = st;
}

bool filterEnabled() { return gFilterOn.load(std::memory_order_relaxed); }

bool filterLive() { return gFilterLive.load(std::memory_order_relaxed); }

FilterStack filterStack() {
    std::lock_guard<std::mutex> lock(gFilterLock);
    return gStack;
}

// ─── GL dispatch ────────────────────────────────────────────────────────────

bool FilterGL::complete() const {
    return CreateShader && ShaderSource && CompileShader && GetShaderiv &&
           GetShaderInfoLog && DeleteShader && CreateProgram && AttachShader &&
           LinkProgram && GetProgramiv && GetProgramInfoLog && DeleteProgram &&
           UseProgram && GetUniformLocation && Uniform1i && Uniform4f &&
           GenFramebuffers && DeleteFramebuffers && BindFramebuffer &&
           FramebufferTexture2D && GenVertexArrays && DeleteVertexArrays &&
           BindVertexArray && Uniform1f && GenTextures && DeleteTextures &&
           TexStorage2D &&
           GenerateMipmap && ActiveTexture && BindTexture && TexParameteri &&
           Viewport && DrawArrays && Disable;
}

// ─── The passes ─────────────────────────────────────────────────────────────

namespace {

GLuint compile(const FilterGL& gl, GLenum type, const char* src,
               const char* what) {
    GLuint sh = gl.CreateShader(type);
    if (!sh) return 0;
    gl.ShaderSource(sh, 1, &src, nullptr);
    gl.CompileShader(sh);
    GLint ok = 0;
    gl.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {};
        gl.GetShaderInfoLog(sh, sizeof(log), nullptr, log);
        ALOGE("AFME: filter shader '%s' compile failed: %s", what, log);
        gl.DeleteShader(sh);
        return 0;
    }
    return sh;
}

GLuint link(const FilterGL& gl, GLuint vs, const char* fragSrc,
            const char* what) {
    GLuint fs = compile(gl, GL_FRAGMENT_SHADER, fragSrc, what);
    if (!fs) return 0;

    GLuint prog = gl.CreateProgram();
    gl.AttachShader(prog, vs);
    gl.AttachShader(prog, fs);
    gl.LinkProgram(prog);

    GLint linked = 0;
    gl.GetProgramiv(prog, GL_LINK_STATUS, &linked);
    gl.DeleteShader(fs);
    if (!linked) {
        char log[1024] = {};
        gl.GetProgramInfoLog(prog, sizeof(log), nullptr, log);
        ALOGE("AFME: filter program '%s' link failed: %s", what, log);
        gl.DeleteProgram(prog);
        return 0;
    }
    return prog;
}

/** Mip levels for a texture of this size, so TexStorage2D allocates the chain. */
GLsizei mipLevels(uint32_t w, uint32_t h) {
    GLsizei n = 1;
    uint32_t m = (w > h) ? w : h;
    while (m > 1) { m >>= 1; n++; }
    return n;
}

}  // namespace

bool Filter::init(const FilterGL& gl) {
    if (progA_) return true;
    if (failed_) return false;

    if (!gl.complete()) {
        ALOGE("AFME: filter unavailable — GL dispatch incomplete");
        failed_ = true;
        return false;
    }
    gl_ = &gl;

    GLuint vs = compile(gl, GL_VERTEX_SHADER, kVertSrc, "vertex");
    if (!vs) { failed_ = true; return false; }

    progA_      = link(gl, vs, kStageASrc, "stageA");
    progB_      = link(gl, vs, kStageBSrc, "stageB");
    progDown_   = link(gl, vs, kDownSrc, "downsample");
    progBright_ = link(gl, vs, kBrightSrc, "bright");
    gl.DeleteShader(vs);

    if (!progA_ || !progB_ || !progDown_ || !progBright_) {
        destroy();
        failed_ = true;
        return false;
    }

    aSrc_    = gl.GetUniformLocation(progA_, "uSrc");
    aScene_  = gl.GetUniformLocation(progA_, "uScene");
    aBloom_  = gl.GetUniformLocation(progA_, "uBloom");
    aCount_  = gl.GetUniformLocation(progA_, "uCount");
    bSrc_    = gl.GetUniformLocation(progB_, "uSrc");
    bCount_  = gl.GetUniformLocation(progB_, "uCount");
    bTime_   = gl.GetUniformLocation(progB_, "uTime");

    // Array element locations are queried individually: the spec only
    // guarantees that "name[0]" resolves, not that later elements are
    // consecutive, and some drivers do lay them out with gaps.
    for (int i = 0; i < FilterStack::kMaxNodes; i++) {
        char n[32];
        snprintf(n, sizeof(n), "uKind[%d]", i);
        aKind_[i] = gl.GetUniformLocation(progA_, n);
        bKind_[i] = gl.GetUniformLocation(progB_, n);
        snprintf(n, sizeof(n), "uP0[%d]", i);
        aP0_[i] = gl.GetUniformLocation(progA_, n);
        bP0_[i] = gl.GetUniformLocation(progB_, n);
        snprintf(n, sizeof(n), "uP1[%d]", i);
        aP1_[i] = gl.GetUniformLocation(progA_, n);
    }

    downSrc_      = gl.GetUniformLocation(progDown_, "uSrc");
    brightSrc_    = gl.GetUniformLocation(progBright_, "uSrc");
    brightParams_ = gl.GetUniformLocation(progBright_, "uParams");

    gl.GenFramebuffers(1, &fbo_);
    gl.GenVertexArrays(1, &vao_);

    ALOGI("AFME: color filter initialized (A=%u B=%u down=%u bright=%u)",
          progA_, progB_, progDown_, progBright_);
    return true;
}

bool Filter::buildMips(uint32_t w, uint32_t h) {
    const FilterGL& gl = *gl_;
    const uint32_t hw = (w >= 2) ? w / 2 : 1;
    const uint32_t hh = (h >= 2) ? h / 2 : 1;

    if (sceneTex_ && (mipW_ != hw || mipH_ != hh)) {
        gl.DeleteTextures(1, &sceneTex_);
        gl.DeleteTextures(1, &bloomTex_);
        sceneTex_ = 0;
        bloomTex_ = 0;
    }

    if (!sceneTex_) {
        const uint32_t qw = (hw >= 2) ? hw / 2 : 1;
        const uint32_t qh = (hh >= 2) ? hh / 2 : 1;

        gl.GenTextures(1, &sceneTex_);
        gl.BindTexture(GL_TEXTURE_2D, sceneTex_);
        gl.TexStorage2D(GL_TEXTURE_2D, mipLevels(hw, hh), GL_RGBA8,
                        (GLsizei)hw, (GLsizei)hh);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                         GL_LINEAR_MIPMAP_LINEAR);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        gl.GenTextures(1, &bloomTex_);
        gl.BindTexture(GL_TEXTURE_2D, bloomTex_);
        gl.TexStorage2D(GL_TEXTURE_2D, mipLevels(qw, qh), GL_RGBA8,
                        (GLsizei)qw, (GLsizei)qh);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                         GL_LINEAR_MIPMAP_LINEAR);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        gl.BindTexture(GL_TEXTURE_2D, 0);
        mipW_ = hw;
        mipH_ = hh;

        if (!sceneTex_ || !bloomTex_) return false;
        ALOGI("AFME: filter mip chains %ux%u / %ux%u", hw, hh, qw, qh);
    }
    return true;
}

void Filter::runPass(GLuint prog, GLuint dstTex, uint32_t w, uint32_t h) {
    const FilterGL& gl = *gl_;
    gl.UseProgram(prog);
    gl.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, dstTex, 0);
    gl.Viewport(0, 0, (GLsizei)w, (GLsizei)h);
    gl.Disable(GL_DEPTH_TEST);
    gl.Disable(GL_BLEND);
    gl.Disable(GL_SCISSOR_TEST);
    gl.BindVertexArray(vao_);
    gl.DrawArrays(GL_TRIANGLES, 0, 3);
}

void Filter::applyStageA(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                         const FilterStack& st) {
    if (!progA_ || !gl_) return;
    const FilterGL& gl = *gl_;

    const bool mips = st.needsMips() && buildMips(w, h);

    if (mips) {
        // Half-res copy, then the driver's mip reduction gives clarity a wide,
        // cheap blur reference for free.
        gl.UseProgram(progDown_);
        gl.ActiveTexture(GL_TEXTURE0);
        gl.BindTexture(GL_TEXTURE_2D, srcTex);
        gl.Uniform1i(downSrc_, 0);
        runPass(progDown_, sceneTex_, mipW_, mipH_);
        gl.BindTexture(GL_TEXTURE_2D, sceneTex_);
        gl.GenerateMipmap(GL_TEXTURE_2D);

        float threshold = 0.75f;
        bool wantBloom = false;
        for (int i = 0; i < st.count; i++) {
            if (st.nodes[i].kind == kDetails && st.nodes[i].p[3] > 1e-4f) {
                threshold = st.nodes[i].p[4];
                wantBloom = true;
            }
        }
        if (wantBloom) {
            gl.UseProgram(progBright_);
            gl.ActiveTexture(GL_TEXTURE0);
            gl.BindTexture(GL_TEXTURE_2D, sceneTex_);
            gl.Uniform1i(brightSrc_, 0);
            gl.Uniform4f(brightParams_, threshold, 0.25f, 0.0f, 0.0f);
            const uint32_t qw = (mipW_ >= 2) ? mipW_ / 2 : 1;
            const uint32_t qh = (mipH_ >= 2) ? mipH_ / 2 : 1;
            runPass(progBright_, bloomTex_, qw, qh);
            gl.BindTexture(GL_TEXTURE_2D, bloomTex_);
            gl.GenerateMipmap(GL_TEXTURE_2D);
        }
    }

    gl.UseProgram(progA_);

    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, srcTex);
    // NEAREST on the 1:1 tap: LINEAR would let the driver resolve sample
    // positions half a texel off and soften the whole image for free. The
    // sharpen taps offset by exact texels, so they need no filtering either.
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.Uniform1i(aSrc_, 0);

    // Bind the chains unconditionally: sampling an incomplete texture is
    // undefined even on a branch the uniforms never take.
    gl.ActiveTexture(GL_TEXTURE1);
    gl.BindTexture(GL_TEXTURE_2D, mips ? sceneTex_ : srcTex);
    gl.Uniform1i(aScene_, 1);
    gl.ActiveTexture(GL_TEXTURE2);
    gl.BindTexture(GL_TEXTURE_2D, (mips && bloomTex_) ? bloomTex_ : srcTex);
    gl.Uniform1i(aBloom_, 2);
    gl.ActiveTexture(GL_TEXTURE0);

    // Upload the stage-A nodes in stack order, skipping the screen-space ones
    // (they belong to stage B). Order is what makes reordering in the UI mean
    // something, so it is preserved exactly.
    int n = 0;
    for (int i = 0; i < st.count && n < FilterStack::kMaxNodes; i++) {
        const FilterNode& node = st.nodes[i];
        if (isScreenSpace(node.kind)) continue;
        float clarity = node.p[1];
        float bloom   = node.p[3];
        if (node.kind == kDetails && !mips) {
            // The chains were not built, so these two would sample garbage.
            clarity = 0.0f;
            bloom   = 0.0f;
        }
        gl.Uniform1i(aKind_[n], node.kind);
        gl.Uniform4f(aP0_[n], node.p[0], clarity, node.p[2], bloom);
        gl.Uniform4f(aP1_[n], node.p[4], node.p[5], 0.0f, 0.0f);
        n++;
    }
    gl.Uniform1i(aCount_, n);

    // No VRS on any of this, deliberately. 2x2 shading would quantize the grade
    // to 2x2 blocks — plainly visible as banding on gradients, and it would
    // defeat the sharpen pass outright.
    runPass(progA_, dstTex, w, h);

    gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    gl.BindVertexArray(0);
    gl.UseProgram(0);
}

void Filter::applyStageB(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                         const FilterStack& st, uint64_t frameIdx) {
    if (!progB_ || !gl_) return;
    const FilterGL& gl = *gl_;

    gl.UseProgram(progB_);
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, srcTex);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl.Uniform1i(bSrc_, 0);

    int n = 0;
    for (int i = 0; i < st.count && n < FilterStack::kMaxNodes; i++) {
        const FilterNode& node = st.nodes[i];
        if (!isScreenSpace(node.kind)) continue;
        gl.Uniform1i(bKind_[n], node.kind);
        gl.Uniform4f(bP0_[n], node.p[0], node.p[1], node.p[2], node.p[3]);
        n++;
    }
    gl.Uniform1i(bCount_, n);

    // Wrap the frame counter well inside float precision: at 120fps a raw
    // counter would lose its low bits within the hour and the grain would
    // gradually freeze.
    gl.Uniform1f(bTime_, (float)(frameIdx % 4096u) * 0.0173f);

    runPass(progB_, dstTex, w, h);

    gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    gl.BindVertexArray(0);
    gl.UseProgram(0);
}

void Filter::destroy() {
    if (!gl_) return;
    if (progA_)      gl_->DeleteProgram(progA_);
    if (progB_)      gl_->DeleteProgram(progB_);
    if (progDown_)   gl_->DeleteProgram(progDown_);
    if (progBright_) gl_->DeleteProgram(progBright_);
    if (fbo_)        gl_->DeleteFramebuffers(1, &fbo_);
    if (vao_)        gl_->DeleteVertexArrays(1, &vao_);
    if (sceneTex_)   gl_->DeleteTextures(1, &sceneTex_);
    if (bloomTex_)   gl_->DeleteTextures(1, &bloomTex_);
    progA_ = progB_ = progDown_ = progBright_ = 0;
    fbo_ = vao_ = sceneTex_ = bloomTex_ = 0;
    mipW_ = mipH_ = 0;
}

}  // namespace afme
