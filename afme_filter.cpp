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
FilterParams      gParams;
std::atomic<bool> gFilterOn{false};
std::atomic<bool> gFilterLive{false};

bool nearly(float a, float b) { return fabsf(a - b) < 1e-4f; }

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

// ── Stage B: screen-space only. Pointwise, no neighbourhood — this is why it
//    is affordable on every present.
const char* kStageBSrc = R"(#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec4 uParams;   // vignette, grain, letterbox, time
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

    if (uParams.x > 1e-4) {
        // Distance from centre, aspect-agnostic: a circular falloff on a
        // 20:9 panel would crush the sides long before the top.
        vec2 d = (vUV - 0.5) * 2.0;
        float r = dot(d, d) * 0.5;
        c *= mix(1.0, clamp(1.0 - r, 0.0, 1.0), uParams.x);
    }

    if (uParams.y > 1e-4) {
        float n = hash(vUV, uParams.w) - 0.5;
        // Scale grain by (1-luma) so it sits in the shadows and mid-tones the
        // way film does, instead of speckling blown highlights.
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c += n * uParams.y * 0.25 * (1.0 - l * 0.8);
    }

    if (uParams.z > 1e-4 && (vUV.y < uParams.z || vUV.y > 1.0 - uParams.z)) {
        c = vec3(0.0);
    }

    outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
})";

// ── Stage A: grade + HDR toning + sharpen + clarity + bloom.
//
// Branches are on uniforms, so every invocation in a draw takes the same path
// and the GPU does not diverge — the cost of a disabled effect is one compare.
const char* kStageASrc = R"(#version 300 es
precision highp float;

uniform mediump sampler2D uSrc;
uniform mediump sampler2D uScene;   // half-res + mips (clarity)
uniform mediump sampler2D uBloom;   // quarter-res + mips (bloom)

uniform vec4 uTone1;    // exposure, brightness, contrast, gamma
uniform vec4 uTone2;    // black, white, shadows, highlights
uniform vec4 uColor1;   // saturation, vibrance, temperature, tint
uniform vec4 uColor2;   // hue, mono, sepia, intensity
uniform vec4 uDetail;   // sharpen, clarity, bloom, hdrToning
uniform vec4 uMisc;     // cbMode, cbStrength, split, unused

in  vec2 vUV;
out vec4 outColor;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
float luma(vec3 c) { return dot(c, kLuma); }

vec3 hueRotate(vec3 c, float a) {
    const vec3 k = vec3(0.57735027);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

vec3 whiteBalance(vec3 c, float temp, float tint) {
    vec3 g = vec3(1.0 + 0.30 * temp,
                  1.0 - 0.15 * tint,
                  1.0 - 0.30 * temp);
    return c * g;
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

// ACES filmic approximation (Narkowicz). Cheap, and its shoulder is what makes
// HDR toning read as "more range" rather than just "darker".
vec3 acesTonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

void main() {
    vec3 orig = texture(uSrc, vUV).rgb;
    vec3 c = orig;

    // ── Sharpen (contrast-adaptive, AMD CAS style) ──
    // Runs on the SOURCE neighbourhood rather than the graded result: grading
    // nine taps would cost nine times the ALU, and sharpen commutes with a
    // monotone grade closely enough that nobody can see the difference.
    if (uDetail.x > 1e-4) {
        vec2 ts = 1.0 / vec2(textureSize(uSrc, 0));
        vec3 a = texture(uSrc, vUV + vec2(0.0, -ts.y)).rgb;
        vec3 b = texture(uSrc, vUV + vec2(-ts.x, 0.0)).rgb;
        vec3 d = texture(uSrc, vUV + vec2( ts.x, 0.0)).rgb;
        vec3 e = texture(uSrc, vUV + vec2(0.0,  ts.y)).rgb;
        vec3 mn = min(c, min(min(a, b), min(d, e)));
        vec3 mx = max(c, max(max(a, b), max(d, e)));
        // Sharpen less where the local range is already wide, which is what
        // keeps CAS from ringing on high-contrast edges.
        vec3 amp = clamp(min(mn, 1.0 - mx) / max(mx, 1e-4), 0.0, 1.0);
        amp = sqrt(amp);
        float peak = -0.125 * mix(1.0, 0.2, uDetail.x);
        vec3 w = amp * peak;
        c = clamp((c + (a + b + d + e) * w) / (1.0 + 4.0 * w), 0.0, 1.0);
    }

    // ── Clarity: wide unsharp mask against the mip chain ──
    if (abs(uDetail.y) > 1e-4) {
        vec3 wide = textureLod(uScene, vUV, 4.0).rgb;
        c = clamp(c + (c - wide) * uDetail.y, 0.0, 1.0);
    }

    // ── Bloom: sum several mips of the bright-pass ──
    if (uDetail.z > 1e-4) {
        vec3 b = textureLod(uBloom, vUV, 1.0).rgb * 0.40
               + textureLod(uBloom, vUV, 2.0).rgb * 0.30
               + textureLod(uBloom, vUV, 3.0).rgb * 0.20
               + textureLod(uBloom, vUV, 4.0).rgb * 0.10;
        c += b * uDetail.z;
    }

    // ── HDR toning ──
    // The only part of the chain that leaves the sampled space: a tonemap curve
    // applied to already-encoded values is meaningless, so linearize, tonemap,
    // re-encode.
    if (uDetail.w > 1e-4) {
        vec3 lin = pow(max(c, 0.0), vec3(2.2));
        vec3 mapped = pow(acesTonemap(lin * 1.8), vec3(1.0 / 2.2));
        c = mix(c, mapped, uDetail.w);
    }

    // ── Tone ──
    c *= exp2(uTone1.x);
    c += uTone1.y;
    c = (c - uTone2.x) / max(uTone2.y - uTone2.x, 1e-3);
    c = pow(max(c, 0.0), vec3(1.0 / max(uTone1.w, 1e-3)));
    c = (c - 0.5) * uTone1.z + 0.5;

    float l = luma(clamp(c, 0.0, 1.0));
    float sMask = 1.0 - smoothstep(0.0, 0.5, l);
    float hMask = smoothstep(0.5, 1.0, l);
    c *= 1.0 + uTone2.z * sMask + uTone2.w * hMask;

    // ── Color ──
    c = whiteBalance(c, uColor1.z, uColor1.w);
    if (abs(uColor2.x) > 1e-4) c = hueRotate(c, uColor2.x);

    float lc = luma(c);
    // Vibrance before saturation: it must see the pre-saturation spread, or a
    // high saturation setting starves it of the low-saturation pixels it targets.
    if (abs(uColor1.y) > 1e-4) {
        float mx2 = max(c.r, max(c.g, c.b));
        float mn2 = min(c.r, min(c.g, c.b));
        c = mix(vec3(lc), c, 1.0 + uColor1.y * (1.0 - (mx2 - mn2)));
    }
    c = mix(vec3(lc), c, uColor1.x);
    c = mix(c, vec3(luma(c)), uColor2.y);

    if (uColor2.z > 1e-4) {
        float sl = luma(c);
        c = mix(c, sl * vec3(1.07, 0.87, 0.66), uColor2.z);
    }

    c = colorBlind(c, int(uMisc.x + 0.5), uMisc.y);

    c = mix(orig, c, uColor2.w);
    c = clamp(c, 0.0, 1.0);

    // Compare divider last, so it shows the true before/after.
    if (uMisc.z >= 0.0 && vUV.x < uMisc.z) c = orig;

    outColor = vec4(c, 1.0);
})";

}  // namespace

// ─── Parameters ─────────────────────────────────────────────────────────────

bool FilterParams::hasStageB() const {
    return vignette > 1e-4f || grain > 1e-4f || letterbox > 1e-4f;
}

bool FilterParams::needsMips() const {
    return fabsf(clarity) > 1e-4f || bloom > 1e-4f;
}

bool FilterParams::isIdentity() const {
    return nearly(exposure, 0.0f) && nearly(brightness, 0.0f) &&
           nearly(contrast, 1.0f) && nearly(gamma, 1.0f) &&
           nearly(black, 0.0f) && nearly(white, 1.0f) &&
           nearly(shadows, 0.0f) && nearly(highlights, 0.0f) &&
           nearly(saturation, 1.0f) && nearly(vibrance, 0.0f) &&
           nearly(temperature, 0.0f) && nearly(tint, 0.0f) &&
           nearly(hue, 0.0f) && nearly(mono, 0.0f) && nearly(sepia, 0.0f) &&
           nearly(intensity, 1.0f) &&
           nearly(sharpen, 0.0f) && nearly(clarity, 0.0f) &&
           nearly(bloom, 0.0f) && nearly(hdrToning, 0.0f) &&
           !hasStageB() &&
           (cbMode == 0 || nearly(cbStrength, 0.0f)) &&
           split < 0.0f;
}

void pollFilterProps() {
    char value[PROPERTY_VALUE_MAX];
    property_get("persist.sys.afme.filter", value, "0");
    gFilterOn.store(value[0] == '1');
    property_get("persist.sys.afme.filter.live", value, "0");
    gFilterLive.store(value[0] == '1');
    if (!gFilterOn.load()) return;

    FilterParams p;
    float tone[8]  = { p.exposure, p.brightness, p.contrast, p.gamma,
                       p.black, p.white, p.shadows, p.highlights };
    float color[8] = { p.saturation, p.vibrance, p.temperature, p.tint,
                       p.hue, p.mono, p.sepia, p.intensity };
    float fx[8]    = { (float)p.cbMode, p.cbStrength, p.split, p.sharpen,
                       p.clarity, p.bloom, p.bloomThreshold, p.hdrToning };
    float fx2[3]   = { p.vignette, p.grain, p.letterbox };

    parseCsv("persist.sys.afme.filter.tone", tone, 8);
    parseCsv("persist.sys.afme.filter.color", color, 8);
    parseCsv("persist.sys.afme.filter.fx", fx, 8);
    parseCsv("persist.sys.afme.filter.fx2", fx2, 3);

    p.exposure = tone[0]; p.brightness = tone[1];
    p.contrast = tone[2]; p.gamma      = tone[3];
    p.black    = tone[4]; p.white      = tone[5];
    p.shadows  = tone[6]; p.highlights = tone[7];

    p.saturation  = color[0]; p.vibrance  = color[1];
    p.temperature = color[2]; p.tint      = color[3];
    p.hue         = color[4]; p.mono      = color[5];
    p.sepia       = color[6]; p.intensity = color[7];

    p.cbMode         = (int)(fx[0] + 0.5f);
    p.cbStrength     = fx[1];
    p.split          = fx[2];
    p.sharpen        = fx[3];
    p.clarity        = fx[4];
    p.bloom          = fx[5];
    p.bloomThreshold = fx[6];
    p.hdrToning      = fx[7];

    p.vignette  = fx2[0];
    p.grain     = fx2[1];
    p.letterbox = fx2[2];

    // A zero white point would divide the image away; a zero gamma divides by
    // zero in the exponent. Clamp rather than trust the caller.
    if (p.white - p.black < 1e-3f) { p.black = 0.0f; p.white = 1.0f; }
    if (p.gamma < 1e-3f) p.gamma = 1.0f;
    if (p.cbMode < 0 || p.cbMode > 3) p.cbMode = 0;
    // A letterbox past a quarter each side leaves nothing to look at.
    if (p.letterbox > 0.25f) p.letterbox = 0.25f;
    if (p.letterbox < 0.0f) p.letterbox = 0.0f;

    std::lock_guard<std::mutex> lock(gFilterLock);
    gParams = p;
}

bool filterEnabled() { return gFilterOn.load(std::memory_order_relaxed); }

bool filterLive() { return gFilterLive.load(std::memory_order_relaxed); }

FilterParams filterParams() {
    std::lock_guard<std::mutex> lock(gFilterLock);
    return gParams;
}

// ─── GL dispatch ────────────────────────────────────────────────────────────

bool FilterGL::complete() const {
    return CreateShader && ShaderSource && CompileShader && GetShaderiv &&
           GetShaderInfoLog && DeleteShader && CreateProgram && AttachShader &&
           LinkProgram && GetProgramiv && GetProgramInfoLog && DeleteProgram &&
           UseProgram && GetUniformLocation && Uniform1i && Uniform4f &&
           GenFramebuffers && DeleteFramebuffers && BindFramebuffer &&
           FramebufferTexture2D && GenVertexArrays && DeleteVertexArrays &&
           BindVertexArray && GenTextures && DeleteTextures && TexStorage2D &&
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
    aTone1_  = gl.GetUniformLocation(progA_, "uTone1");
    aTone2_  = gl.GetUniformLocation(progA_, "uTone2");
    aColor1_ = gl.GetUniformLocation(progA_, "uColor1");
    aColor2_ = gl.GetUniformLocation(progA_, "uColor2");
    aDetail_ = gl.GetUniformLocation(progA_, "uDetail");
    aMisc_   = gl.GetUniformLocation(progA_, "uMisc");

    bSrc_    = gl.GetUniformLocation(progB_, "uSrc");
    bParams_ = gl.GetUniformLocation(progB_, "uParams");

    downSrc_     = gl.GetUniformLocation(progDown_, "uSrc");
    brightSrc_   = gl.GetUniformLocation(progBright_, "uSrc");
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
                         const FilterParams& p) {
    if (!progA_ || !gl_) return;
    const FilterGL& gl = *gl_;

    bool mips = p.needsMips() && buildMips(w, h);

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

        if (p.bloom > 1e-4f) {
            gl.UseProgram(progBright_);
            gl.ActiveTexture(GL_TEXTURE0);
            gl.BindTexture(GL_TEXTURE_2D, sceneTex_);
            gl.Uniform1i(brightSrc_, 0);
            gl.Uniform4f(brightParams_, p.bloomThreshold, 0.25f, 0.0f, 0.0f);
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
    // sharpen taps offset by exact texels, so they do not need filtering either.
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

    gl.Uniform4f(aTone1_, p.exposure, p.brightness, p.contrast, p.gamma);
    gl.Uniform4f(aTone2_, p.black, p.white, p.shadows, p.highlights);
    gl.Uniform4f(aColor1_, p.saturation, p.vibrance, p.temperature, p.tint);
    gl.Uniform4f(aColor2_, p.hue, p.mono, p.sepia, p.intensity);
    gl.Uniform4f(aDetail_, p.sharpen, mips ? p.clarity : 0.0f,
                 mips ? p.bloom : 0.0f, p.hdrToning);
    gl.Uniform4f(aMisc_, (float)p.cbMode, p.cbStrength, p.split, 0.0f);

    // No VRS on any of this, deliberately. 2x2 shading would quantize the grade
    // to 2x2 blocks — plainly visible as banding on gradients, and it would
    // defeat the sharpen pass outright.
    runPass(progA_, dstTex, w, h);

    gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    gl.BindVertexArray(0);
    gl.UseProgram(0);
}

void Filter::applyStageB(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                         const FilterParams& p, uint64_t frameIdx) {
    if (!progB_ || !gl_) return;
    const FilterGL& gl = *gl_;

    gl.UseProgram(progB_);
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, srcTex);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl.Uniform1i(bSrc_, 0);
    // Wrap the frame counter well inside float precision: at 120fps a raw
    // counter would lose its low bits within the hour and the grain would
    // gradually freeze.
    gl.Uniform4f(bParams_, p.vignette, p.grain, p.letterbox,
                 (float)(frameIdx % 4096u) * 0.0173f);

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
