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
 * Fields left empty or missing keep whatever @p out already holds, so a short
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

// ─── Shaders ────────────────────────────────────────────────────────────────

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

const char* kFragSrc = R"(#version 300 es
precision highp float;

uniform mediump sampler2D uSrc;
uniform vec4 uTone1;   // exposure, brightness, contrast, gamma
uniform vec4 uTone2;   // black, white, shadows, highlights
uniform vec4 uColor1;  // saturation, vibrance, temperature, tint
uniform vec4 uColor2;  // hue, mono, sepia, intensity
uniform vec4 uMisc;    // cbMode, cbStrength, split, unused

in  vec2 vUV;
out vec4 outColor;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);

float luma(vec3 c) { return dot(c, kLuma); }

// Rotation about the grey axis (Rodrigues) — cheaper and more stable than a
// round trip through HSV, and it cannot produce out-of-gamut hue wrap.
vec3 hueRotate(vec3 c, float a) {
    const vec3 k = vec3(0.57735027);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

// Approximate white balance. Not a Planckian locus fit: a straight opponent
// gain reads the way the slider is labelled and costs three multiplies.
vec3 whiteBalance(vec3 c, float temp, float tint) {
    vec3 g = vec3(1.0 + 0.30 * temp,
                  1.0 + 0.15 * tint,
                  1.0 - 0.30 * temp);
    g.g = 1.0 - 0.15 * tint + (g.g - 1.0);
    return c * g;
}

// Daltonize: simulate the confusion axis, then push the resulting error back
// into channels the viewer can still separate.
vec3 colorBlind(vec3 c, int mode, float strength) {
    if (mode == 0 || strength <= 0.0) return c;
    mat3 sim;
    if (mode == 1) {          // protanopia
        sim = mat3(0.567, 0.433, 0.000,
                   0.558, 0.442, 0.000,
                   0.000, 0.242, 0.758);
    } else if (mode == 2) {   // deuteranopia
        sim = mat3(0.625, 0.375, 0.000,
                   0.700, 0.300, 0.000,
                   0.000, 0.300, 0.700);
    } else {                  // tritanopia
        sim = mat3(0.950, 0.050, 0.000,
                   0.000, 0.433, 0.567,
                   0.000, 0.475, 0.525);
    }
    vec3 seen = c * sim;
    vec3 err  = c - seen;
    vec3 fix  = vec3(0.0,
                     err.r * 0.7 + err.g * 1.0,
                     err.r * 0.7 + err.b * 1.0);
    return clamp(c + fix * strength, 0.0, 1.0);
}

void main() {
    vec3 orig = texture(uSrc, vUV).rgb;
    vec3 c = orig;

    // ── Tone ──
    c *= exp2(uTone1.x);
    c += uTone1.y;

    // Levels: remap [black, white] onto [0,1].
    c = (c - uTone2.x) / max(uTone2.y - uTone2.x, 1e-3);

    c = pow(max(c, 0.0), vec3(1.0 / max(uTone1.w, 1e-3)));
    c = (c - 0.5) * uTone1.z + 0.5;

    // Shadows / highlights, masked by luminance so each end moves alone.
    float l = luma(clamp(c, 0.0, 1.0));
    float sMask = 1.0 - smoothstep(0.0, 0.5, l);
    float hMask = smoothstep(0.5, 1.0, l);
    c *= 1.0 + uTone2.z * sMask + uTone2.w * hMask;

    // ── Color ──
    c = whiteBalance(c, uColor1.z, uColor1.w);
    if (abs(uColor2.x) > 1e-4) c = hueRotate(c, uColor2.x);

    float lc = luma(c);

    // Vibrance first: it must see the pre-saturation spread, otherwise a high
    // saturation setting starves it of the low-saturation pixels it targets.
    if (abs(uColor1.y) > 1e-4) {
        float mx = max(c.r, max(c.g, c.b));
        float mn = min(c.r, min(c.g, c.b));
        c = mix(vec3(lc), c, 1.0 + uColor1.y * (1.0 - (mx - mn)));
    }
    c = mix(vec3(lc), c, uColor1.x);

    c = mix(c, vec3(luma(c)), uColor2.y);

    if (uColor2.z > 1e-4) {
        float sl = luma(c);
        c = mix(c, sl * vec3(1.07, 0.87, 0.66), uColor2.z);
    }

    c = colorBlind(c, int(uMisc.x + 0.5), uMisc.y);

    // ── Global ──
    c = mix(orig, c, uColor2.w);
    c = clamp(c, 0.0, 1.0);

    // Compare divider: left of it stays untouched. Applied last so it shows the
    // true before/after including the master intensity.
    if (uMisc.z >= 0.0 && vUV.x < uMisc.z) c = orig;

    outColor = vec4(c, 1.0);
})";

}  // namespace

// ─── Parameters ─────────────────────────────────────────────────────────────

bool FilterParams::isIdentity() const {
    return nearly(exposure, 0.0f) && nearly(brightness, 0.0f) &&
           nearly(contrast, 1.0f) && nearly(gamma, 1.0f) &&
           nearly(black, 0.0f) && nearly(white, 1.0f) &&
           nearly(shadows, 0.0f) && nearly(highlights, 0.0f) &&
           nearly(saturation, 1.0f) && nearly(vibrance, 0.0f) &&
           nearly(temperature, 0.0f) && nearly(tint, 0.0f) &&
           nearly(hue, 0.0f) && nearly(mono, 0.0f) && nearly(sepia, 0.0f) &&
           (cbMode == 0 || nearly(cbStrength, 0.0f)) &&
           nearly(intensity, 1.0f) && split < 0.0f;
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
    float fx[3]    = { (float)p.cbMode, p.cbStrength, p.split };

    parseCsv("persist.sys.afme.filter.tone", tone, 8);
    parseCsv("persist.sys.afme.filter.color", color, 8);
    parseCsv("persist.sys.afme.filter.fx", fx, 3);

    p.exposure = tone[0]; p.brightness = tone[1];
    p.contrast = tone[2]; p.gamma      = tone[3];
    p.black    = tone[4]; p.white      = tone[5];
    p.shadows  = tone[6]; p.highlights = tone[7];

    p.saturation  = color[0]; p.vibrance = color[1];
    p.temperature = color[2]; p.tint     = color[3];
    p.hue         = color[4]; p.mono     = color[5];
    p.sepia       = color[6]; p.intensity = color[7];

    p.cbMode     = (int)(fx[0] + 0.5f);
    p.cbStrength = fx[1];
    p.split      = fx[2];

    // A zero white point would divide the whole image away; a zero gamma would
    // divide by zero in the exponent. Clamp rather than trust the caller.
    if (p.white - p.black < 1e-3f) { p.black = 0.0f; p.white = 1.0f; }
    if (p.gamma < 1e-3f) p.gamma = 1.0f;
    if (p.cbMode < 0 || p.cbMode > 3) p.cbMode = 0;

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
           BindVertexArray && ActiveTexture && BindTexture && TexParameteri &&
           Viewport && DrawArrays && Disable;
}

// ─── The pass ───────────────────────────────────────────────────────────────

namespace {

GLuint compile(const FilterGL& gl, GLenum type, const char* src) {
    GLuint sh = gl.CreateShader(type);
    if (!sh) return 0;
    gl.ShaderSource(sh, 1, &src, nullptr);
    gl.CompileShader(sh);
    GLint ok = 0;
    gl.GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        gl.GetShaderInfoLog(sh, sizeof(log), nullptr, log);
        ALOGE("AFME: filter %s shader compile failed: %s",
              type == GL_VERTEX_SHADER ? "vertex" : "fragment", log);
        gl.DeleteShader(sh);
        return 0;
    }
    return sh;
}

}  // namespace

bool Filter::init(const FilterGL& gl) {
    if (program_) return true;
    if (failed_) return false;

    if (!gl.complete()) {
        ALOGE("AFME: filter unavailable — GL dispatch incomplete");
        failed_ = true;
        return false;
    }
    gl_ = &gl;

    GLuint vs = compile(gl, GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compile(gl, GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs) {
        if (vs) gl.DeleteShader(vs);
        if (fs) gl.DeleteShader(fs);
        failed_ = true;
        return false;
    }

    program_ = gl.CreateProgram();
    gl.AttachShader(program_, vs);
    gl.AttachShader(program_, fs);
    gl.LinkProgram(program_);

    GLint linked = 0;
    gl.GetProgramiv(program_, GL_LINK_STATUS, &linked);
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);

    if (!linked) {
        char log[512] = {};
        gl.GetProgramInfoLog(program_, sizeof(log), nullptr, log);
        ALOGE("AFME: filter program link failed: %s", log);
        gl.DeleteProgram(program_);
        program_ = 0;
        failed_  = true;
        return false;
    }

    uSrc_    = gl.GetUniformLocation(program_, "uSrc");
    uTone1_  = gl.GetUniformLocation(program_, "uTone1");
    uTone2_  = gl.GetUniformLocation(program_, "uTone2");
    uColor1_ = gl.GetUniformLocation(program_, "uColor1");
    uColor2_ = gl.GetUniformLocation(program_, "uColor2");
    uMisc_   = gl.GetUniformLocation(program_, "uMisc");

    gl.GenFramebuffers(1, &fbo_);
    gl.GenVertexArrays(1, &vao_);

    ALOGI("AFME: color filter initialized (program=%u)", program_);
    return true;
}

void Filter::apply(GLuint srcTex, GLuint dstTex, uint32_t w, uint32_t h,
                   const FilterParams& p) {
    if (!program_ || !gl_) return;
    const FilterGL& gl = *gl_;

    gl.UseProgram(program_);

    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_2D, srcTex);
    // NEAREST: this is a 1:1 pass, and LINEAR would let the driver resolve
    // sample positions half a texel off, softening the whole image for free.
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.Uniform1i(uSrc_, 0);

    gl.Uniform4f(uTone1_, p.exposure, p.brightness, p.contrast, p.gamma);
    gl.Uniform4f(uTone2_, p.black, p.white, p.shadows, p.highlights);
    gl.Uniform4f(uColor1_, p.saturation, p.vibrance, p.temperature, p.tint);
    gl.Uniform4f(uColor2_, p.hue, p.mono, p.sepia, p.intensity);
    gl.Uniform4f(uMisc_, (float)p.cbMode, p.cbStrength, p.split, 0.0f);

    gl.BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, dstTex, 0);
    gl.Viewport(0, 0, (GLsizei)w, (GLsizei)h);

    gl.Disable(GL_DEPTH_TEST);
    gl.Disable(GL_BLEND);
    gl.Disable(GL_SCISSOR_TEST);

    // No VRS here, deliberately. 2x2 shading would quantize the grade to 2x2
    // blocks, which is plainly visible as banding on gradients — acceptable for
    // a sharpening pass, not for color.
    gl.BindVertexArray(vao_);
    gl.DrawArrays(GL_TRIANGLES, 0, 3);

    gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    gl.BindVertexArray(0);
    gl.UseProgram(0);
}

void Filter::destroy() {
    if (!gl_) return;
    if (program_) gl_->DeleteProgram(program_);
    if (fbo_)     gl_->DeleteFramebuffers(1, &fbo_);
    if (vao_)     gl_->DeleteVertexArrays(1, &vao_);
    program_ = 0;
    fbo_     = 0;
    vao_     = 0;
}

}  // namespace afme
