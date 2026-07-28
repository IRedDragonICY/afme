/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME core — see afme_core.h.
 *
 * Every constant in the control law below was measured on device (onyx,
 * Adreno 840, 120Hz) rather than chosen. Where a value looks arbitrary the
 * comment says what it was tuned against; please do not "clean them up".
 */
#include "afme_core.h"

#include <cmath>
#include <cstdlib>
#include <ctime>

#include <android/log.h>
#include <cutils/properties.h>

#ifndef LOG_TAG
#define LOG_TAG "AFME"
#endif
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace afme {
namespace {

/** Sub-millisecond intervals are measurement noise; 200ms+ is a hitch. */
bool plausibleIntervalMs(float ms) { return ms > 0.05f && ms < 200.0f; }

/** Standard EMA used for every smoothed quantity here (alpha = 0.1). */
void feedEma(float& ema, float sample) {
    ema = (ema <= 0.0f) ? sample : ema * 0.9f + sample * 0.1f;
}

bool propBool(const char* name, const char* def) {
    char value[PROPERTY_VALUE_MAX];
    property_get(name, value, def);
    return value[0] == '1';
}

int propInt(const char* name, const char* def) {
    char value[PROPERTY_VALUE_MAX];
    property_get(name, value, def);
    return atoi(value);
}

}  // namespace

// ─── Clock ──────────────────────────────────────────────────────────────────

int64_t nowNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

bool sleepNs(int64_t ns, int64_t maxNs) {
    if (ns <= 300000LL || ns >= maxNs) return false;
    struct timespec req = { (time_t)(ns / 1000000000LL),
                            (long)(ns % 1000000000LL) };
    nanosleep(&req, nullptr);
    return true;
}

// ─── Configuration ──────────────────────────────────────────────────────────

Config& config() {
    static Config instance;
    return instance;
}

void Config::poll() {
    enabled.store(propBool("persist.sys.afme.enable", "0"));

    // Invalid values keep the previous setting rather than snapping to a
    // default: a transiently empty property during a GameSpace write must not
    // drop a running 4x session to 2x for a frame.
    const int m = propInt("persist.sys.afme.multiplier", "2");
    if (m >= 2 && m <= kMaxMultiplier) multiplier.store(m);

    // Anything unrecognised means extrapolation, which needs no shaders and so
    // can never fail to initialize.
    method.store(propInt("persist.sys.afme.method", "0") == kMotion
                     ? kMotion : kExtrapolate);

    // Empty or out of range means auto — compute the phase from the multiplier.
    char value[PROPERTY_VALUE_MAX];
    property_get("persist.sys.afme.factor", value, "");
    const float f = strtof(value, nullptr);
    factorOverride.store((f > 0.0f && f <= 2.0f) ? f : 0.0f);

    const int hz = propInt("persist.sys.afme.display_hz", "120");
    if (hz >= 30 && hz <= 240) displayHz.store(hz);

    const int sgsr = propInt("persist.sys.sgsr.mode", "0");
    if (sgsr >= 0 && sgsr <= 3) sgsrMode.store(sgsr);

    spacing.store(propBool("persist.sys.afme.spacing", "1"));
    pacing.store(propBool("persist.sys.afme.pacing", "0"));
    limiter.store(propBool("persist.sys.afme.limiter", "1"));
    vrsFg.store(propBool("persist.sys.afme.vrs_fg", "1"));
    hudMask.store(propBool("persist.sys.afme.hud_mask", "1"));
    antiGhost.store(propBool("persist.sys.afme.anti_ghost", "1"));
    smoothMotion.store(propBool("persist.sys.afme.smooth_motion", "1"));
}

bool Config::wantMotion() const {
    // sgsr.mode==3 is the legacy spelling of the same request; existing
    // per-game configs still carry it.
    return method.load() == kMotion || sgsrMode.load() == 3;
}

// ─── Engagement gate ────────────────────────────────────────────────────────

bool EngagementGate::check(int64_t now, int w, int h) {
    if (engaged_) return true;

    if (lastNs_ > 0) {
        const float ms = (float)(now - lastNs_) / 1e6f;
        if (ms > 0.05f && ms < kMaxIntervalMs) {
            run_++;
            emaMs_ = (emaMs_ <= 0.0f) ? ms : emaMs_ * 0.8f + ms * 0.2f;
        } else {
            run_ = 0;
            emaMs_ = 0.0f;
        }
    }
    lastNs_ = now;

    if (run_ >= kRunFrames) {
        engaged_ = true;
        ALOGI("AFME: engaged on %dx%d target (%.1ffps sustained over %d frames)",
              w, h, emaMs_ > 0.0f ? 1000.0f / emaMs_ : 0.0f, run_);
    }
    return engaged_;
}

void EngagementGate::reset() {
    engaged_ = false;
    run_     = 0;
    lastNs_  = 0;
    emaMs_   = 0.0f;
}

// ─── Pacer ──────────────────────────────────────────────────────────────────

Pacer::Tier Pacer::selectTier(int mult, int hz) const {
    Tier t;
    // Until a capability estimate exists, honour the requested multiplier and
    // stay unpaceable — the limiter must not sleep a game we know nothing about.
    t.numGen   = mult - 1;
    t.paceable = false;

    const float capFps = (emaWorkMs_ > 0.1f) ? 1000.0f / emaWorkMs_ : 0.0f;
    if (capFps <= 1.0f) return t;

    // Smallest sustainable n wins: highest base rate, lowest latency, fewest
    // artifacts, and total presents still fill every vsync slot.
    int chosen = -1;
    for (int n = 0; n <= mult - 1; n++) {
        const float base = (float)hz / (float)(n + 1);
        if (capFps >= base * 1.02f) { chosen = n; break; }
    }

    if (chosen < 0) {
        // No tier is fully sustainable, but a FRAME-RATE-CAPPED game still
        // belongs on the panel grid — the limiter locks it to the tier's base
        // and total presents == panel Hz. Snap to the first tier within a 15%
        // pull-out margin: a 60fps cap on 120Hz picks n=1 (60+60), not n=3
        // (which would force it down to 30).
        //
        // The previous best-effort fallback mapped a capability flickering by
        // +/-1fps around a tier boundary onto DIFFERENT gen counts every few
        // frames; measured on a steady 30fps game the total oscillated
        // 120<->60<->85, so generation "worked" but was invisible.
        for (int n = 0; n <= mult - 1; n++) {
            const float base = (float)hz / (float)(n + 1);
            if (capFps >= base * 0.85f) { chosen = n; break; }
        }
    }

    if (chosen >= 0) {
        t.paceable = true;
        t.numGen   = chosen;
        return t;
    }

    // Genuinely slower than the deepest tier's reach: best-effort generation at
    // whatever still fits the panel, unpaced.
    const int slots  = (int)(((float)hz * 1.02f) / capFps);
    int       maxGen = slots - 1;
    if (maxGen < 0) maxGen = 0;
    t.numGen = (maxGen < mult - 1) ? maxGen : (mult - 1);
    return t;
}

int Pacer::applyHysteresis(int want, int mult) {
    // A capability straddling a tier boundary would flip the display cadence
    // every few seconds, which is worse than either steady state. MORE
    // generation means a LOWER paced base, i.e. the safer direction — so
    // respond to capability drops fast (12 frames) and to gains slowly
    // (90 frames, about 2s).
    if (stableGen_ < 0) stableGen_ = want;

    if (want > stableGen_) {
        genDown_ = 0;
        if (++genUp_ >= 12) { stableGen_ = want; genUp_ = 0; }
    } else if (want < stableGen_) {
        genUp_ = 0;
        if (++genDown_ >= 90) { stableGen_ = want; genDown_ = 0; }
    } else {
        genUp_   = 0;
        genDown_ = 0;
    }

    // If the multiplier was lowered mid-session the committed tier can exceed
    // what is now allocated; keep the fresh value until it catches up.
    return (stableGen_ <= mult - 1) ? stableGen_ : want;
}

void Pacer::updateSpacingGovernor() {
    // A spacing sleep takes wall-clock time from the game thread. That is free
    // when the game is capped below the panel — the usual frame-generation
    // case, where its own limiter simply sleeps less — and expensive when the
    // game is work-bound, where it directly costs real frames. Both look
    // identical in any single measurement (a 30fps cap and a 30fps GPU limit
    // have the same interval), so instead of trying to classify, regulate on
    // the quantity that must not be damaged: the real frame rate.
    if (emaFrameMs_ <= 0.0f) return;

    const float realFps = 1000.0f / emaFrameMs_;
    // Decay the reference so a genuine scene-complexity drop re-baselines
    // instead of pinning the scale at zero forever.
    bestRealFps_ = (realFps > bestRealFps_) ? realFps : bestRealFps_ * 0.999f;

    if (bestRealFps_ > 1.0f && realFps < bestRealFps_ * 0.94f) {
        spacingScale_ -= 0.08f;   // back off fast: we are costing frames
    } else {
        spacingScale_ += 0.02f;   // creep up: 50 frames to full
    }

    if (spacingScale_ < 0.0f) spacingScale_ = 0.0f;
    if (spacingScale_ > 1.0f) spacingScale_ = 1.0f;
}

Pacer::Tier Pacer::beginPresent(int64_t now, int mult, int hz) {
    if (lastPresentNs_ > 0) {
        const float ms = (float)(now - lastPresentNs_) / 1e6f;
        if (plausibleIntervalMs(ms)) feedEma(emaFrameMs_, ms);
    }
    lastPresentNs_ = now;

    updateSpacingGovernor();

    if (lastReturnNs_ > 0) {
        const float workMs = (float)(now - lastReturnNs_) / 1e6f;
        if (plausibleIntervalMs(workMs)) feedEma(emaWorkMs_, workMs);
    }

    Tier t = selectTier(mult, hz);
    if (t.numGen > 0) t.numGen = applyHysteresis(t.numGen, mult);
    return t;
}

void Pacer::abortPresent(int64_t now) {
    nextDeadlineNs_ = 0;
    lastReturnNs_   = now;  // a passthrough adds no work of its own
}

void Pacer::spaceSynth(int index, int numGen) const {
    if (!config().spacing.load(std::memory_order_relaxed)) return;
    if (spacingScale_ <= 0.01f || emaFrameMs_ <= 0.0f) return;
    if (realPresentNs_ <= 0) return;

    const int64_t intervalNs = (int64_t)(emaFrameMs_ * 1e6f);
    const int64_t slotNs     = intervalNs / (int64_t)(numGen + 1);
    const int64_t offsetNs   = (int64_t)((float)(slotNs * (int64_t)(index + 1))
                                         * spacingScale_);

    // 50ms ceiling: past that the interval estimate is wrong and sleeping on it
    // would stall the render thread outright.
    sleepNs((realPresentNs_ + offsetNs) - nowNs(), 50000000LL);
}

int64_t Pacer::endPresent(int64_t end, int numGen, bool paceable,
                          int64_t vsyncNs) {
    if (!config().limiter.load(std::memory_order_relaxed) || numGen <= 0 ||
        !paceable) {
        nextDeadlineNs_ = 0;
        lastReturnNs_   = end;
        return end;
    }

    const int64_t targetNs = (int64_t)(numGen + 1) * vsyncNs;

    if (nextDeadlineNs_ <= 0) {
        nextDeadlineNs_ = end + targetNs;
    } else {
        // 100ms ceiling here rather than 50ms: a deep tier at a low panel rate
        // legitimately asks for a longer hold than a spacing slot ever does.
        if (sleepNs(nextDeadlineNs_ - end, 100000000LL)) end = nowNs();
        nextDeadlineNs_ += targetNs;
        // Fell behind on a heavy frame — resync instead of rushing to catch up,
        // which would bunch presents together.
        if (nextDeadlineNs_ < end) nextDeadlineNs_ = end + targetNs;
    }

    lastReturnNs_ = end;
    return end;
}

void Pacer::reset() {
    lastPresentNs_  = 0;
    emaFrameMs_     = 0.0f;
    emaWorkMs_      = 0.0f;
    lastReturnNs_   = 0;
    stableGen_      = -1;
    genUp_          = 0;
    genDown_        = 0;
    realPresentNs_  = 0;
    spacingScale_   = 0.0f;
    bestRealFps_    = 0.0f;
    nextDeadlineNs_ = 0;
}

// ─── Stats ──────────────────────────────────────────────────────────────────

void Stats::publish(int64_t now) {
    if (baseNs_ == 0) {
        baseNs_   = now;
        baseReal_ = real_;
        baseGen_  = gen_;
        return;
    }

    const double elapsed = (double)(now - baseNs_) / 1e9;
    if (elapsed < 1.0) return;

    const int realFps = (int)((double)(real_ - baseReal_) / elapsed + 0.5);
    const int genFps  = (int)((double)(gen_ - baseGen_) / elapsed + 0.5);

    ALOGI("AFME-STATS real=%d gen=%d total=%d", realFps, genFps,
          realFps + genFps);

    baseNs_   = now;
    baseReal_ = real_;
    baseGen_  = gen_;
}

void Stats::reset() {
    baseNs_   = 0;
    baseReal_ = 0;
    baseGen_  = 0;
    real_     = 0;
    gen_      = 0;
}

}  // namespace afme
