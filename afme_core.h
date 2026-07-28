/*
 * Copyright (C) 2025-2026 IRedDragonICY
 * SPDX-License-Identifier: Apache-2.0
 *
 * AFME core — the parts of frame generation that are not graphics.
 *
 * Both AFME layers solve the same scheduling problem and differ only in how
 * they get a frame onto the screen. Everything here is that shared problem:
 * which properties are set, how many frames to synthesize, when to release
 * them, and what to report. It has no EGL, GLES or Vulkan dependency, which is
 * what lets it link into both layers — the GLES layer resolves GL through
 * eglGetProcAddress and deliberately does not link libGLESv3, so any shared
 * code that touched GL directly would fail to link there (-Wl,--no-undefined).
 *
 * The control law implemented here came from the Vulkan layer, where it was
 * tuned on device against ZZZ and Genshin. The GLES layer previously carried a
 * much cruder version (a single headroom clamp, no work-time estimate, no
 * hysteresis, no limiter, no engagement gate); routing both through this file
 * is what brings the two paths to parity.
 */
#ifndef AFME_CORE_H
#define AFME_CORE_H

#include <atomic>
#include <cstdint>

namespace afme {

// ─── Limits ─────────────────────────────────────────────────────────────────

inline constexpr int kMaxMultiplier = 4;

// ─── Clock ──────────────────────────────────────────────────────────────────

/** CLOCK_MONOTONIC in nanoseconds — the time base every deadline here uses. */
int64_t nowNs();

/**
 * Sleep for @p ns, or not at all.
 *
 * Below ~0.3ms the syscall costs more than the wait buys. Above @p maxNs the
 * estimate that produced the request is wrong, and hanging the game's render
 * thread on a bad estimate is far worse than skipping the wait — so the bound
 * is explicit at every call site rather than a shared default. Returns true if
 * it actually slept.
 */
bool sleepNs(int64_t ns, int64_t maxNs);

// ─── Frame-generation method ────────────────────────────────────────────────

enum Method {
    kExtrapolate = 0,  // glExtrapolateTex2DQCOM — driver black box, one call
    kMotion      = 1,  // glTexEstimateMotionQCOM + our own warp
};

// ─── Configuration ──────────────────────────────────────────────────────────

/**
 * The persist.sys.afme.* surface, parsed once for the whole process.
 *
 * Polled every kPollInterval presents by the layers. Fields are atomic because
 * a process can present from more than one thread; they are independent scalars
 * and a torn read across two of them costs at most one frame of mixed settings.
 */
class Config {
public:
    /** Re-read every property. Cheap: property_get is a shared-memory read. */
    void poll();

    /** True when the motion-estimation pipeline should be built and used. */
    bool wantMotion() const;

    // Master
    std::atomic<bool>  enabled{false};        // persist.sys.afme.enable
    std::atomic<int>   multiplier{2};         // 2 / 3 / 4
    std::atomic<int>   method{kExtrapolate};
    std::atomic<float> factorOverride{0.0f};  // 0 = auto phase
    std::atomic<int>   displayHz{120};        // staged panel rate

    // Cadence
    std::atomic<bool> spacing{true};   // spacing sleeps between synth presents
    std::atomic<bool> pacing{false};   // desiredPresentTime stamping
    std::atomic<bool> limiter{true};   // Swappy-style base-rate lock

    // Quality
    std::atomic<bool> vrsFg{true};        // VRS on our own fragment passes
    std::atomic<bool> hudMask{true};      // HUD ghost protection (motion path)
    std::atomic<bool> antiGhost{true};    // moving-content anti-ghost
    std::atomic<bool> smoothMotion{true};
    std::atomic<int>  sgsrMode{0};        // 0=off 1=SGSR1 2=SGSR2 3=legacy motion
};

/** The process-wide instance. */
Config& config();

/** Presents between property polls. Power of two: callers mask rather than modulo. */
inline constexpr uint64_t kPollInterval = 64;

// ─── Engagement gate ────────────────────────────────────────────────────────

/**
 * Is this presentation target a game render loop, or a UI window?
 *
 * Both layers get armed for the target package because which graphics API a
 * game presents with is not knowable before it runs. So the Vulkan layer gets
 * loaded into GLES games, where the only Vulkan swapchain belongs to HWUI
 * drawing the Activity behind the game's SurfaceView — and the GLES layer gets
 * loaded into Vulkan games, where the only EGL surface is that same HWUI
 * window. Accelerating those is pure waste: GPU time spent duplicating a
 * near-static window, plus a second context reporting into the stats channel
 * the overlay reads.
 *
 * The discriminator is behavioural, not identity: a game render loop presents
 * continuously, a UI window presents only when something changes. Thread names
 * are NOT usable here — Unreal Engine also calls its render thread
 * "RenderThread", exactly like HWUI.
 *
 * A loading screen presenting slowly just delays engagement, which is the
 * behaviour we want anyway.
 */
class EngagementGate {
public:
    /**
     * Feed one present. True once this target looks like a game loop.
     * @p w, @p h are for the one-shot log line only.
     */
    bool check(int64_t now, int w, int h);

    bool engaged() const { return engaged_; }
    void reset();

private:
    static constexpr float kMaxIntervalMs = 50.0f;  // 20fps floor
    static constexpr int   kRunFrames     = 24;

    bool    engaged_ = false;
    int     run_     = 0;
    int64_t lastNs_  = 0;
    float   emaMs_   = 0.0f;
};

// ─── Pacer ──────────────────────────────────────────────────────────────────

/**
 * The cadence control law: how many frames to synthesize and when to release
 * them. One instance per presentation target (Vulkan swapchain / EGL surface).
 *
 * The governing idea is that a stable base rate matters more than a high one.
 * base(n) = Hz/(n+1), so every tier the game can sustain fills EVERY vsync slot
 * — total presents == panel Hz, a perfectly even cadence — and the smallest
 * sustainable n has the highest base, meaning lowest latency and fewest
 * artifacts. Generation that is not evenly spaced is generation the user cannot
 * see, which is the failure mode this class exists to prevent.
 */
class Pacer {
public:
    struct Tier {
        int  numGen   = 0;      // synthetic frames to emit this interval
        bool paceable = false;  // the chosen base rate is sustainable
    };

    /**
     * Open a real present: fold in timing, then choose the tier.
     *
     * @param now   CLOCK_MONOTONIC now, taken once by the caller.
     * @param mult  requested multiplier (2/3/4).
     * @param hz    effective panel rate — the MEASURED refresh cycle where the
     *              caller has one, not the staged property. Running this math
     *              on the wrong grid is how generation becomes invisible.
     */
    Tier beginPresent(int64_t now, int mult, int hz);

    /**
     * The caller is taking a passthrough path (no headroom, not engaged).
     *
     * Deliberately does NOT clear the committed tier: transient no-headroom
     * windows — one skipped frame, one hitch — used to wipe it, and the next
     * non-zero window re-committed whatever the noisy capability said at that
     * instant, which fed the gen-count oscillation this class prevents.
     */
    void abortPresent(int64_t now);

    /**
     * Anchor the spacing schedule.
     *
     * Extrapolation calls this right after the real present is issued.
     * Interpolation calls it at the START of synth emission, because there the
     * real frame goes out last — anchoring on it would leave the anchor at zero
     * and silently disable spacing for the whole motion path.
     */
    void anchorReal(int64_t ns) { realPresentNs_ = ns; }

    /**
     * Hold synthetic frame @p index until its slot inside the frame interval.
     *
     * Presenting it immediately lands it in the vsync right after the real
     * frame: at 30fps on a 120Hz panel the real frame is visible for 8ms and
     * the synthetic one for 25ms (measured present-to-present: 8ms x856,
     * 24ms x594). That is 60 presents a second that still read as 30fps with a
     * strobe. Spacing is what makes generation visible.
     */
    void spaceSynth(int index, int numGen) const;

    /**
     * Close the present, applying the frame limiter, and return the time the
     * game's next frame of work begins.
     *
     * Locks the game to (numGen+1) vsync slots so total presents fill every
     * slot: 120Hz / 3 = a locked 40fps base and a steady 120 total. A precise
     * sleep here beats FIFO back-pressure, which blocks the game at
     * unpredictable points (measured: a 24-58ms stall tail at SurfaceFlinger).
     * Only engages when the tier is sustainable — sleeping a game that cannot
     * reach the target would just add latency.
     */
    int64_t endPresent(int64_t end, int numGen, bool paceable, int64_t vsyncNs);

    void reset();

    /** Smoothed real-frame interval, milliseconds. 0 until established. */
    float intervalMs() const { return emaFrameMs_; }

private:
    void updateSpacingGovernor();
    Tier selectTier(int mult, int hz) const;
    int  applyHysteresis(int want, int mult);

    // Timing
    int64_t lastPresentNs_ = 0;
    float   emaFrameMs_    = 0.0f;

    // Capability: work time per frame = interval minus our own pacing sleep.
    // Unlike the raw interval this stays valid while the limiter is engaged,
    // which breaks the pacing-measures-pacing feedback loop. Choosing the tier
    // from the raw interval creates a dead zone: measured on ZZZ city at
    // 39-45fps free-run on 120Hz, too fast for the <=40.8fps two-synth clamp and
    // too slow for one synth to fill the panel, giving 86 uneven presents.
    float   emaWorkMs_    = 0.0f;
    int64_t lastReturnNs_ = 0;

    // Committed tier + hysteresis counters
    int stableGen_   = -1;
    int genUp_       = 0;
    int genDown_     = 0;

    // Spacing
    int64_t realPresentNs_ = 0;
    float   spacingScale_  = 0.0f;
    float   bestRealFps_   = 0.0f;

    // Limiter
    int64_t nextDeadlineNs_ = 0;
};

// ─── Stats ──────────────────────────────────────────────────────────────────

/**
 * The AFME-STATS channel GameSpace's overlay parses out of logcat.
 *
 * property_set() from a game process can never work: platform sepolicy carries
 * `neverallow all_untrusted_apps property_type:property_service set`. A log
 * line is the only channel out of an untrusted_app process that GameSpace
 * (system_app, READ_LOGS) can read.
 *
 * State is per-target, never file-scope: games recreate presentation targets
 * (ZZZ makes several swapchains at startup) and a static baseline from the old
 * one underflows against the new target's small counters — observed on device
 * as "real=2147483647 total=-2".
 */
class Stats {
public:
    void addReal(uint32_t n = 1) { real_ += n; }
    void addGen(uint32_t n = 1)  { gen_  += n; }

    /** Emits at most one line per second. */
    void publish(int64_t now);

    void reset();

private:
    int64_t  baseNs_   = 0;
    uint64_t baseReal_ = 0;
    uint64_t baseGen_  = 0;
    uint64_t real_     = 0;
    uint64_t gen_      = 0;
};

}  // namespace afme

#endif  // AFME_CORE_H
