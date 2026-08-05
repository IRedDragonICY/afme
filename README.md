# AFME — Adreno Frame Motion Engine

Open-source GPU frame generation for Android, built into the OS.
No zygisk, no magisk module, no app-side injection hacks — two native
layers (EGL + Vulkan) shipped on `system_ext`, armed per game from the
platform's own GPU debug-layer machinery, controlled from GameSpace.

Target device: POCO F7 (onyx) — Snapdragon 8s Gen 4 ("tuna"), Adreno 840,
120 Hz panel, Android 16 (bp4a). The code is device-agnostic wherever the
Adreno driver exposes the QCOM frame-generation extensions.

```
┌─────────┐   eglSwapBuffers      ┌──────────────────────┐
│  Game   │ / vkQueuePresentKHR → │       AFME           │
│ (30-60  │                       │  libAFME_layer.so    │  GLES path
│   fps)  │                       │  libVkLayer_AFME.so  │  Vulkan path
└─────────┘                       └──────────┬───────────┘
                                             │ HW primitives:
                                             │  glExtrapolateTex2DQCOM
                                             │  glTexEstimateMotionQCOM
                                             │  MobFGSR ME→dilate→reproject→warp
                                             ▼
                                     SurfaceFlinger → 90/120Hz panel
```

## Features

- **Two generation methods**, per game:
  - `extrapolate` — `glExtrapolateTex2DQCOM`, one driver call, near-zero cost.
  - `motion` — `glTexEstimateMotionQCOM` (Adreno HW block motion estimation,
    8x8 blocks) plus the **MobFGSR** compute pipeline
    (luma→ME→MV upsample→nearest-depth dilate→atomic reproject→hole fill→blend warp)
    running on a hidden GLES 3.1+ context.
- **Multiplier** 2x/3x/4x with an adaptive tier lock: measured capability
  (work-time EMA) is snapped to the panel grid — a 30fps game gets 30+90=120,
  a 60fps game gets 60+60 — and the frame limiter paces the game to the
  chosen base so *total presents == display Hz* (even cadence, what makes
  frame generation actually visible).
- **Vsync-grid calibration** via `vkGetRefreshCycleDurationGOOGLE` (measured
  panel cycle overrides the staged prop; re-calibrated every 600 presents so
  dynamic refresh switches are followed).
- **Zero added input latency** in extrapolate mode (real frame presented
  first, synths after); interpolation mode holds the real frame one present
  (standard interpolation ordering).
- **GPU-side VK↔GLES sync** through `EGL_ANDROID_native_fence_sync` ⇆
  `VK_KHR_external_{fence,semaphore}_fd` (sync_fd export/import) — with
  graceful CPU-wait fallback.
- **VRS** (`glShadingRateQCOM` 2x2) on the generation fragment passes
  (~4x cheaper shading; compute passes unaffected).
- **HDR + Transfer bits** swapchain image inflation, MAILBOX→FIFO rewrite
  (synthetic frames must survive to latch).
- **SGSR1** (Qualcomm SGSR 2.1) spatial sharpening pass as a pre-stage.
- Stats channel (`AFME-STATS real=X gen=Y total=Z` logline) parsed by the
  GameSpace overlay.

## Control surface

Runtime props (read every 64 presents by the layers):

| prop | default | meaning |
|---|---|---|
| `persist.sys.afme.enable` | 0 | master enable (arm per game) |
| `persist.sys.afme.app` | "" | package filter for layer injection |
| `persist.sys.afme.multiplier` | 2 | 2/3/4 |
| `persist.sys.afme.method` | 0 | 0=extrapolate, 1=motion (MobFGSR) |
| `persist.sys.afme.factor` | "" | phase override ("" = auto) |
| `persist.sys.afme.display_hz` | 120 | staged panel rate (measured cycle wins if available) |
| `persist.sys.afme.pacing` | 0 | present-time stamping (`VkPresentTimesInfoGOOGLE` / `eglPresentationTimeANDROID`). **OFF by default** — bp4a SurfaceFlinger drops >50% of delayed buffers; spacing sleeps are the working mechanism. |
| `persist.sys.afme.spacing` | 1 | spacing sleeps between synth presents (main pacing mechanism) |
| `persist.sys.afme.limiter` | 1 | Swappy-style frame limiter locking game to `Hz/(gen+1)` |
| `persist.sys.afme.vrs_fg` | 1 | glShadingRateQCOM 2x2 on generation fragment passes |
| `persist.sys.afme.hud_mask` | 1 | HUD ghost protection (v7): screen-static UI regions pinned to the real frame in the MobFGSR warp |
| `persist.sys.afme.anti_ghost` | 1 | v8 moving-content anti-ghost: deep holes anchor to real pixel; high-disagreement blends damp toward nearer sample |
| `persist.sys.afme.session` | 0 | session marker for init trigger (GPU devfreq floor) |
| `persist.sys.sgsr.mode` | 0 | 0/1/2/3 (3 = legacy spelling of method=motion) |

Per-game keys in GameSpace (`Settings.System` JSON maps):
`gamespace_afme_multiplier`, `gamespace_afme_factor`,
`gamespace_afme_method`, `gamespace_afme_vulkan`, `gamespace_afme_display_hz`,
`gamespace_afme_pacing`, `gamespace_afme_vrs_fg` — forwarded by
`GameStateDispatcher` (system_server) at game start/end.

## Directory layout

```
hardware/xiaomi/afme/
├── afme_layer.cpp      # Android GLES layer (eglSwapBuffers hook)
├── afme_vk_layer.cpp   # Vulkan layer (vkQueuePresentKHR hook)
└── Android.bp
```

Discovery paths (patched loaders):
- `frameworks/native/opengl/libs/EGL/egl_layers.cpp` — `kAfmeLayerLibraryDir` searched when `persist.sys.afme.enable=1`
- `frameworks/native/vulkan/libvulkan/layers_extensions.cpp` — same for `libVkLayer_AFME`
- `frameworks/base/core/java/android/os/GraphicsEnvironment.java` — per-app layer arming (bypass when AFME staged)
- `frameworks/base/services/core/java/com/android/server/wm/GameStateDispatcher.java` — session mgmt + prop forwarding from GameSpace settings

Install: Android.bp `relative_install_path: "afme"` →
`/system_ext/lib64/afme/{libAFME_layer.so,libVkLayer_AFME.so}`.

## Verifying it works (device)

```
adb shell setprop persist.sys.afme.enable 1   # normally staged by GameSpace
# launch the armored game, then:
adb logcat -s AFME:* vulkan:* libEGL:* | tee afme.log
```

Expected trail (Vulkan path):
1. `AFME VK Layer v6: Instance created / Device created`
2. `AFME: CreateSwapchain WxH images 5→9 (for 4x)`
3. `AFME: EGL init — HW AVAILABLE`
4. `AFME: engaged on WxH swapchain`
5. `AFME: Measured panel refresh cycle 8333333 ns (~120 Hz)`
6. **no** `MobFGSR shader compilation failed` (motion method)
7. Steady: `AFME-STATS real=30 gen=90 total=120` (30fps game, 4x, 120Hz)
8. `(3 gen/frame)` stable in `Frame N` lines; `factor=0.25/0.75` phases
   corresponding to the *actual* per-interval present count.

Ground truth on screen:
`adb shell dumpsys SurfaceFlinger --latency <game-window>` →
consecutive latch deltas ≈ vsync period (8.3 ms for 120 Hz).

## Design history

- v1–v5: initial layers, spacing/limiter/governor iteration (measured on ZZZ
  & Genshin).
- **v6 (2026-07-29):** fix set from the runtime analysis in
  `frame-gen-reference/reports/AFME-analysis-2026-07-28.md` — sticky tier
  lock (30-cap game oscillated 3↔1 gen/frame → invisible interpolation),
  factor math `/(numGenFrames+1)`, spacing anchor for interpolation mode,
  MAILBOX→FIFO rewrite, wait-semaphore clamp, MobFGSR fallback visibility,
  vsync-grid calibration, VRS on FG passes.
- **v7 (2026-07-29):** HUD ghost protection — block-grid static-content mask
  (9-tap Δluma + MV ≈ 0, asymmetric accumulation 10-frame lock / 2-frame
  release) blended into the MobFGSR warp so minimap/bars/buttons stay pinned
  to the real frame (no classic FG "shadow trail" on UI). Static world
  regions are visually identical either way, so the mask is safe there too.
  Toggle: `persist.sys.afme.hud_mask` + GameSpace per-game switch.
  Also: refresh-cycle query jitter hysteresis (≥0.1% before adopt/log).
- **v8 (2026-07-29):** moving-content anti-ghost (the running character in
  the middle of the screen). Two warp fixes: (1) deep-occlusion HOLES — the
  scatter-reprojection's uncovered pixels — anchored to the real current
  pixel instead of blending two samples along the previous frame's MV (the
  exact double-exposure recipe behind character silhouette trails);
  (2) high-disagreement warped blends (|c0−c1| ≫ threshold ⇒ unreliable MV)
  damp 75% toward the temporally nearer sample instead of a straight
  50/50 mix. Both default ON via `persist.sys.afme.anti_ghost` (no latency
  added; real-frame path untouched).

## References

- `frame-gen-reference/MobFGSR` — MIT/BSD-style mobile FG+SR (ported shaders).
- `frame-gen-reference/snapdragon-gsr` — Qualcomm SGSR SDK (SGSR1 fragment).
- `frame-gen-reference/LSFG-Android` — alternative (optical-flow) layer
  architecture for cross-checking.

## License

AFME layer code: Apache-2.0 (© 2025-2026 IRedDragonICY).
MobFGSR-derived shaders: retain upstream attribution.
SGSR1 pass: Qualcomm SGSR SDK (BSD-3).
