/*
 * AFME Vulkan Layer — Adreno Frame Motion Engine
 *
 * Copyright (C) 2025-2026 IRedDragonICY
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * HW-accelerated frame generation using Adreno glExtrapolateTex2DQCOM
 * via EGL/GLES interop with AHardwareBuffer shared images.
 * Supports 2x, 3x, 4x frame multiplier via persist.sys.afme.multiplier
 */

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#include <vulkan/vk_layer.h>

#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>

// EGL/GLES for AFME HW interop
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl31.h>    // Compute shaders, image load/store, memory barriers
#include <GLES3/gl32.h>    // glCopyImageSubData
#include <GLES3/gl3ext.h>
#include <GLES2/gl2ext.h>

// AHardwareBuffer
#include <android/hardware_buffer.h>

#include <android/log.h>
#include <cutils/properties.h>

#include "afme_core.h"
#include "afme_filter.h"

// ─── Constants ──────────────────────────────────────────────────────────────
#define LOG_TAG "AFME"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

static const char kLayerName[] = "VK_LAYER_AFME_frame_gen";
static const char kLayerDescription[] = "AFME Frame Generation Layer (IRedDragonICY)";
static const uint32_t kLayerImplVersion = 5;
static const uint32_t kLayerSpecVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);


// Maximum pre-allocated semaphores for frame generation (per swapchain)
// Each synth frame needs 2 sems (acquire + signal); deferred drain means up to
// 2 frames of sems may be in-flight simultaneously. 4x * 4 = 16 covers worst case.
static constexpr int kMaxSemaphorePool = afme::kMaxMultiplier * 4;
// Pre-allocated command buffer ring (avoid hot-path alloc/free)
static constexpr int kCmdRingSize = afme::kMaxMultiplier * 2 + 2;  // +2 for copy steps

typedef void (GL_APIENTRYP PFNGLEXTRAPOLATETEX2DQCOMPROC)(GLuint, GLuint, GLuint, float);
typedef void (GL_APIENTRYP PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)(GLenum, void*);
typedef void* (EGLAPIENTRYP PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC)(const struct AHardwareBuffer*);
typedef void* (EGLAPIENTRYP PFNEGLCREATEIMAGEKHRPROC)(EGLDisplay, EGLContext, EGLenum,
    EGLClientBuffer, const EGLint*);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYIMAGEKHRPROC)(EGLDisplay, void*);

// QCOM motion estimation + depth estimation HW accelerators
typedef void (GL_APIENTRYP PFNGLTEXESTIMATEMOTIONQCOMPROC)(GLuint, GLuint, GLuint);
typedef void (GL_APIENTRYP PFNGLTEXGENERATEDISPARITYQCOMPROC)(GLuint, GLuint);
typedef void (GL_APIENTRYP PFNGLSHADINGRATEQCOMPROC)(GLenum);

// QCOM motion estimation search block size query tokens
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM 0x8C90
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM 0x8C91

// GL_QCOM_shading_rate — ABI values match ANGLE's include/GLES2/gl2ext.h
#ifndef GL_SHADING_RATE_1X1_PIXELS_QCOM
#define GL_SHADING_RATE_1X1_PIXELS_QCOM 0x96A6
#endif
#ifndef GL_SHADING_RATE_2X2_PIXELS_QCOM
#define GL_SHADING_RATE_2X2_PIXELS_QCOM 0x96A9
#endif

// ─── Global next-layer function pointers ────────────────────────────────────
namespace {

PFN_vkGetInstanceProcAddr  next_vkGetInstanceProcAddr{};
PFN_vkGetDeviceProcAddr    next_vkGetDeviceProcAddr{};

PFN_vkCreateInstance       next_vkCreateInstance{};
PFN_vkDestroyInstance      next_vkDestroyInstance{};
PFN_vkCreateDevice         next_vkCreateDevice{};
PFN_vkDestroyDevice        next_vkDestroyDevice{};

PFN_vkQueuePresentKHR      next_vkQueuePresentKHR{};
PFN_vkQueueSubmit          next_vkQueueSubmit{};
PFN_vkQueueWaitIdle        next_vkQueueWaitIdle{};
PFN_vkGetDeviceQueue       next_vkGetDeviceQueue{};

PFN_vkCreateSwapchainKHR   next_vkCreateSwapchainKHR{};
PFN_vkDestroySwapchainKHR  next_vkDestroySwapchainKHR{};
PFN_vkGetSwapchainImagesKHR next_vkGetSwapchainImagesKHR{};
PFN_vkAcquireNextImageKHR   next_vkAcquireNextImageKHR{};

PFN_vkCreateCommandPool    next_vkCreateCommandPool{};
PFN_vkDestroyCommandPool   next_vkDestroyCommandPool{};
PFN_vkAllocateCommandBuffers next_vkAllocateCommandBuffers{};
PFN_vkFreeCommandBuffers   next_vkFreeCommandBuffers{};
PFN_vkResetCommandBuffer   next_vkResetCommandBuffer{};
PFN_vkBeginCommandBuffer   next_vkBeginCommandBuffer{};
PFN_vkEndCommandBuffer     next_vkEndCommandBuffer{};

PFN_vkCmdPipelineBarrier   next_vkCmdPipelineBarrier{};
PFN_vkCmdBlitImage         next_vkCmdBlitImage{};

PFN_vkCreateImage          next_vkCreateImage{};
PFN_vkDestroyImage         next_vkDestroyImage{};
PFN_vkGetImageMemoryRequirements next_vkGetImageMemoryRequirements{};
PFN_vkAllocateMemory       next_vkAllocateMemory{};
PFN_vkFreeMemory           next_vkFreeMemory{};
PFN_vkBindImageMemory      next_vkBindImageMemory{};

PFN_vkCreateFence          next_vkCreateFence{};
PFN_vkDestroyFence         next_vkDestroyFence{};
PFN_vkWaitForFences        next_vkWaitForFences{};
PFN_vkResetFences          next_vkResetFences{};
PFN_vkCreateSemaphore      next_vkCreateSemaphore{};
PFN_vkDestroySemaphore     next_vkDestroySemaphore{};

PFN_vkGetAndroidHardwareBufferPropertiesANDROID next_vkGetAndroidHardwareBufferProperties{};
PFN_vkGetPhysicalDeviceMemoryProperties next_vkGetPhysicalDeviceMemoryProperties{};
PFN_vkGetPhysicalDeviceProperties next_vkGetPhysicalDeviceProperties{};
PFN_vkGetPhysicalDeviceFeatures next_vkGetPhysicalDeviceFeatures{};
PFN_vkCreateSampler        next_vkCreateSampler{};
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR next_vkGetPhysicalDeviceSurfaceCapabilities{};
PFN_vkEnumerateDeviceExtensionProperties next_vkEnumerateDeviceExtensionProperties{};
PFN_vkGetFenceFdKHR           next_vkGetFenceFdKHR{};
PFN_vkImportSemaphoreFdKHR    next_vkImportSemaphoreFdKHR{};
PFN_vkGetRefreshCycleDurationGOOGLE next_vkGetRefreshCycleDurationGOOGLE{};

// ─── Global State ───────────────────────────────────────────────────────────

std::mutex gLock;

// ─── SGSR1 Shader Sources (Qualcomm BSD-3) ──────────────────────────────────
// Fullscreen triangle vertex shader — no VBO needed, 3 vertices cover viewport
static const char* kSGSR1VertSrc = R"(
#version 300 es
out highp vec4 in_TEXCOORD0;
void main() {
    float x = float((gl_VertexID & 1) << 1);
    float y = float((gl_VertexID >> 1) & 1) * 2.0;
    gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    in_TEXCOORD0 = vec4(x, y, 0.0, 0.0);
}
)";

// SGSR1 fragment shader — 12-tap Lanczos-like upscaling + adaptive sharpening
// Source: snapdragon-gsr/sgsr/v1/include/glsl/sgsr1_shader_mobile.frag
static const char* kSGSR1FragSrc = R"(
#version 300 es
precision mediump float;
precision highp int;

#define OperationMode 1
#define EdgeThreshold 8.0/255.0
#define EdgeSharpness 2.0

uniform highp vec4 ViewportInfo[1];
uniform mediump sampler2D ps0;

layout(location=0) in highp vec4 in_TEXCOORD0;
layout(location=0) out vec4 out_Target0;

float fastLanczos2(float x) {
    float wA = x - 4.0;
    float wB = x * wA - wA;
    wA *= wA;
    return wB * wA;
}

vec2 weightY(float dx, float dy, float c, float std) {
    float x = ((dx*dx)+(dy*dy))*0.55 + clamp(abs(c)*std, 0.0, 1.0);
    float w = fastLanczos2(x);
    return vec2(w, w * c);
}

void main() {
    int mode = OperationMode;
    float edgeThreshold = EdgeThreshold;
    float edgeSharpness = EdgeSharpness;
    vec4 color;
    if(mode == 1)
        color.xyz = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyz;
    else
        color.xyzw = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyzw;

    if (mode != 4) {
        highp vec2 imgCoord = ((in_TEXCOORD0.xy*ViewportInfo[0].zw)+vec2(-0.5,0.5));
        highp vec2 imgCoordPixel = floor(imgCoord);
        highp vec2 coord = (imgCoordPixel*ViewportInfo[0].xy);
        vec2 pl = (imgCoord+(-imgCoordPixel));
        vec4 left = textureGather(ps0, coord, mode);

        float edgeVote = abs(left.z - left.y) + abs(color[mode] - left.y) + abs(color[mode] - left.z);
        if(edgeVote > edgeThreshold) {
            coord.x += ViewportInfo[0].x;
            vec4 right = textureGather(ps0, coord + highp vec2(ViewportInfo[0].x, 0.0), mode);
            vec4 upDown;
            upDown.xy = textureGather(ps0, coord + highp vec2(0.0, -ViewportInfo[0].y), mode).wz;
            upDown.zw = textureGather(ps0, coord + highp vec2(0.0, ViewportInfo[0].y), mode).yx;

            float mean = (left.y+left.z+right.x+right.w)*0.25;
            left = left - vec4(mean);
            right = right - vec4(mean);
            upDown = upDown - vec4(mean);
            color.w = color[mode] - mean;

            float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));
            float std = 2.181818/sum;

            vec2 aWY = weightY(pl.x, pl.y+1.0, upDown.x, std);
            aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y, std);
            aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z, std);
            aWY += weightY(pl.x, pl.y-2.0, upDown.w, std);
            aWY += weightY(pl.x+1.0, pl.y-1.0, left.x, std);
            aWY += weightY(pl.x, pl.y-1.0, left.y, std);
            aWY += weightY(pl.x, pl.y, left.z, std);
            aWY += weightY(pl.x+1.0, pl.y, left.w, std);
            aWY += weightY(pl.x-1.0, pl.y-1.0, right.x, std);
            aWY += weightY(pl.x-2.0, pl.y-1.0, right.y, std);
            aWY += weightY(pl.x-2.0, pl.y, right.z, std);
            aWY += weightY(pl.x-1.0, pl.y, right.w, std);

            float finalY = aWY.y/aWY.x;
            float maxY = max(max(left.y,left.z),max(right.x,right.w));
            float minY = min(min(left.y,left.z),min(right.x,right.w));
            finalY = clamp(edgeSharpness*finalY, minY, maxY);
            float deltaY = finalY - color.w;
            deltaY = clamp(deltaY, -23.0/255.0, 23.0/255.0);

            color.x = clamp((color.x+deltaY), 0.0, 1.0);
            color.y = clamp((color.y+deltaY), 0.0, 1.0);
            color.z = clamp((color.z+deltaY), 0.0, 1.0);
        }
    }
    color.w = 1.0;
    out_Target0 = color;
}
)";

// ─── Helpers ────────────────────────────────────────────────────────────────

template<typename T>
bool initInstanceFunc(VkInstance instance, const char* name, T* func) {
    *func = reinterpret_cast<T>(next_vkGetInstanceProcAddr(instance, name));
    if (!*func) { ALOGW("AFME: No func: %s", name); return false; }
    return true;
}

template<typename T>
bool initDeviceFunc(VkDevice device, const char* name, T* func) {
    *func = reinterpret_cast<T>(next_vkGetDeviceProcAddr(device, name));
    if (!*func) { ALOGW("AFME: No func: %s", name); return false; }
    return true;
}

// The filter runs on this layer's private GLES context, so the entry points are
// simply the ones we link. (The GLES layer fills the same struct from
// eglGetProcAddress — see afme_filter.h for why it is a struct at all.)
static afme::FilterGL gFilterGl;
static std::once_flag gFilterGlOnce;

static void initFilterGl() {
    std::call_once(gFilterGlOnce, [] {
        gFilterGl.CreateShader = glCreateShader;
        gFilterGl.ShaderSource = glShaderSource;
        gFilterGl.CompileShader = glCompileShader;
        gFilterGl.GetShaderiv = glGetShaderiv;
        gFilterGl.GetShaderInfoLog = glGetShaderInfoLog;
        gFilterGl.DeleteShader = glDeleteShader;
        gFilterGl.CreateProgram = glCreateProgram;
        gFilterGl.AttachShader = glAttachShader;
        gFilterGl.LinkProgram = glLinkProgram;
        gFilterGl.GetProgramiv = glGetProgramiv;
        gFilterGl.GetProgramInfoLog = glGetProgramInfoLog;
        gFilterGl.DeleteProgram = glDeleteProgram;
        gFilterGl.UseProgram = glUseProgram;
        gFilterGl.GetUniformLocation = glGetUniformLocation;
        gFilterGl.Uniform1i = glUniform1i;
        gFilterGl.Uniform4f = glUniform4f;
        gFilterGl.Uniform1f = glUniform1f;
        gFilterGl.GenFramebuffers = glGenFramebuffers;
        gFilterGl.DeleteFramebuffers = glDeleteFramebuffers;
        gFilterGl.BindFramebuffer = glBindFramebuffer;
        gFilterGl.FramebufferTexture2D = glFramebufferTexture2D;
        gFilterGl.GenVertexArrays = glGenVertexArrays;
        gFilterGl.DeleteVertexArrays = glDeleteVertexArrays;
        gFilterGl.BindVertexArray = glBindVertexArray;
        gFilterGl.GenTextures = glGenTextures;
        gFilterGl.DeleteTextures = glDeleteTextures;
        gFilterGl.TexStorage2D = glTexStorage2D;
        gFilterGl.GenerateMipmap = glGenerateMipmap;
        gFilterGl.ActiveTexture = glActiveTexture;
        gFilterGl.BindTexture = glBindTexture;
        gFilterGl.TexParameteri = glTexParameteri;
        gFilterGl.Viewport = glViewport;
        gFilterGl.DrawArrays = glDrawArrays;
        gFilterGl.Disable = glDisable;
    });
}

// The AHB staging buffer is allocated AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, so
// a 10-bit or FP16 swapchain would round-trip the REAL frame through 8 bits.
// That is invisible while we only read the frame for generation, but the filter
// copies its result back — so refuse rather than quietly degrade the game.
static bool is8BitSwapchain(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return true;
        default:
            return false;
    }
}

static uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memProps,
                                uint32_t typeFilter, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    // Fallback: any compatible type
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (typeFilter & (1 << i)) return i;
    }
    return 0;
}

// ─── AHB Image: shared between Vulkan and GLES ─────────────────────────────

struct AHBImage {
    AHardwareBuffer* ahb = nullptr;
    VkImage vkImage = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    EGLImageKHR eglImage = EGL_NO_IMAGE_KHR;
    GLuint glTex = 0;
    bool valid = false;
};

// ─── AFME Context: one per swapchain ────────────────────────────────────────

struct AFMEContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent = {0, 0};
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    std::vector<VkImage> swapchainImages;
    uint32_t queueFamilyIndex = 0;

    // Vulkan resources
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkFence copyFence = VK_NULL_HANDLE;  // Fence for copy-to-AHB step
    VkPhysicalDeviceMemoryProperties memProps = {};

    // Pre-allocated semaphore pool (avoid per-frame create/destroy)
    VkSemaphore semPool[kMaxSemaphorePool] = {};
    uint32_t semPoolSize = 0;
    uint32_t semPoolCursor = 0;

    // Pre-allocated command buffer ring (avoid hot-path alloc/free)
    VkCommandBuffer cmdRing[kCmdRingSize] = {};
    uint32_t cmdRingSize = 0;
    uint32_t cmdRingCursor = 0;

    // SGSR1 GLES resources
    GLuint sgsrProgram = 0;
    GLuint sgsrFbo = 0;
    GLuint sgsrVao = 0;
    GLint sgsrViewportInfoLoc = -1;
    GLint sgsrPs0Loc = -1;
    bool sgsrInitialized = false;
    AHBImage sharpenedFrame;  // Output of SGSR1 sharpening

    // Fence signaled by the LAST synth copy submit of a frame; waited at the
    // start of the next frame instead of vkQueueWaitIdle (which also waited
    // for the game's freshly submitted rendering and destroyed pipelining).
    VkFence drainFence = VK_NULL_HANDLE;
    bool drainPending = false;

    // Cadence control law, statistics and the game-loop discriminator. Shared
    // with the GLES layer — see afme_core.h. Per context, never file-scope:
    // games recreate swapchains (ZZZ makes several at startup) and a static
    // baseline from the old one underflows against the new context's small
    // counters (observed: "real=2147483647 total=-2" for one window).
    afme::Pacer pacer;
    afme::Stats stats;
    afme::EngagementGate gate;

    // Color filter. stageFrame holds the untouched present image; the filter
    // grades it INTO currFrame, so generation and the real frame both see the
    // graded result and the grade costs one pass per real frame at any
    // multiplier. Allocated lazily — a session that never enables the filter
    // pays nothing.
    AHBImage stageFrame;
    // Stage B output. Only allocated when a screen-space effect is actually
    // set, because it is a full extra frame of memory.
    AHBImage presentFrame;
    // Generation scratch: with stage B on, synthesis writes here and stage B
    // copies out to the synth AHB, since a pass cannot read and write one
    // texture. Without stage B, synthesis writes the AHB directly as before.
    GLuint genScratchTex = 0;
    afme::Filter filter;
    bool filterUnsupported = false;   // non-8-bit swapchain: refuse, do not degrade

    // MobFGSR is built at swapchain creation, but the method property can flip
    // mid-session; this lets us build it on first use without retrying forever.
    bool mobfgsrAttempted = false;

    // VK_GOOGLE_display_timing pacing for synthetic frames
    bool hasDisplayTiming = false;
    uint32_t presentId = 0;
    // Measured panel refresh cycle (vkGetRefreshCycleDurationGOOGLE); 0 = use
    // the afme::config().displayHz prop instead. Vsync grid
    // calibration: the panel CAN differ from the staged prop (e.g. Battery
    // Saver votes 60Hz over the GameStateDispatcher force), and tier/limiter
    // math against the wrong grid produced exactly the invisible-generation
    // cadence seen on device.
    uint64_t refreshCycleNs = 0;
    PFNGLSHADINGRATEQCOMPROC glShadingRate = nullptr;

    // Native-fence (sync_fd) VK↔GLES interop — replaces vkWaitForFences and
    // glFinish on the game's render thread with GPU-side waits.
    bool hasNativeFenceSync = false;
    bool copyFencePending = false;  // copyFence submitted, not yet CPU-waited
    VkSemaphore genDoneSem = VK_NULL_HANDLE;   // GLES gen complete → VK copy
    PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR_ = nullptr;
    PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR_ = nullptr;
    PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR_ = nullptr;
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFD_ = nullptr;

    // AHB frames for AFME
    AHBImage prevFrame;
    AHBImage currFrame;
    AHBImage synthFrames[afme::kMaxMultiplier - 1]; // up to 3 for 4x

    // EGL/GLES context (owned by this swapchain context)
    EGLDisplay eglDpy = EGL_NO_DISPLAY;
    EGLContext eglCtx = EGL_NO_CONTEXT;
    EGLSurface eglSurf = EGL_NO_SURFACE;

    // GLES extension pointers
    PFNGLEXTRAPOLATETEX2DQCOMPROC glExtrapolateTex2D = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2D = nullptr;
    PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBuffer = nullptr;
    PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR = nullptr;

    // QCOM HW motion estimation + depth estimation
    PFNGLTEXESTIMATEMOTIONQCOMPROC glTexEstimateMotion = nullptr;
    PFNGLTEXGENERATEDISPARITYQCOMPROC glTexGenerateDisparity = nullptr;
    int motionBlockX = 16;  // queried at runtime
    int motionBlockY = 16;

    // ─── MobFGSR resources ───────────────────────────────────
    bool mobfgsrInitialized = false;
    GLuint lumaConvertProg = 0;    // RGB→R8 luminance (render pass, not compute)
    GLuint lumaFbo = 0;            // render target for the luminance pass
    GLuint lumaVao = 0;            // empty VAO for the fullscreen triangle
    GLint  lumaSrcLoc = -1;
    GLuint mvUpsampleProg = 0;     // Block-level MV → per-pixel MV
    GLuint dilateProg = 0;         // Nearest-depth dilation
    GLuint clearProg = 0;          // Clear reprojection buffer
    GLuint reprojectProg = 0;      // Scatter reproject with atomicMin
    GLuint fillProg = 0;           // Fill reprojection holes
    GLuint warpProg = 0;           // Warp + blend final output
    GLuint mobfgsrUBO = 0;         // Uniform buffer (binding=10)

    // MobFGSR GLES textures (NOT AHB — pure GLES on our private context)
    GLuint prevLumaTex = 0;        // R8 luminance, render res
    GLuint currLumaTex = 0;        // R8 luminance, render res
    GLuint motionVecBlockTex = 0;  // RGBA16F, W/blockX × H/blockY
    GLuint motionVecTex = 0;       // RG16F, per-pixel (upsampled)
    GLuint prevMotionVecTex = 0;   // RG16F, previous frame
    GLuint depthTex = 0;           // R32F, estimated depth
    GLuint prevDepthTex = 0;       // R32F, previous depth
    GLuint dilatedDepthTex = 0;    // R32F, dilated current
    GLuint dilatedMVTex = 0;       // RG16F, dilated current
    GLuint prevDilatedDepthTex = 0; // R32F, dilated previous
    GLuint prevDilatedMVTex = 0;   // RG16F, dilated previous
    GLuint reprojectionTex = 0;    // R32UI, reproject scatter buf
    GLuint filledReprojTex = 0;    // R32UI, filled reprojection
    GLuint fgResultTex = 0;        // RGBA8, frame gen result

    // ── HUD ghost protection ────────────────────────────────────────────
    // Per-block (motion-grid) accumulation of screen-STATIC content
    // (minimap, HP bars, buttons, subtitles). Moving world pixels fail the
    // "static" test, so the warp keeps blending them normally; HUD pixels
    // are locked to the current real frame instead of the warped/interpolated
    // value — which is what stops the classic FG "shadow trail" on UI.
    GLuint hudMaskProg = 0;        // block-grid static accumulation
    GLuint hudMaskTex = 0;         // R32F, W/blockX × H/blockY (write side)
    GLuint prevHudMaskTex = 0;     // R32F, read side (baseline, swapped)

    bool initialized = false;
    bool afmeHWAvailable = false;
    bool hasPrevFrame = false;
    int allocatedMult = 2;       // Multiplier at init time (# synth frames = allocatedMult-1)
    uint64_t frameIdx = 0;
    uint64_t genFrames = 0;
    uint64_t skippedFrames = 0;  // Frames skipped due to scene change

    // Semaphore pool helpers — cursor resets each frame to ensure drain
    VkSemaphore acquireSem() {
        if (semPoolSize == 0) return VK_NULL_HANDLE;
        VkSemaphore s = semPool[semPoolCursor % semPoolSize];
        semPoolCursor++;
        return s;
    }

    // Command buffer ring — cursor resets each frame
    VkCommandBuffer acquireCmd() {
        if (cmdRingSize == 0) return VK_NULL_HANDLE;
        VkCommandBuffer cb = cmdRing[cmdRingCursor % cmdRingSize];
        cmdRingCursor++;
        return cb;
    }
};

// Effective panel rate: prefer the measured refresh cycle (display_timing)
// over the staged prop — the two can disagree (Battery Saver 60Hz vote,
// 90Hz override sessions, future panels), and running the tier/limiter math
// on the wrong grid is how generation becomes invisible.
static inline int effectiveHz(const AFMEContext& ctx) {
    if (ctx.refreshCycleNs >= 2000000 && ctx.refreshCycleNs <= 100000000) { // 10..500 Hz
        int hz = (int)(1000000000.0 / (double)ctx.refreshCycleNs + 0.5);
        if (hz >= 10 && hz <= 500) return hz;
    }
    return afme::config().displayHz.load(std::memory_order_relaxed);
}

// Set fragment shading rate for our passes when the driver supports it.
static inline void setFgShadingRate(const AFMEContext& ctx, GLenum rate) {
    if (ctx.glShadingRate && afme::config().vrsFg.load(std::memory_order_relaxed)) {
        ctx.glShadingRate(rate);
    }
}

// Swapchain → context mapping
std::unordered_map<VkSwapchainKHR, AFMEContext> gSwapchainContexts;
// Device → state mapping
std::unordered_map<VkDevice, VkPhysicalDevice> gDeviceToPhysical;
// Device → memProps mapping
std::unordered_map<VkDevice, VkPhysicalDeviceMemoryProperties> gDeviceMemProps;
// Device → VK_GOOGLE_display_timing enabled (for synth-frame pacing)
std::unordered_map<VkDevice, bool> gDeviceHasGoogleTiming;
// Device → VK_KHR_external_fence_fd + VK_KHR_external_semaphore_fd enabled
// (for GPU-side VK↔GLES sync without blocking the game thread)
std::unordered_map<VkDevice, bool> gDeviceHasNativeFence;

// ─── EGL Context Setup ─────────────────────────────────────────────────────

static bool initEGLContext(AFMEContext& ctx) {
    ctx.eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (ctx.eglDpy == EGL_NO_DISPLAY) {
        ALOGE("AFME: eglGetDisplay failed");
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(ctx.eglDpy, &major, &minor)) {
        ALOGE("AFME: eglInitialize failed: 0x%x", eglGetError());
        return false;
    }

    EGLint configAttribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(ctx.eglDpy, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
        ALOGE("AFME: eglChooseConfig failed");
        return false;
    }

    EGLint surfAttribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    ctx.eglSurf = eglCreatePbufferSurface(ctx.eglDpy, config, surfAttribs);

    EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    ctx.eglCtx = eglCreateContext(ctx.eglDpy, config, EGL_NO_CONTEXT, ctxAttribs);
    if (ctx.eglCtx == EGL_NO_CONTEXT) {
        ALOGE("AFME: eglCreateContext failed: 0x%x", eglGetError());
        return false;
    }

    // Resolve extension functions (don't make current yet — we save/restore)
    ctx.glExtrapolateTex2D = (PFNGLEXTRAPOLATETEX2DQCOMPROC)
        eglGetProcAddress("glExtrapolateTex2DQCOM");
    ctx.glEGLImageTargetTexture2D = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
        eglGetProcAddress("glEGLImageTargetTexture2DOES");
    ctx.eglGetNativeClientBuffer = (PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC)
        eglGetProcAddress("eglGetNativeClientBufferANDROID");
    ctx.eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)
        eglGetProcAddress("eglCreateImageKHR");
    ctx.eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)
        eglGetProcAddress("eglDestroyImageKHR");

    ctx.afmeHWAvailable = ctx.glExtrapolateTex2D
        && ctx.glEGLImageTargetTexture2D
        && ctx.eglGetNativeClientBuffer
        && ctx.eglCreateImageKHR;

    ALOGI("AFME: EGL init — HW %s (glExtrapolateTex2DQCOM=%p)",
          ctx.afmeHWAvailable ? "AVAILABLE" : "UNAVAILABLE", ctx.glExtrapolateTex2D);

    // Resolve QCOM motion estimation. RE of libGLESv2_adreno.so confirms
    // glTexEstimateMotionQCOM is impl(ctx, uint prev, uint curr, uint outMV)
    // — exactly our 3-arg typedef, and GL_QCOM_motion_estimation is an
    // advertised extension. Safe to call.
    ctx.glTexEstimateMotion = (PFNGLTEXESTIMATEMOTIONQCOMPROC)
        eglGetProcAddress("glTexEstimateMotionQCOM");

    // VRS for our own fragment passes (generation cost drops ~4x on the
    // shaded passes)
    ctx.glShadingRate = (PFNGLSHADINGRATEQCOMPROC)
        eglGetProcAddress("glShadingRateQCOM");

    // glTexGenerateDisparityQCOM is deliberately NOT resolved. RE of the
    // driver export shows its real ABI is
    //   (uint,uint,uint,uint,uint,uint,uint,uint, float,float)  — 8 uint + 2f
    // not the (uint,uint) this layer previously assumed. It is also NOT in
    // the driver's advertised GL_QCOM_* extension list (it's a hidden stereo
    // disparity primitive, not a monocular depth estimator). Calling it with
    // 2 args passed undefined register values as the other 8 params →
    // UB/corrupt output. MobFGSR runs its clean degraded path instead
    // (depthTex stays 0 → occlusion handling off, interpolation still valid).
    ctx.glTexGenerateDisparity = nullptr;

    // EGL_ANDROID_native_fence_sync + EGL_KHR_wait_sync for GPU-side VK↔GLES
    // synchronization (no game-thread blocking)
    ctx.eglCreateSyncKHR_ = (PFNEGLCREATESYNCKHRPROC)
        eglGetProcAddress("eglCreateSyncKHR");
    ctx.eglDestroySyncKHR_ = (PFNEGLDESTROYSYNCKHRPROC)
        eglGetProcAddress("eglDestroySyncKHR");
    ctx.eglWaitSyncKHR_ = (PFNEGLWAITSYNCKHRPROC)
        eglGetProcAddress("eglWaitSyncKHR");
    ctx.eglDupNativeFenceFD_ = (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)
        eglGetProcAddress("eglDupNativeFenceFDANDROID");

    ALOGI("AFME: HW MotionEstimation=%p DepthEstimation=%p",
          ctx.glTexEstimateMotion, ctx.glTexGenerateDisparity);

    return true;
}

// ─── SGSR1 Shader Initialization ────────────────────────────────────────────

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        ALOGE("AFME: SGSR1 shader compile error: %s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static bool initSGSR(AFMEContext& ctx) {
    // Must be called with AFME's EGL context current
    GLuint vs = compileShader(GL_VERTEX_SHADER, kSGSR1VertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kSGSR1FragSrc);
    if (!vs || !fs) {
        ALOGE("AFME: SGSR1 shader compilation failed");
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return false;
    }

    ctx.sgsrProgram = glCreateProgram();
    glAttachShader(ctx.sgsrProgram, vs);
    glAttachShader(ctx.sgsrProgram, fs);
    glLinkProgram(ctx.sgsrProgram);

    GLint linked = 0;
    glGetProgramiv(ctx.sgsrProgram, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512];
        glGetProgramInfoLog(ctx.sgsrProgram, sizeof(log), nullptr, log);
        ALOGE("AFME: SGSR1 program link error: %s", log);
        glDeleteProgram(ctx.sgsrProgram);
        ctx.sgsrProgram = 0;
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }

    glDeleteShader(vs);  // Safe to delete after linking
    glDeleteShader(fs);

    ctx.sgsrViewportInfoLoc = glGetUniformLocation(ctx.sgsrProgram, "ViewportInfo");
    ctx.sgsrPs0Loc = glGetUniformLocation(ctx.sgsrProgram, "ps0");

    glGenFramebuffers(1, &ctx.sgsrFbo);
    glGenVertexArrays(1, &ctx.sgsrVao);

    ctx.sgsrInitialized = true;
    ALOGI("AFME: SGSR1 shader initialized (program=%u, viewportInfo=%d, ps0=%d)",
          ctx.sgsrProgram, ctx.sgsrViewportInfoLoc, ctx.sgsrPs0Loc);
    return true;
}

// Apply SGSR1 sharpening: inputTex → outputTex (same resolution)
// Must be called with AFME's EGL context current
static void applySGSR1(AFMEContext& ctx, GLuint inputTex, GLuint outputTex,
                       uint32_t w, uint32_t h) {
    glUseProgram(ctx.sgsrProgram);

    // ViewportInfo = {1/w, 1/h, w, h}
    float viewportInfo[4] = {1.0f/(float)w, 1.0f/(float)h, (float)w, (float)h};
    glUniform4fv(ctx.sgsrViewportInfoLoc, 1, viewportInfo);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glUniform1i(ctx.sgsrPs0Loc, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, ctx.sgsrFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, outputTex, 0);
    glViewport(0, 0, w, h);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);

    setFgShadingRate(ctx, GL_SHADING_RATE_2X2_PIXELS_QCOM);
    glBindVertexArray(ctx.sgsrVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    setFgShadingRate(ctx, GL_SHADING_RATE_1X1_PIXELS_QCOM);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

// ─── MobFGSR Compute Shaders (ported to GLES 310 es) ───────────────────────
//
// Pipeline: HW motion estimation → dilate → clear → reproject → fill → warp
// Based on MobFGSR (BSD-3 license) adapted for Adreno 840 HW primitives.

// RGB → R8 luminance for motion estimation input.
//
// Done as a render-to-texture pass, not a compute imageStore: `r8` is not a
// required image format in GLSL ES 3.1/3.2, and this Adreno build does not
// expose GL_NV_image_formats, so `layout(r8, ...)` fails to compile with
// "not a legal layout qualifier id". R8 *is* colour-renderable though, so a
// plain fragment shader writing into an FBO works and keeps the single-channel
// format glTexEstimateMotionQCOM expects for its inputs.
static const char* kLumaVertSrc = R"(#version 300 es
out vec2 vUV;
void main() {
    // ids 0,1,2 -> (-1,-1), (3,-1), (-1,3): one triangle covering the viewport
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
})";

static const char* kLumaFragSrc = R"(#version 300 es
precision mediump float;
uniform mediump sampler2D uSrc;
in vec2 vUV;
out vec4 outColor;
void main() {
    vec3 c = texture(uSrc, vUV).rgb;
    outColor = vec4(0.299 * c.r + 0.587 * c.g + 0.114 * c.b, 0.0, 0.0, 1.0);
})";

// Compute shader: Bilinear upsample block-level MVs → per-pixel MVs
// glTexEstimateMotionQCOM outputs pixel displacements at block granularity.
// MobFGSR expects UV-space (normalized) motion vectors at per-pixel resolution.
static const char* kMVUpsampleSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
uniform mediump sampler2D blockMV;
layout(rgba16f, binding=0) writeonly uniform mediump image2D perPixelMV;
uniform ivec2 renderSize;
uniform ivec2 blockSize;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, renderSize))) return;
    // Map pixel position to UV in [0,1] — bilinear sampling handles block boundaries
    vec2 uv = (vec2(p) + 0.5) / vec2(renderSize);
    // Sample block-level MV texture (bilinear interpolation upsamples automatically)
    vec2 mvPixels = textureLod(blockMV, uv, 0.0).xy;
    // Convert from pixel displacement to UV-space for MobFGSR shaders
    vec2 mvUV = mvPixels / vec2(renderSize);
    imageStore(perPixelMV, p, vec4(mvUV, 0.0, 0.0));
})";

// Compute shader: Nearest-depth dilation (from MobFGSR Dilate.comp)
static const char* kDilateSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform mediump sampler2D r_depth;
layout(binding=1) uniform mediump sampler2D r_mv;
layout(r32f, binding=0) writeonly uniform highp image2D rw_dilated_depth;
layout(rgba16f, binding=1) writeonly uniform mediump image2D rw_dilated_mv;
uniform ivec2 renderSize;
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, renderSize))) return;
    const ivec2 offsets[8] = ivec2[8](
        ivec2(-1,-1),ivec2(-1,0),ivec2(-1,1),ivec2(0,-1),
        ivec2(0,1),ivec2(1,-1),ivec2(1,0),ivec2(1,1));
    ivec2 nearestPos = pos;
    float nearestDepth = texelFetch(r_depth, pos, 0).x;
    for (int i=0; i<8; i++) {
        ivec2 sp = clamp(pos + offsets[i], ivec2(0), renderSize - ivec2(1));
        float d = texelFetch(r_depth, sp, 0).x;
        if (d < nearestDepth) { nearestPos = sp; nearestDepth = d; }
    }
    vec2 mv = texelFetch(r_mv, nearestPos, 0).xy;
    imageStore(rw_dilated_depth, pos, vec4(nearestDepth));
    imageStore(rw_dilated_mv, pos, vec4(mv, 0.0, 0.0));
})";

// Compute shader: Clear reprojection buffer
static const char* kClearSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
layout(r32ui, binding=0) writeonly uniform highp uimage2D rw_reproj;
void main() {
    imageStore(rw_reproj, ivec2(gl_GlobalInvocationID.xy), uvec4(0xFFFFFFFFu));
})";

// Compute shader: Reproject with atomicMin (from MobFGSR Reproject_I.comp)
static const char* kReprojectSrc = R"(#version 310 es
// imageAtomicMin is not core until ES 3.2; on 3.1 it must be asked for by name.
// Adreno 840 advertises GL_OES_shader_image_atomic, so this compiles here —
// without it the shader failed with "requires extension ... to be enabled" and
// took the whole MobFGSR pipeline down with it.
#extension GL_OES_shader_image_atomic : require
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform mediump sampler2D r_depth;
layout(binding=1) uniform mediump sampler2D r_cur_mv;
layout(binding=2) uniform mediump sampler2D r_prev_mv;
layout(r32ui, binding=0) coherent uniform highp uimage2D rw_reproj;
uniform ivec2 renderSize;
uniform float delta;
const uint depthBits = 11u;
const uint xBits = 11u;
const uint yBits = 10u;
const uint maxDepth = (1u << depthBits) - 1u;
const int minX = -(1 << (int(xBits)-1));
const int minY = -(1 << (int(yBits)-1));
const int maxX = (1 << (int(xBits)-1)) - 1;
const int maxY_ = (1 << (int(yBits)-1)) - 1;
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, renderSize))) return;
    vec2 rsInv = 1.0 / vec2(renderSize);
    vec2 uv = (vec2(pos) + 0.5) * rsInv;
    vec2 mv_t1 = texelFetch(r_cur_mv, pos, 0).xy;
    ivec2 pos_t0 = ivec2((uv - mv_t1) * vec2(renderSize));
    vec2 mv_t0 = texelFetch(r_prev_mv, clamp(pos_t0,ivec2(0),renderSize-ivec2(1)), 0).xy;
    float d2 = delta * 0.5;
    float dw = delta * delta * 0.5;
    vec2 uvDelta = uv + (-1.0+d2+dw)*mv_t1 + (d2-dw)*mv_t0;
    if (all(greaterThanEqual(uvDelta, vec2(0.0))) && all(lessThanEqual(uvDelta, vec2(1.0)))) {
        ivec2 posDelta = ivec2(uvDelta * vec2(renderSize));
        float depth = texelFetch(r_depth, pos, 0).x;
        uint uD = uint(float(maxDepth) * depth);
        ivec2 rel = clamp(posDelta - pos, ivec2(minX,minY), ivec2(maxX,maxY_));
        uvec2 uRel = uvec2(rel - ivec2(minX,minY));
        uint data = (uD << (32u-depthBits)) | (uRel.x << yBits) | uRel.y;
        imageAtomicMin(rw_reproj, posDelta, data);
    }
})";

// Compute shader: Fill holes in reprojection (from MobFGSR Fill.comp)
static const char* kFillSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform highp usampler2D r_reproj;
layout(r32ui, binding=0) writeonly uniform highp uimage2D rw_filled;
uniform ivec2 renderSize;
const uint INV = 0xFFFFFFFFu;
const uint depthBits = 11u;
float unpackDepth(uint d) { return float(d >> (32u-depthBits)) / float((1u<<depthBits)-1u); }
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, renderSize))) return;
    uint center = texelFetch(r_reproj, pos, 0).x;
    float cDepth = unpackDepth(center);
    float nearest = 1.0;
    uint selected = INV;
    uint mask = 1u << 4u;
    const ivec2 off[9] = ivec2[9](
        ivec2(-1,-1),ivec2(-1,0),ivec2(-1,1),
        ivec2(0,-1),ivec2(0,0),ivec2(0,1),
        ivec2(1,-1),ivec2(1,0),ivec2(1,1));
    for (int i=0; i<9; i++) {
        if (i==4) continue;
        ivec2 np = clamp(pos+off[i], ivec2(0), renderSize-ivec2(1));
        uint nd = texelFetch(r_reproj, np, 0).x;
        float nDepth = unpackDepth(nd);
        float diff = cDepth - nDepth;
        if (nd != INV && diff > 0.0005) {
            if (nDepth < nearest) { nearest = nDepth; selected = nd; }
        } else { mask |= (1u << uint(i)); }
    }
    const uint rej[4] = uint[4](
        (1u<<0u)|(1u<<1u)|(1u<<3u)|(1u<<4u),
        (1u<<1u)|(1u<<2u)|(1u<<4u)|(1u<<5u),
        (1u<<3u)|(1u<<4u)|(1u<<6u)|(1u<<7u),
        (1u<<4u)|(1u<<5u)|(1u<<7u)|(1u<<8u));
    bool reject = ((mask&rej[0])==rej[0])||((mask&rej[1])==rej[1])||
                  ((mask&rej[2])==rej[2])||((mask&rej[3])==rej[3]);
    uint result;
    if (reject) { result = (center != INV) ? center : INV; }
    else { result = selected; }
    imageStore(rw_filled, pos, uvec4(result));
})";

// Compute shader: HUD mask — per-BLOCK accumulation of screen-static content
// (minimap frames, HP bars, buttons, prompts). Runs on the motion-estimation
// block grid (8x8 px on Adreno 840), so one texel per ME block; the warp
// shader bilinearly upsamples it for free.
//
// Two tests, both required — luma alone is fooled by static sky/ground, MV
// alone is fooled by small-magnitude real motion:
//   1. 9-tap mean |Δluma| at the block centre ≈ 0   (content isn't changing)
//   2. the block's motion vector length < threshold (ME agrees it isn't moving)
// A static block accumulates toward 1 (+0.10/frame ≈ 10 frames to lock), a
// moved block releases fast (−0.50/frame ≈ 2 frames) so it can never smear.
// Note this is SAFE for static WORLD content too: static world pixels are
// identical in curr and prev, so pinning them to curr is exact, not a guess.
static const char* kHudMaskSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform mediump sampler2D r_curr_luma;
layout(binding=1) uniform mediump sampler2D r_prev_luma;
layout(binding=2) uniform highp sampler2D r_block_mv;
layout(binding=3) uniform mediump sampler2D r_prev_mask;
// r32f, NOT r8: ESSL 3.10 only guarantees rgba32f/rgba16f/r32f/rgba8/
// rgba8_snorm/rgba*ui/r32ui/rgba*i/r32i as image formats. The Adreno compiler
// rejects r8 ("not a legal layout qualifier id"), which failed the whole of
// initMobFGSR and silently downgraded method=motion to extrapolation — the
// interpolation engine never ran once on device.
layout(r32f, binding=0) writeonly uniform highp image2D rw_mask;
uniform ivec2 lumaSize;
uniform ivec2 blockSize;
uniform float lumaThr;
uniform float mvThr;
uniform float upRate;
uniform float downRate;
void main() {
    ivec2 blk = ivec2(gl_GlobalInvocationID.xy);
    ivec2 maskSize = imageSize(rw_mask);
    if (any(greaterThanEqual(blk, maskSize))) return;
    ivec2 c = blk * blockSize + blockSize / 2;
    float diff = 0.0;
    for (int dy = -2; dy <= 2; dy += 2) {
        for (int dx = -2; dx <= 2; dx += 2) {
            ivec2 p = clamp(c + ivec2(dx, dy), ivec2(0), lumaSize - ivec2(1));
            diff += abs(texelFetch(r_curr_luma, p, 0).x -
                        texelFetch(r_prev_luma, p, 0).x);
        }
    }
    diff /= 9.0;
    vec2 mvPx = texelFetch(r_block_mv, blk, 0).xy;
    bool isStatic = (diff < lumaThr) && (length(mvPx) < mvThr);
    float prev = texelFetch(r_prev_mask, blk, 0).x;
    float m = clamp(prev + (isStatic ? upRate : -downRate), 0.0, 1.0);
    imageStore(rw_mask, blk, vec4(m));
})";

// Compute shader: Warp + blend interpolation (from MobFGSR Warp_I.comp)
static const char* kWarpSrc = R"(#version 310 es
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform highp usampler2D r_filled;
layout(binding=1) uniform mediump sampler2D r_cur_color;
layout(binding=2) uniform mediump sampler2D r_prev_color;
layout(binding=3) uniform mediump sampler2D r_cur_depth;
layout(binding=4) uniform mediump sampler2D r_prev_depth;
layout(binding=5) uniform mediump sampler2D r_cur_mv;
layout(binding=6) uniform mediump sampler2D r_prev_mv;
layout(binding=7) uniform mediump sampler2D r_hud_mask;
layout(rgba8, binding=0) writeonly uniform mediump image2D rw_result;
uniform ivec2 renderSize;
uniform float delta;
uniform float depthThreshold;
uniform float colorThreshold;
uniform float hudStrength;
uniform float holeAnchor;    // 1 = pin deep-occlusion holes to the real pixel
uniform float ghostStrength; // 1 = damp blend when warped samples disagree
const uint INV = 0xFFFFFFFFu;
const uint depthBits = 11u;
const uint xBits = 11u;
const uint yBits = 10u;
const int minX = -(1 << (int(xBits)-1));
const int minY = -(1 << (int(yBits)-1));
ivec2 unpackSrc(uint d, ivec2 tgt) {
    uint ux = (d >> yBits) & ((1u<<xBits)-1u);
    uint uy = d & ((1u<<yBits)-1u);
    return tgt - (ivec2(ux,uy) + ivec2(minX,minY));
}
void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, renderSize))) return;
    vec2 rsInv = 1.0 / vec2(renderSize);
    vec2 uv = (vec2(pos) + 0.5) * rsInv;
    uint packed = texelFetch(r_filled, pos, 0).x;
    vec2 mv_t1;
    ivec2 srcPos = ivec2(-1);
    if (packed == INV) {
        // DEEP HOLE: the scatter-reprojection placed no source here — this is
        // a disocclusion (background revealed behind the moving character,
        // content entering past the frame edge). The legacy behaviour guessed
        // an MV from the *previous* frame's field and blended two warped
        // samples along it — the exact manufacturing recipe for the classic
        // frame-gen double-exposure "shadow trail" trailing the character.
        // Anchoring to the real current pixel is artifact-free: the content
        // at that exact location is the CORRECT instantaneous content.
        if (holeAnchor > 0.5) {
            vec3 px = texelFetch(r_cur_color, pos, 0).rgb;
            imageStore(rw_result, pos, vec4(px, 1.0));
            return;
        }
        mv_t1 = texelFetch(r_prev_mv, pos, 0).xy;
    } else {
        srcPos = unpackSrc(packed, pos);
        srcPos = clamp(srcPos, ivec2(0), renderSize - ivec2(1));
        mv_t1 = texelFetch(r_cur_mv, srcPos, 0).xy;
    }
    vec2 sampleUV_t1, sampleUV_t0;
    if (srcPos.x < 0) {
        sampleUV_t1 = uv + mv_t1 * (1.0 - delta);
        sampleUV_t0 = uv - mv_t1 * delta;
    } else {
        vec2 uv_t1 = (vec2(srcPos) + 0.5) * rsInv;
        vec2 uv_t0 = uv_t1 - mv_t1;
        vec2 mv_t0 = textureLod(r_prev_mv, uv_t0, 0.0).xy;
        float d2 = delta * 0.5;
        float dw = delta * delta * 0.5;
        sampleUV_t0 = uv + (-d2 - dw) * mv_t1 + (-d2 + dw) * mv_t0;
        sampleUV_t1 = mv_t1 + sampleUV_t0;
    }
    vec3 c1 = textureLod(r_cur_color, clamp(sampleUV_t1, vec2(0.0), vec2(1.0)), 0.0).rgb;
    vec3 c0 = textureLod(r_prev_color, clamp(sampleUV_t0, vec2(0.0), vec2(1.0)), 0.0).rgb;
    float d1 = texelFetch(r_cur_depth, clamp(ivec2(sampleUV_t1*vec2(renderSize)),ivec2(0),renderSize-ivec2(1)), 0).x;
    float d0 = texelFetch(r_prev_depth, clamp(ivec2(sampleUV_t0*vec2(renderSize)),ivec2(0),renderSize-ivec2(1)), 0).x;
    vec3 color;
    float dd = abs(d0 - d1);
    if (any(lessThan(sampleUV_t0, vec2(0.0))) || any(greaterThan(sampleUV_t0, vec2(1.0)))) {
        color = c1;
    } else if (any(lessThan(sampleUV_t1, vec2(0.0))) || any(greaterThan(sampleUV_t1, vec2(1.0)))) {
        color = c0;
    } else if (dd < depthThreshold) {
        vec3 cd = abs(c1 - c0);
        float ld = cd.r * 0.5 + cd.b * 0.5 + cd.g;
        if (ld < colorThreshold) {
            color = delta < 0.5 ? c0 : c1;
        } else {
            color = mix(c0, c1, delta);
            // Anti-ghost damping: a large disagreement between the two warped
            // samples means this pixel's MV is unreliable (silhouette edge of
            // a running character, animated/damage text, ME miss). Blending
            // them 50/50 manufactures two offset ghosts; damping toward the
            // temporally closer sample keeps ONE clean image instead.
            float ghost = clamp((ld - colorThreshold) * 8.0, 0.0, 1.0)
                          * ghostStrength;
            if (ghost > 0.0) {
                vec3 near_ = delta < 0.5 ? c0 : c1;
                color = mix(color, near_, ghost * 0.75);
            }
        }
    } else {
        color = d1 > d0 ? c1 : c0;
    }

    // HUD ghost protection: screen-static pixels (minimap, bars, prompts)
    // must NOT show the interpolated blend — anchor them to the real current
    // frame at their exact position. Static world pixels are unaffected in
    // practice (they match curr anyway).
    float hud = textureLod(r_hud_mask, uv, 0.0).x * hudStrength;
    if (hud > 0.0) {
        vec3 px = texelFetch(r_cur_color, pos, 0).rgb;
        color = mix(color, px, clamp(hud, 0.0, 1.0));
    }
    imageStore(rw_result, pos, vec4(color, 1.0));
})";

// Helper: compile a GLES compute shader program from source
static GLuint compileComputeProgram(const char* src, const char* name) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        ALOGE("AFME: %s compile error: %s", name, log);
        glDeleteShader(shader);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glDeleteShader(shader);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        ALOGE("AFME: %s link error: %s", name, log);
        glDeleteProgram(prog);
        return 0;
    }
    ALOGI("AFME: %s compiled OK (prog=%u)", name, prog);
    return prog;
}

// Helper: create a GLES texture with specific format
static GLuint createGLTex(GLenum internalFormat, uint32_t w, uint32_t h, GLenum filter) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static bool initMobFGSR(AFMEContext& ctx) {
    // Must be called with AFME's EGL context current
    if (!ctx.glTexEstimateMotion) {
        ALOGW("AFME: MobFGSR init skipped — no HW motion estimation");
        return false;
    }

    uint32_t w = ctx.extent.width;
    uint32_t h = ctx.extent.height;

    // Query HW motion estimation block size
    glGetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &ctx.motionBlockX);
    glGetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &ctx.motionBlockY);
    ALOGI("AFME: Motion estimation block size: %dx%d", ctx.motionBlockX, ctx.motionBlockY);

    // Compile all compute shaders
    // Luminance is a render pass (see kLumaFragSrc for why it cannot be compute)
    {
        GLuint vs = compileShader(GL_VERTEX_SHADER, kLumaVertSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, kLumaFragSrc);
        if (vs && fs) {
            ctx.lumaConvertProg = glCreateProgram();
            glAttachShader(ctx.lumaConvertProg, vs);
            glAttachShader(ctx.lumaConvertProg, fs);
            glLinkProgram(ctx.lumaConvertProg);
            GLint linked = 0;
            glGetProgramiv(ctx.lumaConvertProg, GL_LINK_STATUS, &linked);
            if (!linked) {
                char log[512];
                glGetProgramInfoLog(ctx.lumaConvertProg, sizeof(log), nullptr, log);
                ALOGE("AFME: Luma program link error: %s", log);
                glDeleteProgram(ctx.lumaConvertProg);
                ctx.lumaConvertProg = 0;
            } else {
                ctx.lumaSrcLoc = glGetUniformLocation(ctx.lumaConvertProg, "uSrc");
                glGenFramebuffers(1, &ctx.lumaFbo);
                glGenVertexArrays(1, &ctx.lumaVao);
                ALOGI("AFME: Luma render program ready (prog=%u)", ctx.lumaConvertProg);
            }
        }
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
    }
    ctx.mvUpsampleProg = compileComputeProgram(kMVUpsampleSrc, "MVUpsample");
    ctx.dilateProg = compileComputeProgram(kDilateSrc, "Dilate");
    ctx.clearProg = compileComputeProgram(kClearSrc, "Clear");
    ctx.reprojectProg = compileComputeProgram(kReprojectSrc, "Reproject");
    ctx.fillProg = compileComputeProgram(kFillSrc, "Fill");
    ctx.warpProg = compileComputeProgram(kWarpSrc, "Warp");
    ctx.hudMaskProg = compileComputeProgram(kHudMaskSrc, "HudMask");

    if (!ctx.lumaConvertProg || !ctx.mvUpsampleProg || !ctx.dilateProg ||
        !ctx.clearProg || !ctx.reprojectProg || !ctx.fillProg ||
        !ctx.warpProg || !ctx.hudMaskProg) {
        ALOGE("AFME: MobFGSR shader compilation failed");
        return false;
    }

    // Create GLES textures
    uint32_t mvW = w / ctx.motionBlockX;
    uint32_t mvH = h / ctx.motionBlockY;

    ctx.prevLumaTex = createGLTex(GL_R8, w, h, GL_NEAREST);
    ctx.currLumaTex = createGLTex(GL_R8, w, h, GL_NEAREST);
    ctx.motionVecBlockTex = createGLTex(GL_RGBA16F, mvW, mvH, GL_LINEAR);
    ctx.motionVecTex = createGLTex(GL_RGBA16F, w, h, GL_NEAREST);
    ctx.prevMotionVecTex = createGLTex(GL_RGBA16F, w, h, GL_NEAREST);
    ctx.depthTex = createGLTex(GL_R32F, w, h, GL_NEAREST);
    ctx.prevDepthTex = createGLTex(GL_R32F, w, h, GL_NEAREST);
    ctx.dilatedDepthTex = createGLTex(GL_R32F, w, h, GL_NEAREST);
    ctx.dilatedMVTex = createGLTex(GL_RGBA16F, w, h, GL_NEAREST);
    ctx.prevDilatedDepthTex = createGLTex(GL_R32F, w, h, GL_NEAREST);
    ctx.prevDilatedMVTex = createGLTex(GL_RGBA16F, w, h, GL_NEAREST);
    ctx.reprojectionTex = createGLTex(GL_R32UI, w, h, GL_NEAREST);
    ctx.filledReprojTex = createGLTex(GL_R32UI, w, h, GL_NEAREST);
    ctx.fgResultTex = createGLTex(GL_RGBA8, w, h, GL_LINEAR);

    // HUD mask ping-pong at the ME block grid; LINEAR so the warp pass gets
    // soft mask edges for free (hard 8px block edges would halo visibly).
    ctx.hudMaskTex = createGLTex(GL_R32F, mvW, mvH, GL_LINEAR);
    ctx.prevHudMaskTex = createGLTex(GL_R32F, mvW, mvH, GL_LINEAR);

    // Initialize both mask sides to 0 (no HUD assumed) — garbage would
    // otherwise pollute the first seconds of accumulation. Uploaded rather than
    // cleared through an FBO: R32F is only color-renderable with
    // EXT_color_buffer_float, and an incomplete-FBO clear would leave the masks
    // undefined on any driver that lacks it.
    {
        std::vector<float> zeros((size_t)mvW * (size_t)mvH, 0.0f);
        for (GLuint t : {ctx.hudMaskTex, ctx.prevHudMaskTex}) {
            glBindTexture(GL_TEXTURE_2D, t);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)mvW, (GLsizei)mvH,
                            GL_RED, GL_FLOAT, zeros.data());
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    ctx.mobfgsrInitialized = true;
    ALOGI("AFME: MobFGSR initialized (%ux%u, MV block %dx%d → %ux%u, hud mask %ux%u)",
          w, h, ctx.motionBlockX, ctx.motionBlockY, mvW, mvH, mvW, mvH);
    return true;
}

// MobFGSR stage 1 — per-GAME-frame preparation (delta-independent):
// luminance conversion, HW motion estimation, depth estimation, MV upsample,
// dilation. Run ONCE per real frame; the per-delta warp stage below reuses
// the results for every generated frame (3x/4x no longer recompute ME).
// Must be called with AFME's EGL context current.
static void applyMobFGSRPrepare(AFMEContext& ctx) {
    uint32_t w = ctx.extent.width;
    uint32_t h = ctx.extent.height;
    int gx = (w + 7) / 8;
    int gy = (h + 7) / 8;

    // Step 1: Convert current frame to luminance (R8) for HW motion estimation.
    // Render pass rather than compute — r8 is not a legal image-store format
    // here (no GL_NV_image_formats on this driver).
    glUseProgram(ctx.lumaConvertProg);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ctx.lumaFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, ctx.currLumaTex, 0);
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.currFrame.glTex);
    if (ctx.lumaSrcLoc >= 0) glUniform1i(ctx.lumaSrcLoc, 0);
    glBindVertexArray(ctx.lumaVao);
    // Luma is already a downsampled input for block ME — shade it at 2x2.
    setFgShadingRate(ctx, GL_SHADING_RATE_2X2_PIXELS_QCOM);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    setFgShadingRate(ctx, GL_SHADING_RATE_1X1_PIXELS_QCOM);
    // Detach before glTexEstimateMotionQCOM samples currLumaTex: reading a
    // texture still attached to the bound framebuffer is a feedback loop.
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // Step 2: HW motion estimation (Adreno accelerated!)
    ctx.glTexEstimateMotion(ctx.prevLumaTex, ctx.currLumaTex, ctx.motionVecBlockTex);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Step 3: HW depth estimation (or fallback)
    if (ctx.glTexGenerateDisparity) {
        ctx.glTexGenerateDisparity(ctx.currFrame.glTex, ctx.depthTex);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    }
    // If no HW disparity: depthTex stays at 0.0 (initialized by glTexStorage2D).
    // With uniform depth=0: Dilate→no-op, Warp depth diff → always 0 → blends normally.
    // This is a valid degraded mode: interpolation works but without occlusion handling.

    // Step 4: Upsample block-level MVs to per-pixel
    glUseProgram(ctx.mvUpsampleProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.motionVecBlockTex);
    glUniform1i(glGetUniformLocation(ctx.mvUpsampleProg, "blockMV"), 0);
    glBindImageTexture(0, ctx.motionVecTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glUniform2i(glGetUniformLocation(ctx.mvUpsampleProg, "renderSize"), w, h);
    glUniform2i(glGetUniformLocation(ctx.mvUpsampleProg, "blockSize"),
                ctx.motionBlockX, ctx.motionBlockY);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Step 5: Dilate depth + motion vectors (nearest-depth selection)
    glUseProgram(ctx.dilateProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.depthTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ctx.motionVecTex);
    glBindImageTexture(0, ctx.dilatedDepthTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(1, ctx.dilatedMVTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glUniform2i(glGetUniformLocation(ctx.dilateProg, "renderSize"), w, h);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Step 6: HUD mask accumulation on the block grid (cheap: ~w/8 × h/8
    // invocations). Reads prev-luma vs curr-luma + the just-estimated block
    // MVs, so it sees exactly what motion estimation saw this frame.
    {
        uint32_t mvW = w / ctx.motionBlockX;
        uint32_t mvH = h / ctx.motionBlockY;
        if (mvW == 0 || mvH == 0) return;
        int bgx = (int)((mvW + 7) / 8);
        int bgy = (int)((mvH + 7) / 8);

        glUseProgram(ctx.hudMaskProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ctx.currLumaTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, ctx.prevLumaTex);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, ctx.motionVecBlockTex);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, ctx.prevHudMaskTex);
        glBindImageTexture(0, ctx.hudMaskTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glUniform2i(glGetUniformLocation(ctx.hudMaskProg, "lumaSize"), (GLint)w, (GLint)h);
        glUniform2i(glGetUniformLocation(ctx.hudMaskProg, "blockSize"),
                    ctx.motionBlockX, ctx.motionBlockY);
        // Slightly above the VRS 2x2 luma noise floor.
        glUniform1f(glGetUniformLocation(ctx.hudMaskProg, "lumaThr"), 6.0f / 255.0f);
        glUniform1f(glGetUniformLocation(ctx.hudMaskProg, "mvThr"), 1.25f); // px
        glUniform1f(glGetUniformLocation(ctx.hudMaskProg, "upRate"), 0.10f);
        glUniform1f(glGetUniformLocation(ctx.hudMaskProg, "downRate"), 0.50f);
        glDispatchCompute(bgx, bgy, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    }
}

// MobFGSR stage 2 — per-GENERATED-frame warp for one interpolation position.
// Inputs: results of applyMobFGSRPrepare + prev/curr color.
// Output: fgResultTex (RGBA8).
// Must be called with AFME's EGL context current.
static void applyMobFGSRWarp(AFMEContext& ctx, float delta) {
    uint32_t w = ctx.extent.width;
    uint32_t h = ctx.extent.height;
    int gx = (w + 7) / 8;
    int gy = (h + 7) / 8;

    // Step 6: Clear reprojection buffer
    glUseProgram(ctx.clearProg);
    glBindImageTexture(0, ctx.reprojectionTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Step 7: Reproject — scatter current pixels to interpolated positions
    glUseProgram(ctx.reprojectProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.dilatedDepthTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ctx.dilatedMVTex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, ctx.prevDilatedMVTex);
    glBindImageTexture(0, ctx.reprojectionTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUniform2i(glGetUniformLocation(ctx.reprojectProg, "renderSize"), w, h);
    glUniform1f(glGetUniformLocation(ctx.reprojectProg, "delta"), delta);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Step 8: Fill — fill gaps in reprojection
    glUseProgram(ctx.fillProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.reprojectionTex);
    glBindImageTexture(0, ctx.filledReprojTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glUniform2i(glGetUniformLocation(ctx.fillProg, "renderSize"), w, h);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Step 9: Warp — sample from both frames, blend by delta
    glUseProgram(ctx.warpProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ctx.filledReprojTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ctx.currFrame.glTex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, ctx.prevFrame.glTex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, ctx.dilatedDepthTex);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, ctx.prevDilatedDepthTex);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, ctx.dilatedMVTex);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, ctx.prevDilatedMVTex);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, ctx.hudMaskTex);   // fresh mask from prepare
    glBindImageTexture(0, ctx.fgResultTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform2i(glGetUniformLocation(ctx.warpProg, "renderSize"), w, h);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "delta"), delta);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "depthThreshold"), 0.004f);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "colorThreshold"), 0.01f);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "hudStrength"),
                afme::config().hudMask.load(std::memory_order_relaxed) ? 1.0f : 0.0f);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "holeAnchor"),
                afme::config().antiGhost.load(std::memory_order_relaxed) ? 1.0f : 0.0f);
    glUniform1f(glGetUniformLocation(ctx.warpProg, "ghostStrength"),
                afme::config().antiGhost.load(std::memory_order_relaxed) ? 1.0f : 0.0f);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

// Swap MobFGSR prev/curr textures after each real frame
static void swapMobFGSRBuffers(AFMEContext& ctx) {
    std::swap(ctx.prevLumaTex, ctx.currLumaTex);
    std::swap(ctx.prevDepthTex, ctx.depthTex);
    std::swap(ctx.prevMotionVecTex, ctx.motionVecTex);
    std::swap(ctx.prevDilatedDepthTex, ctx.dilatedDepthTex);
    std::swap(ctx.prevDilatedMVTex, ctx.dilatedMVTex);
    std::swap(ctx.prevHudMaskTex, ctx.hudMaskTex);
}

// ─── AHB Image Creation ────────────────────────────────────────────────────

static bool createAHBImage(AFMEContext& ctx, AHBImage& img, uint32_t w, uint32_t h) {
    AHardwareBuffer_Desc desc = {};
    desc.width = w;
    desc.height = h;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
                 AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT |
                 AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;

    if (AHardwareBuffer_allocate(&desc, &img.ahb) != 0) {
        ALOGE("AFME: AHB alloc failed %ux%u", w, h);
        return false;
    }

    // Import AHB into Vulkan
    // Create VkImage with external memory handle type for AHB
    VkExternalMemoryImageCreateInfo extMemInfo = {};
    extMemInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extMemInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    // Match the swapchain's transfer function: for sRGB swapchains, view the
    // AHB as sRGB so vkCmdBlitImage does a raw copy instead of a lossy
    // linearize→re-encode round trip through 8-bit UNORM (crushes shadows on
    // every generated frame). The AHB byte layout is identical either way.
    VkFormat ahbViewFormat = (ctx.format == VK_FORMAT_R8G8B8A8_SRGB ||
                              ctx.format == VK_FORMAT_B8G8R8A8_SRGB)
                                 ? VK_FORMAT_R8G8B8A8_SRGB
                                 : VK_FORMAT_R8G8B8A8_UNORM;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &extMemInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = ahbViewFormat;
    imageInfo.extent = {w, h, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (next_vkCreateImage(ctx.device, &imageInfo, nullptr, &img.vkImage) != VK_SUCCESS) {
        ALOGE("AFME: VkImage create failed");
        AHardwareBuffer_release(img.ahb); img.ahb = nullptr;
        return false;
    }

    // Use vkGetImageMemoryRequirements to get alloc size + type bits
    // (avoids dependency on vkGetAndroidHardwareBufferPropertiesANDROID
    //  which may not be available if extension wasn't enabled by game)
    VkMemoryRequirements memReqs = {};
    next_vkGetImageMemoryRequirements(ctx.device, img.vkImage, &memReqs);

    VkImportAndroidHardwareBufferInfoANDROID importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = img.ahb;
    VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.pNext = &importInfo;
    dedicatedInfo.image = img.vkImage;
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx.memProps,
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (next_vkAllocateMemory(ctx.device, &allocInfo, nullptr, &img.vkMemory) != VK_SUCCESS) {
        ALOGE("AFME: VkMemory alloc failed");
        next_vkDestroyImage(ctx.device, img.vkImage, nullptr);
        AHardwareBuffer_release(img.ahb);
        img.vkImage = VK_NULL_HANDLE; img.ahb = nullptr;
        return false;
    }
    next_vkBindImageMemory(ctx.device, img.vkImage, img.vkMemory, 0);

    // Import into GLES via EGLImage — need EGL context current
    EGLDisplay oldDpy = eglGetCurrentDisplay();
    EGLSurface oldRead = eglGetCurrentSurface(EGL_READ);
    EGLSurface oldDraw = eglGetCurrentSurface(EGL_DRAW);
    EGLContext oldCtx = eglGetCurrentContext();

    eglMakeCurrent(ctx.eglDpy, ctx.eglSurf, ctx.eglSurf, ctx.eglCtx);

    EGLClientBuffer clientBuf = ctx.eglGetNativeClientBuffer(img.ahb);
    if (!clientBuf) {
        ALOGE("AFME: eglGetNativeClientBuffer failed");
        eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
        return false;
    }

    EGLint attrs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE };
    img.eglImage = ctx.eglCreateImageKHR(ctx.eglDpy, EGL_NO_CONTEXT,
                                          EGL_NATIVE_BUFFER_ANDROID, clientBuf, attrs);
    if (img.eglImage == EGL_NO_IMAGE_KHR) {
        ALOGE("AFME: eglCreateImageKHR failed: 0x%x", eglGetError());
        eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
        return false;
    }

    glGenTextures(1, &img.glTex);
    glBindTexture(GL_TEXTURE_2D, img.glTex);
    ctx.glEGLImageTargetTexture2D(GL_TEXTURE_2D, img.eglImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Restore original EGL context
    if (oldCtx != EGL_NO_CONTEXT) {
        eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
    } else {
        eglMakeCurrent(ctx.eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    img.valid = true;
    ALOGD("AFME: AHB %ux%u gl=%u vk=%p", w, h, img.glTex, (void*)img.vkImage);
    return true;
}

// ─── Vulkan Command Helpers ─────────────────────────────────────────────────

static void cmdBarrier(VkCommandBuffer cmd, VkImage image,
                       VkImageLayout oldLayout, VkImageLayout newLayout,
                       VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                       VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    next_vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

static VkCommandBuffer allocCmdBuf(AFMEContext& ctx) {
    VkCommandBufferAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = ctx.cmdPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    next_vkAllocateCommandBuffers(ctx.device, &info, &cmd);
    return cmd;
}

static void beginCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    next_vkBeginCommandBuffer(cmd, &info);
}

// Copy src image → dst image via blit (handles format conversion)
static void cmdCopyImage(VkCommandBuffer cmd, VkImage src, VkImage dst,
                         uint32_t w, uint32_t h,
                         VkImageLayout srcStartLayout, bool srcPresentable,
                         VkImageLayout dstStartLayout, bool dstPresentable) {
    // src → TRANSFER_SRC
    cmdBarrier(cmd, src,
        srcStartLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        srcPresentable ? 0 : VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // dst → TRANSFER_DST
    cmdBarrier(cmd, dst,
        dstStartLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    next_vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &region, VK_FILTER_NEAREST);

    // Restore layouts
    if (srcPresentable) {
        cmdBarrier(cmd, src,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }
    if (dstPresentable) {
        cmdBarrier(cmd, dst,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }
}

// ─── AFME Context Initialization ────────────────────────────────────────────

static bool initAFMEContext(AFMEContext& ctx) {
    if (ctx.initialized) return true;
    if (!ctx.afmeHWAvailable) return false;

    uint32_t w = ctx.extent.width;
    uint32_t h = ctx.extent.height;
    if (w == 0 || h == 0) return false;

    ALOGI("AFME: Initializing context %ux%u", w, h);

    // The native-fence path needs the EGL side too
    if (ctx.hasNativeFenceSync &&
        !(ctx.eglCreateSyncKHR_ && ctx.eglDestroySyncKHR_ &&
          ctx.eglWaitSyncKHR_ && ctx.eglDupNativeFenceFD_)) {
        ALOGW("AFME: EGL native fence functions missing — CPU sync fallback");
        ctx.hasNativeFenceSync = false;
    }

    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
    if (next_vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool) != VK_SUCCESS) {
        ALOGE("AFME: CreateCommandPool failed");
        return false;
    }

    // Create fence for copy operations — exportable as a sync_fd when the
    // native-fence path is available so GLES can wait for it on the GPU.
    VkExportFenceCreateInfo exportFenceInfo = {};
    exportFenceInfo.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
    exportFenceInfo.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (ctx.hasNativeFenceSync) fenceInfo.pNext = &exportFenceInfo;
    if (next_vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.copyFence) != VK_SUCCESS) {
        if (ctx.hasNativeFenceSync) {
            // Retry as a plain fence and fall back to CPU waits
            ALOGW("AFME: exportable copy fence failed, falling back to CPU sync");
            ctx.hasNativeFenceSync = false;
            fenceInfo.pNext = nullptr;
        }
        if (next_vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.copyFence) != VK_SUCCESS) {
            ALOGE("AFME: CreateFence (copy) failed");
            return false;
        }
    }

    // Reusable semaphore for GLES-generation-complete → Vulkan-copy ordering
    // (payload replaced every frame via temporary sync_fd import)
    if (ctx.hasNativeFenceSync) {
        VkSemaphoreCreateInfo genSemInfo = {};
        genSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (next_vkCreateSemaphore(ctx.device, &genSemInfo, nullptr,
                                   &ctx.genDoneSem) != VK_SUCCESS) {
            ALOGW("AFME: genDoneSem creation failed, falling back to glFinish");
            ctx.genDoneSem = VK_NULL_HANDLE;
        }
    }

    // Create the per-frame drain fence (signaled by the last synth submit)
    if (next_vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.drainFence) != VK_SUCCESS) {
        ALOGE("AFME: CreateFence (drain) failed");
        return false;
    }

    // Pre-allocate semaphore pool
    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ctx.semPoolSize = kMaxSemaphorePool;
    ctx.semPoolCursor = 0;
    for (uint32_t i = 0; i < ctx.semPoolSize; i++) {
        if (next_vkCreateSemaphore(ctx.device, &semInfo, nullptr, &ctx.semPool[i]) != VK_SUCCESS) {
            ALOGE("AFME: CreateSemaphore pool[%u] failed", i);
            ctx.semPoolSize = i;
            break;
        }
    }

    // Pre-allocate command buffer ring
    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = ctx.cmdPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = kCmdRingSize;
    ctx.cmdRingSize = kCmdRingSize;
    ctx.cmdRingCursor = 0;
    if (next_vkAllocateCommandBuffers(ctx.device, &cbInfo, ctx.cmdRing) != VK_SUCCESS) {
        ALOGE("AFME: AllocateCommandBuffers ring failed");
        ctx.cmdRingSize = 0;
    }

    // Create AHB images: prev, curr, and synth frames
    if (!createAHBImage(ctx, ctx.prevFrame, w, h)) return false;
    if (!createAHBImage(ctx, ctx.currFrame, w, h)) return false;

    int mult = afme::config().multiplier.load();
    ctx.allocatedMult = mult;  // Track what we actually allocated
    for (int i = 0; i < mult - 1; i++) {
        if (!createAHBImage(ctx, ctx.synthFrames[i], w, h)) {
            ALOGE("AFME: Failed to create synthFrame[%d]", i);
            return false;
        }
    }

    // Initialize GLES-based features (SGSR1, MobFGSR)
    // Need EGL context current for shader compilation + texture creation
    {
        EGLDisplay oldDpy = eglGetCurrentDisplay();
        EGLSurface oldRead = eglGetCurrentSurface(EGL_READ);
        EGLSurface oldDraw = eglGetCurrentSurface(EGL_DRAW);
        EGLContext oldCtx = eglGetCurrentContext();

        eglMakeCurrent(ctx.eglDpy, ctx.eglSurf, ctx.eglSurf, ctx.eglCtx);

        // SGSR1: adaptive sharpening
        if (afme::config().sgsrMode.load() >= 1) {
            if (initSGSR(ctx)) {
                // Create sharpened frame buffer for SGSR1 output
                // (uses createGLTex since it's GLES-only, not AHB)
                ctx.sharpenedFrame.glTex = createGLTex(GL_RGBA8, w, h, GL_LINEAR);
                ctx.sharpenedFrame.valid = (ctx.sharpenedFrame.glTex != 0);
                ALOGI("AFME: SGSR1 sharpen buffer created (tex=%u)", ctx.sharpenedFrame.glTex);
            }
        }

        // MobFGSR: motion-vector based interpolation (method=1)
        if (afme::config().wantMotion()) {
            ctx.mobfgsrAttempted = true;
            initMobFGSR(ctx);
        }

        // Restore EGL
        if (oldCtx != EGL_NO_CONTEXT) {
            eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
        } else {
            eglMakeCurrent(ctx.eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
    }

    // Vsync-grid calibration: ask the display pipeline for the TRUE refresh
    // cycle instead of trusting the staged prop (the EGL equivalent is
    // eglGetCompositorTimingANDROID — this is the Vulkan path).
    if (ctx.hasDisplayTiming && next_vkGetRefreshCycleDurationGOOGLE) {
        VkRefreshCycleDurationGOOGLE dur = {};
        if (next_vkGetRefreshCycleDurationGOOGLE(ctx.device, ctx.swapchain, &dur)
                == VK_SUCCESS) {
            ctx.refreshCycleNs = dur.refreshDuration;
            ALOGI("AFME: Measured panel refresh cycle %llu ns (~%d Hz, prop Hz=%d)",
                  (unsigned long long)ctx.refreshCycleNs, effectiveHz(ctx),
                  afme::config().displayHz.load());
        }
    }

    ctx.initialized = true;
    ALOGI("AFME: Context initialized (%ux%u, %d synth frames for %dx, %u sems, "
          "sgsr=%d, mobfgsr=%d, nativeFence=%d, displayTiming=%d)",
          w, h, mult - 1, mult, ctx.semPoolSize, ctx.sgsrInitialized ? 1 : 0,
          ctx.mobfgsrInitialized ? 1 : 0, ctx.hasNativeFenceSync ? 1 : 0,
          ctx.hasDisplayTiming ? 1 : 0);
    return true;
}

// ─── Vulkan Hooks ───────────────────────────────────────────────────────────

static VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateInstance(
        const VkInstanceCreateInfo* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkInstance* pInstance) {
    auto* layerInfo = reinterpret_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext);
    while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                         layerInfo->function != VK_LAYER_LINK_INFO)) {
        layerInfo = reinterpret_cast<const VkLayerInstanceCreateInfo*>(layerInfo->pNext);
    }
    if (!layerInfo) {
        ALOGE("AFME: No layer link info in vkCreateInstance");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    next_vkGetInstanceProcAddr = layerInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(layerInfo)->u.pLayerInfo =
        layerInfo->u.pLayerInfo->pNext;

    next_vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        next_vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));

    VkResult result = next_vkCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    initInstanceFunc(*pInstance, "vkDestroyInstance", &next_vkDestroyInstance);
    initInstanceFunc(*pInstance, "vkCreateDevice", &next_vkCreateDevice);
    initInstanceFunc(*pInstance, "vkGetPhysicalDeviceMemoryProperties",
                     &next_vkGetPhysicalDeviceMemoryProperties);
    initInstanceFunc(*pInstance, "vkGetPhysicalDeviceProperties",
                     &next_vkGetPhysicalDeviceProperties);
    initInstanceFunc(*pInstance, "vkGetPhysicalDeviceFeatures",
                     &next_vkGetPhysicalDeviceFeatures);
    initInstanceFunc(*pInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                     &next_vkGetPhysicalDeviceSurfaceCapabilities);
    initInstanceFunc(*pInstance, "vkEnumerateDeviceExtensionProperties",
                     &next_vkEnumerateDeviceExtensionProperties);

    afme::config().poll();
    ALOGI("AFME VK Layer v8: Instance created (enabled=%d fg=%d multiplier=%dx af=%dx)",
          afme::config().enabled.load(), afme::config().fg.load(),
          afme::config().multiplier.load(), afme::config().af.load());
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_vkDestroyInstance(
        VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    if (next_vkDestroyInstance) next_vkDestroyInstance(instance, pAllocator);
}

// ─── Anisotropic filtering override ─────────────────────────────────────────
//
// There is no driver property for AF on Adreno — see the matching section in
// afme_layer.cpp for the evidence. The override therefore happens where the
// state is actually created: every sampler the game builds for a mipmapped
// texture gets its anisotropy raised on the way through.
//
// Set at device creation, read at sampler creation.
std::atomic<bool>  gAfAnisoEnabled{false};  // samplerAnisotropy on this device
std::atomic<float> gAfDriverMax{1.0f};      // limits.maxSamplerAnisotropy

static VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateSampler(
        VkDevice device,
        const VkSamplerCreateInfo* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkSampler* pSampler) {
    if (!next_vkCreateSampler) return VK_ERROR_INITIALIZATION_FAILED;

    const int want = afme::config().af.load(std::memory_order_relaxed);
    const float driverMax = gAfDriverMax.load(std::memory_order_relaxed);

    // Every condition here is a spec requirement or a case anisotropy cannot
    // help, not a heuristic:
    //   - the feature must be enabled on the device (VUID-anisotropyEnable)
    //   - unnormalizedCoordinates forbids anisotropy outright
    //   - cubic filtering forbids it too
    //   - maxLod == minLod means the sampler never leaves one mip level, so
    //     there is no minification footprint to sample along
    const bool eligible =
        want > 1 &&
        gAfAnisoEnabled.load(std::memory_order_relaxed) &&
        driverMax > 1.0f &&
        pCreateInfo != nullptr &&
        pCreateInfo->anisotropyEnable == VK_FALSE &&
        pCreateInfo->unnormalizedCoordinates == VK_FALSE &&
        pCreateInfo->magFilter != VK_FILTER_CUBIC_EXT &&
        pCreateInfo->minFilter != VK_FILTER_CUBIC_EXT &&
        pCreateInfo->maxLod > pCreateInfo->minLod;

    if (!eligible) {
        return next_vkCreateSampler(device, pCreateInfo, pAllocator, pSampler);
    }

    VkSamplerCreateInfo info = *pCreateInfo;
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = (float)want < driverMax ? (float)want : driverMax;
    // Same reasoning as the GLES path: extra taps are wasted if the mip
    // transition itself is a hard cut.
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    return next_vkCreateSampler(device, &info, pAllocator, pSampler);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateDevice(
        VkPhysicalDevice physicalDevice,
        const VkDeviceCreateInfo* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDevice* pDevice) {
    auto* layerInfo = reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext);
    while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                         layerInfo->function != VK_LAYER_LINK_INFO)) {
        layerInfo = reinterpret_cast<const VkLayerDeviceCreateInfo*>(layerInfo->pNext);
    }
    if (!layerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipaNext = layerInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    next_vkGetDeviceProcAddr = layerInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(layerInfo)->u.pLayerInfo =
        layerInfo->u.pLayerInfo->pNext;

    // Inject AHB extensions needed for AFME frame sharing
    std::vector<const char*> extensions(
        pCreateInfo->ppEnabledExtensionNames,
        pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);

    auto hasExt = [&](const char* name) {
        for (auto& e : extensions) if (strcmp(e, name) == 0) return true;
        return false;
    };
    if (!hasExt("VK_ANDROID_external_memory_android_hardware_buffer"))
        extensions.push_back("VK_ANDROID_external_memory_android_hardware_buffer");
    if (!hasExt("VK_KHR_external_memory"))
        extensions.push_back("VK_KHR_external_memory");
    if (!hasExt("VK_KHR_dedicated_allocation"))
        extensions.push_back("VK_KHR_dedicated_allocation");
    if (!hasExt("VK_KHR_get_memory_requirements2"))
        extensions.push_back("VK_KHR_get_memory_requirements2");
    if (!hasExt("VK_KHR_sampler_ycbcr_conversion"))
        extensions.push_back("VK_KHR_sampler_ycbcr_conversion");
    if (!hasExt("VK_KHR_bind_memory2"))
        extensions.push_back("VK_KHR_bind_memory2");
    if (!hasExt("VK_KHR_maintenance1"))
        extensions.push_back("VK_KHR_maintenance1");

    // VK_GOOGLE_display_timing (implemented by the Android loader when the
    // driver exposes present timestamps): lets us schedule synthetic frames
    // at the temporal midpoint of the game-frame interval instead of the
    // very next vsync slot. Without it, presents pace as
    // [real][synth]————gap————[real][synth] (measured: 8ms/16-24ms bimodal
    // present2present) which strobes instead of smoothing. Only inject when
    // actually supported so the create doesn't fall back and lose AHB too.
    bool hasGoogleTiming = false;
    bool hasFenceFd = false;
    bool hasSemFd = false;
    if (next_vkEnumerateDeviceExtensionProperties) {
        uint32_t extCount = 0;
        next_vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr,
                                                  &extCount, nullptr);
        std::vector<VkExtensionProperties> extProps(extCount);
        if (extCount > 0) {
            next_vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr,
                                                      &extCount, extProps.data());
            for (const auto& p : extProps) {
                if (strcmp(p.extensionName, "VK_GOOGLE_display_timing") == 0)
                    hasGoogleTiming = true;
                else if (strcmp(p.extensionName, "VK_KHR_external_fence_fd") == 0)
                    hasFenceFd = true;
                else if (strcmp(p.extensionName, "VK_KHR_external_semaphore_fd") == 0)
                    hasSemFd = true;
            }
        }
    }
    if (hasGoogleTiming && !hasExt("VK_GOOGLE_display_timing"))
        extensions.push_back("VK_GOOGLE_display_timing");

    // sync_fd interop: lets the GLES generation pass wait for the Vulkan copy
    // ON THE GPU (and vice versa) instead of vkWaitForFences/glFinish on the
    // game's render thread — the CPU stalls that cost real FPS when AFME is on.
    if (hasFenceFd && !hasExt("VK_KHR_external_fence_fd"))
        extensions.push_back("VK_KHR_external_fence_fd");
    if (hasSemFd && !hasExt("VK_KHR_external_semaphore_fd"))
        extensions.push_back("VK_KHR_external_semaphore_fd");
    bool hasNativeFence = hasFenceFd && hasSemFd;

    VkDeviceCreateInfo modifiedCreateInfo = *pCreateInfo;
    modifiedCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    modifiedCreateInfo.ppEnabledExtensionNames = extensions.data();

    // ── Anisotropic filtering: enable the feature the override needs ────────
    //
    // vkCreateSampler may only set anisotropyEnable if samplerAnisotropy was
    // enabled at device creation, and a game that never uses AF has no reason to
    // ask for it. So turn it on here, once, and remember whether it took — the
    // sampler hook must not touch anisotropy on a device that lacks it.
    //
    // pEnabledFeatures and a VkPhysicalDeviceFeatures2 in pNext are mutually
    // exclusive, so exactly one of the two branches applies.
    VkPhysicalDeviceFeatures afFeatures{};
    VkPhysicalDeviceFeatures2* appFeatures2 = nullptr;
    VkBool32 savedFeatures2Aniso = VK_FALSE;
    bool wantAniso = false;
    if (next_vkGetPhysicalDeviceFeatures) {
        VkPhysicalDeviceFeatures supported{};
        next_vkGetPhysicalDeviceFeatures(physicalDevice, &supported);
        wantAniso = supported.samplerAnisotropy == VK_TRUE;
    }
    if (wantAniso) {
        for (auto* p = reinterpret_cast<const VkBaseInStructure*>(pCreateInfo->pNext);
             p != nullptr; p = p->pNext) {
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                appFeatures2 = const_cast<VkPhysicalDeviceFeatures2*>(
                    reinterpret_cast<const VkPhysicalDeviceFeatures2*>(p));
                break;
            }
        }
        if (appFeatures2) {
            // Flip it in place for the duration of the call and restore after:
            // the chain is the app's memory and it may create a second device
            // from the same structs.
            savedFeatures2Aniso = appFeatures2->features.samplerAnisotropy;
            appFeatures2->features.samplerAnisotropy = VK_TRUE;
        } else {
            if (pCreateInfo->pEnabledFeatures) afFeatures = *pCreateInfo->pEnabledFeatures;
            afFeatures.samplerAnisotropy = VK_TRUE;
            modifiedCreateInfo.pEnabledFeatures = &afFeatures;
        }
    }

    auto createDevice = reinterpret_cast<PFN_vkCreateDevice>(
        gipaNext(VK_NULL_HANDLE, "vkCreateDevice"));
    VkResult result = createDevice(physicalDevice, &modifiedCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        // Fallback: try without our extensions
        ALOGW("AFME: Device creation with AHB extensions failed, trying original");
        if (appFeatures2) appFeatures2->features.samplerAnisotropy = savedFeatures2Aniso;
        result = createDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
        if (result != VK_SUCCESS) return result;
        hasGoogleTiming = false;  // our extension list was not applied
        hasNativeFence = false;
        // The retry used the app's own create info, so the feature is only on if
        // the app asked for it itself.
        wantAniso = (appFeatures2 && savedFeatures2Aniso == VK_TRUE)
                 || (!appFeatures2 && pCreateInfo->pEnabledFeatures
                     && pCreateInfo->pEnabledFeatures->samplerAnisotropy == VK_TRUE);
    } else if (appFeatures2) {
        appFeatures2->features.samplerAnisotropy = savedFeatures2Aniso;
    }
    gAfAnisoEnabled.store(wantAniso, std::memory_order_relaxed);
    if (wantAniso && next_vkGetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props{};
        next_vkGetPhysicalDeviceProperties(physicalDevice, &props);
        gAfDriverMax.store(props.limits.maxSamplerAnisotropy, std::memory_order_relaxed);
    }

    // Resolve device functions
    initDeviceFunc(*pDevice, "vkDestroyDevice", &next_vkDestroyDevice);
    initDeviceFunc(*pDevice, "vkCreateSampler", &next_vkCreateSampler);
    initDeviceFunc(*pDevice, "vkQueuePresentKHR", &next_vkQueuePresentKHR);
    initDeviceFunc(*pDevice, "vkQueueSubmit", &next_vkQueueSubmit);
    initDeviceFunc(*pDevice, "vkGetDeviceQueue", &next_vkGetDeviceQueue);
    initDeviceFunc(*pDevice, "vkCreateSwapchainKHR", &next_vkCreateSwapchainKHR);
    initDeviceFunc(*pDevice, "vkDestroySwapchainKHR", &next_vkDestroySwapchainKHR);
    initDeviceFunc(*pDevice, "vkGetSwapchainImagesKHR", &next_vkGetSwapchainImagesKHR);
    initDeviceFunc(*pDevice, "vkAcquireNextImageKHR", &next_vkAcquireNextImageKHR);
    initDeviceFunc(*pDevice, "vkCreateCommandPool", &next_vkCreateCommandPool);
    initDeviceFunc(*pDevice, "vkDestroyCommandPool", &next_vkDestroyCommandPool);
    initDeviceFunc(*pDevice, "vkAllocateCommandBuffers", &next_vkAllocateCommandBuffers);
    initDeviceFunc(*pDevice, "vkFreeCommandBuffers", &next_vkFreeCommandBuffers);
    initDeviceFunc(*pDevice, "vkResetCommandBuffer", &next_vkResetCommandBuffer);
    initDeviceFunc(*pDevice, "vkBeginCommandBuffer", &next_vkBeginCommandBuffer);
    initDeviceFunc(*pDevice, "vkEndCommandBuffer", &next_vkEndCommandBuffer);
    initDeviceFunc(*pDevice, "vkCmdPipelineBarrier", &next_vkCmdPipelineBarrier);
    initDeviceFunc(*pDevice, "vkCmdBlitImage", &next_vkCmdBlitImage);
    initDeviceFunc(*pDevice, "vkCreateImage", &next_vkCreateImage);
    initDeviceFunc(*pDevice, "vkDestroyImage", &next_vkDestroyImage);
    initDeviceFunc(*pDevice, "vkGetImageMemoryRequirements", &next_vkGetImageMemoryRequirements);
    initDeviceFunc(*pDevice, "vkAllocateMemory", &next_vkAllocateMemory);
    initDeviceFunc(*pDevice, "vkFreeMemory", &next_vkFreeMemory);
    initDeviceFunc(*pDevice, "vkBindImageMemory", &next_vkBindImageMemory);
    initDeviceFunc(*pDevice, "vkCreateFence", &next_vkCreateFence);
    initDeviceFunc(*pDevice, "vkDestroyFence", &next_vkDestroyFence);
    initDeviceFunc(*pDevice, "vkWaitForFences", &next_vkWaitForFences);
    initDeviceFunc(*pDevice, "vkResetFences", &next_vkResetFences);
    initDeviceFunc(*pDevice, "vkCreateSemaphore", &next_vkCreateSemaphore);
    initDeviceFunc(*pDevice, "vkDestroySemaphore", &next_vkDestroySemaphore);
    initDeviceFunc(*pDevice, "vkQueueWaitIdle", &next_vkQueueWaitIdle);
    initDeviceFunc(*pDevice, "vkGetAndroidHardwareBufferPropertiesANDROID",
                   &next_vkGetAndroidHardwareBufferProperties);
    if (hasNativeFence) {
        initDeviceFunc(*pDevice, "vkGetFenceFdKHR", &next_vkGetFenceFdKHR);
        initDeviceFunc(*pDevice, "vkImportSemaphoreFdKHR",
                       &next_vkImportSemaphoreFdKHR);
        if (!next_vkGetFenceFdKHR || !next_vkImportSemaphoreFdKHR)
            hasNativeFence = false;
    }
    if (hasGoogleTiming) {
        initDeviceFunc(*pDevice, "vkGetRefreshCycleDurationGOOGLE",
                       &next_vkGetRefreshCycleDurationGOOGLE);
    }

    {
        std::lock_guard<std::mutex> lock(gLock);
        gDeviceToPhysical[*pDevice] = physicalDevice;
        gDeviceHasGoogleTiming[*pDevice] = hasGoogleTiming;
        gDeviceHasNativeFence[*pDevice] = hasNativeFence;
        if (next_vkGetPhysicalDeviceMemoryProperties) {
            next_vkGetPhysicalDeviceMemoryProperties(physicalDevice, &gDeviceMemProps[*pDevice]);
        }
    }

    ALOGI("AFME VK Layer v8: Device created");
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_vkDestroyDevice(
        VkDevice device, const VkAllocationCallbacks* pAllocator) {
    {
        std::lock_guard<std::mutex> lock(gLock);
        gDeviceToPhysical.erase(device);
        gDeviceMemProps.erase(device);
        gDeviceHasGoogleTiming.erase(device);
        gDeviceHasNativeFence.erase(device);
    }
    if (next_vkDestroyDevice) next_vkDestroyDevice(device, pAllocator);
}

// ═══ HOOK: vkCreateSwapchainKHR ═══

static VKAPI_ATTR VkResult VKAPI_CALL layer_vkCreateSwapchainKHR(
        VkDevice device,
        const VkSwapchainCreateInfoKHR* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkSwapchainKHR* pSwapchain) {
    if (!next_vkCreateSwapchainKHR || !afme::config().enabled.load(std::memory_order_relaxed)) {
        return next_vkCreateSwapchainKHR ?
            next_vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain) :
            VK_ERROR_INITIALIZATION_FAILED;
    }

    int mult = afme::config().multiplier.load();

    // Modify swapchain: increase image count + add TRANSFER usage for blit ops
    VkSwapchainCreateInfoKHR modifiedInfo = *pCreateInfo;
    uint32_t requestedCount = pCreateInfo->minImageCount + (uint32_t)(mult - 1) + 1;

    // Cap to maxImageCount from surface capabilities (issue found in interpolation engines)
    VkSurfaceCapabilitiesKHR surfCaps = {};
    if (next_vkGetPhysicalDeviceSurfaceCapabilities) {
        VkPhysicalDevice physDev = VK_NULL_HANDLE;
        {
            std::lock_guard<std::mutex> lock(gLock);
            auto it = gDeviceToPhysical.find(device);
            if (it != gDeviceToPhysical.end()) physDev = it->second;
        }
        if (physDev && next_vkGetPhysicalDeviceSurfaceCapabilities(
                physDev, pCreateInfo->surface, &surfCaps) == VK_SUCCESS) {
            if (surfCaps.maxImageCount > 0 && requestedCount > surfCaps.maxImageCount) {
                ALOGW("AFME: Capping images %u→%u (driver max)",
                      requestedCount, surfCaps.maxImageCount);
                requestedCount = surfCaps.maxImageCount;
            }
        }
    }
    modifiedInfo.minImageCount = requestedCount;
    modifiedInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // Frame generation requires FIFO: MAILBOX/IMMEDIATE overwrite the pending
    // buffer before latch, so synthetic frames would be discarded while the
    // stats still count them (bug E from the 2026-07-28 analysis — generation
    // "working" invisibly). FIFO keeps every present a definite vsync occupant.
    if (afme::config().enabled.load(std::memory_order_relaxed) &&
        modifiedInfo.presentMode != VK_PRESENT_MODE_FIFO_KHR) {
        ALOGI("AFME: Forcing presentMode FIFO (was %d) — synth frames need "
              "reliable vsync latching", (int)pCreateInfo->presentMode);
        modifiedInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }

    ALOGI("AFME: CreateSwapchain %ux%u images %u→%u (for %dx)",
          pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
          pCreateInfo->minImageCount, modifiedInfo.minImageCount, mult);

    // Retire old swapchain context
    if (pCreateInfo->oldSwapchain) {
        std::lock_guard<std::mutex> lock(gLock);
        gSwapchainContexts.erase(pCreateInfo->oldSwapchain);
    }

    VkResult result = next_vkCreateSwapchainKHR(device, &modifiedInfo, pAllocator, pSwapchain);
    if (result != VK_SUCCESS) {
        // Fallback: try original params
        ALOGW("AFME: Modified CreateSwapchain failed, trying original");
        result = next_vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
        if (result != VK_SUCCESS) return result;
    }

    // Create AFME context for this swapchain
    {
        std::lock_guard<std::mutex> lock(gLock);
        AFMEContext& ctx = gSwapchainContexts[*pSwapchain];
        ctx = AFMEContext{};
        ctx.device = device;
        ctx.swapchain = *pSwapchain;
        ctx.extent = pCreateInfo->imageExtent;
        ctx.format = pCreateInfo->imageFormat;

        auto devIt = gDeviceToPhysical.find(device);
        if (devIt != gDeviceToPhysical.end()) ctx.physicalDevice = devIt->second;

        auto timingIt = gDeviceHasGoogleTiming.find(device);
        ctx.hasDisplayTiming = (timingIt != gDeviceHasGoogleTiming.end())
                                   && timingIt->second;

        auto fenceIt = gDeviceHasNativeFence.find(device);
        ctx.hasNativeFenceSync = (fenceIt != gDeviceHasNativeFence.end())
                                     && fenceIt->second;

        auto memIt = gDeviceMemProps.find(device);
        if (memIt != gDeviceMemProps.end()) ctx.memProps = memIt->second;

        // Find queue family from create info
        if (pCreateInfo->queueFamilyIndexCount > 0 && pCreateInfo->pQueueFamilyIndices) {
            ctx.queueFamilyIndex = pCreateInfo->pQueueFamilyIndices[0];
        }

        // Get swapchain images
        uint32_t imageCount = 0;
        next_vkGetSwapchainImagesKHR(device, *pSwapchain, &imageCount, nullptr);
        ctx.swapchainImages.resize(imageCount);
        next_vkGetSwapchainImagesKHR(device, *pSwapchain, &imageCount, ctx.swapchainImages.data());

        // Initialize EGL
        initEGLContext(ctx);

        ALOGI("AFME: Swapchain context created (%u images, %ux%u, HW=%d)",
              imageCount, ctx.extent.width, ctx.extent.height, ctx.afmeHWAvailable);
    }

    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL layer_vkDestroySwapchainKHR(
        VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    {
        std::lock_guard<std::mutex> lock(gLock);
        gSwapchainContexts.erase(swapchain);
    }
    if (next_vkDestroySwapchainKHR) next_vkDestroySwapchainKHR(device, swapchain, pAllocator);
}

// ═══ CORE: vkQueuePresentKHR — Frame Generation ═══

static VKAPI_ATTR VkResult VKAPI_CALL layer_vkQueuePresentKHR(
        VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    if (!next_vkQueuePresentKHR) return VK_ERROR_DEVICE_LOST;

    static std::atomic<uint64_t> sPresentCount{0};
    const uint64_t fc = sPresentCount.fetch_add(1);
    if ((fc % afme::kPollInterval) == 0) {
        afme::config().poll();
        afme::pollFilterProps();
    } else if (afme::filterLive()) {
        // GameSpace has the filter panel open: follow every slider movement.
        afme::pollFilterProps();
    }

    if (!afme::config().enabled.load(std::memory_order_relaxed)) {
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    // Only handle single-swapchain presents for now
    if (pPresentInfo->swapchainCount != 1 || pPresentInfo->swapchainCount == 0) {
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    VkSwapchainKHR swapchain = pPresentInfo->pSwapchains[0];
    uint32_t presentIdx = pPresentInfo->pImageIndices[0];
    int mult = afme::config().multiplier.load();

    AFMEContext* ctxPtr = nullptr;
    {
        std::lock_guard<std::mutex> lock(gLock);
        auto it = gSwapchainContexts.find(swapchain);
        if (it != gSwapchainContexts.end()) ctxPtr = &it->second;
    }

    if (!ctxPtr) {
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    AFMEContext& ctx = *ctxPtr;

    // ── Engagement gate ─────────────────────────────────────────────────
    // Runs before the lazy resource init below, so a swapchain we never
    // accelerate costs nothing but this counter.
    if (!ctx.gate.check(afme::nowNs(), (int)ctx.extent.width,
                        (int)ctx.extent.height)) {
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    // Dynamic synth frame reallocation: if user increased multiplier
    // mid-game (e.g. 2x→3x), allocate the missing synth frames now.
    if (mult > ctx.allocatedMult && ctx.initialized) {
        ALOGI("AFME: Multiplier changed %dx→%dx, allocating %d more synth frames",
              ctx.allocatedMult, mult, (mult - 1) - (ctx.allocatedMult - 1));
        bool ok = true;
        for (int i = ctx.allocatedMult - 1; i < mult - 1; i++) {
            if (!createAHBImage(ctx, ctx.synthFrames[i],
                                ctx.extent.width, ctx.extent.height)) {
                ALOGE("AFME: Failed to create synthFrame[%d] for %dx", i, mult);
                ok = false;
                break;
            }
        }
        if (ok) {
            ctx.allocatedMult = mult;
        } else {
            // Fallback: clamp to what we have
            mult = ctx.allocatedMult;
        }
    } else if (mult > ctx.allocatedMult) {
        // Not initialized yet — clamp for safety
        mult = ctx.allocatedMult;
    }
    // Lazy init AFME resources on first present
    if (!ctx.initialized && ctx.afmeHWAvailable) {
        if (!initAFMEContext(ctx)) {
            ALOGW("AFME: Init failed, passthrough");
            return next_vkQueuePresentKHR(queue, pPresentInfo);
        }
    }

    // Re-calibrate the vsync grid occasionally — the panel can switch modes
    // mid-session (dynamic refresh, Battery Saver vote down, etc.)
    if (ctx.hasDisplayTiming && next_vkGetRefreshCycleDurationGOOGLE &&
        (fc % 600) == 599) {
        VkRefreshCycleDurationGOOGLE dur = {};
        if (next_vkGetRefreshCycleDurationGOOGLE(ctx.device, ctx.swapchain, &dur)
                == VK_SUCCESS &&
            dur.refreshDuration != ctx.refreshCycleNs) {
            // Hysteresis: the driver reports ±few-ns jitter per query — a
            // 2ns "change" at 8.3ms is noise, not a mode switch, and logging
            // it spammed logcat every 10s on device. Adopt only ≥0.1% deltas.
            double oldNs = (double)ctx.refreshCycleNs;
            double dAbs = (double)dur.refreshDuration - oldNs;
            if (dAbs < 0.0) dAbs = -dAbs;
            double rel = (oldNs > 0.0) ? dAbs / oldNs : 1.0;
            if (rel >= 0.001) {
                ALOGI("AFME: Panel refresh cycle changed %llu → %llu ns (~%d Hz)",
                      (unsigned long long)ctx.refreshCycleNs,
                      (unsigned long long)dur.refreshDuration,
                      (int)(1000000000.0 / (double)dur.refreshDuration + 0.5));
                ctx.refreshCycleNs = dur.refreshDuration;
            }
        }
    }

    if (!ctx.initialized || !ctx.afmeHWAvailable) {
        if ((fc % 300) == 0) {
            ALOGI("AFME: Frame %lu — passthrough (HW=%d init=%d)",
                  (unsigned long)fc, ctx.afmeHWAvailable, ctx.initialized);
        }
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    // ═══════════════════════════════════════════════════════════════════
    // AFME Frame Generation Pipeline v6
    //
    // Two orderings, chosen by generation mode:
    //
    //  EXTRAPOLATION (modes 0-2, glExtrapolateTex2DQCOM): synthetic frames
    //  depict time t+f AFTER the current frame, so the REAL frame is
    //  presented FIRST (no added latency) and synth frames fill the
    //  following vsync slots. v5.2 presented synth first — the display
    //  stepped backwards in time once per real frame (visible judder).
    //
    //  INTERPOLATION (mode 3, MobFGSR): synthetic frames lie BETWEEN prev
    //  and curr, so they are presented BEFORE the real frame (standard
    //  ordering; costs ~1 vsync of latency — inherent to interpolation).
    //
    // Sync design:
    //  - The real present waits the swapchain→AHB copy on the GPU timeline
    //    via a semaphore; the CPU no longer blocks before the real present.
    //  - vkQueueWaitIdle is GONE: it also waited for the game's freshly
    //    submitted next-frame rendering, serializing CPU and GPU every
    //    frame. A fence on our LAST synth copy submit is waited at the
    //    start of the next present instead.
    //  - The generation ratio adapts to panel headroom: on a 60Hz-locked
    //    panel (Battery Saver) a 60fps game gets 0 synth frames instead of
    //    being throttled to 30fps by FIFO back-pressure.
    // ═══════════════════════════════════════════════════════════════════

    // Step 0: Drain previous frame's synth submissions (fence, not queue
    // idle). Pools are sized so entries are reused ≥2 frames later; cursors
    // wrap instead of resetting.
    if (ctx.drainPending) {
        next_vkWaitForFences(ctx.device, 1, &ctx.drainFence, VK_TRUE, 50000000ULL);
        next_vkResetFences(ctx.device, 1, &ctx.drainFence);
        ctx.drainPending = false;
    }

    const int64_t nowNs = afme::nowNs();
    const afme::Pacer::Tier tier =
            ctx.pacer.beginPresent(nowNs, mult, effectiveHz(ctx));
    // Frame generation and the color filter are independent features: a
    // filter-only session is legitimate, and generation must not be a
    // precondition for grading.
    const bool fgOn = afme::config().fg.load(std::memory_order_relaxed);
    const int numGenFrames = fgOn ? tier.numGen : 0;

    const afme::FilterStack fp = afme::filterStack();
    bool filterOn = afme::filterEnabled() && !fp.empty() &&
                    !ctx.filterUnsupported && !ctx.filter.failed();
    if (filterOn && !is8BitSwapchain(ctx.format)) {
        ALOGW("AFME: color filter disabled — swapchain format %d is not 8-bit, "
              "and the staging buffer would truncate the real frame",
              (int)ctx.format);
        ctx.filterUnsupported = true;
        filterOn = false;
    }
    if (filterOn && !ctx.stageFrame.valid) {
        if (!createAHBImage(ctx, ctx.stageFrame, ctx.extent.width,
                            ctx.extent.height)) {
            ALOGE("AFME: filter staging buffer allocation failed");
            ctx.filterUnsupported = true;
            filterOn = false;
        }
    }

    // Stage B holds the screen-space effects — vignette, grain, letterbox —
    // which must run AFTER generation on every present, or the motion field
    // would warp them off the screen and the grain would be read as motion.
    bool stageB = filterOn && fp.hasScreenSpace();
    if (stageB && !ctx.presentFrame.valid) {
        if (!createAHBImage(ctx, ctx.presentFrame, ctx.extent.width,
                            ctx.extent.height)) {
            ALOGE("AFME: stage-B buffer allocation failed — "
                  "screen-space effects disabled");
            stageB = false;
        }
    }

    // Every present the game makes is a real frame, whether or not we generate
    // from it. Counting it only on the generating path reported real=0 for the
    // whole time the panel had no headroom — the GLES layer already did this
    // correctly, and the overlay reads one channel from both.
    ctx.stats.addReal();

    if (numGenFrames == 0 && !filterOn) {
        // Nothing to do: the game already fills the panel and no grade is set.
        // The committed tier deliberately survives this path — see
        // afme::Pacer::abortPresent().
        ctx.hasPrevFrame = false;
        ctx.pacer.abortPresent(nowNs);
        ctx.stats.publish(nowNs);
        if ((fc % 300) == 0) {
            ALOGI("AFME: no display headroom @ %dHz — passthrough",
                  effectiveHz(ctx));
        }
        return next_vkQueuePresentKHR(queue, pPresentInfo);
    }

    VkImage swapImg = ctx.swapchainImages[presentIdx];

    const int sgsrMode = afme::config().sgsrMode.load();
    // The motion method interpolates, so it must hold the real frame back until
    // the synthetic one is ready; extrapolation presents the real frame first.
    // Falls back to extrapolation if MobFGSR could not be built for this ctx.
    const bool interpolation = numGenFrames > 0 &&
                               afme::config().wantMotion() && ctx.mobfgsrInitialized;
    // Visibility for a silent downgrade: a failed MobFGSR build used to look
    // identical in logcat to interpolation working. One WARN per 300 frames.
    if (afme::config().wantMotion() && !ctx.mobfgsrInitialized && ctx.mobfgsrAttempted
        && (fc % 300) == 0) {
        ALOGW("AFME: method=motion requested but MobFGSR unavailable for this "
              "swapchain — falling back to extrapolation (see init errors above)");
    }
    // Only generate when generation is on AND a tier was chosen for it.
    const bool generate = ctx.hasPrevFrame && numGenFrames > 0;

    // With the filter on, the present image is copied into stageFrame and the
    // grade writes into currFrame; without it, straight into currFrame as before.
    const VkImage copyDst = filterOn ? ctx.stageFrame.vkImage
                                     : ctx.currFrame.vkImage;

    // Step 1: Copy swapchain → currFrame (Vulkan blit). Waits the game's
    // present semaphores on the GPU; signals copyDoneSem for the
    // extrapolation-mode real present. The CPU does NOT wait here.
    // With the filter on, the real present waits the grade's write-back instead
    // (signalled further down); a semaphore signalled but never waited would
    // leak pool state, so do not take one.
    VkSemaphore copyDoneSem =
            (interpolation || filterOn) ? VK_NULL_HANDLE : ctx.acquireSem();
    {
        VkCommandBuffer cmd = ctx.acquireCmd();
        if (cmd == VK_NULL_HANDLE) cmd = allocCmdBuf(ctx); // fallback
        next_vkResetCommandBuffer(cmd, 0);
        beginCmd(cmd);
        cmdCopyImage(cmd, swapImg, copyDst,
                     ctx.extent.width, ctx.extent.height,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, true,
                     VK_IMAGE_LAYOUT_UNDEFINED, false);
        // Leave the destination in GENERAL for GLES to read
        cmdBarrier(cmd, copyDst,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        next_vkEndCommandBuffer(cmd);

        // Wait for game's render to finish (via present semaphores)
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        // Clamp to the fixed stage array: a game presenting with >8 wait
        // semaphores would otherwise exceed waitStages' declared bounds.
        if (pPresentInfo->waitSemaphoreCount > 8) {
            ALOGW("AFME: game presented with %u wait semaphores — clamping to 8",
                  pPresentInfo->waitSemaphoreCount);
        }
        submitInfo.waitSemaphoreCount =
            (pPresentInfo->waitSemaphoreCount > 8) ? 8 : pPresentInfo->waitSemaphoreCount;
        submitInfo.pWaitSemaphores = pPresentInfo->pWaitSemaphores;
        VkPipelineStageFlags waitStages[8] = {};
        for (uint32_t i = 0; i < pPresentInfo->waitSemaphoreCount && i < 8; i++)
            waitStages[i] = VK_PIPELINE_STAGE_TRANSFER_BIT;
        submitInfo.pWaitDstStageMask = waitStages;
        if (copyDoneSem != VK_NULL_HANDLE) {
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &copyDoneSem;
        }

        // On the native-fence path nothing CPU-waits copyFence during the
        // frame, and vkResetFences on an in-flight fence is invalid. Waiting
        // here is ~free: the GPU had the whole previous frame to finish the
        // blit, so the fence is virtually always already signaled.
        if (ctx.copyFencePending) {
            next_vkWaitForFences(ctx.device, 1, &ctx.copyFence,
                                 VK_TRUE, 16000000ULL);
        }
        next_vkResetFences(ctx.device, 1, &ctx.copyFence);
        next_vkQueueSubmit(queue, 1, &submitInfo, ctx.copyFence);
        ctx.copyFencePending = true;
    }

    VkResult finalResult = VK_SUCCESS;
    bool synthSubmitted = false;

    // ── Step 1b: grade ───────────────────────────────────────────────────
    // stageFrame → currFrame on the private GLES context, then currFrame back
    // into the present image so the REAL frame is graded too. Generation later
    // reads currFrame/prevFrame, which are now graded, so synthetic frames
    // inherit the grade at no extra cost.
    VkSemaphore filterDoneSem = VK_NULL_HANDLE;
    if (filterOn) {
        // GLES must not read stageFrame before the copy lands. Same GPU-side
        // wait the generation path uses; CPU fence wait as the fallback.
        int copyFd = -2;
        if (ctx.hasNativeFenceSync && next_vkGetFenceFdKHR) {
            VkFenceGetFdInfoKHR fdInfo = {};
            fdInfo.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR;
            fdInfo.fence = ctx.copyFence;
            fdInfo.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
            if (next_vkGetFenceFdKHR(ctx.device, &fdInfo, &copyFd) != VK_SUCCESS)
                copyFd = -2;
        }
        if (copyFd == -2)
            next_vkWaitForFences(ctx.device, 1, &ctx.copyFence, VK_TRUE,
                                 16000000ULL);

        EGLDisplay oldDpy = eglGetCurrentDisplay();
        EGLSurface oldRead = eglGetCurrentSurface(EGL_READ);
        EGLSurface oldDraw = eglGetCurrentSurface(EGL_DRAW);
        EGLContext oldCtx = eglGetCurrentContext();
        eglMakeCurrent(ctx.eglDpy, ctx.eglSurf, ctx.eglSurf, ctx.eglCtx);

        if (copyFd >= 0) {
            EGLint syncAttrs[] = { EGL_SYNC_NATIVE_FENCE_FD_ANDROID, copyFd,
                                   EGL_NONE };
            EGLSyncKHR copySync = ctx.eglCreateSyncKHR_(
                    ctx.eglDpy, EGL_SYNC_NATIVE_FENCE_ANDROID, syncAttrs);
            if (copySync != EGL_NO_SYNC_KHR) {
                ctx.eglWaitSyncKHR_(ctx.eglDpy, copySync, 0);
                ctx.eglDestroySyncKHR_(ctx.eglDpy, copySync);
            } else {
                close(copyFd);
                next_vkWaitForFences(ctx.device, 1, &ctx.copyFence, VK_TRUE,
                                     16000000ULL);
            }
        }

        initFilterGl();
        if (ctx.filter.init(gFilterGl)) {
        ctx.filter.applyStageA(ctx.stageFrame.glTex, ctx.currFrame.glTex,
                                   ctx.extent.width, ctx.extent.height, fp);
            if (stageB) {
                ctx.filter.applyStageB(ctx.currFrame.glTex,
                                       ctx.presentFrame.glTex,
                                       ctx.extent.width, ctx.extent.height,
                                       fp, ctx.frameIdx);
            }
        } else {
            filterOn = false;
        }

        // Hand the grade back to Vulkan the same way generation does.
        bool filterSemValid = false;
        if (filterOn && ctx.hasNativeFenceSync &&
            ctx.genDoneSem != VK_NULL_HANDLE && next_vkImportSemaphoreFdKHR) {
            EGLSyncKHR fSync = ctx.eglCreateSyncKHR_(
                    ctx.eglDpy, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
            if (fSync != EGL_NO_SYNC_KHR) {
                glFlush();
                int fFd = ctx.eglDupNativeFenceFD_(ctx.eglDpy, fSync);
                ctx.eglDestroySyncKHR_(ctx.eglDpy, fSync);
                if (fFd >= 0) {
                    VkImportSemaphoreFdInfoKHR imp = {};
                    imp.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                    imp.semaphore = ctx.genDoneSem;
                    imp.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                    imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                    imp.fd = fFd;
                    if (next_vkImportSemaphoreFdKHR(ctx.device, &imp) == VK_SUCCESS)
                        filterSemValid = true;
                    else
                        close(fFd);
                }
            }
        }
        if (filterOn && !filterSemValid) glFinish();

        if (oldCtx != EGL_NO_CONTEXT) {
            eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
        } else {
            eglMakeCurrent(ctx.eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
                           EGL_NO_CONTEXT);
        }

        if (filterOn) {
            // Write the graded frame back into the image the game is presenting.
            filterDoneSem = ctx.acquireSem();
            VkCommandBuffer cmd = ctx.acquireCmd();
            if (cmd == VK_NULL_HANDLE) cmd = allocCmdBuf(ctx);
            next_vkResetCommandBuffer(cmd, 0);
            beginCmd(cmd);
            cmdCopyImage(cmd,
                         stageB ? ctx.presentFrame.vkImage
                                : ctx.currFrame.vkImage,
                         swapImg,
                         ctx.extent.width, ctx.extent.height,
                         VK_IMAGE_LAYOUT_GENERAL, false,
                         VK_IMAGE_LAYOUT_UNDEFINED, true);
            next_vkEndCommandBuffer(cmd);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo si = {};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            si.waitSemaphoreCount = filterSemValid ? 1u : 0u;
            si.pWaitSemaphores = &ctx.genDoneSem;
            si.pWaitDstStageMask = &waitStage;
            if (filterDoneSem != VK_NULL_HANDLE) {
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &filterDoneSem;
            }
            next_vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        }
    }

    // Step 2a (extrapolation): REAL frame goes out immediately — the
    // presentation engine waits copyDoneSem, the game thread keeps running.
    if (!interpolation) {
        VkPresentInfoKHR realPresent = {};
        realPresent.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        realPresent.pNext = pPresentInfo->pNext;
        const VkSemaphore realWait =
                (filterDoneSem != VK_NULL_HANDLE) ? filterDoneSem : copyDoneSem;
        realPresent.waitSemaphoreCount = (realWait != VK_NULL_HANDLE) ? 1u : 0u;
        realPresent.pWaitSemaphores = &realWait;
        realPresent.swapchainCount = 1;
        realPresent.pSwapchains = &swapchain;
        realPresent.pImageIndices = &presentIdx;
        realPresent.pResults = pPresentInfo->pResults;
        finalResult = next_vkQueuePresentKHR(queue, &realPresent);

        // Anchor for spacing the synthetic frames that follow.
        ctx.pacer.anchorReal(afme::nowNs());
    }

    // Step 2b: Generate synth frames (both modes)
    if (generate) {
        // GLES must not read currFrame before the swapchain→AHB copy lands.
        // Preferred: export the copy fence as a sync_fd and have the GLES
        // context wait for it ON THE GPU (eglWaitSyncKHR) — the game thread
        // keeps running. Fallback: CPU fence wait (blocks until the game's
        // render + our blit finish; the stall that costs real FPS).
        int copyFd = -2;  // -2 = use the CPU fallback
        if (ctx.hasNativeFenceSync && next_vkGetFenceFdKHR) {
            VkFenceGetFdInfoKHR fdInfo = {};
            fdInfo.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR;
            fdInfo.fence = ctx.copyFence;
            fdInfo.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
            if (next_vkGetFenceFdKHR(ctx.device, &fdInfo, &copyFd) != VK_SUCCESS)
                copyFd = -2;
        }
        if (copyFd == -2)
            next_vkWaitForFences(ctx.device, 1, &ctx.copyFence, VK_TRUE, 16000000ULL);

        // Save EGL state
        EGLDisplay oldDpy = eglGetCurrentDisplay();
        EGLSurface oldRead = eglGetCurrentSurface(EGL_READ);
        EGLSurface oldDraw = eglGetCurrentSurface(EGL_DRAW);
        EGLContext oldCtx = eglGetCurrentContext();

        eglMakeCurrent(ctx.eglDpy, ctx.eglSurf, ctx.eglSurf, ctx.eglCtx);

        // GPU-side wait for the swapchain→AHB copy. fd == -1 means the fence
        // was already signaled at export time — nothing to wait on.
        if (copyFd >= 0) {
            EGLint syncAttrs[] = { EGL_SYNC_NATIVE_FENCE_FD_ANDROID, copyFd,
                                   EGL_NONE };
            EGLSyncKHR copySync = ctx.eglCreateSyncKHR_(
                ctx.eglDpy, EGL_SYNC_NATIVE_FENCE_ANDROID, syncAttrs);
            if (copySync != EGL_NO_SYNC_KHR) {
                ctx.eglWaitSyncKHR_(ctx.eglDpy, copySync, 0);
                ctx.eglDestroySyncKHR_(ctx.eglDpy, copySync);
            } else {
                close(copyFd);
                next_vkWaitForFences(ctx.device, 1, &ctx.copyFence,
                                     VK_TRUE, 16000000ULL);
            }
        }

        // The method property can flip mid-session (GameSpace writes it while
        // the game runs), but MobFGSR is otherwise only built at swapchain
        // creation — so selecting "motion" on a running game used to silently
        // keep extrapolating. Build it on first use instead. AFME's EGL
        // context is current here, which is what initMobFGSR requires.
        if (afme::config().wantMotion() && !ctx.mobfgsrInitialized && !ctx.mobfgsrAttempted) {
            ctx.mobfgsrAttempted = true;
            ALOGI("AFME: method=motion selected at runtime — building MobFGSR");
            initMobFGSR(ctx);
        }

        // Generation reads the RAW frame pair.
        //
        // This used to sharpen currFrame and hand the sharpened copy to
        // generation as src2 while src1 stayed raw. Sharpening exactly one of
        // the two inputs injects a full-frame high-frequency delta, and QCOM's
        // block matcher scores that delta as MOTION: every synthetic frame was
        // displaced along a bogus global vector, and alternating with untouched
        // real frames the whole image snapped top-left <-> bottom-right once per
        // real frame. Confirmed on onyx/Genshin — persist.sys.sgsr.mode=0 made
        // the artifact vanish completely, and moving the pass here kept the
        // sharpening while removing the artifact.
        //
        // SGSR1 is a presentation sharpener, so it belongs on the generated
        // RESULT: same visual gain, motion estimation left honest.
        GLuint srcPrevTex = ctx.prevFrame.glTex;
        GLuint srcCurrTex = ctx.currFrame.glTex;
        bool sharpen = sgsrMode >= 1 && ctx.sgsrInitialized &&
                       ctx.sharpenedFrame.glTex;

        // Motion estimation / dilation run ONCE per real frame; only the
        // warp is repeated per generated frame (3x/4x no longer pay ME 2-3×)
        if (interpolation) {
            applyMobFGSRPrepare(ctx);
        }

        // Interpolation already lands in fgResultTex, so only the extrapolation
        // path needs somewhere to write before a post pass reads it back.
        if (!interpolation && (stageB || sharpen) && !ctx.genScratchTex) {
            ctx.genScratchTex = createGLTex(GL_RGBA8, ctx.extent.width,
                                            ctx.extent.height, GL_LINEAR);
            if (!ctx.genScratchTex) { stageB = false; sharpen = false; }
        }

        float userFactor = afme::config().factorOverride.load();  // 0 = auto
        for (int i = 0; i < numGenFrames; i++) {
            // Divide by the ACTUAL number of presents per interval
            // (numGenFrames+1), not the requested multiplier. With the tier
            // clamp reducing generation below mult-1, /mult placed synth
            // frames at the wrong temporal phase (measured: 1 gen at 4x
            // produced factor=0.25 — quarter-phase motion displayed in the
            // half-phase slot, the interpolation was literally half-strength).
            float autoFactor = (float)(i + 1) / (float)(numGenFrames + 1);
            float factor = (userFactor > 0.0f) ? autoFactor * userFactor : autoFactor;

            // ── produce the raw generated frame ──────────────────────────
            GLuint rawTex;
            if (interpolation) {
                // ═══ MobFGSR interpolation ═══
                applyMobFGSRWarp(ctx, factor);
                rawTex = ctx.fgResultTex;
            } else {
                // ═══ HW Extrapolation (modes 0-2) ═══
                // A pass cannot read and write one texture, so whenever a post
                // pass follows, the driver writes scratch and the post pass
                // reads it back out into the synth AHB.
                rawTex = (sharpen || stageB) ? ctx.genScratchTex
                                             : ctx.synthFrames[i].glTex;
                ctx.glExtrapolateTex2D(srcPrevTex, srcCurrTex, rawTex, factor);
            }

            // ── post chain: sharpen → screen-space effects → synth AHB ────
            // Each step writes the next stage's input; the last one must land
            // in synthFrames[i], which is what gets blitted to the swapchain.
            if (sharpen && stageB) {
                applySGSR1(ctx, rawTex, ctx.sharpenedFrame.glTex,
                           ctx.extent.width, ctx.extent.height);
                ctx.filter.applyStageB(ctx.sharpenedFrame.glTex,
                                       ctx.synthFrames[i].glTex,
                                       ctx.extent.width, ctx.extent.height,
                                       fp, ctx.frameIdx * 8 + (uint64_t)i);
            } else if (sharpen) {
                applySGSR1(ctx, rawTex, ctx.synthFrames[i].glTex,
                           ctx.extent.width, ctx.extent.height);
            } else if (stageB) {
                // Stage B REPLACES the result→AHB copy rather than adding a
                // pass: it has to read rawTex and write the AHB either way.
                ctx.filter.applyStageB(rawTex, ctx.synthFrames[i].glTex,
                                       ctx.extent.width, ctx.extent.height,
                                       fp, ctx.frameIdx * 8 + (uint64_t)i);
            } else if (interpolation) {
                glCopyImageSubData(
                    rawTex, GL_TEXTURE_2D, 0, 0, 0, 0,
                    ctx.synthFrames[i].glTex, GL_TEXTURE_2D, 0, 0, 0, 0,
                    ctx.extent.width, ctx.extent.height, 1);
            }

            if ((fc % 300) == 0 && i == 0) {
                ALOGI("AFME: Gen frame %d/%d factor=%.2f method=%s sgsr=%d "
                      "(frame %lu, %dx)",
                      i+1, numGenFrames, factor,
                      interpolation ? "motion" : "extrapolate", sgsrMode,
                      (unsigned long)fc, mult);
            }
        }

        // GLES→Vulkan sync. Preferred: wrap the generation work in a native
        // fence and import it into genDoneSem — the first synth copy waits
        // for it ON THE GPU. Fallback: glFinish() (CPU block on the game
        // thread; measured ~0.2ms 2x HW, ~1.5ms 2x MobFGSR — plus however
        // long the GPU still needs, which is the real cost when loaded).
        bool genSemValid = false;
        if (ctx.hasNativeFenceSync && ctx.genDoneSem != VK_NULL_HANDLE &&
            next_vkImportSemaphoreFdKHR) {
            EGLSyncKHR genSync = ctx.eglCreateSyncKHR_(
                ctx.eglDpy, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
            if (genSync != EGL_NO_SYNC_KHR) {
                glFlush();  // makes the native fence fd available
                int genFd = ctx.eglDupNativeFenceFD_(ctx.eglDpy, genSync);
                ctx.eglDestroySyncKHR_(ctx.eglDpy, genSync);
                if (genFd >= 0) {
                    VkImportSemaphoreFdInfoKHR importInfo = {};
                    importInfo.sType =
                        VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                    importInfo.semaphore = ctx.genDoneSem;
                    importInfo.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                    importInfo.handleType =
                        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                    importInfo.fd = genFd;
                    if (next_vkImportSemaphoreFdKHR(ctx.device, &importInfo)
                            == VK_SUCCESS) {
                        genSemValid = true;  // fd ownership moved to semaphore
                    } else {
                        close(genFd);
                    }
                }
            }
        }
        if (!genSemValid) glFinish();

        // Restore EGL context — minimize EGL hold time
        if (oldCtx != EGL_NO_CONTEXT) {
            eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
        } else {
            eglMakeCurrent(ctx.eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }

        // Present the synth frames. The LAST copy submit signals drainFence —
        // next frame's drain point.
        //
        // Interpolation mode anchor: the real frame is presented AFTER the
        // synths, so the anchor used by the spacing sleep must be relative to
        // the START of the synth emission — otherwise the anchor stays 0 and
        // the spacing path is silently dead for method=1 (bug D).
        if (interpolation) ctx.pacer.anchorReal(afme::nowNs());
        for (int i = 0; i < numGenFrames; i++) {
            VkSemaphore acquireSem = ctx.acquireSem();
            if (acquireSem == VK_NULL_HANDLE) {
                ALOGW("AFME: Semaphore pool exhausted at gen frame %d", i);
                break;
            }

            uint32_t newIdx = 0;
            // 2ms, not 100ms: this runs on the game's render thread, and a
            // starved swapchain used to stall it for a twelfth of a second.
            // A synthetic frame is always worth dropping instead.
            VkResult acqResult = next_vkAcquireNextImageKHR(
                ctx.device, swapchain, 2000000ULL,
                acquireSem, VK_NULL_HANDLE, &newIdx);

            if (acqResult != VK_SUCCESS && acqResult != VK_SUBOPTIMAL_KHR) {
                ALOGW("AFME: AcquireNextImage failed: %d", acqResult);
                break;
            }

            VkImage newSwapImg = ctx.swapchainImages[newIdx];
            VkSemaphore copySem = ctx.acquireSem();

            VkCommandBuffer cmd = ctx.acquireCmd();
            if (cmd == VK_NULL_HANDLE) cmd = allocCmdBuf(ctx); // fallback
            next_vkResetCommandBuffer(cmd, 0);
            beginCmd(cmd);
            cmdCopyImage(cmd, ctx.synthFrames[i].vkImage, newSwapImg,
                         ctx.extent.width, ctx.extent.height,
                         VK_IMAGE_LAYOUT_GENERAL, false,
                         VK_IMAGE_LAYOUT_UNDEFINED, true);
            next_vkEndCommandBuffer(cmd);

            // The first synth copy also waits the GLES generation fence
            // (imported into genDoneSem); later submits are ordered behind
            // it by same-queue submission order.
            VkSemaphore submitWaits[2] = { acquireSem, ctx.genDoneSem };
            VkPipelineStageFlags submitWaitStages[2] = {
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            submitInfo.waitSemaphoreCount = (i == 0 && genSemValid) ? 2u : 1u;
            submitInfo.pWaitSemaphores = submitWaits;
            submitInfo.pWaitDstStageMask = submitWaitStages;
            if (copySem != VK_NULL_HANDLE) {
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = &copySem;
            }

            // Fence on the LAST synth submit = next frame's drain point
            VkFence submitFence = (i == numGenFrames - 1) ? ctx.drainFence
                                                          : VK_NULL_HANDLE;
            next_vkQueueSubmit(queue, 1, &submitInfo, submitFence);
            if (submitFence != VK_NULL_HANDLE) synthSubmitted = true;

            VkPresentInfoKHR genPresent = {};
            genPresent.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            if (copySem != VK_NULL_HANDLE) {
                genPresent.waitSemaphoreCount = 1;
                genPresent.pWaitSemaphores = &copySem;
            }
            genPresent.swapchainCount = 1;
            genPresent.pSwapchains = &swapchain;
            genPresent.pImageIndices = &newIdx;

            // PACING: without a desired present time, this synth frame lands
            // in the very next vsync slot after the real frame — measured
            // present2present is bimodal 8ms / 16-24ms, which strobes.
            // Schedule it at its temporal position inside the game-frame
            // interval ((i+1)/(numGenFrames+1) of the interval AFTER the
            // real frame), minus ~half a 120Hz vsync so SurfaceFlinger's
            // latch quantization rounds to the nearest slot.
            //
            // NOTE: on bp4a/onyx SurfaceFlinger was measured to DROP >50% of
            // delayed synth buffers instead of holding them — the spacing
            // sleep below is the mechanism that works here; this stamp stays
            // opt-in via persist.sys.afme.pacing for future SF releases.
            VkPresentTimeGOOGLE presentTime = {};
            VkPresentTimesInfoGOOGLE timesInfo = {};
            if (afme::config().pacing.load(std::memory_order_relaxed) &&
                ctx.hasDisplayTiming && ctx.pacer.intervalMs() > 0.0f) {
                int64_t offsetNs = (int64_t)(ctx.pacer.intervalMs() * 1e6f *
                                             (float)(i + 1) / (float)(numGenFrames + 1))
                                   - 4000000LL;
                if (offsetNs < 0) offsetNs = 0;
                presentTime.presentID = ++ctx.presentId;
                presentTime.desiredPresentTime = (uint64_t)(nowNs + offsetNs);
                timesInfo.sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
                timesInfo.swapchainCount = 1;
                timesInfo.pTimes = &presentTime;
                genPresent.pNext = &timesInfo;
            }

            // Hold this synthetic frame until its temporal slot inside the
            // game's frame interval. Presenting it now would land it in the
            // vsync immediately after the real frame, which is what makes
            // generation invisible: the real frame gets one vsync of screen
            // time and the synthetic one gets the rest.
            //
            // Scaled by the governor, so if the sleep turns out to cost the
            // game real frames this decays to a no-op on its own.
            ctx.pacer.spaceSynth(i, numGenFrames, !interpolation);

            next_vkQueuePresentKHR(queue, &genPresent);
            ctx.stats.addGen();
        }

        ctx.drainPending = synthSubmitted;
    }

    // Step 2c (interpolation): the REAL frame goes out AFTER the in-between
    // frames — correct temporal order for interpolation, +1 vsync latency.
    if (interpolation) {
        // The game's semaphores were consumed by our copy submit, so make
        // sure render+copy finished before presenting the real frame. (On
        // the native-fence path the fence was never CPU-waited during the
        // frame; by now it is already signaled, so this is ~free.)
        next_vkWaitForFences(ctx.device, 1, &ctx.copyFence, VK_TRUE, 16000000ULL);
        // The real frame owns the LAST slot of the interval. Without this hold
        // it goes out immediately behind the final synth, so at 2x half of all
        // presents arrive as a bunched pair.
        ctx.pacer.spaceRealTail(numGenFrames);
        VkPresentInfoKHR realPresent = {};
        realPresent.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        realPresent.pNext = pPresentInfo->pNext;
        realPresent.waitSemaphoreCount =
                (filterDoneSem != VK_NULL_HANDLE) ? 1u : 0u;
        realPresent.pWaitSemaphores = &filterDoneSem;
        realPresent.swapchainCount = 1;
        realPresent.pSwapchains = &swapchain;
        realPresent.pImageIndices = &presentIdx;
        realPresent.pResults = pPresentInfo->pResults;
        finalResult = next_vkQueuePresentKHR(queue, &realPresent);
    }

    // Step 3: Rotate prev ↔ curr (and MobFGSR buffers if active)
    std::swap(ctx.prevFrame, ctx.currFrame);
    if (ctx.mobfgsrInitialized) {
        swapMobFGSRBuffers(ctx);
    }
    ctx.hasPrevFrame = true;
    ctx.frameIdx++;

    if ((fc % 300) == 0) {
        ALOGI("AFME: Frame %lu — %dx active (%d gen/frame), gen=%lu total, "
              "paceable=%d spacingScale=%.2f cap=%.1ffps interval=%.2fms",
              (unsigned long)fc, mult, numGenFrames,
              (unsigned long)ctx.frameIdx, (int)tier.paceable,
              ctx.pacer.spacingScale(), ctx.pacer.capabilityFps(),
              ctx.pacer.intervalMs());
    }

    // ── FPS stats ────────────────────────────────────────────────────
    // Published as a log line because property_set() from a game process can
    // never work: platform sepolicy carries `neverallow all_untrusted_apps
    // property_type:property_service set`. GameSpace (system_app, READ_LOGS)
    // parses "AFME-STATS" out of logcat for the overlay.
    ctx.stats.publish(afme::nowNs());

    // Pace the game to exactly (numGen+1) vsync slots so total presents fill
    // every slot: 120Hz / 3 = a locked 40fps base, a steady 120 total. Prefer
    // the measured vsync period; fall back to inverting the staged prop.
    const int64_t vsyncNs = (ctx.refreshCycleNs >= 2000000 &&
                             ctx.refreshCycleNs <= 100000000)
                                ? (int64_t)ctx.refreshCycleNs
                                : 1000000000LL / (int64_t)effectiveHz(ctx);
    ctx.pacer.endPresent(afme::nowNs(), numGenFrames, tier.paceable, vsyncNs);

    return finalResult;
}

} // anonymous namespace

// ─── GetProcAddr dispatchers ────────────────────────────────────────────────

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_vkGetInstanceProcAddr(
        VkInstance instance, const char* pName);
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_vkGetDeviceProcAddr(
        VkDevice device, const char* pName);

static const std::unordered_map<std::string, PFN_vkVoidFunction> kLayerFunctions = {
    {"vkCreateInstance",    reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateInstance)},
    {"vkDestroyInstance",   reinterpret_cast<PFN_vkVoidFunction>(&layer_vkDestroyInstance)},
    {"vkCreateDevice",      reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateDevice)},
    {"vkDestroyDevice",     reinterpret_cast<PFN_vkVoidFunction>(&layer_vkDestroyDevice)},
    {"vkGetInstanceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(&layer_vkGetInstanceProcAddr)},
    {"vkGetDeviceProcAddr",   reinterpret_cast<PFN_vkVoidFunction>(&layer_vkGetDeviceProcAddr)},
};

static const std::unordered_map<std::string, PFN_vkVoidFunction> kDeviceHooks = {
    {"vkQueuePresentKHR",       reinterpret_cast<PFN_vkVoidFunction>(&layer_vkQueuePresentKHR)},
    {"vkCreateSwapchainKHR",    reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateSwapchainKHR)},
    {"vkDestroySwapchainKHR",   reinterpret_cast<PFN_vkVoidFunction>(&layer_vkDestroySwapchainKHR)},
    {"vkDestroyDevice",         reinterpret_cast<PFN_vkVoidFunction>(&layer_vkDestroyDevice)},
    // Claimed unconditionally: the dispatch table is built once at device
    // creation, and AF is switched live, per game, long after that.
    {"vkCreateSampler",         reinterpret_cast<PFN_vkVoidFunction>(&layer_vkCreateSampler)},
};

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_vkGetInstanceProcAddr(
        VkInstance instance, const char* pName) {
    if (!pName) return nullptr;

    auto it = kLayerFunctions.find(pName);
    if (it != kLayerFunctions.end()) return it->second;

    auto hit = kDeviceHooks.find(pName);
    if (hit != kDeviceHooks.end()) return hit->second;

    if (strcmp(pName, "vkEnumerateInstanceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&vkEnumerateInstanceLayerProperties);
    if (strcmp(pName, "vkEnumerateInstanceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&vkEnumerateInstanceExtensionProperties);

    if (next_vkGetInstanceProcAddr) return next_vkGetInstanceProcAddr(instance, pName);
    return nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_vkGetDeviceProcAddr(
        VkDevice device, const char* pName) {
    if (!pName) return nullptr;

    auto it = kLayerFunctions.find(pName);
    if (it != kLayerFunctions.end()) return it->second;

    auto hit = kDeviceHooks.find(pName);
    if (hit != kDeviceHooks.end()) return hit->second;

    if (next_vkGetDeviceProcAddr) return next_vkGetDeviceProcAddr(device, pName);
    return nullptr;
}

// ─── Exported layer enumeration ─────────────────────────────────────────────

extern "C" {

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
        uint32_t* pPropertyCount, VkLayerProperties* pProperties) {
    if (!pProperties) { *pPropertyCount = 1; return VK_SUCCESS; }
    if (*pPropertyCount < 1) return VK_INCOMPLETE;
    *pPropertyCount = 1;
    memset(pProperties, 0, sizeof(VkLayerProperties));
    strncpy(pProperties->layerName, kLayerName, sizeof(pProperties->layerName) - 1);
    strncpy(pProperties->description, kLayerDescription, sizeof(pProperties->description) - 1);
    pProperties->specVersion = kLayerSpecVersion;
    pProperties->implementationVersion = kLayerImplVersion;
    return VK_SUCCESS;
}

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
        const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties*) {
    if (pLayerName && strcmp(pLayerName, kLayerName) == 0) {
        *pPropertyCount = 0; return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
        VkPhysicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties) {
    return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
        VkPhysicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties*) {
    if (pLayerName && strcmp(pLayerName, kLayerName) == 0) {
        *pPropertyCount = 0; return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

__attribute__((visibility("default")))
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL VK_LAYER_AFME_frame_genGetInstanceProcAddr(
        VkInstance instance, const char* pName) {
    return layer_vkGetInstanceProcAddr(instance, pName);
}

__attribute__((visibility("default")))
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL VK_LAYER_AFME_frame_genGetDeviceProcAddr(
        VkDevice device, const char* pName) {
    return layer_vkGetDeviceProcAddr(device, pName);
}

__attribute__((visibility("default")))
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
        VkInstance instance, const char* pName) {
    return layer_vkGetInstanceProcAddr(instance, pName);
}

__attribute__((visibility("default")))
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
        VkDevice device, const char* pName) {
    return layer_vkGetDeviceProcAddr(device, pName);
}

} // extern "C"
