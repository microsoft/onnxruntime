// Copyright (c) Microsoft Corporation. All rights reserved.਍⼀⼀ 䰀椀挀攀渀猀攀搀 甀渀搀攀爀 琀栀攀 䴀䤀吀 䰀椀挀攀渀猀攀⸀ഀഀ
਍⌀瀀爀愀最洀愀 漀渀挀攀ഀഀ
਍⌀椀渀挀氀甀搀攀 㰀洀攀洀漀爀礀㸀ഀഀ
#include <mutex>਍⌀椀渀挀氀甀搀攀 㰀漀瀀琀椀漀渀愀氀㸀ഀഀ
਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀眀攀戀最瀀甀开攀砀琀攀爀渀愀氀开栀攀愀搀攀爀⸀栀∀ഀഀ
਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀挀漀洀洀漀渀⼀挀漀洀洀漀渀⸀栀∀ഀഀ
#include "core/providers/webgpu/buffer_manager.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀瀀爀漀最爀愀洀开洀愀渀愀最攀爀⸀栀∀ഀഀ
#include "core/providers/webgpu/webgpu_utils.h"਍ഀഀ
#if defined(ENABLE_PIX_FOR_WEBGPU_EP)਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀眀攀戀最瀀甀开瀀椀砀开昀爀愀洀攀开最攀渀攀爀愀琀漀爀⸀栀∀ഀഀ
#endif  // ENABLE_PIX_FOR_WEBGPU_EP਍ഀഀ
namespace onnxruntime {਍挀氀愀猀猀 吀攀渀猀漀爀㬀ഀഀ
਍渀愀洀攀猀瀀愀挀攀 眀攀戀最瀀甀 笀ഀഀ
class WebGpuContext;਍挀氀愀猀猀 䌀漀洀瀀甀琀攀䌀漀渀琀攀砀琀䈀愀猀攀㬀ഀഀ
class ProgramBase;਍ഀഀ
// PendingKernelInfo stores profiling information for a kernel execution਍猀琀爀甀挀琀 倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀 笀ഀഀ
  PendingKernelInfo(std::string_view kernel_name,਍                    猀琀搀㨀㨀猀琀爀椀渀最开瘀椀攀眀 欀攀爀渀攀氀开琀礀瀀攀Ⰰഀഀ
                    std::string_view program_name,਍                    猀琀搀㨀㨀猀琀爀椀渀最开瘀椀攀眀 挀愀挀栀攀开欀攀礀Ⰰഀഀ
                    const std::vector<ProgramInput>& inputs,਍                    挀漀渀猀琀 猀琀搀㨀㨀瘀攀挀琀漀爀㰀倀爀漀最爀愀洀伀甀琀瀀甀琀㸀☀ 漀甀琀瀀甀琀猀⤀ഀഀ
      : name{absl::StrJoin({kernel_name, kernel_type, program_name}, "&")}, cache_key{cache_key} {਍    ⼀⼀ 匀琀漀爀攀 猀栀愀瀀攀 椀渀昀漀爀洀愀琀椀漀渀 椀渀猀琀攀愀搀 漀昀 琀攀渀猀漀爀 瀀漀椀渀琀攀爀猀 琀漀 愀瘀漀椀搀 愀挀挀攀猀猀椀渀最 爀攀氀攀愀猀攀搀 琀攀渀猀漀爀猀ഀഀ
    input_shapes.reserve(inputs.size());਍    昀漀爀 ⠀挀漀渀猀琀 愀甀琀漀☀ 椀渀瀀甀琀 㨀 椀渀瀀甀琀猀⤀ 笀ഀഀ
      input_shapes.emplace_back(input.use_override_shape ? input.override_shape : input.tensor->Shape());਍    紀ഀഀ
    output_shapes.reserve(outputs.size());਍    昀漀爀 ⠀挀漀渀猀琀 愀甀琀漀☀ 漀甀琀瀀甀琀 㨀 漀甀琀瀀甀琀猀⤀ 笀ഀഀ
      output_shapes.emplace_back(output.use_override_shape ? output.override_shape : output.tensor->Shape());਍    紀ഀഀ
  }਍ഀഀ
  PendingKernelInfo(const PendingKernelInfo&) = default;਍  倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀☀ 漀瀀攀爀愀琀漀爀㴀⠀挀漀渀猀琀 倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀☀⤀ 㴀 搀攀昀愀甀氀琀㬀ഀഀ
  PendingKernelInfo(PendingKernelInfo&&) = default;਍  倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀☀ 漀瀀攀爀愀琀漀爀㴀⠀倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀☀☀⤀ 㴀 搀攀昀愀甀氀琀㬀ഀഀ
਍  猀琀搀㨀㨀猀琀爀椀渀最 渀愀洀攀㬀ഀഀ
  std::string cache_key;਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀吀攀渀猀漀爀匀栀愀瀀攀㸀 椀渀瀀甀琀开猀栀愀瀀攀猀㬀ഀഀ
  std::vector<TensorShape> output_shapes;਍紀㬀ഀഀ
਍⼀⼀ 䐀攀昀椀渀椀琀椀漀渀 昀漀爀 䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀 椀渀 琀栀攀 眀攀戀最瀀甀 渀愀洀攀猀瀀愀挀攀ഀഀ
struct CapturedCommandInfo {਍  眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀椀瀀攀氀椀渀攀 挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀㬀ഀഀ
  WGPUBindGroup bind_group;਍  圀䜀倀唀䈀椀渀搀䜀爀漀甀瀀䰀愀礀漀甀琀 戀椀渀搀开最爀漀甀瀀开氀愀礀漀甀琀㬀ഀഀ
  std::array<uint32_t, 3> dispatch_group;਍  ⼀⼀ 圀䜀倀唀䈀甀昀昀攀爀 昀漀爀 椀渀搀椀爀攀挀琀 搀椀猀瀀愀琀挀栀Ⰰ 渀甀氀氀瀀琀爀 椀昀 渀漀琀 甀猀椀渀最 椀渀搀椀爀攀挀琀 搀椀猀瀀愀琀挀栀ഀഀ
  WGPUBuffer indirect_buffer;਍  ⼀⼀ 伀瀀琀椀漀渀愀氀 瀀爀漀昀椀氀椀渀最 搀愀琀愀ഀഀ
  std::optional<PendingKernelInfo> pending_kernel_info;਍紀㬀ഀഀ
਍猀琀爀甀挀琀 圀攀戀䜀瀀甀䈀甀昀昀攀爀䌀愀挀栀攀䌀漀渀昀椀最 笀ഀഀ
  struct ConfigEntry {਍    䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀 洀漀搀攀㬀ഀഀ
    std::string config_string;  // preserved for customized configuration, eg. bucket sizes਍  紀㬀ഀഀ
  ConfigEntry storage{BufferCacheMode::Bucket, {}};਍  䌀漀渀昀椀最䔀渀琀爀礀 甀渀椀昀漀爀洀笀䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀匀椀洀瀀氀攀Ⰰ 笀紀紀㬀ഀഀ
  ConfigEntry query_resolve{BufferCacheMode::Disabled, {}};਍  䌀漀渀昀椀最䔀渀琀爀礀 搀攀昀愀甀氀琀开攀渀琀爀礀笀䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀䐀椀猀愀戀氀攀搀Ⰰ 笀紀紀㬀ഀഀ
};਍ഀഀ
/// <summary>਍⼀⼀⼀ 刀攀瀀爀攀猀攀渀琀猀 琀栀攀 挀漀渀昀椀最甀爀愀琀椀漀渀 漀瀀琀椀漀渀猀 昀漀爀 挀爀攀愀琀椀渀最 愀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀⸀ഀഀ
/// </summary>਍猀琀爀甀挀琀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䌀漀渀昀椀最 笀ഀഀ
  int context_id{0};਍  圀䜀倀唀䤀渀猀琀愀渀挀攀 椀渀猀琀愀渀挀攀笀渀甀氀氀瀀琀爀紀㬀ഀഀ
  WGPUDevice device{nullptr};਍  挀漀渀猀琀 瘀漀椀搀⨀ 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀笀渀甀氀氀瀀琀爀紀㬀ഀഀ
  ValidationMode validation_mode{਍⌀椀昀渀搀攀昀 一䐀䔀䈀唀䜀ഀഀ
      webgpu::ValidationMode::Full  // for debug build, enable full validation by default਍⌀攀氀猀攀ഀഀ
      webgpu::ValidationMode::Basic  // for release build, enable basic validation by default਍⌀攀渀搀椀昀  ⼀⼀ ℀一䐀䔀䈀唀䜀ഀഀ
  };਍  戀漀漀氀 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀笀昀愀氀猀攀紀㬀ഀഀ
  bool preserve_device{false};਍  甀椀渀琀㘀㐀开琀 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀笀　紀㬀ഀഀ
  WebGpuBufferCacheConfig buffer_cache_config{};਍  椀渀琀 瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀笀猀琀愀琀椀挀开挀愀猀琀㰀椀渀琀㸀⠀圀䜀倀唀倀漀眀攀爀倀爀攀昀攀爀攀渀挀攀开䠀椀最栀倀攀爀昀漀爀洀愀渀挀攀⤀紀㬀ഀഀ
  int backend_type{਍⌀椀昀搀攀昀 开圀䤀一㌀㈀ഀഀ
  // Setup Windows default backend type based on the build configuration਍⌀椀昀 搀攀昀椀渀攀搀⠀䐀䄀圀一开䔀一䄀䈀䰀䔀开䐀㌀䐀㄀㈀⤀ഀഀ
      static_cast<int>(WGPUBackendType_D3D12)਍⌀攀氀椀昀 搀攀昀椀渀攀搀⠀䐀䄀圀一开䔀一䄀䈀䰀䔀开嘀唀䰀䬀䄀一⤀ഀഀ
      static_cast<int>(WGPUBackendType_Vulkan)਍⌀攀氀猀攀ഀഀ
      0਍⌀攀渀搀椀昀ഀഀ
#else਍      　ഀഀ
#endif਍  紀㬀ഀഀ
};਍ഀഀ
class WebGpuContextFactory {਍ 瀀甀戀氀椀挀㨀ഀഀ
  struct WebGpuContextInfo {਍    猀琀搀㨀㨀甀渀椀焀甀攀开瀀琀爀㰀圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㸀 挀漀渀琀攀砀琀㬀ഀഀ
    int ref_count;਍  紀㬀ഀഀ
਍  ⼀⼀⼀ 㰀猀甀洀洀愀爀礀㸀ഀഀ
  /// Create a new WebGPU context for the specified context ID if not present, or return the existing one. (ref-count based)਍  ⼀⼀⼀ 㰀⼀猀甀洀洀愀爀礀㸀ഀഀ
  static WebGpuContext& CreateContext(const WebGpuContextConfig& config);਍ഀഀ
  /// <summary>਍  ⼀⼀⼀ 䜀攀琀 琀栀攀 圀攀戀䜀倀唀 挀漀渀琀攀砀琀 昀漀爀 琀栀攀 猀瀀攀挀椀昀椀攀搀 挀漀渀琀攀砀琀 䤀䐀⸀ 吀栀爀漀眀 椀昀 渀漀琀 瀀爀攀猀攀渀琀⸀ഀഀ
  /// </summary>਍  猀琀愀琀椀挀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀☀ 䜀攀琀䌀漀渀琀攀砀琀⠀椀渀琀 挀漀渀琀攀砀琀开椀搀⤀㬀ഀഀ
਍  ⼀⼀⼀ 㰀猀甀洀洀愀爀礀㸀ഀഀ
  /// Release the WebGPU context. (ref-count based)਍  ⼀⼀⼀ 㰀⼀猀甀洀洀愀爀礀㸀ഀഀ
  static void ReleaseContext(int context_id);਍ഀഀ
  static void Cleanup();਍ഀഀ
  /// <summary>਍  ⼀⼀⼀ 刀攀琀甀爀渀 琀栀攀 搀攀昀愀甀氀琀 挀漀渀琀攀砀琀⸀ 䌀爀攀愀琀攀 椀昀 渀漀琀 瀀爀攀猀攀渀琀⸀ഀഀ
  /// </summary>਍  猀琀愀琀椀挀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀☀ 䐀攀昀愀甀氀琀䌀漀渀琀攀砀琀⠀⤀㬀ഀഀ
਍ 瀀爀椀瘀愀琀攀㨀ഀഀ
  WebGpuContextFactory() {}਍ഀഀ
  static std::mutex mutex_;਍  猀琀愀琀椀挀 猀琀搀㨀㨀漀渀挀攀开昀氀愀最 椀渀椀琀开搀攀昀愀甀氀琀开昀氀愀最开㬀ഀഀ
਍  ⼀⼀ 唀猀攀 瀀漀椀渀琀攀爀猀 琀漀 栀攀愀瀀ⴀ愀氀氀漀挀愀琀攀搀 漀戀樀攀挀琀猀 猀漀 琀栀愀琀 琀栀攀椀爀 搀攀猀琀爀甀挀琀漀爀猀 搀漀 一伀吀 爀甀渀ഀഀ
  // during static destruction at process exit. This avoids crashes when dependent਍  ⼀⼀ 䐀䰀䰀猀 ⠀攀⸀最⸀ 搀砀挀漀洀瀀椀氀攀爀⸀搀氀氀⤀ 栀愀瘀攀 愀氀爀攀愀搀礀 戀攀攀渀 甀渀氀漀愀搀攀搀 戀礀 琀栀攀 伀匀⸀ഀഀ
  // Cleanup() explicitly deletes them during normal unload. In the shared-library਍  ⼀⼀ 戀甀椀氀搀 琀栀椀猀 椀猀 爀攀愀挀栀攀搀 瘀椀愀 刀攀氀攀愀猀攀䔀瀀䘀愀挀琀漀爀礀Ⰰ 愀渀搀 椀渀 琀栀攀 圀攀戀䜀倀唀 猀琀愀琀椀挀ⴀ氀椀戀 戀甀椀氀搀ഀഀ
  // it is reached from OrtEnv::~OrtEnv via CleanupWebGpuContexts().਍  ⼀⼀ 伀渀 愀戀渀漀爀洀愀氀⼀瀀爀漀挀攀猀猀 琀攀爀洀椀渀愀琀椀漀渀 琀栀攀礀 猀椀洀瀀氀礀 氀攀愀欀Ⰰ 眀栀椀挀栀 椀猀 猀愀昀攀⸀ഀഀ
  static std::unordered_map<int32_t, WebGpuContextInfo>* contexts_;਍  猀琀愀琀椀挀 圀䜀倀唀䤀渀猀琀愀渀挀攀 搀攀昀愀甀氀琀开椀渀猀琀愀渀挀攀开㬀ഀഀ
};਍ഀഀ
// Class WebGpuContext includes all necessary resources for the context.਍挀氀愀猀猀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀 昀椀渀愀氀 笀ഀഀ
 public:਍  匀琀愀琀甀猀 圀愀椀琀⠀眀最瀀甀㨀㨀䘀甀琀甀爀攀 昀⤀㬀ഀഀ
਍  挀漀渀猀琀 眀最瀀甀㨀㨀䐀攀瘀椀挀攀☀ 䐀攀瘀椀挀攀⠀⤀ 挀漀渀猀琀 笀 爀攀琀甀爀渀 搀攀瘀椀挀攀开㬀 紀ഀഀ
਍  挀漀渀猀琀 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀䤀渀昀漀☀ 䄀搀愀瀀琀攀爀䤀渀昀漀⠀⤀ 挀漀渀猀琀 笀 爀攀琀甀爀渀 愀搀愀瀀琀攀爀开椀渀昀漀开㬀 紀ഀഀ
  const wgpu::Limits& DeviceLimits() const { return device_limits_; }਍  戀漀漀氀 䐀攀瘀椀挀攀䠀愀猀䘀攀愀琀甀爀攀⠀眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀 昀攀愀琀甀爀攀⤀ 挀漀渀猀琀 笀 爀攀琀甀爀渀 搀攀瘀椀挀攀开昀攀愀琀甀爀攀猀开⸀挀漀渀琀愀椀渀猀⠀昀攀愀琀甀爀攀⤀㬀 紀ഀഀ
#if !defined(__wasm__)਍  挀漀渀猀琀 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀倀爀漀瀀攀爀琀椀攀猀匀甀戀最爀漀甀瀀䴀愀琀爀椀砀䌀漀渀昀椀最猀☀ 匀甀戀最爀漀甀瀀䴀愀琀爀椀砀䌀漀渀昀椀最猀⠀⤀ 挀漀渀猀琀 笀 爀攀琀甀爀渀 猀甀戀最爀漀甀瀀开洀愀琀爀椀砀开挀漀渀昀椀最猀开㬀 紀ഀഀ
#endif਍ഀഀ
  const wgpu::CommandEncoder& GetCommandEncoder() {਍    椀昀 ⠀℀挀甀爀爀攀渀琀开挀漀洀洀愀渀搀开攀渀挀漀搀攀爀开⤀ 笀ഀഀ
      current_command_encoder_ = device_.CreateCommandEncoder();਍    紀ഀഀ
    return current_command_encoder_;਍  紀ഀഀ
਍  挀漀渀猀琀 眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀☀ 䜀攀琀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀⠀⤀ 笀ഀഀ
    if (!current_compute_pass_encoder_) {਍      愀甀琀漀☀ 挀漀洀洀愀渀搀开攀渀挀漀搀攀爀 㴀 䜀攀琀䌀漀洀洀愀渀搀䔀渀挀漀搀攀爀⠀⤀㬀ഀഀ
਍      眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀愀猀猀䐀攀猀挀爀椀瀀琀漀爀 挀漀洀瀀甀琀攀开瀀愀猀猀开搀攀猀挀笀紀㬀ഀഀ
਍      椀昀 ⠀椀猀开瀀爀漀昀椀氀椀渀最开 ☀☀ 焀甀攀爀礀开琀礀瀀攀开 㴀㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀䄀琀倀愀猀猀攀猀 ☀☀ 最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀愀琀攀开 ℀㴀 䜀爀愀瀀栀䌀愀瀀琀甀爀攀匀琀愀琀攀㨀㨀䌀愀瀀琀甀爀椀渀最⤀ 笀ഀഀ
        wgpu::PassTimestampWrites timestampWrites = {਍            渀甀氀氀瀀琀爀Ⰰഀഀ
            query_set_,਍            渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 ⨀ ㈀Ⰰഀഀ
            num_pending_dispatches_ * 2 + 1};਍        挀漀洀瀀甀琀攀开瀀愀猀猀开搀攀猀挀⸀琀椀洀攀猀琀愀洀瀀圀爀椀琀攀猀 㴀 ☀琀椀洀攀猀琀愀洀瀀圀爀椀琀攀猀㬀ഀഀ
      }਍ഀഀ
      current_compute_pass_encoder_ = command_encoder.BeginComputePass(&compute_pass_desc);਍    紀ഀഀ
    return current_compute_pass_encoder_;਍  紀ഀഀ
਍  瘀漀椀搀 䔀渀搀䌀漀洀瀀甀琀攀倀愀猀猀⠀⤀ 笀ഀഀ
    if (current_compute_pass_encoder_) {਍      挀甀爀爀攀渀琀开挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀开⸀䔀渀搀⠀⤀㬀ഀഀ
      current_compute_pass_encoder_ = nullptr;਍    紀ഀഀ
  }਍  瘀漀椀搀 䌀愀瀀琀甀爀攀䈀攀最椀渀⠀猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀攀戀最瀀甀㨀㨀䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀㸀⨀ 挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀Ⰰ 挀漀渀猀琀 眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀☀ 戀甀昀昀攀爀开洀愀渀愀最攀爀⤀㬀ഀഀ
  void CaptureEnd();਍  瘀漀椀搀 刀攀瀀氀愀礀⠀挀漀渀猀琀 猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀攀戀最瀀甀㨀㨀䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀㸀☀ 挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀Ⰰ 挀漀渀猀琀 眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀☀ 戀甀昀昀攀爀开洀愀渀愀最攀爀⤀㬀ഀഀ
  void ReleaseGraphResources(std::vector<webgpu::CapturedCommandInfo>& captured_commands);਍ഀഀ
  void Flush(const webgpu::BufferManager& buffer_mgr);਍ഀഀ
  /**਍   ⨀ 䜀攀琀 琀栀攀 戀甀昀昀攀爀 洀愀渀愀最攀爀⸀ഀഀ
   */਍  眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀☀ 䈀甀昀昀攀爀䴀愀渀愀最攀爀⠀⤀ 挀漀渀猀琀 笀 爀攀琀甀爀渀 ⨀戀甀昀昀攀爀开洀最爀开㬀 紀ഀഀ
਍  ⼀⨀⨀ഀഀ
   * Get the initializer buffer manager.਍   ⨀ഀഀ
   * This buffer manager is used for read-only buffers (e.g. initializers).਍   ⨀⼀ഀഀ
  webgpu::BufferManager& InitializerBufferManager() const { return *initializer_buffer_mgr_; }਍ഀഀ
  inline webgpu::ValidationMode ValidationMode() const {਍    爀攀琀甀爀渀 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开㬀ഀഀ
  }਍ഀഀ
  //਍  ⼀⼀ 䜀攀琀 匀瀀氀椀琀ⴀ䬀 挀漀渀昀椀最甀爀愀琀椀漀渀⸀ഀഀ
  //਍  挀漀渀猀琀 匀瀀氀椀琀䬀䌀漀渀昀椀最☀ 䜀攀琀匀瀀氀椀琀䬀䌀漀渀昀椀最⠀⤀ 挀漀渀猀琀 笀ഀഀ
    return *split_k_config_;਍  紀ഀഀ
਍  瘀漀椀搀 匀琀愀爀琀倀爀漀昀椀氀椀渀最⠀⤀㬀ഀഀ
  // Collect pending GPU profiling data into the given events vector.਍  瘀漀椀搀 䌀漀氀氀攀挀琀倀爀漀昀椀氀椀渀最䐀愀琀愀⠀瀀爀漀昀椀氀椀渀最㨀㨀䔀瘀攀渀琀猀☀ 攀瘀攀渀琀猀⤀㬀ഀഀ
  // Collect pending GPU profiling data into the shared events_ vector (run-level).਍  瘀漀椀搀 䌀漀氀氀攀挀琀倀爀漀昀椀氀椀渀最䐀愀琀愀⠀⤀㬀ഀഀ
  void EndProfiling(TimePoint, profiling::Events& events);਍ഀഀ
  //਍  ⼀⼀ 倀甀猀栀 攀爀爀漀爀 猀挀漀瀀攀⸀ഀഀ
  //਍  ⼀⼀ 吀栀椀猀 椀猀 甀猀攀昀甀氀 漀渀氀礀 眀栀攀渀 ∀猀欀椀瀀开瘀愀氀椀搀愀琀椀漀渀∀ 椀猀 渀漀琀 猀攀琀⸀ഀഀ
  //਍  瘀漀椀搀 倀甀猀栀䔀爀爀漀爀匀挀漀瀀攀⠀⤀㬀ഀഀ
਍  ⼀⼀ഀഀ
  // Pop error scope.਍  ⼀⼀ഀഀ
  // This is useful only when "skip_validation" is not set.਍  ⼀⼀ഀഀ
  Status PopErrorScope();਍ഀഀ
  Status Run(ComputeContextBase& context, const ProgramBase& program);਍ഀഀ
#if defined(ENABLE_PIX_FOR_WEBGPU_EP)਍  猀琀搀㨀㨀甀渀椀焀甀攀开瀀琀爀㰀圀攀戀䜀瀀甀倀䤀堀䘀爀愀洀攀䜀攀渀攀爀愀琀漀爀㸀 䌀爀攀愀琀攀倀䤀堀䘀爀愀洀攀䜀攀渀攀爀愀琀漀爀⠀⤀ 笀ഀഀ
    return std::make_unique<WebGpuPIXFrameGenerator>(instance_,਍                                                     䐀攀瘀椀挀攀⠀⤀⤀㬀ഀഀ
  }਍⌀攀渀搀椀昀  ⼀⼀ 䔀一䄀䈀䰀䔀开倀䤀堀开䘀伀刀开圀䔀䈀䜀倀唀开䔀倀ഀഀ
਍ 瀀爀椀瘀愀琀攀㨀ഀഀ
  enum class TimestampQueryType {਍    一漀渀攀 㴀 　Ⰰഀഀ
    InsidePasses,਍    䄀琀倀愀猀猀攀猀ഀഀ
  };਍ഀഀ
  WebGpuContext(WGPUInstance instance,਍                圀䜀倀唀䐀攀瘀椀挀攀 搀攀瘀椀挀攀Ⰰഀഀ
                webgpu::ValidationMode validation_mode,਍                戀漀漀氀 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀Ⰰഀഀ
                bool preserve_device,਍                甀椀渀琀㘀㐀开琀 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀⤀ഀഀ
      : instance_{instance},਍        搀攀瘀椀挀攀开笀搀攀瘀椀挀攀紀Ⰰഀഀ
        validation_mode_{validation_mode},਍        瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀开笀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀紀Ⰰഀഀ
        query_type_{TimestampQueryType::None},਍        瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀开笀瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀紀Ⰰഀഀ
        max_storage_buffer_binding_size_{max_storage_buffer_binding_size} {਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开 㴀㴀 　 簀簀 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开 㸀㴀 ㄀㌀㐀㈀㄀㜀㜀㈀㠀Ⰰഀഀ
                "max_storage_buffer_binding_size must be 0 or at least 128MB");਍  紀ഀഀ
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuContext);਍ഀഀ
  void Initialize(const WebGpuContextConfig& config);਍ഀഀ
  void LaunchComputePipeline(const wgpu::ComputePassEncoder& compute_pass_encoder,਍                             挀漀渀猀琀 猀琀搀㨀㨀瘀攀挀琀漀爀㰀圀䜀倀唀䈀甀昀昀攀爀㸀☀ 戀椀渀搀开戀甀昀昀攀爀猀Ⰰഀഀ
                             const std::vector<uint32_t>& bind_buffers_segments,਍                             挀漀渀猀琀 倀爀漀最爀愀洀䄀爀琀椀昀愀挀琀☀ 瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀Ⰰഀഀ
                             uint32_t x, uint32_t y, uint32_t z,਍                             挀漀渀猀琀 吀攀渀猀漀爀⨀ 椀渀搀椀爀攀挀琀开搀椀猀瀀愀琀挀栀开琀攀渀猀漀爀 㴀 渀甀氀氀瀀琀爀⤀㬀ഀഀ
਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀挀漀渀猀琀 挀栀愀爀⨀㸀 䜀攀琀䔀渀愀戀氀攀搀䄀搀愀瀀琀攀爀吀漀最最氀攀猀⠀⤀ 挀漀渀猀琀㬀ഀഀ
  std::vector<const char*> GetEnabledDeviceToggles() const;਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀挀漀渀猀琀 挀栀愀爀⨀㸀 䜀攀琀䐀椀猀愀戀氀攀搀䐀攀瘀椀挀攀吀漀最最氀攀猀⠀⤀ 挀漀渀猀琀㬀ഀഀ
  std::vector<wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) const;਍  眀最瀀甀㨀㨀䰀椀洀椀琀猀 䜀攀琀刀攀焀甀椀爀攀搀䰀椀洀椀琀猀⠀挀漀渀猀琀 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀☀ 愀搀愀瀀琀攀爀⤀ 挀漀渀猀琀㬀ഀഀ
  void WriteTimestamp(uint32_t query_index);਍ഀഀ
  struct PendingQueryInfo {਍    倀攀渀搀椀渀最儀甀攀爀礀䤀渀昀漀⠀猀琀搀㨀㨀瘀攀挀琀漀爀㰀倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀㸀☀☀ 欀攀爀渀攀氀猀Ⰰ 眀最瀀甀㨀㨀䈀甀昀昀攀爀 焀甀攀爀礀开戀甀昀昀攀爀⤀ഀഀ
        : kernels{std::move(kernels)}, query_buffer{query_buffer} {}਍ഀഀ
    PendingQueryInfo(PendingQueryInfo&&) = default;਍    倀攀渀搀椀渀最儀甀攀爀礀䤀渀昀漀☀ 漀瀀攀爀愀琀漀爀㴀⠀倀攀渀搀椀渀最儀甀攀爀礀䤀渀昀漀☀☀⤀ 㴀 搀攀昀愀甀氀琀㬀ഀഀ
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(PendingQueryInfo);਍ഀഀ
    std::vector<PendingKernelInfo> kernels;਍    眀最瀀甀㨀㨀䈀甀昀昀攀爀 焀甀攀爀礀开戀甀昀昀攀爀㬀ഀഀ
  };਍ഀഀ
  friend class WebGpuContextFactory;਍ഀഀ
  std::once_flag init_flag_;਍ഀഀ
  wgpu::Instance instance_;਍  眀最瀀甀㨀㨀䐀攀瘀椀挀攀 搀攀瘀椀挀攀开㬀ഀഀ
਍  眀攀戀最瀀甀㨀㨀嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开㬀ഀഀ
  bool validation_mode_explicitly_set_;਍ഀഀ
  wgpu::Queue device_queue_;਍  眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀䤀渀昀漀 愀搀愀瀀琀攀爀开椀渀昀漀开㬀ഀഀ
  wgpu::Limits device_limits_;਍  猀琀搀㨀㨀甀渀漀爀搀攀爀攀搀开猀攀琀㰀眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀㸀 搀攀瘀椀挀攀开昀攀愀琀甀爀攀猀开㬀ഀഀ
#if !defined(__wasm__)਍  眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀倀爀漀瀀攀爀琀椀攀猀匀甀戀最爀漀甀瀀䴀愀琀爀椀砀䌀漀渀昀椀最猀 猀甀戀最爀漀甀瀀开洀愀琀爀椀砀开挀漀渀昀椀最猀开㬀ഀഀ
#endif਍ഀഀ
  wgpu::CommandEncoder current_command_encoder_;਍  眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀 挀甀爀爀攀渀琀开挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀开㬀ഀഀ
਍  猀琀搀㨀㨀甀渀椀焀甀攀开瀀琀爀㰀眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀㸀 戀甀昀昀攀爀开洀最爀开㬀ഀഀ
  std::unique_ptr<webgpu::BufferManager> initializer_buffer_mgr_;਍  猀琀搀㨀㨀甀渀椀焀甀攀开瀀琀爀㰀倀爀漀最爀愀洀䴀愀渀愀最攀爀㸀 瀀爀漀最爀愀洀开洀最爀开㬀ഀഀ
਍  甀椀渀琀㌀㈀开琀 渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 㴀 　㬀ഀഀ
  const uint32_t max_num_pending_dispatches_ = 16;਍ഀഀ
  std::unique_ptr<SplitKConfig> split_k_config_;਍ഀഀ
  // profiling਍  吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀 焀甀攀爀礀开琀礀瀀攀开㬀ഀഀ
  wgpu::QuerySet query_set_;਍  眀最瀀甀㨀㨀䈀甀昀昀攀爀 焀甀攀爀礀开爀攀猀漀氀瘀攀开戀甀昀昀攀爀开㬀ഀഀ
਍  ⼀⼀ 椀渀昀漀 漀昀 欀攀爀渀攀氀猀 瀀攀渀搀椀渀最 猀甀戀洀椀猀猀椀漀渀 昀漀爀 愀 猀椀渀最氀攀 戀愀琀挀栀ഀഀ
  std::vector<PendingKernelInfo> pending_kernels_;਍  ⼀⼀ 椀渀昀漀 漀昀 焀甀攀爀椀攀猀 瀀攀渀搀椀渀最 愀瀀瀀攀渀搀椀渀最 琀漀 瀀爀漀昀椀氀椀渀最 攀瘀攀渀琀猀ഀഀ
  std::vector<PendingQueryInfo> pending_queries_;਍ഀഀ
  uint64_t gpu_timestamp_offset_ = 0;਍  戀漀漀氀 椀猀开瀀爀漀昀椀氀椀渀最开 㴀 昀愀氀猀攀㬀ഀഀ
  // Shared GPU profiling events for run-level profiling.਍  瀀爀漀昀椀氀椀渀最㨀㨀䔀瘀攀渀琀猀 攀瘀攀渀琀猀开㬀ഀഀ
  bool preserve_device_;਍  甀椀渀琀㘀㐀开琀 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开㬀ഀഀ
  GraphCaptureState graph_capture_state_{GraphCaptureState::Default};਍ഀഀ
  // External vector to store captured commands, owned by EP਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀攀戀最瀀甀㨀㨀䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀㸀⨀ 攀砀琀攀爀渀愀氀开挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀开 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
};਍ഀഀ
}  // namespace webgpu਍紀  ⼀⼀ 渀愀洀攀猀瀀愀挀攀 漀渀渀砀爀甀渀琀椀洀攀ഀഀ
