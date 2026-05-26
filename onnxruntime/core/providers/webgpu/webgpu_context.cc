// Copyright (c) Microsoft Corporation. All rights reserved.਍⼀⼀ 䰀椀挀攀渀猀攀搀 甀渀搀攀爀 琀栀攀 䴀䤀吀 䰀椀挀攀渀猀攀⸀ഀഀ
਍⌀椀渀挀氀甀搀攀 㰀洀攀洀漀爀礀㸀ഀഀ
#include <cmath>਍⌀椀渀挀氀甀搀攀 㰀猀琀爀椀渀最㸀ഀഀ
਍⌀椀昀 搀攀昀椀渀攀搀⠀开开䜀一唀䌀开开⤀ഀഀ
#pragma GCC diagnostic push਍⌀瀀爀愀最洀愀 䜀䌀䌀 搀椀愀最渀漀猀琀椀挀 椀最渀漀爀攀搀 ∀ⴀ圀猀琀爀椀挀琀ⴀ愀氀椀愀猀椀渀最∀ഀഀ
#endif਍ഀഀ
#if !defined(__wasm__)਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀䈀唀䤀䰀䐀开䐀䄀圀一开匀䠀䄀刀䔀䐀开䰀䤀䈀刀䄀刀夀⤀ഀഀ
#include "dawn/dawn_proc.h"਍⌀攀渀搀椀昀ഀഀ
#if !defined(USE_EXTERNAL_DAWN)਍⌀椀渀挀氀甀搀攀 ∀搀愀眀渀⼀渀愀琀椀瘀攀⼀䐀愀眀渀一愀琀椀瘀攀⸀栀∀ഀഀ
#endif਍⌀攀渀搀椀昀ഀഀ
#if defined(__GNUC__)਍⌀瀀爀愀最洀愀 䜀䌀䌀 搀椀愀最渀漀猀琀椀挀 瀀漀瀀ഀഀ
#endif਍ഀഀ
#include "core/common/common.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀挀漀洀洀漀渀⼀瀀愀琀栀开猀琀爀椀渀最⸀栀∀ഀഀ
#include "core/platform/env.h"਍ഀഀ
#include "core/providers/webgpu/compute_context.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀眀攀戀最瀀甀开挀漀渀琀攀砀琀⸀栀∀ഀഀ
#include "core/providers/webgpu/webgpu_profiler.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀戀甀昀昀攀爀开洀愀渀愀最攀爀⸀栀∀ഀഀ
#include "core/providers/webgpu/webgpu_execution_provider.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀瀀爀漀最爀愀洀⸀栀∀ഀഀ
#include "core/providers/webgpu/program_cache_key.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀瀀爀漀最爀愀洀开洀愀渀愀最攀爀⸀栀∀ഀഀ
#include "core/providers/webgpu/string_macros.h"਍ഀഀ
namespace onnxruntime {਍渀愀洀攀猀瀀愀挀攀 眀攀戀最瀀甀 笀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䤀渀椀琀椀愀氀椀稀攀⠀挀漀渀猀琀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䌀漀渀昀椀最☀ 挀漀渀昀椀最⤀ 笀ഀഀ
  std::call_once(init_flag_, [this, &config]() {਍    椀昀 ⠀搀攀瘀椀挀攀开 㴀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      // Create wgpu::Adapter਍      眀最瀀甀㨀㨀刀攀焀甀攀猀琀䄀搀愀瀀琀攀爀伀瀀琀椀漀渀猀 爀攀焀开愀搀愀瀀琀攀爀开漀瀀琀椀漀渀猀 㴀 笀紀㬀ഀഀ
      req_adapter_options.backendType = static_cast<wgpu::BackendType>(config.backend_type);਍      爀攀焀开愀搀愀瀀琀攀爀开漀瀀琀椀漀渀猀⸀瀀漀眀攀爀倀爀攀昀攀爀攀渀挀攀 㴀 猀琀愀琀椀挀开挀愀猀琀㰀眀最瀀甀㨀㨀倀漀眀攀爀倀爀攀昀攀爀攀渀挀攀㸀⠀挀漀渀昀椀最⸀瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀⤀㬀ഀഀ
਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
      auto enabled_adapter_toggles = GetEnabledAdapterToggles();਍ഀഀ
      wgpu::DawnTogglesDescriptor adapter_toggles_desc = {};਍      愀搀愀瀀琀攀爀开琀漀最最氀攀猀开搀攀猀挀⸀攀渀愀戀氀攀搀吀漀最最氀攀䌀漀甀渀琀 㴀 攀渀愀戀氀攀搀开愀搀愀瀀琀攀爀开琀漀最最氀攀猀⸀猀椀稀攀⠀⤀㬀ഀഀ
      adapter_toggles_desc.enabledToggles = enabled_adapter_toggles.data();਍ഀഀ
      req_adapter_options.nextInChain = &adapter_toggles_desc;਍⌀攀渀搀椀昀ഀഀ
਍      眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀 愀搀愀瀀琀攀爀㬀ഀഀ
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(instance_.RequestAdapter(਍                                                                     ☀爀攀焀开愀搀愀瀀琀攀爀开漀瀀琀椀漀渀猀Ⰰഀഀ
                                                                     wgpu::CallbackMode::WaitAnyOnly,਍                                                                     嬀崀⠀眀最瀀甀㨀㨀刀攀焀甀攀猀琀䄀搀愀瀀琀攀爀匀琀愀琀甀猀 猀琀愀琀甀猀Ⰰ 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀 愀搀愀瀀琀攀爀Ⰰ 眀最瀀甀㨀㨀匀琀爀椀渀最嘀椀攀眀 洀攀猀猀愀最攀Ⰰ 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀⨀ 瀀琀爀⤀ 笀ഀഀ
                                                                       ORT_ENFORCE(status == wgpu::RequestAdapterStatus::Success, "Failed to get a WebGPU adapter: ", std::string_view{message});਍                                                                       ⨀瀀琀爀 㴀 猀琀搀㨀㨀洀漀瘀攀⠀愀搀愀瀀琀攀爀⤀㬀ഀഀ
                                                                     },਍                                                                     ☀愀搀愀瀀琀攀爀⤀Ⰰഀഀ
                                                                 UINT64_MAX));਍      伀刀吀开䔀一䘀伀刀䌀䔀⠀愀搀愀瀀琀攀爀 ℀㴀 渀甀氀氀瀀琀爀Ⰰ ∀䘀愀椀氀攀搀 琀漀 最攀琀 愀 圀攀戀䜀倀唀 愀搀愀瀀琀攀爀⸀∀⤀㬀ഀഀ
਍      ⼀⼀ 䌀爀攀愀琀攀 眀最瀀甀㨀㨀䐀攀瘀椀挀攀ഀഀ
      wgpu::DeviceDescriptor device_desc = {};਍ഀഀ
#if !defined(__wasm__)਍      眀最瀀甀㨀㨀䐀愀眀渀吀漀最最氀攀猀䐀攀猀挀爀椀瀀琀漀爀 搀攀瘀椀挀攀开琀漀最最氀攀猀开搀攀猀挀 㴀 笀紀㬀ഀഀ
      device_desc.nextInChain = &device_toggles_desc;਍ഀഀ
      auto enabled_device_toggles = GetEnabledDeviceToggles();਍      搀攀瘀椀挀攀开琀漀最最氀攀猀开搀攀猀挀⸀攀渀愀戀氀攀搀吀漀最最氀攀䌀漀甀渀琀 㴀 攀渀愀戀氀攀搀开搀攀瘀椀挀攀开琀漀最最氀攀猀⸀猀椀稀攀⠀⤀㬀ഀഀ
      device_toggles_desc.enabledToggles = enabled_device_toggles.data();਍ഀഀ
      auto disabled_device_toggles = GetDisabledDeviceToggles();਍      搀攀瘀椀挀攀开琀漀最最氀攀猀开搀攀猀挀⸀搀椀猀愀戀氀攀搀吀漀最最氀攀䌀漀甀渀琀 㴀 搀椀猀愀戀氀攀搀开搀攀瘀椀挀攀开琀漀最最氀攀猀⸀猀椀稀攀⠀⤀㬀ഀഀ
      device_toggles_desc.disabledToggles = disabled_device_toggles.data();਍⌀攀渀搀椀昀ഀഀ
਍      猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀㸀 爀攀焀甀椀爀攀搀开昀攀愀琀甀爀攀猀 㴀 䜀攀琀䄀瘀愀椀氀愀戀氀攀刀攀焀甀椀爀攀搀䘀攀愀琀甀爀攀猀⠀愀搀愀瀀琀攀爀⤀㬀ഀഀ
      if (!required_features.empty()) {਍        搀攀瘀椀挀攀开搀攀猀挀⸀爀攀焀甀椀爀攀搀䘀攀愀琀甀爀攀猀 㴀 爀攀焀甀椀爀攀搀开昀攀愀琀甀爀攀猀⸀搀愀琀愀⠀⤀㬀ഀഀ
        device_desc.requiredFeatureCount = required_features.size();਍      紀ഀഀ
      wgpu::Limits required_limits = GetRequiredLimits(adapter);਍      搀攀瘀椀挀攀开搀攀猀挀⸀爀攀焀甀椀爀攀搀䰀椀洀椀琀猀 㴀 ☀爀攀焀甀椀爀攀搀开氀椀洀椀琀猀㬀ഀഀ
਍      ⼀⼀ 吀伀䐀伀㨀 爀攀瘀椀猀攀 琀攀洀瀀漀爀愀爀礀 攀爀爀漀爀 栀愀渀搀氀椀渀最ഀഀ
      device_desc.SetUncapturedErrorCallback([](const wgpu::Device& /*device*/, wgpu::ErrorType type, wgpu::StringView message) {਍        椀昀 ⠀氀漀最最椀渀最㨀㨀䰀漀最最椀渀最䴀愀渀愀最攀爀㨀㨀䠀愀猀䐀攀昀愀甀氀琀䰀漀最最攀爀⠀⤀⤀ 笀ഀഀ
          LOGS_DEFAULT(ERROR) << "WebGPU device error(" << int(type) << "): " << std::string_view{message};਍        紀ഀഀ
      });਍      ⼀⼀ 吀伀䐀伀㨀 爀攀瘀椀猀攀 琀攀洀瀀漀爀愀爀礀 搀攀瘀椀挀攀 氀漀猀琀 栀愀渀搀氀椀渀最ഀഀ
      device_desc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous, [](const wgpu::Device& /*device*/, wgpu::DeviceLostReason reason, wgpu::StringView message) {਍        椀昀 ⠀氀漀最最椀渀最㨀㨀䰀漀最最椀渀最䴀愀渀愀最攀爀㨀㨀䠀愀猀䐀攀昀愀甀氀琀䰀漀最最攀爀⠀⤀⤀ 笀ഀഀ
          LOGS_DEFAULT(INFO) << "WebGPU device lost (" << int(reason) << "): " << std::string_view{message};਍        紀ഀഀ
      });਍ഀഀ
      ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter.RequestDevice(਍                                                                     ☀搀攀瘀椀挀攀开搀攀猀挀Ⰰഀഀ
                                                                     wgpu::CallbackMode::WaitAnyOnly,਍                                                                     嬀崀⠀眀最瀀甀㨀㨀刀攀焀甀攀猀琀䐀攀瘀椀挀攀匀琀愀琀甀猀 猀琀愀琀甀猀Ⰰ 眀最瀀甀㨀㨀䐀攀瘀椀挀攀 搀攀瘀椀挀攀Ⰰ 眀最瀀甀㨀㨀匀琀爀椀渀最嘀椀攀眀 洀攀猀猀愀最攀Ⰰ 眀最瀀甀㨀㨀䐀攀瘀椀挀攀⨀ 瀀琀爀⤀ 笀ഀഀ
                                                                       ORT_ENFORCE(status == wgpu::RequestDeviceStatus::Success, "Failed to get a WebGPU device: ", std::string_view{message});਍                                                                       ⨀瀀琀爀 㴀 猀琀搀㨀㨀洀漀瘀攀⠀搀攀瘀椀挀攀⤀㬀ഀഀ
                                                                     },਍                                                                     ☀搀攀瘀椀挀攀开⤀Ⰰഀഀ
                                                                 UINT64_MAX));਍      伀刀吀开䔀一䘀伀刀䌀䔀⠀搀攀瘀椀挀攀开 ℀㴀 渀甀氀氀瀀琀爀Ⰰ ∀䘀愀椀氀攀搀 琀漀 最攀琀 愀 圀攀戀䜀倀唀 搀攀瘀椀挀攀⸀∀⤀㬀ഀഀ
    }਍ഀഀ
    LOGS_DEFAULT(VERBOSE) << "WebGPU EP Context is created for: Instance=" << instance_.Get() << ", Device=" << device_.Get() << ".";਍ഀഀ
    // cache device queue਍    搀攀瘀椀挀攀开焀甀攀甀攀开 㴀 搀攀瘀椀挀攀开⸀䜀攀琀儀甀攀甀攀⠀⤀㬀ഀഀ
    // cache device limits਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀䐀攀瘀椀挀攀⠀⤀⸀䜀攀琀䰀椀洀椀琀猀⠀☀搀攀瘀椀挀攀开氀椀洀椀琀猀开⤀⤀㬀ഀഀ
    // Align maxStorageBufferBindingSize down to minStorageBufferOffsetAlignment so that਍    ⼀⼀ 戀甀昀昀攀爀 猀攀最洀攀渀琀 漀昀昀猀攀琀猀 愀爀攀 愀氀眀愀礀猀 瀀爀漀瀀攀爀氀礀 愀氀椀最渀攀搀 昀漀爀 圀攀戀䜀倀唀 戀椀渀搀 最爀漀甀瀀 挀爀攀愀琀椀漀渀⸀ഀഀ
    if (device_limits_.minStorageBufferOffsetAlignment > 0) {਍      搀攀瘀椀挀攀开氀椀洀椀琀猀开⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀 ⴀ㴀ഀഀ
          (device_limits_.maxStorageBufferBindingSize % device_limits_.minStorageBufferOffsetAlignment);਍    紀ഀഀ
    // cache device features਍    眀最瀀甀㨀㨀匀甀瀀瀀漀爀琀攀搀䘀攀愀琀甀爀攀猀 猀甀瀀瀀漀爀琀攀搀开昀攀愀琀甀爀攀猀㬀ഀഀ
    Device().GetFeatures(&supported_features);਍    昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 猀甀瀀瀀漀爀琀攀搀开昀攀愀琀甀爀攀猀⸀昀攀愀琀甀爀攀䌀漀甀渀琀㬀 椀⬀⬀⤀ 笀ഀഀ
      device_features_.insert(supported_features.features[i]);਍    紀ഀഀ
    // cache adapter info਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
    if (DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {਍      愀搀愀瀀琀攀爀开椀渀昀漀开⸀渀攀砀琀䤀渀䌀栀愀椀渀 㴀 ☀猀甀戀最爀漀甀瀀开洀愀琀爀椀砀开挀漀渀昀椀最猀开㬀ഀഀ
    }਍⌀攀渀搀椀昀ഀഀ
    ORT_ENFORCE(Device().GetAdapterInfo(&adapter_info_));਍ഀഀ
    // create buffer manager਍    戀甀昀昀攀爀开洀最爀开 㴀 䈀甀昀昀攀爀䴀愀渀愀最攀爀䘀愀挀琀漀爀礀㨀㨀䌀爀攀愀琀攀⠀⨀琀栀椀猀Ⰰഀഀ
                                               config.buffer_cache_config.storage.mode,਍                                               挀漀渀昀椀最⸀戀甀昀昀攀爀开挀愀挀栀攀开挀漀渀昀椀最⸀甀渀椀昀漀爀洀⸀洀漀搀攀Ⰰഀഀ
                                               config.buffer_cache_config.query_resolve.mode);਍ഀഀ
    // create initializer buffer manager.਍    椀渀椀琀椀愀氀椀稀攀爀开戀甀昀昀攀爀开洀最爀开 㴀 䈀甀昀昀攀爀䴀愀渀愀最攀爀䘀愀挀琀漀爀礀㨀㨀䌀爀攀愀琀攀⠀⨀琀栀椀猀Ⰰഀഀ
                                                           BufferCacheMode::LazyRelease,਍                                                           䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀䰀愀稀礀刀攀氀攀愀猀攀Ⰰഀഀ
                                                           BufferCacheMode::Disabled);਍ഀഀ
    // create program manager਍    瀀爀漀最爀愀洀开洀最爀开 㴀 猀琀搀㨀㨀洀愀欀攀开甀渀椀焀甀攀㰀倀爀漀最爀愀洀䴀愀渀愀最攀爀㸀⠀⨀琀栀椀猀⤀㬀ഀഀ
਍    ⼀⼀ 挀爀攀愀琀攀 猀瀀氀椀琀ⴀ欀 挀漀渀昀椀最ഀഀ
    split_k_config_ = std::make_unique<SplitKConfig>(adapter_info_);਍ഀഀ
    // set query type਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
    if (DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses)) {਍      焀甀攀爀礀开琀礀瀀攀开 㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀䤀渀猀椀搀攀倀愀猀猀攀猀㬀ഀഀ
    } else਍⌀攀渀搀椀昀ഀഀ
        if (DeviceHasFeature(wgpu::FeatureName::TimestampQuery)) {਍      焀甀攀爀礀开琀礀瀀攀开 㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀䄀琀倀愀猀猀攀猀㬀ഀഀ
    } else {਍      焀甀攀爀礀开琀礀瀀攀开 㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀一漀渀攀㬀ഀഀ
    }਍  紀⤀㬀ഀഀ
}਍ഀഀ
Status WebGpuContext::Wait(wgpu::Future f) {਍  愀甀琀漀 猀琀愀琀甀猀 㴀 椀渀猀琀愀渀挀攀开⸀圀愀椀琀䄀渀礀⠀昀Ⰰ 唀䤀一吀㘀㐀开䴀䄀堀⤀㬀ഀഀ
  if (status == wgpu::WaitStatus::Success) {਍    爀攀琀甀爀渀 匀琀愀琀甀猀㨀㨀伀䬀⠀⤀㬀ഀഀ
  }਍  爀攀琀甀爀渀 伀刀吀开䴀䄀䬀䔀开匀吀䄀吀唀匀⠀伀一一堀刀唀一吀䤀䴀䔀Ⰰ 䘀䄀䤀䰀Ⰰ ∀䘀愀椀氀攀搀 琀漀 眀愀椀琀 昀漀爀 琀栀攀 漀瀀攀爀愀琀椀漀渀㨀∀Ⰰ 甀椀渀琀㌀㈀开琀⠀猀琀愀琀甀猀⤀⤀㬀ഀഀ
}਍ഀഀ
Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {਍  挀漀渀猀琀 愀甀琀漀☀ 椀渀瀀甀琀猀 㴀 瀀爀漀最爀愀洀⸀䤀渀瀀甀琀猀⠀⤀㬀ഀഀ
  const auto& outputs = program.Outputs();਍ഀഀ
  if (outputs.empty()) {਍    爀攀琀甀爀渀 匀琀愀琀甀猀㨀㨀伀䬀⠀⤀㬀ഀഀ
  }਍ഀഀ
  // validate inputs and outputs are on WebGPU buffers਍  椀昀 ⠀嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀⠀⤀ 㸀㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䈀愀猀椀挀⤀ 笀ഀഀ
    ORT_ENFORCE(std::all_of(inputs.begin(), inputs.end(), [](const ProgramInput& input) {਍                  挀漀渀猀琀 愀甀琀漀⨀ 琀攀渀猀漀爀 㴀 椀渀瀀甀琀⸀琀攀渀猀漀爀㬀ഀഀ
                  return tensor != nullptr &&਍                         琀攀渀猀漀爀ⴀ㸀䰀漀挀愀琀椀漀渀⠀⤀⸀洀攀洀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀吀礀瀀攀㨀㨀伀爀琀䴀攀洀吀礀瀀攀䐀攀昀愀甀氀琀 ☀☀ഀഀ
                         tensor->Location().device.Type() == OrtDevice::GPU &&਍                         ℀猀琀爀挀洀瀀⠀琀攀渀猀漀爀ⴀ㸀䰀漀挀愀琀椀漀渀⠀⤀⸀渀愀洀攀⸀挀开猀琀爀⠀⤀Ⰰ 圀䔀䈀䜀倀唀开䈀唀䘀䘀䔀刀⤀㬀ഀഀ
                }),਍                ∀䄀氀氀 椀渀瀀甀琀猀 洀甀猀琀 戀攀 琀攀渀猀漀爀猀 漀渀 圀攀戀䜀倀唀 戀甀昀昀攀爀猀⸀∀⤀㬀ഀഀ
਍    椀昀 ⠀瀀爀漀最爀愀洀⸀䤀渀搀椀爀攀挀琀䐀椀猀瀀愀琀挀栀吀攀渀猀漀爀⠀⤀ ℀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      ORT_ENFORCE(!inputs.empty() && inputs.back().tensor == program.IndirectDispatchTensor(),਍                  ∀吀栀攀 椀渀搀椀爀攀挀琀 搀椀猀瀀愀琀挀栀 琀攀渀猀漀爀 洀甀猀琀 戀攀 琀栀攀 氀愀猀琀 椀渀瀀甀琀⸀ ∀ഀഀ
                  "Ensure no call to program.AddInput() occurs after program.SetIndirectDispatchTensor().");਍    紀ഀഀ
਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀猀琀搀㨀㨀愀氀氀开漀昀⠀漀甀琀瀀甀琀猀⸀戀攀最椀渀⠀⤀Ⰰ 漀甀琀瀀甀琀猀⸀攀渀搀⠀⤀Ⰰ 嬀崀⠀挀漀渀猀琀 倀爀漀最爀愀洀伀甀琀瀀甀琀☀ 漀甀琀瀀甀琀⤀ 笀ഀഀ
                  const auto* tensor = output.tensor;਍                  爀攀琀甀爀渀 琀攀渀猀漀爀 ℀㴀 渀甀氀氀瀀琀爀 ☀☀ഀഀ
                         tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&਍                         琀攀渀猀漀爀ⴀ㸀䰀漀挀愀琀椀漀渀⠀⤀⸀搀攀瘀椀挀攀⸀吀礀瀀攀⠀⤀ 㴀㴀 伀爀琀䐀攀瘀椀挀攀㨀㨀䜀倀唀 ☀☀ഀഀ
                         !strcmp(tensor->Location().name.c_str(), WEBGPU_BUFFER);਍                紀⤀Ⰰഀഀ
                "All outputs must be tensors on WebGPU buffers.");਍  紀ഀഀ
਍  挀漀渀猀琀 倀爀漀最爀愀洀䴀攀琀愀搀愀琀愀☀ 洀攀琀愀搀愀琀愀 㴀 瀀爀漀最爀愀洀⸀䴀攀琀愀搀愀琀愀⠀⤀㬀ഀഀ
਍  ⼀⼀ 瘀愀氀椀搀愀琀攀 瀀爀漀最爀愀洀 洀攀琀愀搀愀琀愀ഀഀ
  if (ValidationMode() >= ValidationMode::Basic) {਍    挀漀渀猀琀 愀甀琀漀☀ 嬀挀漀渀猀琀愀渀琀猀Ⰰ 漀瘀攀爀爀椀搀愀戀氀攀开挀漀渀猀琀愀渀琀猀Ⰰ 甀渀椀昀漀爀洀开瘀愀爀椀愀戀氀攀猀崀 㴀 洀攀琀愀搀愀琀愀㬀ഀഀ
਍    ⼀⼀ 挀栀攀挀欀 漀瘀攀爀爀椀搀愀戀氀攀 挀漀渀猀琀愀渀琀猀ഀഀ
    ORT_RETURN_IF(program.OverridableConstants().size() != overridable_constants.size(),਍                  ∀匀椀稀攀 漀昀 漀瘀攀爀爀椀搀愀戀氀攀 挀漀渀猀琀愀渀琀猀 洀椀猀洀愀琀挀栀 椀渀 瀀爀漀最爀愀洀 尀∀∀Ⰰ 瀀爀漀最爀愀洀⸀一愀洀攀⠀⤀Ⰰഀഀ
                  "\", Expected: ", overridable_constants.size(),਍                  ∀Ⰰ 䄀挀琀甀愀氀㨀 ∀Ⰰ 瀀爀漀最爀愀洀⸀伀瘀攀爀爀椀搀愀戀氀攀䌀漀渀猀琀愀渀琀猀⠀⤀⸀猀椀稀攀⠀⤀⤀㬀ഀഀ
਍    椀昀 ⠀嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀⠀⤀ 㸀㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䘀甀氀氀⤀ 笀ഀഀ
      size_t num_overridable_constants = program.OverridableConstants().size();਍      昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 渀甀洀开漀瘀攀爀爀椀搀愀戀氀攀开挀漀渀猀琀愀渀琀猀㬀 ⬀⬀椀⤀ 笀ഀഀ
        const auto& override_value = program.OverridableConstants()[i];਍        挀漀渀猀琀 愀甀琀漀☀ 搀攀昀椀渀椀琀椀漀渀 㴀 漀瘀攀爀爀椀搀愀戀氀攀开挀漀渀猀琀愀渀琀猀嬀椀崀㬀ഀഀ
        ORT_RETURN_IF(override_value.has_value && override_value.type != definition.type,਍                      ∀伀瘀攀爀爀椀搀愀戀氀攀 漀瘀攀爀爀椀搀攀开瘀愀氀甀攀嬀∀Ⰰ 椀Ⰰ ∀崀 ⠀∀Ⰰ 搀攀昀椀渀椀琀椀漀渀⸀渀愀洀攀Ⰰ ∀⤀ 搀愀琀愀 琀礀瀀攀 洀椀猀洀愀琀挀栀 椀渀 瀀爀漀最爀愀洀 尀∀∀Ⰰ 瀀爀漀最爀愀洀⸀一愀洀攀⠀⤀Ⰰഀഀ
                      "\", Expected: ", definition.type,਍                      ∀Ⰰ 䄀挀琀甀愀氀㨀 ∀Ⰰ 漀瘀攀爀爀椀搀攀开瘀愀氀甀攀⸀琀礀瀀攀⤀㬀ഀഀ
        ORT_RETURN_IF(!override_value.has_value && !definition.has_default_value,਍                      ∀伀瘀攀爀爀椀搀愀戀氀攀 漀瘀攀爀爀椀搀攀开瘀愀氀甀攀嬀∀Ⰰ 椀Ⰰ ∀崀 ⠀∀Ⰰ 搀攀昀椀渀椀琀椀漀渀⸀渀愀洀攀Ⰰ ∀⤀ 渀漀 漀瘀攀爀爀椀搀攀开瘀愀氀甀攀 猀瀀攀挀椀昀椀攀搀 椀渀 瀀爀漀最爀愀洀 尀∀∀Ⰰ 瀀爀漀最爀愀洀⸀一愀洀攀⠀⤀Ⰰഀഀ
                      "\"");਍      紀ഀഀ
    }਍ഀഀ
    // check uniform variables਍    伀刀吀开刀䔀吀唀刀一开䤀䘀⠀瀀爀漀最爀愀洀⸀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀猀⠀⤀⸀猀椀稀攀⠀⤀ ℀㴀 甀渀椀昀漀爀洀开瘀愀爀椀愀戀氀攀猀⸀猀椀稀攀⠀⤀Ⰰഀഀ
                  "Size of uniform_value variables mismatch in program \"", program.Name(),਍                  ∀尀∀Ⰰ 䔀砀瀀攀挀琀攀搀㨀 ∀Ⰰ 甀渀椀昀漀爀洀开瘀愀爀椀愀戀氀攀猀⸀猀椀稀攀⠀⤀Ⰰഀഀ
                  ", Actual: ", program.UniformVariables().size());਍ഀഀ
    if (ValidationMode() >= ValidationMode::Full) {਍      猀椀稀攀开琀 渀甀洀开甀渀椀昀漀爀洀开瘀愀爀椀愀戀氀攀猀 㴀 瀀爀漀最爀愀洀⸀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀猀⠀⤀⸀猀椀稀攀⠀⤀㬀ഀഀ
      for (size_t i = 0; i < num_uniform_variables; ++i) {਍        挀漀渀猀琀 愀甀琀漀☀ 甀渀椀昀漀爀洀开瘀愀氀甀攀 㴀 瀀爀漀最爀愀洀⸀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀猀⠀⤀嬀椀崀㬀ഀഀ
        const auto& definition = uniform_variables[i];਍        伀刀吀开刀䔀吀唀刀一开䤀䘀⠀甀渀椀昀漀爀洀开瘀愀氀甀攀⸀氀攀渀最琀栀 㸀 　 ☀☀ 甀渀椀昀漀爀洀开瘀愀氀甀攀⸀搀愀琀愀开琀礀瀀攀 ℀㴀 搀攀昀椀渀椀琀椀漀渀⸀搀愀琀愀开琀礀瀀攀Ⰰഀഀ
                      "Uniform variable[", i, "] (", definition.name, ") data type mismatch in program \"", program.Name(),਍                      ∀尀∀Ⰰ 䔀砀瀀攀挀琀攀搀㨀 ∀Ⰰ 搀攀昀椀渀椀琀椀漀渀⸀搀愀琀愀开琀礀瀀攀Ⰰഀഀ
                      ", Actual: ", uniform_value.data_type);਍      紀ഀഀ
    }਍  紀ഀഀ
਍  ⼀⼀ ∀匀攀最洀攀渀琀猀∀ 椀猀 愀 昀攀愀琀甀爀攀 琀栀愀琀 愀氀氀漀眀猀 戀椀最 戀甀昀昀攀爀 琀漀 戀攀 甀猀攀搀 椀渀 猀栀愀搀攀爀⸀ഀഀ
  //਍  ⼀⼀ 䘀漀爀 攀砀愀洀瀀氀攀Ⰰ 椀昀 怀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀怀 椀猀 ㄀㈀㠀䴀䈀Ⰰ 愀 ㈀　　䴀䈀 猀椀稀攀搀 椀渀瀀甀琀 戀甀昀昀攀爀 挀愀渀 戀攀 猀瀀氀椀琀 椀渀琀漀 琀眀漀 猀攀最洀攀渀琀猀ഀഀ
  // (128MB + 72MB) to be bound to the shader. In this case, the input segment count is 2. There will be 2 input਍  ⼀⼀ 戀椀渀搀椀渀最猀 椀渀 琀栀攀 猀栀愀搀攀爀 昀漀爀 琀栀椀猀 椀渀瀀甀琀 戀甀昀昀攀爀⸀ഀഀ
  //਍  ⼀⼀ 匀攀攀 栀琀琀瀀猀㨀⼀⼀最椀琀栀甀戀⸀挀漀洀⼀洀椀挀爀漀猀漀昀琀⼀漀渀渀砀爀甀渀琀椀洀攀⼀瀀甀氀氀⼀㈀㔀㤀㘀㈀ 昀漀爀 洀漀爀攀 椀渀昀漀爀洀愀琀椀漀渀⸀ഀഀ
਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀甀椀渀琀㌀㈀开琀㸀 椀渀瀀甀琀猀开猀攀最洀攀渀琀猀㬀ഀഀ
  std::vector<uint32_t> outputs_segments;਍  伀刀吀开刀䔀吀唀刀一开䤀䘀开䔀刀刀伀刀⠀瀀爀漀最爀愀洀开洀最爀开ⴀ㸀䌀愀氀挀甀氀愀琀攀匀攀最洀攀渀琀猀䘀漀爀䤀渀瀀甀琀猀䄀渀搀伀甀琀瀀甀琀猀⠀瀀爀漀最爀愀洀Ⰰ 椀渀瀀甀琀猀开猀攀最洀攀渀琀猀Ⰰ 漀甀琀瀀甀琀猀开猀攀最洀攀渀琀猀⤀⤀㬀ഀഀ
਍  甀椀渀琀㌀㈀开琀 砀 㴀 瀀爀漀最爀愀洀⸀䐀椀猀瀀愀琀挀栀䜀爀漀甀瀀匀椀稀攀堀⠀⤀㬀ഀഀ
  uint32_t y = program.DispatchGroupSizeY();਍  甀椀渀琀㌀㈀开琀 稀 㴀 瀀爀漀最爀愀洀⸀䐀椀猀瀀愀琀挀栀䜀爀漀甀瀀匀椀稀攀娀⠀⤀㬀ഀഀ
਍  ⼀⼀ 匀欀椀瀀 渀漀爀洀愀氀椀稀愀琀椀漀渀 昀漀爀 椀渀搀椀爀攀挀琀 搀椀猀瀀愀琀挀栀 猀椀渀挀攀 搀椀洀攀渀猀椀漀渀猀 愀爀攀 搀攀琀攀爀洀椀渀攀搀 戀礀 琀栀攀 椀渀搀椀爀攀挀琀 戀甀昀昀攀爀ഀഀ
  if (program.IndirectDispatchTensor() == nullptr) {਍    伀刀吀开刀䔀吀唀刀一开䤀䘀开䔀刀刀伀刀⠀瀀爀漀最爀愀洀开洀最爀开ⴀ㸀一漀爀洀愀氀椀稀攀䐀椀猀瀀愀琀挀栀䜀爀漀甀瀀匀椀稀攀⠀砀Ⰰ 礀Ⰰ 稀⤀⤀㬀ഀഀ
  } else {਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀砀 㴀㴀 　 ☀☀ 礀 㴀㴀 　 ☀☀ 稀 㴀㴀 　Ⰰഀഀ
                "Only one of SetIndirectDispatchTensor and SetDispatchGroupSize should be called for program", program.Name());਍  紀ഀഀ
਍  戀漀漀氀 椀猀开㄀搀开搀椀猀瀀愀琀挀栀 㴀 ⠀礀 㴀㴀 ㄀ ☀☀ 稀 㴀㴀 ㄀⤀㬀ഀഀ
਍  愀甀琀漀 欀攀礀 㴀 䌀愀氀挀甀氀愀琀攀倀爀漀最爀愀洀䌀愀挀栀攀䬀攀礀⠀瀀爀漀最爀愀洀Ⰰ 椀渀瀀甀琀猀开猀攀最洀攀渀琀猀Ⰰ 漀甀琀瀀甀琀猀开猀攀最洀攀渀琀猀Ⰰ 椀猀开㄀搀开搀椀猀瀀愀琀挀栀⤀㬀ഀഀ
਍  䰀伀䜀匀⠀挀漀渀琀攀砀琀⸀䰀漀最最攀爀⠀⤀Ⰰ 䤀一䘀伀⤀ 㰀㰀 ∀匀琀愀爀琀椀渀最 瀀爀漀最爀愀洀 尀∀∀ 㰀㰀 欀攀礀 㰀㰀 ∀尀∀ ⠀∀ 㰀㰀 砀 㰀㰀 ∀Ⰰ ∀ 㰀㰀 礀 㰀㰀 ∀Ⰰ ∀ 㰀㰀 稀 㰀㰀 ∀⤀∀㬀ഀഀ
਍  挀漀渀猀琀 愀甀琀漀⨀ 瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀 㴀 瀀爀漀最爀愀洀开洀最爀开ⴀ㸀䜀攀琀⠀欀攀礀⤀㬀ഀഀ
  if (program_artifact == nullptr) {਍    眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀椀瀀攀氀椀渀攀 挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀㬀ഀഀ
    std::vector<int> shape_uniform_ranks;਍    愀甀琀漀 猀琀愀琀甀猀 㴀 瀀爀漀最爀愀洀开洀最爀开ⴀ㸀䈀甀椀氀搀⠀瀀爀漀最爀愀洀Ⰰഀഀ
                                      metadata,਍                                      椀渀瀀甀琀猀开猀攀最洀攀渀琀猀Ⰰഀഀ
                                      outputs_segments,਍⌀椀昀渀搀攀昀 一䐀䔀䈀唀䜀  ⼀⼀ 椀昀 搀攀戀甀最 戀甀椀氀搀ഀഀ
                                      key,਍⌀攀渀搀椀昀ഀഀ
                                      x,਍                                      礀Ⰰഀഀ
                                      z,਍                                      挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀Ⰰഀഀ
                                      shape_uniform_ranks);਍    伀刀吀开刀䔀吀唀刀一开䤀䘀开䔀刀刀伀刀⠀猀琀愀琀甀猀⤀㬀ഀഀ
    program_artifact = program_mgr_->Set(key, ProgramArtifact{program,਍                                                              猀琀搀㨀㨀洀漀瘀攀⠀挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀⤀Ⰰഀഀ
                                                              std::move(shape_uniform_ranks)});਍⌀椀昀渀搀攀昀 一䐀䔀䈀唀䜀  ⼀⼀ 椀昀 搀攀戀甀最 戀甀椀氀搀ഀഀ
    ORT_ENFORCE(program_artifact != nullptr, "Program artifact should not be nullptr.");਍⌀攀渀搀椀昀ഀഀ
  }਍ഀഀ
  // prepare shape uniforms for shader variables (if any) and user defined uniforms਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀倀爀漀最爀愀洀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀嘀愀氀甀攀㸀 猀栀愀瀀攀开甀渀椀昀漀爀洀猀㬀ഀഀ
  shape_uniforms.reserve(program_artifact->shape_uniform_ranks.size() * 2);਍  椀昀 ⠀嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀⠀⤀ 㸀㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䈀愀猀椀挀⤀ 笀ഀഀ
    ORT_RETURN_IF_NOT(program_artifact->shape_uniform_ranks.size() == inputs.size() + outputs.size() + program.Indices().size(),਍                      ∀䤀渀瘀愀氀椀搀 瀀爀漀最爀愀洀 愀爀琀椀昀愀挀琀㨀 瘀愀爀椀愀戀氀攀 猀椀稀攀 ⠀∀Ⰰ 瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀ⴀ㸀猀栀愀瀀攀开甀渀椀昀漀爀洀开爀愀渀欀猀⸀猀椀稀攀⠀⤀Ⰰഀഀ
                      ") does not match current program (input: ", inputs.size(),਍                      ∀Ⰰ 漀甀琀瀀甀琀㨀 ∀Ⰰ 漀甀琀瀀甀琀猀⸀猀椀稀攀⠀⤀Ⰰഀഀ
                      ", indices: ", program.Indices().size(), ")");਍  紀ഀഀ
਍  愀甀琀漀 愀瀀瀀攀渀搀开猀栀愀瀀攀开甀渀椀昀漀爀洀猀 㴀 嬀☀猀栀愀瀀攀开甀渀椀昀漀爀洀猀Ⰰ 瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀崀⠀猀椀稀攀开琀 椀Ⰰ 挀漀渀猀琀 吀攀渀猀漀爀匀栀愀瀀攀☀ 猀栀愀瀀攀⤀ 笀ഀഀ
    if (program_artifact->shape_uniform_ranks[i] > 0) {਍      猀椀稀攀开琀 攀砀瀀攀挀琀攀搀开爀愀渀欀 㴀 猀琀愀琀椀挀开挀愀猀琀㰀猀椀稀攀开琀㸀⠀瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀ⴀ㸀猀栀愀瀀攀开甀渀椀昀漀爀洀开爀愀渀欀猀嬀椀崀⤀㬀ഀഀ
      ORT_RETURN_IF(expected_rank != shape.NumDimensions(),਍                    ∀䤀渀瘀愀氀椀搀 瀀爀漀最爀愀洀 愀爀琀椀昀愀挀琀㨀 瘀愀爀椀愀戀氀攀嬀∀Ⰰ 椀Ⰰ ∀崀 爀愀渀欀 洀椀猀洀愀琀挀栀⸀ 䔀砀瀀攀挀琀攀搀㨀 ∀Ⰰ 攀砀瀀攀挀琀攀搀开爀愀渀欀Ⰰഀഀ
                    ", Actual: ", shape.NumDimensions());਍ഀഀ
      std::vector<uint32_t> dims(expected_rank);਍      猀琀搀㨀㨀瘀攀挀琀漀爀㰀甀椀渀琀㌀㈀开琀㸀 猀琀爀椀搀攀⠀攀砀瀀攀挀琀攀搀开爀愀渀欀 ⴀ ㄀⤀㬀ഀഀ
      for (size_t j = 0; j < expected_rank; ++j) {਍        搀椀洀猀嬀樀崀 㴀 漀渀渀砀爀甀渀琀椀洀攀㨀㨀渀愀爀爀漀眀㰀甀椀渀琀㌀㈀开琀㸀⠀猀栀愀瀀攀嬀樀崀⤀㬀ഀഀ
        if (j < expected_rank - 1) {਍          猀琀爀椀搀攀嬀樀崀 㴀 漀渀渀砀爀甀渀琀椀洀攀㨀㨀渀愀爀爀漀眀㰀甀椀渀琀㌀㈀开琀㸀⠀猀栀愀瀀攀⸀匀椀稀攀䘀爀漀洀䐀椀洀攀渀猀椀漀渀⠀樀 ⬀ ㄀⤀⤀㬀ഀഀ
        }਍      紀ഀഀ
਍      猀栀愀瀀攀开甀渀椀昀漀爀洀猀⸀攀洀瀀氀愀挀攀开戀愀挀欀⠀最猀氀㨀㨀洀愀欀攀开猀瀀愀渀⠀搀椀洀猀⤀⤀㬀ഀഀ
      if (expected_rank > 1) {਍        猀栀愀瀀攀开甀渀椀昀漀爀洀猀⸀攀洀瀀氀愀挀攀开戀愀挀欀⠀最猀氀㨀㨀洀愀欀攀开猀瀀愀渀⠀猀琀爀椀搀攀⤀⤀㬀ഀഀ
      }਍    紀ഀഀ
    return Status::OK();਍  紀㬀ഀഀ
਍  昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 椀渀瀀甀琀猀⸀猀椀稀攀⠀⤀㬀 椀⬀⬀⤀ 笀ഀഀ
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i,਍                                              椀渀瀀甀琀猀嬀椀崀⸀甀猀攀开漀瘀攀爀爀椀搀攀开猀栀愀瀀攀 㼀 椀渀瀀甀琀猀嬀椀崀⸀漀瘀攀爀爀椀搀攀开猀栀愀瀀攀 㨀 椀渀瀀甀琀猀嬀椀崀⸀琀攀渀猀漀爀ⴀ㸀匀栀愀瀀攀⠀⤀⤀⤀㬀ഀഀ
  }਍  昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 漀甀琀瀀甀琀猀⸀猀椀稀攀⠀⤀㬀 椀⬀⬀⤀ 笀ഀഀ
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size(),਍                                              漀甀琀瀀甀琀猀嬀椀崀⸀甀猀攀开漀瘀攀爀爀椀搀攀开猀栀愀瀀攀 㼀 漀甀琀瀀甀琀猀嬀椀崀⸀漀瘀攀爀爀椀搀攀开猀栀愀瀀攀 㨀 漀甀琀瀀甀琀猀嬀椀崀⸀琀攀渀猀漀爀ⴀ㸀匀栀愀瀀攀⠀⤀⤀⤀㬀ഀഀ
  }਍  昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 瀀爀漀最爀愀洀⸀䤀渀搀椀挀攀猀⠀⤀⸀猀椀稀攀⠀⤀㬀 椀⬀⬀⤀ 笀ഀഀ
    ORT_RETURN_IF_ERROR(append_shape_uniforms(i + inputs.size() + outputs.size(), program.Indices()[i]));਍  紀ഀഀ
਍  挀漀渀猀琀 猀椀稀攀开琀 甀渀椀昀漀爀洀开挀漀甀渀琀 㴀 猀栀愀瀀攀开甀渀椀昀漀爀洀猀⸀猀椀稀攀⠀⤀ ⬀ 瀀爀漀最爀愀洀⸀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀猀⠀⤀⸀猀椀稀攀⠀⤀㬀ഀഀ
  size_t current_offset = 0;਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀猀琀搀㨀㨀琀甀瀀氀攀㰀挀漀渀猀琀 倀爀漀最爀愀洀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀嘀愀氀甀攀☀Ⰰ 猀椀稀攀开琀㸀㸀 甀渀椀昀漀爀洀开愀渀搀开漀昀昀猀攀琀猀㬀ഀഀ
  uniform_and_offsets.reserve(uniform_count);਍  昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 甀渀椀昀漀爀洀开挀漀甀渀琀㬀 椀⬀⬀⤀ 笀ഀഀ
    const auto& uniform = i < shape_uniforms.size() ? shape_uniforms[i]਍                                                    㨀 瀀爀漀最爀愀洀⸀唀渀椀昀漀爀洀嘀愀爀椀愀戀氀攀猀⠀⤀嬀椀 ⴀ 猀栀愀瀀攀开甀渀椀昀漀爀洀猀⸀猀椀稀攀⠀⤀崀㬀ഀഀ
    size_t length = uniform.length;਍    椀昀 ⠀氀攀渀最琀栀 㴀㴀 　⤀ 笀  ⼀⼀ 猀欀椀瀀 稀攀爀漀ⴀ氀攀渀最琀栀 甀渀椀昀漀爀洀ഀഀ
      continue;਍    紀ഀഀ
਍    ⼀⼀ 䌀愀氀挀甀氀愀琀攀 琀栀攀 猀椀稀攀 愀渀搀 愀氀椀最渀洀攀渀琀 漀昀 琀栀攀 甀渀椀昀漀爀洀 瘀愀爀椀愀戀氀攀⸀ഀഀ
    //਍    ⼀⼀ 栀琀琀瀀猀㨀⼀⼀眀眀眀⸀眀㌀⸀漀爀最⼀吀刀⼀圀䜀匀䰀⼀⌀愀氀椀最渀漀昀ഀഀ
    //਍    ⼀⼀ 䘀漀爀 昀㄀㘀㨀ഀഀ
    // - length > 8      : array<vec4<u32>, N>   (align 16) (size 16 * N, N = ceil(length / 8))਍    ⼀⼀ ⴀ 氀攀渀最琀栀 㴀㴀 㜀 漀爀 㠀㨀 瘀攀挀㐀㰀甀㌀㈀㸀             ⠀愀氀椀最渀 ㄀㘀⤀ ⠀猀椀稀攀 ㄀㘀⤀ഀഀ
    // - length == 5 or 6: vec3<u32>             (align 16) (size 12)਍    ⼀⼀ ⴀ 氀攀渀最琀栀 㴀㴀 ㌀ 漀爀 㐀㨀 瘀攀挀㈀㰀甀㌀㈀㸀             ⠀愀氀椀最渀 㠀⤀  ⠀猀椀稀攀 㠀⤀ഀഀ
    // - length == 1 or 2: u32                   (align 4)  (size 4)਍    ⼀⼀ഀഀ
    // For other types (i32, u32, f32):਍    ⼀⼀ ⴀ 氀攀渀最琀栀 㸀 㐀      㨀 愀爀爀愀礀㰀瘀攀挀㐀㰀吀㸀Ⰰ 一㸀     ⠀愀氀椀最渀 ㄀㘀⤀ ⠀猀椀稀攀 ㄀㘀 ⨀ 一Ⰰ 一 㴀 挀攀椀氀⠀氀攀渀最琀栀 ⼀ 㐀⤀⤀ഀഀ
    // - length == 4     : vec4<T>               (align 16) (size 16)਍    ⼀⼀ ⴀ 氀攀渀最琀栀 㴀㴀 ㌀     㨀 瘀攀挀㌀㰀吀㸀               ⠀愀氀椀最渀 ㄀㘀⤀ ⠀猀椀稀攀 ㄀㈀⤀ഀഀ
    // - length == 2     : vec2<T>               (align 8)  (size 8)਍    ⼀⼀ ⴀ 氀攀渀最琀栀 㴀㴀 ㄀     㨀 吀                     ⠀愀氀椀最渀 㐀⤀  ⠀猀椀稀攀 㐀⤀ഀഀ
    //਍ഀഀ
    const bool is_f16 = uniform.data_type == ProgramUniformVariableDataType::Float16;਍ഀഀ
    size_t variable_alignment = 4;  // default alignment for scalar types਍    猀椀稀攀开琀 瘀愀爀椀愀戀氀攀开猀椀稀攀 㴀 㐀㬀       ⼀⼀ 搀攀昀愀甀氀琀 猀椀稀攀 昀漀爀 猀挀愀氀愀爀 琀礀瀀攀猀ഀഀ
਍    椀昀 ⠀椀猀开昀㄀㘀⤀ 笀ഀഀ
      if (length > 6) {਍        瘀愀爀椀愀戀氀攀开愀氀椀最渀洀攀渀琀 㴀 ㄀㘀㬀ഀഀ
        variable_size = 16 * ((length + 7) / 8);਍      紀 攀氀猀攀 椀昀 ⠀氀攀渀最琀栀 㸀 㐀⤀ 笀ഀഀ
        variable_alignment = 16;਍        瘀愀爀椀愀戀氀攀开猀椀稀攀 㴀 ㄀㈀㬀ഀഀ
      } else if (length > 2) {਍        瘀愀爀椀愀戀氀攀开愀氀椀最渀洀攀渀琀 㴀 㠀㬀ഀഀ
        variable_size = 8;਍      紀ഀഀ
    } else {਍      椀昀 ⠀氀攀渀最琀栀 㸀 ㌀⤀ 笀ഀഀ
        variable_alignment = 16;਍        瘀愀爀椀愀戀氀攀开猀椀稀攀 㴀 ㄀㘀 ⨀ ⠀⠀氀攀渀最琀栀 ⬀ ㌀⤀ ⼀ 㐀⤀㬀ഀഀ
      } else if (length > 2) {਍        瘀愀爀椀愀戀氀攀开愀氀椀最渀洀攀渀琀 㴀 ㄀㘀㬀ഀഀ
        variable_size = 12;਍      紀 攀氀猀攀 椀昀 ⠀氀攀渀最琀栀 㸀 ㄀⤀ 笀ഀഀ
        variable_alignment = 8;਍        瘀愀爀椀愀戀氀攀开猀椀稀攀 㴀 㠀㬀ഀഀ
      }਍    紀ഀഀ
    current_offset = (current_offset + variable_alignment - 1) / variable_alignment * variable_alignment;਍    甀渀椀昀漀爀洀开愀渀搀开漀昀昀猀攀琀猀⸀攀洀瀀氀愀挀攀开戀愀挀欀⠀甀渀椀昀漀爀洀Ⰰ 挀甀爀爀攀渀琀开漀昀昀猀攀琀⤀㬀ഀഀ
਍    挀甀爀爀攀渀琀开漀昀昀猀攀琀 ⬀㴀 瘀愀爀椀愀戀氀攀开猀椀稀攀㬀ഀഀ
  }਍ഀഀ
  // Meet alignment of struct here: https://www.w3.org/TR/WGSL/#alignment-and-size. For simplicity, set਍  ⼀⼀ 洀愀砀开愀氀椀最渀洀攀渀琀开漀昀开昀椀攀氀搀 琀漀 ㄀㘀 猀椀渀挀攀 琀栀攀 甀渀搀攀爀氀礀椀渀最 戀甀昀昀攀爀 栀愀猀 戀攀攀渀 爀漀甀渀搀攀搀 甀瀀 琀漀 ㄀㘀⸀ഀഀ
  constexpr size_t max_alignment_of_field = 16;਍  挀漀渀猀琀 猀椀稀攀开琀 甀渀椀昀漀爀洀开戀甀昀昀攀爀开琀漀琀愀氀开猀椀稀攀 㴀 ⠀挀甀爀爀攀渀琀开漀昀昀猀攀琀 ⬀ 洀愀砀开愀氀椀最渀洀攀渀琀开漀昀开昀椀攀氀搀 ⴀ ㄀⤀ ⼀ 洀愀砀开愀氀椀最渀洀攀渀琀开漀昀开昀椀攀氀搀 ⨀ 洀愀砀开愀氀椀最渀洀攀渀琀开漀昀开昀椀攀氀搀㬀ഀഀ
਍  圀䜀倀唀䈀甀昀昀攀爀 甀渀椀昀漀爀洀开戀甀昀昀攀爀 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
  const webgpu::BufferManager& buffer_mgr = ComputeContextBase::BufferManagerAccessor::Get(context);਍  椀昀 ⠀甀渀椀昀漀爀洀开戀甀昀昀攀爀开琀漀琀愀氀开猀椀稀攀 㸀 　⤀ 笀ഀഀ
    std::vector<uint8_t> uniform_data_buffer(uniform_buffer_total_size);਍ഀഀ
    for (auto const& [uniform, offset] : uniform_and_offsets) {਍      洀攀洀挀瀀礀⠀甀渀椀昀漀爀洀开搀愀琀愀开戀甀昀昀攀爀⸀搀愀琀愀⠀⤀ ⬀ 漀昀昀猀攀琀Ⰰ 甀渀椀昀漀爀洀⸀搀愀琀愀⸀搀愀琀愀⠀⤀Ⰰ 甀渀椀昀漀爀洀⸀搀愀琀愀⸀猀椀稀攀⠀⤀⤀㬀ഀഀ
    }਍ഀഀ
    uniform_buffer = buffer_mgr.Create(uniform_buffer_total_size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);਍    搀攀瘀椀挀攀开焀甀攀甀攀开⸀圀爀椀琀攀䈀甀昀昀攀爀⠀甀渀椀昀漀爀洀开戀甀昀昀攀爀Ⰰ 　Ⰰ 甀渀椀昀漀爀洀开搀愀琀愀开戀甀昀昀攀爀⸀搀愀琀愀⠀⤀Ⰰ 甀渀椀昀漀爀洀开戀甀昀昀攀爀开琀漀琀愀氀开猀椀稀攀⤀㬀ഀഀ
  }਍ഀഀ
  const auto& compute_pass_encoder = GetComputePassEncoder();਍ഀഀ
  WriteTimestamp(num_pending_dispatches_ * 2);਍ഀഀ
  const size_t total_buffer_count = inputs.size() + outputs.size() + (uniform_buffer ? 1 : 0);਍ഀഀ
  std::vector<WGPUBuffer> bind_buffers;਍  猀琀搀㨀㨀瘀攀挀琀漀爀㰀甀椀渀琀㌀㈀开琀㸀 戀椀渀搀开戀甀昀昀攀爀猀开猀攀最洀攀渀琀猀㬀ഀഀ
  bind_buffers.reserve(total_buffer_count);਍  戀椀渀搀开戀甀昀昀攀爀猀开猀攀最洀攀渀琀猀⸀爀攀猀攀爀瘀攀⠀琀漀琀愀氀开戀甀昀昀攀爀开挀漀甀渀琀⤀㬀ഀഀ
  for (size_t i = 0; i < inputs.size(); i++) {਍    戀椀渀搀开戀甀昀昀攀爀猀⸀瀀甀猀栀开戀愀挀欀⠀爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀圀䜀倀唀䈀甀昀昀攀爀㸀⠀挀漀渀猀琀开挀愀猀琀㰀瘀漀椀搀⨀㸀⠀椀渀瀀甀琀猀嬀椀崀⸀琀攀渀猀漀爀ⴀ㸀䐀愀琀愀刀愀眀⠀⤀⤀⤀⤀㬀ഀഀ
    bind_buffers_segments.push_back(inputs_segments[i]);਍  紀ഀഀ
  for (size_t i = 0; i < outputs.size(); i++) {਍    戀椀渀搀开戀甀昀昀攀爀猀⸀瀀甀猀栀开戀愀挀欀⠀爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀圀䜀倀唀䈀甀昀昀攀爀㸀⠀漀甀琀瀀甀琀猀嬀椀崀⸀琀攀渀猀漀爀ⴀ㸀䴀甀琀愀戀氀攀䐀愀琀愀刀愀眀⠀⤀⤀⤀㬀ഀഀ
    bind_buffers_segments.push_back(outputs_segments[i]);਍  紀ഀഀ
  if (uniform_buffer) {਍    戀椀渀搀开戀甀昀昀攀爀猀⸀瀀甀猀栀开戀愀挀欀⠀甀渀椀昀漀爀洀开戀甀昀昀攀爀⤀㬀ഀഀ
    bind_buffers_segments.push_back(1);  // uniform buffer defaults to 1 segment਍  紀ഀഀ
਍  䰀愀甀渀挀栀䌀漀洀瀀甀琀攀倀椀瀀攀氀椀渀攀⠀挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀Ⰰ 戀椀渀搀开戀甀昀昀攀爀猀Ⰰ 戀椀渀搀开戀甀昀昀攀爀猀开猀攀最洀攀渀琀猀Ⰰ ⨀瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀Ⰰ 砀Ⰰ 礀Ⰰ 稀Ⰰ 瀀爀漀最爀愀洀⸀䤀渀搀椀爀攀挀琀䐀椀猀瀀愀琀挀栀吀攀渀猀漀爀⠀⤀⤀㬀ഀഀ
  if (uniform_buffer) {਍    戀甀昀昀攀爀开洀最爀⸀刀攀氀攀愀猀攀⠀甀渀椀昀漀爀洀开戀甀昀昀攀爀⤀㬀ഀഀ
  }਍ഀഀ
  WriteTimestamp(num_pending_dispatches_ * 2 + 1);਍  ⬀⬀渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开㬀ഀഀ
਍  ⼀⼀ 唀瀀搀愀琀攀 瀀爀漀昀椀氀椀渀最 搀愀琀愀 愀昀琀攀爀 䰀愀甀渀挀栀䌀漀洀瀀甀琀攀倀椀瀀攀氀椀渀攀ഀഀ
  if (is_profiling_) {਍    倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀 瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀⠀挀漀渀琀攀砀琀⸀一漀搀攀一愀洀攀⠀⤀Ⰰഀഀ
                                          context.OpType(),਍                                          瀀爀漀最爀愀洀⸀一愀洀攀⠀⤀Ⰰഀഀ
                                          key,਍                                          椀渀瀀甀琀猀Ⰰഀഀ
                                          outputs);਍ഀഀ
    if (graph_capture_state_ == GraphCaptureState::Capturing) {਍      ⼀⼀ 唀瀀搀愀琀攀 琀栀攀 氀愀猀琀 挀愀瀀琀甀爀攀搀 挀漀洀洀愀渀搀✀猀 瀀爀漀昀椀氀椀渀最 椀渀昀漀ഀഀ
      if (external_captured_commands_ && !external_captured_commands_->empty()) {਍        攀砀琀攀爀渀愀氀开挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀开ⴀ㸀戀愀挀欀⠀⤀⸀瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀 㴀 猀琀搀㨀㨀洀漀瘀攀⠀瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀⤀㬀ഀഀ
      }਍    紀 攀氀猀攀 笀ഀഀ
      // Add to pending kernels for current run profiling਍      瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开⸀攀洀瀀氀愀挀攀开戀愀挀欀⠀猀琀搀㨀㨀洀漀瘀攀⠀瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀⤀⤀㬀ഀഀ
    }਍  紀ഀഀ
਍  椀昀 ⠀渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 㸀㴀 洀愀砀开渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 簀簀ഀഀ
      (is_profiling_ && query_type_ == TimestampQueryType::AtPasses)) {਍    䔀渀搀䌀漀洀瀀甀琀攀倀愀猀猀⠀⤀㬀ഀഀ
  }਍  椀昀 ⠀渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 㸀㴀 洀愀砀开渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开⤀ 笀ഀഀ
    Flush(buffer_mgr);਍    渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 㴀 　㬀ഀഀ
  }਍ഀഀ
  return Status::OK();਍紀ഀഀ
਍猀琀搀㨀㨀瘀攀挀琀漀爀㰀挀漀渀猀琀 挀栀愀爀⨀㸀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䜀攀琀䔀渀愀戀氀攀搀䄀搀愀瀀琀攀爀吀漀最最氀攀猀⠀⤀ 挀漀渀猀琀 笀ഀഀ
  // See the description of all the toggles in toggles.cpp਍  ⼀⼀ ∀甀猀攀开搀砀挀∀ 昀漀爀 匀栀愀搀攀爀 䴀漀搀攀氀 㘀⬀ 昀攀愀琀甀爀攀猀 ⠀攀⸀最⸀ 昀氀漀愀琀㄀㘀⤀ഀഀ
  // "allow_unsafe_apis" for chromium experimental features਍  挀漀渀猀琀攀砀瀀爀 挀漀渀猀琀 挀栀愀爀⨀ 琀漀最最氀攀猀嬀崀 㴀 笀ഀഀ
      "use_dxc",਍      ∀愀氀氀漀眀开甀渀猀愀昀攀开愀瀀椀猀∀Ⰰഀഀ
      "decompose_uniform_buffers",਍⌀椀昀 搀攀昀椀渀攀搀⠀䐀䄀圀一开䔀一䄀䈀䰀䔀开嘀唀䰀䬀䄀一⤀ഀഀ
      "use_vulkan_memory_model",਍      ∀瘀甀氀欀愀渀开攀渀愀戀氀攀开昀㄀㘀开漀渀开渀瘀椀搀椀愀∀Ⰰഀഀ
#endif਍  紀㬀ഀഀ
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));਍紀ഀഀ
਍猀琀搀㨀㨀瘀攀挀琀漀爀㰀挀漀渀猀琀 挀栀愀爀⨀㸀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䜀攀琀䔀渀愀戀氀攀搀䐀攀瘀椀挀攀吀漀最最氀攀猀⠀⤀ 挀漀渀猀琀 笀ഀഀ
  // Enable / disable other toggles that may affect the performance.਍  ⼀⼀ 伀琀栀攀爀 琀漀最最氀攀猀 琀栀愀琀 洀愀礀 戀攀 甀猀攀昀甀氀㨀 ∀搀甀洀瀀开猀栀愀搀攀爀猀∀Ⰰ ∀搀椀猀愀戀氀攀开猀礀洀戀漀氀开爀攀渀愀洀椀渀最∀ഀഀ
  constexpr const char* toggles[] = {਍      ∀猀欀椀瀀开瘀愀氀椀搀愀琀椀漀渀∀Ⰰഀഀ
      "disable_robustness",਍      ∀搀㌀搀开搀椀猀愀戀氀攀开椀攀攀攀开猀琀爀椀挀琀渀攀猀猀∀Ⰰഀഀ
  };਍⌀椀昀渀搀攀昀 一䐀䔀䈀唀䜀ഀഀ
  return std::vector<const char*>(ValidationMode() >= ValidationMode::WGPUOnly਍                                      㼀 猀琀搀㨀㨀戀攀最椀渀⠀琀漀最最氀攀猀⤀ ⬀ ㄀ഀഀ
                                      : std::begin(toggles),਍                                  猀琀搀㨀㨀攀渀搀⠀琀漀最最氀攀猀⤀⤀㬀ഀഀ
#else਍  ⼀⼀ 䤀渀 爀攀氀攀愀猀攀⼀爀攀氀眀椀琀栀搀攀戀椀渀昀漀 戀甀椀氀搀猀Ⰰ 搀攀昀愀甀氀琀 琀漀 猀欀椀瀀开瘀愀氀椀搀愀琀椀漀渀 昀漀爀 瀀攀爀昀漀爀洀愀渀挀攀Ⰰഀഀ
  // but honor explicit validationMode overrides.਍  椀昀 ⠀℀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀开⤀ 笀ഀഀ
    return std::vector<const char*>(std::begin(toggles), std::end(toggles));਍  紀ഀഀ
  return std::vector<const char*>(ValidationMode() >= ValidationMode::WGPUOnly਍                                      㼀 猀琀搀㨀㨀戀攀最椀渀⠀琀漀最最氀攀猀⤀ ⬀ ㄀ഀഀ
                                      : std::begin(toggles),਍                                  猀琀搀㨀㨀攀渀搀⠀琀漀最最氀攀猀⤀⤀㬀ഀഀ
#endif਍紀ഀഀ
਍猀琀搀㨀㨀瘀攀挀琀漀爀㰀挀漀渀猀琀 挀栀愀爀⨀㸀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䜀攀琀䐀椀猀愀戀氀攀搀䐀攀瘀椀挀攀吀漀最最氀攀猀⠀⤀ 挀漀渀猀琀 笀ഀഀ
  constexpr const char* toggles[] = {਍      ∀氀愀稀礀开挀氀攀愀爀开爀攀猀漀甀爀挀攀开漀渀开昀椀爀猀琀开甀猀攀∀Ⰰഀഀ
      "timestamp_quantization",਍  紀㬀ഀഀ
  return std::vector<const char*>(std::begin(toggles), std::end(toggles));਍紀ഀഀ
਍猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀㸀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䜀攀琀䄀瘀愀椀氀愀戀氀攀刀攀焀甀椀爀攀搀䘀攀愀琀甀爀攀猀⠀挀漀渀猀琀 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀☀ 愀搀愀瀀琀攀爀⤀ 挀漀渀猀琀 笀ഀഀ
  std::vector<wgpu::FeatureName> required_features;਍  挀漀渀猀琀攀砀瀀爀 眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀 昀攀愀琀甀爀攀猀嬀崀笀ഀഀ
#if !defined(__wasm__)਍      眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀㨀㨀䌀栀爀漀洀椀甀洀䔀砀瀀攀爀椀洀攀渀琀愀氀吀椀洀攀猀琀愀洀瀀儀甀攀爀礀䤀渀猀椀搀攀倀愀猀猀攀猀Ⰰഀഀ
      wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,਍⌀攀渀搀椀昀ഀഀ
      wgpu::FeatureName::TimestampQuery,਍      眀最瀀甀㨀㨀䘀攀愀琀甀爀攀一愀洀攀㨀㨀匀栀愀搀攀爀䘀㄀㘀Ⰰഀഀ
      wgpu::FeatureName::Subgroups,਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
      wgpu::FeatureName::BufferMapExtendedUsages,਍⌀攀渀搀椀昀ഀഀ
  };਍  昀漀爀 ⠀愀甀琀漀 昀攀愀琀甀爀攀 㨀 昀攀愀琀甀爀攀猀⤀ 笀ഀഀ
    if (adapter.HasFeature(feature)) {਍      爀攀焀甀椀爀攀搀开昀攀愀琀甀爀攀猀⸀瀀甀猀栀开戀愀挀欀⠀昀攀愀琀甀爀攀⤀㬀ഀഀ
    }਍  紀ഀഀ
  return required_features;਍紀ഀഀ
਍眀最瀀甀㨀㨀䰀椀洀椀琀猀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䜀攀琀刀攀焀甀椀爀攀搀䰀椀洀椀琀猀⠀挀漀渀猀琀 眀最瀀甀㨀㨀䄀搀愀瀀琀攀爀☀ 愀搀愀瀀琀攀爀⤀ 挀漀渀猀琀 笀ഀഀ
  wgpu::Limits required_limits{};਍  眀最瀀甀㨀㨀䰀椀洀椀琀猀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀㬀ഀഀ
  ORT_ENFORCE(adapter.GetLimits(&adapter_limits));਍ഀഀ
  required_limits.maxBindGroups = adapter_limits.maxBindGroups;਍  爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀圀漀爀欀最爀漀甀瀀匀琀漀爀愀最攀匀椀稀攀 㴀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀圀漀爀欀最爀漀甀瀀匀琀漀爀愀最攀匀椀稀攀㬀ഀഀ
  required_limits.maxComputeWorkgroupsPerDimension = adapter_limits.maxComputeWorkgroupsPerDimension;਍  爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀猀倀攀爀匀栀愀搀攀爀匀琀愀最攀 㴀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀猀倀攀爀匀栀愀搀攀爀匀琀愀最攀㬀ഀഀ
਍  椀昀 ⠀洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开 㴀㴀 　⤀ 笀ഀഀ
    // If not set by the user, use the adapter limit.਍    爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀 㴀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀㬀ഀഀ
  } else {਍    爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀 㴀 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开㬀ഀഀ
  }਍ഀഀ
  required_limits.maxBufferSize = adapter_limits.maxBufferSize;਍  爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀䤀渀瘀漀挀愀琀椀漀渀猀倀攀爀圀漀爀欀最爀漀甀瀀 㴀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀䤀渀瘀漀挀愀琀椀漀渀猀倀攀爀圀漀爀欀最爀漀甀瀀㬀ഀഀ
  required_limits.maxComputeWorkgroupSizeX = adapter_limits.maxComputeWorkgroupSizeX;਍  爀攀焀甀椀爀攀搀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀圀漀爀欀最爀漀甀瀀匀椀稀攀夀 㴀 愀搀愀瀀琀攀爀开氀椀洀椀琀猀⸀洀愀砀䌀漀洀瀀甀琀攀圀漀爀欀最爀漀甀瀀匀椀稀攀夀㬀ഀഀ
  required_limits.maxComputeWorkgroupSizeZ = adapter_limits.maxComputeWorkgroupSizeZ;਍ഀഀ
  return required_limits;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀圀爀椀琀攀吀椀洀攀猀琀愀洀瀀⠀甀椀渀琀㌀㈀开琀 焀甀攀爀礀开椀渀搀攀砀⤀ 笀ഀഀ
  if (!is_profiling_ || graph_capture_state_ == GraphCaptureState::Capturing || query_type_ != TimestampQueryType::InsidePasses) {਍    爀攀琀甀爀渀㬀ഀഀ
  }਍ഀഀ
  const auto& compute_pass_encoder = GetComputePassEncoder();਍  挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀⸀圀爀椀琀攀吀椀洀攀猀琀愀洀瀀⠀焀甀攀爀礀开猀攀琀开Ⰰ 焀甀攀爀礀开椀渀搀攀砀⤀㬀ഀഀ
}਍ഀഀ
void WebGpuContext::StartProfiling() {਍  椀昀 ⠀焀甀攀爀礀开琀礀瀀攀开 㴀㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀一漀渀攀⤀ 笀ഀഀ
    return;਍  紀ഀഀ
਍  椀猀开瀀爀漀昀椀氀椀渀最开 㴀 琀爀甀攀㬀ഀഀ
਍  挀漀渀猀琀 甀椀渀琀㌀㈀开琀 焀甀攀爀礀开挀漀甀渀琀 㴀 洀愀砀开渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 ⨀ ㈀㬀ഀഀ
਍  椀昀 ⠀℀焀甀攀爀礀开猀攀琀开⤀ 笀ഀഀ
    // Create query set਍    眀最瀀甀㨀㨀儀甀攀爀礀匀攀琀䐀攀猀挀爀椀瀀琀漀爀 焀甀攀爀礀匀攀琀䐀攀猀挀爀椀瀀琀漀爀㬀ഀഀ
    querySetDescriptor.count = query_count;਍    焀甀攀爀礀匀攀琀䐀攀猀挀爀椀瀀琀漀爀⸀琀礀瀀攀 㴀 眀最瀀甀㨀㨀儀甀攀爀礀吀礀瀀攀㨀㨀吀椀洀攀猀琀愀洀瀀㬀ഀഀ
    query_set_ = device_.CreateQuerySet(&querySetDescriptor);਍  紀ഀഀ
਍  椀昀 ⠀℀焀甀攀爀礀开爀攀猀漀氀瘀攀开戀甀昀昀攀爀开⤀ 笀ഀഀ
    // Create resolve buffer਍    眀最瀀甀㨀㨀䈀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀 戀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀㬀ഀഀ
    bufferDescriptor.size = query_count * sizeof(uint64_t);਍    戀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀⸀甀猀愀最攀 㴀 眀最瀀甀㨀㨀䈀甀昀昀攀爀唀猀愀最攀㨀㨀儀甀攀爀礀刀攀猀漀氀瘀攀 簀 眀最瀀甀㨀㨀䈀甀昀昀攀爀唀猀愀最攀㨀㨀䌀漀瀀礀匀爀挀 簀ഀഀ
                             wgpu::BufferUsage::CopyDst;਍    焀甀攀爀礀开爀攀猀漀氀瘀攀开戀甀昀昀攀爀开 㴀 搀攀瘀椀挀攀开⸀䌀爀攀愀琀攀䈀甀昀昀攀爀⠀☀戀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀⤀㬀ഀഀ
  }਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䌀漀氀氀攀挀琀倀爀漀昀椀氀椀渀最䐀愀琀愀⠀瀀爀漀昀椀氀椀渀最㨀㨀䔀瘀攀渀琀猀☀ 攀瘀攀渀琀猀⤀ 笀ഀഀ
  if (!pending_queries_.empty()) {਍    昀漀爀 ⠀挀漀渀猀琀 愀甀琀漀☀ 瀀攀渀搀椀渀最开焀甀攀爀礀 㨀 瀀攀渀搀椀渀最开焀甀攀爀椀攀猀开⤀ 笀ഀഀ
      const auto& pending_kernels = pending_query.kernels;਍      挀漀渀猀琀 愀甀琀漀☀ 焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀 㴀 瀀攀渀搀椀渀最开焀甀攀爀礀⸀焀甀攀爀礀开戀甀昀昀攀爀㬀ഀഀ
਍      伀刀吀开䔀一䘀伀刀䌀䔀⠀圀愀椀琀⠀焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀⸀䴀愀瀀䄀猀礀渀挀⠀眀最瀀甀㨀㨀䴀愀瀀䴀漀搀攀㨀㨀刀攀愀搀Ⰰഀഀ
                                                  0,਍                                                  猀琀愀琀椀挀开挀愀猀琀㰀猀椀稀攀开琀㸀⠀焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀⸀䜀攀琀匀椀稀攀⠀⤀⤀Ⰰഀഀ
                                                  wgpu::CallbackMode::WaitAnyOnly,਍                                                  嬀崀⠀眀最瀀甀㨀㨀䴀愀瀀䄀猀礀渀挀匀琀愀琀甀猀 猀琀愀琀甀猀Ⰰ 眀最瀀甀㨀㨀匀琀爀椀渀最嘀椀攀眀 洀攀猀猀愀最攀⤀ 笀ഀഀ
                                                    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success, "Failed to download data from buffer: ", std::string_view{message});਍                                                  紀⤀⤀ 㴀㴀 匀琀愀琀甀猀㨀㨀伀䬀⠀⤀⤀㬀ഀഀ
      auto mapped_data = static_cast<const uint64_t*>(query_read_buffer.GetConstMappedRange());਍ഀഀ
      for (size_t i = 0; i < pending_kernels.size(); i++) {਍        挀漀渀猀琀 倀攀渀搀椀渀最䬀攀爀渀攀氀䤀渀昀漀☀ 瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀 㴀 瀀攀渀搀椀渀最开欀攀爀渀攀氀猀嬀椀崀㬀ഀഀ
        const auto& input_shapes = pending_kernel_info.input_shapes;਍        挀漀渀猀琀 愀甀琀漀☀ 漀甀琀瀀甀琀开猀栀愀瀀攀猀 㴀 瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀⸀漀甀琀瀀甀琀开猀栀愀瀀攀猀㬀ഀഀ
਍        匀匀⠀猀栀愀瀀攀猀Ⰰ ㄀㈀㠀⤀㬀ഀഀ
        for (size_t s = 0; s < input_shapes.size(); s++) {਍          猀栀愀瀀攀猀 㰀㰀 ∀椀渀瀀甀琀猀嬀∀ 㰀㰀 猀 㰀㰀 ∀崀 㴀 ∀ 㰀㰀 椀渀瀀甀琀开猀栀愀瀀攀猀嬀猀崀⸀吀漀匀琀爀椀渀最⠀⤀ 㰀㰀 ∀ ∀㬀ഀഀ
        }਍        昀漀爀 ⠀猀椀稀攀开琀 猀 㴀 　㬀 猀 㰀 漀甀琀瀀甀琀开猀栀愀瀀攀猀⸀猀椀稀攀⠀⤀㬀 猀⬀⬀⤀ 笀ഀഀ
          shapes << "outputs[" << s << "] = " << output_shapes[s].ToString() << " ";਍        紀ഀഀ
਍        椀昀 ⠀最瀀甀开琀椀洀攀猀琀愀洀瀀开漀昀昀猀攀琀开 㴀㴀 　⤀ 笀ഀഀ
          gpu_timestamp_offset_ = mapped_data[i * 2];਍          ⼀⼀ 吀伀䐀伀㨀 愀瀀瀀氀礀 䌀倀唀ⴀ䜀倀唀 琀椀洀攀 漀昀昀猀攀琀 猀漀 琀栀愀琀 琀椀洀攀猀琀愀洀瀀猀 愀爀攀 愀氀椀最渀攀搀ഀഀ
        }਍        甀椀渀琀㘀㐀开琀 猀琀愀爀琀开琀椀洀攀 㴀 洀愀瀀瀀攀搀开搀愀琀愀嬀椀 ⨀ ㈀崀 ⴀ 最瀀甀开琀椀洀攀猀琀愀洀瀀开漀昀昀猀攀琀开㬀ഀഀ
        uint64_t end_time = mapped_data[i * 2 + 1] - gpu_timestamp_offset_;਍ഀഀ
        InlinedHashMap<std::string, std::string> event_args = {਍            笀∀猀栀愀瀀攀猀∀Ⰰ 匀匀开䜀䔀吀⠀猀栀愀瀀攀猀⤀紀Ⰰഀഀ
            {"cache_key", pending_kernel_info.cache_key},਍        紀㬀ഀഀ
਍        瀀爀漀昀椀氀椀渀最㨀㨀䔀瘀攀渀琀刀攀挀漀爀搀 攀瘀攀渀琀⠀瀀爀漀昀椀氀椀渀最㨀㨀䄀倀䤀开䔀嘀䔀一吀Ⰰഀഀ
                                     -1,਍                                     ⴀ㄀Ⰰഀഀ
                                     pending_kernel_info.name,਍                                     猀琀愀琀椀挀开挀愀猀琀㰀椀渀琀㘀㐀开琀㸀⠀猀琀搀㨀㨀爀漀甀渀搀⠀猀琀愀爀琀开琀椀洀攀 ⼀ ㄀　　　⸀　⤀⤀Ⰰഀഀ
                                     static_cast<int64_t>(std::round((end_time - start_time) / 1000.0)),਍                                     攀瘀攀渀琀开愀爀最猀⤀㬀ഀഀ
        events.emplace_back(std::move(event));਍      紀ഀഀ
਍      焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀⸀唀渀洀愀瀀⠀⤀㬀ഀഀ
      query_read_buffer.Destroy();਍    紀ഀഀ
਍    瀀攀渀搀椀渀最开焀甀攀爀椀攀猀开⸀挀氀攀愀爀⠀⤀㬀ഀഀ
  }਍ഀഀ
  is_profiling_ = false;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䌀漀氀氀攀挀琀倀爀漀昀椀氀椀渀最䐀愀琀愀⠀⤀ 笀ഀഀ
  CollectProfilingData(events_);਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䔀渀搀倀爀漀昀椀氀椀渀最⠀吀椀洀攀倀漀椀渀琀 ⼀⨀ 琀瀀 ⨀⼀Ⰰ 瀀爀漀昀椀氀椀渀最㨀㨀䔀瘀攀渀琀猀☀ 攀瘀攀渀琀猀⤀ 笀ഀഀ
  // This function is called when no active inference is ongoing.਍  伀刀吀开䔀一䘀伀刀䌀䔀⠀℀椀猀开瀀爀漀昀椀氀椀渀最开Ⰰ ∀倀爀漀昀椀氀椀渀最 椀猀 漀渀最漀椀渀最 椀渀 愀渀 椀渀昀攀爀攀渀挀攀 爀甀渀⸀∀⤀㬀ഀഀ
਍  椀昀 ⠀焀甀攀爀礀开琀礀瀀攀开 ℀㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀一漀渀攀⤀ 笀ഀഀ
    // No pending kernels or queries should be present at this point. They should have been collected in CollectProfilingData.਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开⸀攀洀瀀琀礀⠀⤀ ☀☀ 瀀攀渀搀椀渀最开焀甀攀爀椀攀猀开⸀攀洀瀀琀礀⠀⤀Ⰰ ∀倀攀渀搀椀渀最 欀攀爀渀攀氀猀 漀爀 焀甀攀爀椀攀猀 愀爀攀 渀漀琀 攀洀瀀琀礀⸀∀⤀㬀ഀഀ
਍    攀瘀攀渀琀猀⸀椀渀猀攀爀琀⠀攀瘀攀渀琀猀⸀攀渀搀⠀⤀Ⰰഀഀ
                  std::make_move_iterator(events_.begin()),਍                  猀琀搀㨀㨀洀愀欀攀开洀漀瘀攀开椀琀攀爀愀琀漀爀⠀攀瘀攀渀琀猀开⸀攀渀搀⠀⤀⤀⤀㬀ഀഀ
    events_.clear();਍  紀 攀氀猀攀 笀ഀഀ
    LOGS_DEFAULT(WARNING) << "TimestampQuery is not supported in this device.";਍  紀ഀഀ
}਍ഀഀ
void WebGpuContext::PushErrorScope() { device_.PushErrorScope(wgpu::ErrorFilter::Validation); }਍ഀഀ
Status WebGpuContext::PopErrorScope() {਍  匀琀愀琀甀猀 猀琀愀琀甀猀笀紀㬀ഀഀ
  ORT_RETURN_IF_ERROR(Wait(device_.PopErrorScope(਍      眀最瀀甀㨀㨀䌀愀氀氀戀愀挀欀䴀漀搀攀㨀㨀圀愀椀琀䄀渀礀伀渀氀礀Ⰰഀഀ
      [](wgpu::PopErrorScopeStatus pop_status, wgpu::ErrorType error_type, char const* message, Status* status) {਍        伀刀吀开䔀一䘀伀刀䌀䔀⠀瀀漀瀀开猀琀愀琀甀猀 㴀㴀 眀最瀀甀㨀㨀倀漀瀀䔀爀爀漀爀匀挀漀瀀攀匀琀愀琀甀猀㨀㨀匀甀挀挀攀猀猀Ⰰ ∀䤀渀猀琀愀渀挀攀 搀爀漀瀀瀀攀搀⸀∀⤀㬀ഀഀ
        if (error_type == wgpu::ErrorType::NoError) {਍          爀攀琀甀爀渀㬀ഀഀ
        }਍        ⨀猀琀愀琀甀猀 㴀 伀刀吀开䴀䄀䬀䔀开匀吀䄀吀唀匀⠀伀一一堀刀唀一吀䤀䴀䔀Ⰰ 䘀䄀䤀䰀Ⰰ ∀圀攀戀䜀倀唀 瘀愀氀椀搀愀琀椀漀渀 昀愀椀氀攀搀⸀ ∀Ⰰ 洀攀猀猀愀最攀⤀㬀ഀഀ
      },਍      ☀猀琀愀琀甀猀⤀⤀⤀㬀ഀഀ
  return status;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䘀氀甀猀栀⠀挀漀渀猀琀 眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀☀ 戀甀昀昀攀爀开洀最爀⤀ 笀ഀഀ
  if (!current_command_encoder_) {਍    爀攀琀甀爀渀㬀ഀഀ
  }਍ഀഀ
  EndComputePass();਍ഀഀ
  if (is_profiling_ && num_pending_dispatches_ > 0 && graph_capture_state_ != GraphCaptureState::Capturing) {਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 㴀㴀 瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开⸀猀椀稀攀⠀⤀Ⰰഀഀ
                "Number of pending dispatches (", num_pending_dispatches_,਍                ∀⤀ 搀漀攀猀 渀漀琀 洀愀琀挀栀 瀀攀渀搀椀渀最 欀攀爀渀攀氀猀 猀椀稀攀 ⠀∀Ⰰ 瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开⸀猀椀稀攀⠀⤀Ⰰ ∀⤀∀⤀㬀ഀഀ
਍    甀椀渀琀㌀㈀开琀 焀甀攀爀礀开挀漀甀渀琀 㴀 渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开 ⨀ ㈀㬀ഀഀ
    current_command_encoder_.ResolveQuerySet(਍        焀甀攀爀礀开猀攀琀开Ⰰഀഀ
        0,਍        焀甀攀爀礀开挀漀甀渀琀Ⰰഀഀ
        query_resolve_buffer_,਍        　⤀㬀ഀഀ
਍    眀最瀀甀㨀㨀䈀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀 戀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀㬀ഀഀ
    bufferDescriptor.size = query_count * sizeof(uint64_t);਍    戀甀昀昀攀爀䐀攀猀挀爀椀瀀琀漀爀⸀甀猀愀最攀 㴀 眀最瀀甀㨀㨀䈀甀昀昀攀爀唀猀愀最攀㨀㨀䴀愀瀀刀攀愀搀 簀 眀最瀀甀㨀㨀䈀甀昀昀攀爀唀猀愀最攀㨀㨀䌀漀瀀礀䐀猀琀㬀ഀഀ
    wgpu::Buffer query_read_buffer = device_.CreateBuffer(&bufferDescriptor);਍ഀഀ
    current_command_encoder_.CopyBufferToBuffer(਍        焀甀攀爀礀开爀攀猀漀氀瘀攀开戀甀昀昀攀爀开Ⰰഀഀ
        0,਍        焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀Ⰰഀഀ
        0,਍        焀甀攀爀礀开挀漀甀渀琀 ⨀ 猀椀稀攀漀昀⠀甀椀渀琀㘀㐀开琀⤀⤀㬀ഀഀ
਍    瀀攀渀搀椀渀最开焀甀攀爀椀攀猀开⸀攀洀瀀氀愀挀攀开戀愀挀欀⠀猀琀搀㨀㨀洀漀瘀攀⠀瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开⤀Ⰰ 焀甀攀爀礀开爀攀愀搀开戀甀昀昀攀爀⤀㬀ഀഀ
    pending_kernels_.clear();਍  紀ഀഀ
  auto command_buffer = current_command_encoder_.Finish();਍  搀攀瘀椀挀攀开焀甀攀甀攀开⸀匀甀戀洀椀琀⠀㄀Ⰰ ☀挀漀洀洀愀渀搀开戀甀昀昀攀爀⤀㬀ഀഀ
  if (graph_capture_state_ != GraphCaptureState::Replaying) {਍    戀甀昀昀攀爀开洀最爀⸀刀攀昀爀攀猀栀倀攀渀搀椀渀最䈀甀昀昀攀爀猀⠀最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀愀琀攀开⤀㬀ഀഀ
  }਍  挀甀爀爀攀渀琀开挀漀洀洀愀渀搀开攀渀挀漀搀攀爀开 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
  num_pending_dispatches_ = 0;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䰀愀甀渀挀栀䌀漀洀瀀甀琀攀倀椀瀀攀氀椀渀攀⠀挀漀渀猀琀 眀最瀀甀㨀㨀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀☀ 挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀Ⰰഀഀ
                                          const std::vector<WGPUBuffer>& bind_buffers,਍                                          挀漀渀猀琀 猀琀搀㨀㨀瘀攀挀琀漀爀㰀甀椀渀琀㌀㈀开琀㸀☀ 戀椀渀搀开戀甀昀昀攀爀猀开猀攀最洀攀渀琀猀Ⰰഀഀ
                                          const ProgramArtifact& program_artifact,਍                                          甀椀渀琀㌀㈀开琀 砀Ⰰ 甀椀渀琀㌀㈀开琀 礀Ⰰ 甀椀渀琀㌀㈀开琀 稀Ⰰഀഀ
                                          const Tensor* indirect_dispatch_tensor) {਍  甀椀渀琀㌀㈀开琀 攀渀琀爀礀开椀渀搀攀砀 㴀 　㬀ഀഀ
  std::vector<WGPUBindGroupEntry> bind_group_entries;਍ഀഀ
  const uint64_t kMaxBufferSize = device_limits_.maxStorageBufferBindingSize;਍  昀漀爀 ⠀猀椀稀攀开琀 戀甀昀昀攀爀开椀搀砀 㴀 　㬀 戀甀昀昀攀爀开椀搀砀 㰀 戀椀渀搀开戀甀昀昀攀爀猀⸀猀椀稀攀⠀⤀㬀 ⬀⬀戀甀昀昀攀爀开椀搀砀⤀ 笀ഀഀ
    WGPUBuffer buffer = bind_buffers[buffer_idx];਍    挀漀渀猀琀 甀椀渀琀㌀㈀开琀 琀漀琀愀氀开猀攀最洀攀渀琀猀 㴀 戀椀渀搀开戀甀昀昀攀爀猀开猀攀最洀攀渀琀猀嬀戀甀昀昀攀爀开椀搀砀崀㬀ഀഀ
    // `total_segments` we used is calculated by tensor size, not actual buffer size. Because for bucketed buffer,਍    ⼀⼀ 琀栀攀 愀挀琀甀愀氀 戀甀昀昀攀爀 猀椀稀攀 洀愀礀 戀攀 氀愀爀最攀爀 琀栀愀渀 琀栀攀 琀攀渀猀漀爀 猀椀稀攀Ⰰ 愀渀 攀砀琀爀攀洀攀 挀愀猀攀 椀猀 琀栀愀琀 琀攀渀猀漀爀 猀椀稀攀 㴀 ㄀㈀㜀䴀䈀Ⰰ 戀甀昀昀攀爀 猀椀稀攀 㴀 ㈀㔀㘀䴀䈀Ⰰഀഀ
    // maxStorageBufferBindingSize = 128MB, in this case we only need to bind 1 segment instead of 2 segments because਍    ⼀⼀ 琀栀攀爀攀 椀猀 渀漀 搀愀琀愀 昀漀爀 琀栀攀 猀攀挀漀渀搀 猀攀最洀攀渀琀⸀ഀഀ
    if (total_segments > 1) {਍      甀椀渀琀㘀㐀开琀 漀昀昀猀攀琀 㴀 　㬀ഀഀ
      uint64_t buffer_size = wgpuBufferGetSize(buffer);਍      昀漀爀 ⠀甀椀渀琀㌀㈀开琀 猀攀最洀攀渀琀 㴀 　㬀 猀攀最洀攀渀琀 㰀 琀漀琀愀氀开猀攀最洀攀渀琀猀㬀 ⬀⬀猀攀最洀攀渀琀⤀ 笀ഀഀ
        uint64_t segment_size = std::min(kMaxBufferSize, buffer_size - offset);਍        戀椀渀搀开最爀漀甀瀀开攀渀琀爀椀攀猀⸀瀀甀猀栀开戀愀挀欀⠀笀渀甀氀氀瀀琀爀Ⰰ 攀渀琀爀礀开椀渀搀攀砀⬀⬀Ⰰ 戀甀昀昀攀爀Ⰰ 漀昀昀猀攀琀Ⰰ 猀攀最洀攀渀琀开猀椀稀攀Ⰰ 渀甀氀氀瀀琀爀Ⰰ 渀甀氀氀瀀琀爀紀⤀㬀ഀഀ
        offset += segment_size;਍      紀ഀഀ
    } else {਍      戀椀渀搀开最爀漀甀瀀开攀渀琀爀椀攀猀⸀瀀甀猀栀开戀愀挀欀⠀笀渀甀氀氀瀀琀爀Ⰰ 攀渀琀爀礀开椀渀搀攀砀⬀⬀Ⰰ 戀甀昀昀攀爀Ⰰ 　Ⰰ 圀䜀倀唀开圀䠀伀䰀䔀开匀䤀娀䔀Ⰰ 渀甀氀氀瀀琀爀Ⰰ 渀甀氀氀瀀琀爀紀⤀㬀ഀഀ
    }਍  紀ഀഀ
਍  伀刀吀开䔀一䘀伀刀䌀䔀⠀攀渀琀爀礀开椀渀搀攀砀 㰀 搀攀瘀椀挀攀开氀椀洀椀琀猀开⸀洀愀砀䈀椀渀搀椀渀最猀倀攀爀䈀椀渀搀䜀爀漀甀瀀Ⰰ ∀一甀洀戀攀爀 漀昀 戀椀渀搀 最爀漀甀瀀 攀渀琀爀椀攀猀 ⠀∀Ⰰ 攀渀琀爀礀开椀渀搀攀砀Ⰰഀഀ
              ") exceeds device limit (", device_limits_.maxBindingsPerBindGroup, ").");਍ഀഀ
  WGPUBindGroupLayout bind_group_layout = program_artifact.compute_pipeline.GetBindGroupLayout(0).MoveToCHandle();਍  圀䜀倀唀䈀椀渀搀䜀爀漀甀瀀䐀攀猀挀爀椀瀀琀漀爀 戀椀渀搀开最爀漀甀瀀开搀攀猀挀笀紀㬀ഀഀ
  bind_group_desc.layout = bind_group_layout;਍  戀椀渀搀开最爀漀甀瀀开搀攀猀挀⸀攀渀琀爀礀䌀漀甀渀琀 㴀 戀椀渀搀开最爀漀甀瀀开攀渀琀爀椀攀猀⸀猀椀稀攀⠀⤀㬀ഀഀ
  bind_group_desc.entries = bind_group_entries.data();਍  戀椀渀搀开最爀漀甀瀀开搀攀猀挀⸀氀愀戀攀氀 㴀 笀瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀⸀渀愀洀攀⸀搀愀琀愀⠀⤀Ⰰ 瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀⸀渀愀洀攀⸀氀攀渀最琀栀⠀⤀紀㬀ഀഀ
਍  愀甀琀漀 戀椀渀搀开最爀漀甀瀀 㴀 眀最瀀甀䐀攀瘀椀挀攀䌀爀攀愀琀攀䈀椀渀搀䜀爀漀甀瀀⠀䐀攀瘀椀挀攀⠀⤀⸀䜀攀琀⠀⤀Ⰰ ☀戀椀渀搀开最爀漀甀瀀开搀攀猀挀⤀㬀ഀഀ
  if (graph_capture_state_ == GraphCaptureState::Capturing) {਍    圀䜀倀唀䈀甀昀昀攀爀 椀渀搀椀爀攀挀琀开戀甀昀昀攀爀 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
    if (indirect_dispatch_tensor != nullptr) {਍      椀渀搀椀爀攀挀琀开戀甀昀昀攀爀 㴀 爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀圀䜀倀唀䈀甀昀昀攀爀㸀⠀挀漀渀猀琀开挀愀猀琀㰀瘀漀椀搀⨀㸀⠀椀渀搀椀爀攀挀琀开搀椀猀瀀愀琀挀栀开琀攀渀猀漀爀ⴀ㸀䐀愀琀愀刀愀眀⠀⤀⤀⤀㬀ഀഀ
    }਍ഀഀ
    // Profiling data will be populated in Run() after this call returns.਍    攀砀琀攀爀渀愀氀开挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀开ⴀ㸀瀀甀猀栀开戀愀挀欀⠀笀瀀爀漀最爀愀洀开愀爀琀椀昀愀挀琀⸀挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀Ⰰഀഀ
                                            bind_group,਍                                            戀椀渀搀开最爀漀甀瀀开氀愀礀漀甀琀Ⰰഀഀ
                                            {x, y, z},਍                                            椀渀搀椀爀攀挀琀开戀甀昀昀攀爀Ⰰഀഀ
                                            std::nullopt});਍  紀 攀氀猀攀 笀ഀഀ
    compute_pass_encoder.SetPipeline(program_artifact.compute_pipeline);਍    眀最瀀甀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀匀攀琀䈀椀渀搀䜀爀漀甀瀀⠀挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀⸀䜀攀琀⠀⤀Ⰰ 　Ⰰ 戀椀渀搀开最爀漀甀瀀Ⰰ 　Ⰰ 渀甀氀氀瀀琀爀⤀㬀ഀഀ
਍    椀昀 ⠀椀渀搀椀爀攀挀琀开搀椀猀瀀愀琀挀栀开琀攀渀猀漀爀 ℀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      // Use indirect dispatch਍      圀䜀倀唀䈀甀昀昀攀爀 椀渀搀椀爀攀挀琀开戀甀昀昀攀爀 㴀 爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀圀䜀倀唀䈀甀昀昀攀爀㸀⠀挀漀渀猀琀开挀愀猀琀㰀瘀漀椀搀⨀㸀⠀椀渀搀椀爀攀挀琀开搀椀猀瀀愀琀挀栀开琀攀渀猀漀爀ⴀ㸀䐀愀琀愀刀愀眀⠀⤀⤀⤀㬀ഀഀ
      compute_pass_encoder.DispatchWorkgroupsIndirect(indirect_buffer, 0);਍    紀 攀氀猀攀 笀ഀഀ
      // Use direct dispatch਍      挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀⸀䐀椀猀瀀愀琀挀栀圀漀爀欀最爀漀甀瀀猀⠀砀Ⰰ 礀Ⰰ 稀⤀㬀ഀഀ
    }਍ഀഀ
    wgpuBindGroupRelease(bind_group);਍    眀最瀀甀䈀椀渀搀䜀爀漀甀瀀䰀愀礀漀甀琀刀攀氀攀愀猀攀⠀戀椀渀搀开最爀漀甀瀀开氀愀礀漀甀琀⤀㬀ഀഀ
  }਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀䌀愀瀀琀甀爀攀䈀攀最椀渀⠀猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀攀戀最瀀甀㨀㨀䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀㸀⨀ 挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀Ⰰ 挀漀渀猀琀 眀攀戀最瀀甀㨀㨀䈀甀昀昀攀爀䴀愀渀愀最攀爀☀ 戀甀昀昀攀爀开洀愀渀愀最攀爀⤀ 笀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "CaptureBegin with external storage";਍  ⼀⼀ 䘀氀甀猀栀 愀渀礀 瀀攀渀搀椀渀最 挀漀洀洀愀渀搀猀 戀攀昀漀爀攀 眀攀 挀栀愀渀最攀 琀栀攀 猀琀愀琀甀猀ഀഀ
  Flush(buffer_manager);਍ഀഀ
  external_captured_commands_ = captured_commands;਍ഀഀ
  // Make sure the external vector is empty before we start capturing਍  椀昀 ⠀攀砀琀攀爀渀愀氀开挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀开⤀ 笀ഀഀ
    external_captured_commands_->clear();਍  紀ഀഀ
਍  最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀愀琀攀开 㴀 䜀爀愀瀀栀䌀愀瀀琀甀爀攀匀琀愀琀攀㨀㨀䌀愀瀀琀甀爀椀渀最㬀ഀഀ
}਍ഀഀ
void WebGpuContext::Replay(const std::vector<webgpu::CapturedCommandInfo>& captured_commands, const webgpu::BufferManager& buffer_manager) {਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀刀攀瀀氀愀礀 眀椀琀栀 攀砀琀攀爀渀愀氀 猀琀漀爀愀最攀∀㬀ഀഀ
  graph_capture_state_ = GraphCaptureState::Replaying;਍  ⼀⼀ 刀攀瀀氀愀礀 愀氀氀 挀愀瀀琀甀爀攀搀 挀漀洀洀愀渀搀猀 昀爀漀洀 琀栀攀 瀀爀漀瘀椀搀攀搀 瘀攀挀琀漀爀ഀഀ
  const size_t command_count = captured_commands.size();਍  昀漀爀 ⠀猀椀稀攀开琀 椀 㴀 　㬀 椀 㰀 挀漀洀洀愀渀搀开挀漀甀渀琀㬀 ⬀⬀椀⤀ 笀ഀഀ
    auto& command = captured_commands[i];਍    挀漀渀猀琀 愀甀琀漀☀ 挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀 㴀 䜀攀琀䌀漀洀瀀甀琀攀倀愀猀猀䔀渀挀漀搀攀爀⠀⤀㬀ഀഀ
    WriteTimestamp(num_pending_dispatches_ * 2);਍ഀഀ
    // Restore profiling info when profiling is enabled. All commands are expected਍    ⼀⼀ 琀漀 栀愀瘀攀 瀀爀漀昀椀氀椀渀最 搀愀琀愀 椀渀 琀栀椀猀 洀漀搀攀 琀漀 欀攀攀瀀 瀀攀渀搀椀渀最开欀攀爀渀攀氀猀开 挀漀渀猀椀猀琀攀渀琀ഀഀ
    // with num_pending_dispatches_.਍    椀昀 ⠀椀猀开瀀爀漀昀椀氀椀渀最开⤀ 笀ഀഀ
      ORT_ENFORCE(command.pending_kernel_info.has_value(),਍                  ∀圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀刀攀瀀氀愀礀㨀 瀀爀漀昀椀氀椀渀最 椀猀 攀渀愀戀氀攀搀 戀甀琀 挀愀瀀琀甀爀攀搀 挀漀洀洀愀渀搀 愀琀 椀渀搀攀砀 ∀Ⰰഀഀ
                  i,਍                  ∀ 椀猀 洀椀猀猀椀渀最 瀀攀渀搀椀渀最开欀攀爀渀攀氀开椀渀昀漀⸀∀⤀㬀ഀഀ
      pending_kernels_.emplace_back(*command.pending_kernel_info);਍    紀ഀഀ
਍    挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀⸀匀攀琀倀椀瀀攀氀椀渀攀⠀挀漀洀洀愀渀搀⸀挀漀洀瀀甀琀攀开瀀椀瀀攀氀椀渀攀⤀㬀ഀഀ
    wgpuComputePassEncoderSetBindGroup(compute_pass_encoder.Get(), 0, command.bind_group, 0, nullptr);਍ഀഀ
    if (command.indirect_buffer != nullptr) {਍      ⼀⼀ 唀猀攀 椀渀搀椀爀攀挀琀 搀椀猀瀀愀琀挀栀ഀഀ
      compute_pass_encoder.DispatchWorkgroupsIndirect(command.indirect_buffer, 0);਍    紀 攀氀猀攀 笀ഀഀ
      // Use direct dispatch਍      挀漀洀瀀甀琀攀开瀀愀猀猀开攀渀挀漀搀攀爀⸀䐀椀猀瀀愀琀挀栀圀漀爀欀最爀漀甀瀀猀⠀挀漀洀洀愀渀搀⸀搀椀猀瀀愀琀挀栀开最爀漀甀瀀嬀　崀Ⰰ 挀漀洀洀愀渀搀⸀搀椀猀瀀愀琀挀栀开最爀漀甀瀀嬀㄀崀Ⰰ 挀漀洀洀愀渀搀⸀搀椀猀瀀愀琀挀栀开最爀漀甀瀀嬀㈀崀⤀㬀ഀഀ
    }਍ഀഀ
    WriteTimestamp(num_pending_dispatches_ * 2 + 1);਍    ⬀⬀渀甀洀开瀀攀渀搀椀渀最开搀椀猀瀀愀琀挀栀攀猀开㬀ഀഀ
    if (num_pending_dispatches_ >= max_num_pending_dispatches_ ||਍        ⠀椀猀开瀀爀漀昀椀氀椀渀最开 ☀☀ 焀甀攀爀礀开琀礀瀀攀开 㴀㴀 吀椀洀攀猀琀愀洀瀀儀甀攀爀礀吀礀瀀攀㨀㨀䄀琀倀愀猀猀攀猀⤀⤀ 笀ഀഀ
      EndComputePass();਍    紀ഀഀ
    if (num_pending_dispatches_ >= max_num_pending_dispatches_) {਍      䘀氀甀猀栀⠀戀甀昀昀攀爀开洀愀渀愀最攀爀⤀㬀ഀഀ
      num_pending_dispatches_ = 0;਍    紀ഀഀ
  }਍ഀഀ
  // Flush any remaining commands਍  䘀氀甀猀栀⠀戀甀昀昀攀爀开洀愀渀愀最攀爀⤀㬀ഀഀ
਍  最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀愀琀攀开 㴀 䜀爀愀瀀栀䌀愀瀀琀甀爀攀匀琀愀琀攀㨀㨀䐀攀昀愀甀氀琀㬀ഀഀ
}਍ഀഀ
void WebGpuContext::CaptureEnd() {਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀䌀愀瀀琀甀爀攀䔀渀搀∀㬀ഀഀ
਍  最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀愀琀攀开 㴀 䜀爀愀瀀栀䌀愀瀀琀甀爀攀匀琀愀琀攀㨀㨀䐀攀昀愀甀氀琀㬀ഀഀ
  external_captured_commands_ = nullptr;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㨀㨀刀攀氀攀愀猀攀䜀爀愀瀀栀刀攀猀漀甀爀挀攀猀⠀猀琀搀㨀㨀瘀攀挀琀漀爀㰀眀攀戀最瀀甀㨀㨀䌀愀瀀琀甀爀攀搀䌀漀洀洀愀渀搀䤀渀昀漀㸀☀ 挀愀瀀琀甀爀攀搀开挀漀洀洀愀渀搀猀⤀ 笀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "ReleaseGraphResources: Releasing " << captured_commands.size() << " captured command resources";਍ഀഀ
  for (auto& command : captured_commands) {਍    椀昀 ⠀挀漀洀洀愀渀搀⸀戀椀渀搀开最爀漀甀瀀 ℀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      wgpuBindGroupRelease(command.bind_group);਍      挀漀洀洀愀渀搀⸀戀椀渀搀开最爀漀甀瀀 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
    }਍ഀഀ
    if (command.bind_group_layout != nullptr) {਍      眀最瀀甀䈀椀渀搀䜀爀漀甀瀀䰀愀礀漀甀琀刀攀氀攀愀猀攀⠀挀漀洀洀愀渀搀⸀戀椀渀搀开最爀漀甀瀀开氀愀礀漀甀琀⤀㬀ഀഀ
      command.bind_group_layout = nullptr;਍    紀ഀഀ
  }਍紀ഀഀ
਍猀琀搀㨀㨀洀甀琀攀砀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀洀甀琀攀砀开㬀ഀഀ
std::once_flag WebGpuContextFactory::init_default_flag_;਍ഀഀ
std::unordered_map<int32_t, WebGpuContextFactory::WebGpuContextInfo>* WebGpuContextFactory::contexts_ = nullptr;਍圀䜀倀唀䤀渀猀琀愀渀挀攀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀搀攀昀愀甀氀琀开椀渀猀琀愀渀挀攀开 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
਍圀攀戀䜀瀀甀䌀漀渀琀攀砀琀☀ 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䌀爀攀愀琀攀䌀漀渀琀攀砀琀⠀挀漀渀猀琀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䌀漀渀昀椀最☀ 挀漀渀昀椀最⤀ 笀ഀഀ
  const int context_id = config.context_id;਍  圀䜀倀唀䤀渀猀琀愀渀挀攀 椀渀猀琀愀渀挀攀 㴀 挀漀渀昀椀最⸀椀渀猀琀愀渀挀攀㬀ഀഀ
  WGPUDevice device = config.device;਍ഀഀ
  std::call_once(init_default_flag_, [਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
                                         dawn_proc_table = config.dawn_proc_table਍⌀攀渀搀椀昀ഀഀ
  ]() {਍  ⼀⼀ 匀攀琀甀瀀 搀愀眀渀 瀀爀漀挀 琀愀戀氀攀 ⠀漀渀氀礀 昀漀爀 渀漀渀ⴀ圀䄀匀䴀 戀甀椀氀搀⤀ഀഀ
਍⌀椀昀 ℀搀攀昀椀渀攀搀⠀开开眀愀猀洀开开⤀ഀഀ
    const DawnProcTable* dawn_procs = reinterpret_cast<const DawnProcTable*>(dawn_proc_table);਍⌀椀昀 搀攀昀椀渀攀搀⠀䈀唀䤀䰀䐀开䐀䄀圀一开匀䠀䄀刀䔀䐀开䰀䤀䈀刀䄀刀夀⤀ഀഀ
    ORT_ENFORCE(dawn_procs == nullptr, "setting DawnProcTable is not allowed when dynamically linked to webgpu_dawn.");਍⌀攀氀猀攀ഀഀ
#if !defined(USE_EXTERNAL_DAWN)਍    椀昀 ⠀搀愀眀渀开瀀爀漀挀猀 㴀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      dawn_procs = &dawn::native::GetProcs();਍    紀ഀഀ
#else਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀搀愀眀渀开瀀爀漀挀猀 ℀㴀 渀甀氀氀瀀琀爀Ⰰ ∀䐀愀眀渀倀爀漀挀吀愀戀氀攀 洀甀猀琀 戀攀 瀀爀漀瘀椀搀攀搀⸀∀⤀㬀ഀഀ
#endif਍    搀愀眀渀倀爀漀挀匀攀琀倀爀漀挀猀⠀搀愀眀渀开瀀爀漀挀猀⤀㬀ഀഀ
#endif਍⌀攀渀搀椀昀ഀഀ
  });਍ഀഀ
  std::lock_guard<std::mutex> lock(mutex_);਍ഀഀ
  if (default_instance_ == nullptr) {਍    ⼀⼀ 䌀爀攀愀琀攀 眀最瀀甀㨀㨀䤀渀猀琀愀渀挀攀ഀഀ
    wgpu::InstanceFeatureName required_instance_features[] = {wgpu::InstanceFeatureName::TimedWaitAny};਍    眀最瀀甀㨀㨀䤀渀猀琀愀渀挀攀䐀攀猀挀爀椀瀀琀漀爀 椀渀猀琀愀渀挀攀开搀攀猀挀笀紀㬀ഀഀ
    instance_desc.requiredFeatures = required_instance_features;਍    椀渀猀琀愀渀挀攀开搀攀猀挀⸀爀攀焀甀椀爀攀搀䘀攀愀琀甀爀攀䌀漀甀渀琀 㴀 猀椀稀攀漀昀⠀爀攀焀甀椀爀攀搀开椀渀猀琀愀渀挀攀开昀攀愀琀甀爀攀猀⤀ ⼀ 猀椀稀攀漀昀⠀爀攀焀甀椀爀攀搀开椀渀猀琀愀渀挀攀开昀攀愀琀甀爀攀猀嬀　崀⤀㬀ഀഀ
    default_instance_ = wgpu::CreateInstance(&instance_desc).MoveToCHandle();਍ഀഀ
    ORT_ENFORCE(default_instance_ != nullptr, "Failed to create wgpu::Instance.");਍  紀ഀഀ
਍  椀昀 ⠀挀漀渀琀攀砀琀开椀搀 㴀㴀 　⤀ 笀ഀഀ
    // context ID is preserved for the default context. User cannot use context ID 0 as a custom context.਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀椀渀猀琀愀渀挀攀 㴀㴀 渀甀氀氀瀀琀爀 ☀☀ 搀攀瘀椀挀攀 㴀㴀 渀甀氀氀瀀琀爀Ⰰഀഀ
                "WebGPU EP default context (contextId=0) must not have custom WebGPU instance or device.");਍ഀഀ
    instance = default_instance_;਍  紀 攀氀猀攀 笀ഀഀ
    // for context ID > 0, user must provide custom WebGPU instance and device.਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀椀渀猀琀愀渀挀攀 ℀㴀 渀甀氀氀瀀琀爀 ☀☀ 搀攀瘀椀挀攀 ℀㴀 渀甀氀氀瀀琀爀Ⰰഀഀ
                "WebGPU EP custom context (contextId>0) must have custom WebGPU instance and device.");਍  紀ഀഀ
਍  ⼀⼀ 䰀愀稀礀ⴀ愀氀氀漀挀愀琀攀 琀栀攀 挀漀渀琀攀砀琀猀 洀愀瀀 漀渀 昀椀爀猀琀 甀猀攀 ⠀栀攀愀瀀ⴀ愀氀氀漀挀愀琀攀搀 琀漀 愀瘀漀椀搀 猀琀愀琀椀挀 搀攀猀琀爀甀挀琀椀漀渀 挀爀愀猀栀⤀⸀ഀഀ
  if (contexts_ == nullptr) {਍    挀漀渀琀攀砀琀猀开 㴀 渀攀眀 猀琀搀㨀㨀甀渀漀爀搀攀爀攀搀开洀愀瀀㰀椀渀琀㌀㈀开琀Ⰰ 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䤀渀昀漀㸀⠀⤀㬀ഀഀ
  }਍ഀഀ
  auto it = contexts_->find(context_id);਍  椀昀 ⠀椀琀 㴀㴀 挀漀渀琀攀砀琀猀开ⴀ㸀攀渀搀⠀⤀⤀ 笀ഀഀ
    GSL_SUPPRESS(r.11)਍    愀甀琀漀 挀漀渀琀攀砀琀 㴀 猀琀搀㨀㨀甀渀椀焀甀攀开瀀琀爀㰀圀攀戀䜀瀀甀䌀漀渀琀攀砀琀㸀⠀渀攀眀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀⠀椀渀猀琀愀渀挀攀Ⰰഀഀ
                                                                    device,਍                                                                    挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀Ⰰഀഀ
                                                                    config.validation_mode_explicitly_set,਍                                                                    挀漀渀昀椀最⸀瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀Ⰰഀഀ
                                                                    config.max_storage_buffer_binding_size));਍    椀琀 㴀 挀漀渀琀攀砀琀猀开ⴀ㸀攀洀瀀氀愀挀攀⠀挀漀渀琀攀砀琀开椀搀Ⰰ 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䤀渀昀漀笀猀琀搀㨀㨀洀漀瘀攀⠀挀漀渀琀攀砀琀⤀Ⰰ 　紀⤀⸀昀椀爀猀琀㬀ഀഀ
  } else if (context_id != 0) {਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀椀琀ⴀ㸀猀攀挀漀渀搀⸀挀漀渀琀攀砀琀ⴀ㸀椀渀猀琀愀渀挀攀开⸀䜀攀琀⠀⤀ 㴀㴀 椀渀猀琀愀渀挀攀 ☀☀ഀഀ
                    it->second.context->device_.Get() == device,਍                ∀圀攀戀䜀倀唀 䔀倀 挀漀渀琀攀砀琀 䤀䐀 ∀Ⰰ 挀漀渀琀攀砀琀开椀搀Ⰰ ∀ 椀猀 愀氀爀攀愀搀礀 挀爀攀愀琀攀搀 眀椀琀栀 搀椀昀昀攀爀攀渀琀 圀攀戀䜀倀唀 椀渀猀琀愀渀挀攀 漀爀 搀攀瘀椀挀攀⸀∀⤀㬀ഀഀ
  }਍  椀琀ⴀ㸀猀攀挀漀渀搀⸀爀攀昀开挀漀甀渀琀⬀⬀㬀ഀഀ
਍  ⼀⼀ 瀀攀爀昀漀爀洀 椀渀椀琀椀愀氀椀稀愀琀椀漀渀ഀഀ
  it->second.context->Initialize(config);਍ഀഀ
  return *it->second.context;਍紀ഀഀ
਍圀攀戀䜀瀀甀䌀漀渀琀攀砀琀☀ 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䜀攀琀䌀漀渀琀攀砀琀⠀椀渀琀 挀漀渀琀攀砀琀开椀搀⤀ 笀ഀഀ
  std::lock_guard<std::mutex> lock(mutex_);਍ഀഀ
  ORT_ENFORCE(contexts_ != nullptr, "WebGPU contexts have not been initialized or have been cleaned up.");਍  愀甀琀漀 椀琀 㴀 挀漀渀琀攀砀琀猀开ⴀ㸀昀椀渀搀⠀挀漀渀琀攀砀琀开椀搀⤀㬀ഀഀ
  ORT_ENFORCE(it != contexts_->end(), "WebGPU EP context ID ", context_id, " is not found.");਍ഀഀ
  return *it->second.context;਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀刀攀氀攀愀猀攀䌀漀渀琀攀砀琀⠀椀渀琀 挀漀渀琀攀砀琀开椀搀⤀ 笀ഀഀ
  std::lock_guard<std::mutex> lock(mutex_);਍ഀഀ
  ORT_ENFORCE(contexts_ != nullptr, "WebGPU contexts have not been initialized or have been cleaned up.");਍  愀甀琀漀 椀琀 㴀 挀漀渀琀攀砀琀猀开ⴀ㸀昀椀渀搀⠀挀漀渀琀攀砀琀开椀搀⤀㬀ഀഀ
  ORT_ENFORCE(it != contexts_->end(), "WebGPU EP context ID ", context_id, " is not found.");਍ഀഀ
  if (--it->second.ref_count == 0 && !it->second.context->preserve_device_) {਍    挀漀渀琀攀砀琀猀开ⴀ㸀攀爀愀猀攀⠀椀琀⤀㬀ഀഀ
  }਍紀ഀഀ
਍瘀漀椀搀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䌀氀攀愀渀甀瀀⠀⤀ 笀ഀഀ
  std::lock_guard<std::mutex> lock(mutex_);਍ഀഀ
  if (contexts_ != nullptr) {਍    搀攀氀攀琀攀 挀漀渀琀攀砀琀猀开㬀ഀഀ
    contexts_ = nullptr;਍  紀ഀഀ
਍  椀昀 ⠀搀攀昀愀甀氀琀开椀渀猀琀愀渀挀攀开 ℀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
    wgpuInstanceRelease(default_instance_);਍    搀攀昀愀甀氀琀开椀渀猀琀愀渀挀攀开 㴀 渀甀氀氀瀀琀爀㬀ഀഀ
  }਍紀ഀഀ
਍圀攀戀䜀瀀甀䌀漀渀琀攀砀琀☀ 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䐀攀昀愀甀氀琀䌀漀渀琀攀砀琀⠀⤀ 笀ഀഀ
  WebGpuContextConfig config{};਍  爀攀琀甀爀渀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䌀爀攀愀琀攀䌀漀渀琀攀砀琀⠀挀漀渀昀椀最⤀㬀ഀഀ
}਍ഀഀ
void CleanupWebGpuContexts() {਍  圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䌀氀攀愀渀甀瀀⠀⤀㬀ഀഀ
}਍ഀഀ
WGPUDevice GetDevice(int context_id) {਍  爀攀琀甀爀渀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䜀攀琀䌀漀渀琀攀砀琀⠀挀漀渀琀攀砀琀开椀搀⤀⸀䐀攀瘀椀挀攀⠀⤀⸀䜀攀琀⠀⤀㬀ഀഀ
}਍ഀഀ
}  // namespace webgpu਍紀  ⼀⼀ 渀愀洀攀猀瀀愀挀攀 漀渀渀砀爀甀渀琀椀洀攀ഀഀ
