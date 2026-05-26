// Copyright (c) Microsoft Corporation. All rights reserved.਍⼀⼀ 䰀椀挀攀渀猀攀搀 甀渀搀攀爀 琀栀攀 䴀䤀吀 䰀椀挀攀渀猀攀⸀ഀഀ
਍⌀椀渀挀氀甀搀攀 㰀挀栀愀爀挀漀渀瘀㸀ഀഀ
#include <mutex>਍ഀഀ
#include "core/framework/error_code_helper.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀戀甀昀昀攀爀开洀愀渀愀最攀爀⸀栀∀ഀഀ
#include "core/providers/webgpu/webgpu_execution_provider.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀眀攀戀最瀀甀开瀀爀漀瘀椀搀攀爀开昀愀挀琀漀爀礀开挀爀攀愀琀漀爀⸀栀∀ഀഀ
#include "core/providers/webgpu/webgpu_context.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀猀攀猀猀椀漀渀⼀愀戀椀开猀攀猀猀椀漀渀开漀瀀琀椀漀渀猀开椀洀瀀氀⸀栀∀ഀഀ
#include "core/session/ort_apis.h"਍ഀഀ
#include "core/providers/webgpu/webgpu_provider_options.h"਍⌀椀渀挀氀甀搀攀 ∀挀漀爀攀⼀瀀爀漀瘀椀搀攀爀猀⼀眀攀戀最瀀甀⼀搀愀琀愀开琀爀愀渀猀昀攀爀⸀栀∀ഀഀ
਍甀猀椀渀最 渀愀洀攀猀瀀愀挀攀 漀渀渀砀爀甀渀琀椀洀攀㨀㨀眀攀戀最瀀甀㬀ഀഀ
using namespace onnxruntime::webgpu::options;਍ഀഀ
namespace onnxruntime {਍猀琀爀甀挀琀 圀攀戀䜀瀀甀倀爀漀瘀椀搀攀爀䘀愀挀琀漀爀礀 㨀 䤀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀䘀愀挀琀漀爀礀 笀ഀഀ
  WebGpuProviderFactory(int context_id, WebGpuContext& context, WebGpuExecutionProviderConfig&& webgpu_ep_config)਍      㨀 挀漀渀琀攀砀琀开椀搀开笀挀漀渀琀攀砀琀开椀搀紀Ⰰ 挀漀渀琀攀砀琀开笀挀漀渀琀攀砀琀紀Ⰰ 挀漀渀昀椀最开笀猀琀搀㨀㨀洀漀瘀攀⠀眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⤀紀 笀ഀഀ
  }਍ഀഀ
  std::unique_ptr<IExecutionProvider> CreateProvider() override {਍    爀攀琀甀爀渀 猀琀搀㨀㨀洀愀欀攀开甀渀椀焀甀攀㰀圀攀戀䜀瀀甀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀㸀⠀挀漀渀琀攀砀琀开椀搀开Ⰰ 挀漀渀琀攀砀琀开Ⰰ 猀琀搀㨀㨀洀漀瘀攀⠀挀漀渀昀椀最开⤀⤀㬀ഀഀ
  }਍ഀഀ
 private:਍  椀渀琀 挀漀渀琀攀砀琀开椀搀开㬀ഀഀ
  WebGpuContext& context_;਍  圀攀戀䜀瀀甀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀䌀漀渀昀椀最 挀漀渀昀椀最开㬀ഀഀ
};਍ഀഀ
namespace {਍ഀഀ
WebGpuExecutionProviderConfig ParseEpConfig(const ConfigOptions& config_options) {਍  圀攀戀䜀瀀甀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀䌀漀渀昀椀最 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最笀紀㬀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 瀀爀攀昀攀爀爀攀搀开氀愀礀漀甀琀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kPreferredLayout, preferred_layout_str)) {਍    椀昀 ⠀瀀爀攀昀攀爀爀攀搀开氀愀礀漀甀琀开猀琀爀 㴀㴀 欀倀爀攀昀攀爀爀攀搀䰀愀礀漀甀琀开一䠀圀䌀⤀ 笀ഀഀ
      webgpu_ep_config.data_layout = DataLayout::NHWC;਍    紀 攀氀猀攀 椀昀 ⠀瀀爀攀昀攀爀爀攀搀开氀愀礀漀甀琀开猀琀爀 㴀㴀 欀倀爀攀昀攀爀爀攀搀䰀愀礀漀甀琀开一䌀䠀圀⤀ 笀ഀഀ
      webgpu_ep_config.data_layout = DataLayout::NCHW;਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid preferred layout: ", preferred_layout_str);਍    紀ഀഀ
  }਍ഀഀ
  if (std::string enable_graph_capture_str;਍      挀漀渀昀椀最开漀瀀琀椀漀渀猀⸀吀爀礀䜀攀琀䌀漀渀昀椀最䔀渀琀爀礀⠀欀䔀渀愀戀氀攀䜀爀愀瀀栀䌀愀瀀琀甀爀攀Ⰰ 攀渀愀戀氀攀开最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀爀⤀⤀ 笀ഀഀ
    if (enable_graph_capture_str == kEnableGraphCapture_ON) {਍      眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀攀渀愀戀氀攀开最爀愀瀀栀开挀愀瀀琀甀爀攀 㴀 琀爀甀攀㬀ഀഀ
    } else if (enable_graph_capture_str == kEnableGraphCapture_OFF) {਍      眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀攀渀愀戀氀攀开最爀愀瀀栀开挀愀瀀琀甀爀攀 㴀 昀愀氀猀攀㬀ഀഀ
    } else {਍      伀刀吀开吀䠀刀伀圀⠀∀䤀渀瘀愀氀椀搀 攀渀愀戀氀攀 最爀愀瀀栀 挀愀瀀琀甀爀攀㨀 ∀Ⰰ 攀渀愀戀氀攀开最爀愀瀀栀开挀愀瀀琀甀爀攀开猀琀爀⤀㬀ഀഀ
    }਍  紀ഀഀ
਍  猀琀搀㨀㨀猀琀爀椀渀最 攀渀愀戀氀攀开椀渀琀㘀㐀开猀琀爀㬀ഀഀ
  if (config_options.TryGetConfigEntry(kEnableInt64, enable_int64_str)) {਍    椀昀 ⠀攀渀愀戀氀攀开椀渀琀㘀㐀开猀琀爀 㴀㴀 欀䔀渀愀戀氀攀䤀渀琀㘀㐀开伀一⤀ 笀ഀഀ
      webgpu_ep_config.enable_int64 = true;਍    紀 攀氀猀攀 椀昀 ⠀攀渀愀戀氀攀开椀渀琀㘀㐀开猀琀爀 㴀㴀 欀䔀渀愀戀氀攀䤀渀琀㘀㐀开伀䘀䘀⤀ 笀ഀഀ
      webgpu_ep_config.enable_int64 = false;਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid enableInt64 value: ", enable_int64_str);਍    紀ഀഀ
  }਍ഀഀ
  std::string multi_rotary_cache_concat_offset_str;਍  椀昀 ⠀挀漀渀昀椀最开漀瀀琀椀漀渀猀⸀吀爀礀䜀攀琀䌀漀渀昀椀最䔀渀琀爀礀⠀欀䴀甀氀琀椀刀漀琀愀爀礀䌀愀挀栀攀䌀漀渀挀愀琀伀昀昀猀攀琀Ⰰ 洀甀氀琀椀开爀漀琀愀爀礀开挀愀挀栀攀开挀漀渀挀愀琀开漀昀昀猀攀琀开猀琀爀⤀⤀ 笀ഀഀ
    uint32_t offset_value = 0;਍    愀甀琀漀 爀攀猀甀氀琀 㴀 猀琀搀㨀㨀昀爀漀洀开挀栀愀爀猀⠀洀甀氀琀椀开爀漀琀愀爀礀开挀愀挀栀攀开挀漀渀挀愀琀开漀昀昀猀攀琀开猀琀爀⸀搀愀琀愀⠀⤀Ⰰഀഀ
                                  multi_rotary_cache_concat_offset_str.data() + multi_rotary_cache_concat_offset_str.size(),਍                                  漀昀昀猀攀琀开瘀愀氀甀攀⤀㬀ഀഀ
    if (result.ec == std::errc{}) {਍      眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀洀甀氀琀椀开爀漀琀愀爀礀开挀愀挀栀攀开挀漀渀挀愀琀开漀昀昀猀攀琀 㴀 漀昀昀猀攀琀开瘀愀氀甀攀㬀ഀഀ
    } else {਍      伀刀吀开吀䠀刀伀圀⠀∀䤀渀瘀愀氀椀搀 洀甀氀琀椀刀漀琀愀爀礀䌀愀挀栀攀䌀漀渀挀愀琀伀昀昀猀攀琀 瘀愀氀甀攀㨀 ∀Ⰰ 洀甀氀琀椀开爀漀琀愀爀礀开挀愀挀栀攀开挀漀渀挀愀琀开漀昀昀猀攀琀开猀琀爀Ⰰ ∀⸀ 䴀甀猀琀 戀攀 愀 渀漀渀ⴀ渀攀最愀琀椀瘀攀 椀渀琀攀最攀爀⸀∀⤀㬀ഀഀ
    }਍  紀ഀഀ
਍  ⼀⼀ 瀀愀爀猀攀 昀漀爀挀攀 䌀倀唀 渀漀搀攀 渀愀洀攀猀ഀഀ
  // The force CPU node names are separated by EOL (\n or \r\n) in the config entry.਍  ⼀⼀ 攀愀挀栀 氀椀渀攀 椀猀 愀 渀漀搀攀 渀愀洀攀 琀栀愀琀 眀椀氀氀 戀攀 昀漀爀挀攀搀 琀漀 爀甀渀 漀渀 䌀倀唀⸀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 昀漀爀挀攀开挀瀀甀开渀漀搀攀开渀愀洀攀猀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kForceCpuNodeNames, force_cpu_node_names_str)) {਍    ⼀⼀ 猀瀀氀椀琀 琀栀攀 猀琀爀椀渀最 戀礀 䔀伀䰀 ⠀尀渀 漀爀 尀爀尀渀⤀ഀഀ
    std::istringstream ss(force_cpu_node_names_str);਍    猀琀搀㨀㨀猀琀爀椀渀最 氀椀渀攀㬀ഀഀ
    while (std::getline(ss, line)) {਍      ⼀⼀ 猀欀椀瀀 攀洀瀀琀礀 氀椀渀攀猀ഀഀ
      if (line.empty()) {਍        挀漀渀琀椀渀甀攀㬀ഀഀ
      }਍ഀഀ
      webgpu_ep_config.force_cpu_node_names.push_back(line);਍    紀ഀഀ
  }਍ഀഀ
  // enable pix capture਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 攀渀愀戀氀攀开瀀椀砀开挀愀瀀琀甀爀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kEnablePIXCapture, enable_pix_capture_str)) {਍    椀昀 ⠀攀渀愀戀氀攀开瀀椀砀开挀愀瀀琀甀爀攀开猀琀爀 㴀㴀 欀䔀渀愀戀氀攀倀䤀堀䌀愀瀀琀甀爀攀开伀一⤀ 笀ഀഀ
      webgpu_ep_config.enable_pix_capture = true;਍    紀 攀氀猀攀 椀昀 ⠀攀渀愀戀氀攀开瀀椀砀开挀愀瀀琀甀爀攀开猀琀爀 㴀㴀 欀䔀渀愀戀氀攀倀䤀堀䌀愀瀀琀甀爀攀开伀䘀䘀⤀ 笀ഀഀ
      webgpu_ep_config.enable_pix_capture = false;਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid enable pix capture: ", enable_pix_capture_str);਍    紀ഀഀ
  }਍ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP preferred layout: " << int(webgpu_ep_config.data_layout);਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 最爀愀瀀栀 挀愀瀀琀甀爀攀 攀渀愀戀氀攀㨀 ∀ 㰀㰀 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀攀渀愀戀氀攀开最爀愀瀀栀开挀愀瀀琀甀爀攀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP force CPU node count: " << webgpu_ep_config.force_cpu_node_names.size();਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 瀀椀砀 挀愀瀀琀甀爀攀 攀渀愀戀氀攀㨀 ∀ 㰀㰀 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀攀渀愀戀氀攀开瀀椀砀开挀愀瀀琀甀爀攀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP enable int64: " << webgpu_ep_config.enable_int64;਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 洀甀氀琀椀 爀漀琀愀爀礀 挀愀挀栀攀 挀漀渀挀愀琀 漀昀昀猀攀琀㨀 ∀ 㰀㰀 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最⸀洀甀氀琀椀开爀漀琀愀爀礀开挀愀挀栀攀开挀漀渀挀愀琀开漀昀昀猀攀琀㬀ഀഀ
਍  爀攀琀甀爀渀 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最㬀ഀഀ
}਍ഀഀ
WebGpuContextConfig ParseWebGpuContextConfig(const ConfigOptions& config_options) {਍  圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䌀漀渀昀椀最 挀漀渀昀椀最笀紀㬀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 挀漀渀琀攀砀琀开椀搀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kDeviceId, context_id_str)) {਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀猀琀搀㨀㨀攀爀爀挀笀紀 㴀㴀ഀഀ
                std::from_chars(context_id_str.data(), context_id_str.data() + context_id_str.size(), config.context_id).ec);਍  紀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 眀攀戀最瀀甀开椀渀猀琀愀渀挀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kWebGpuInstance, webgpu_instance_str)) {਍    猀琀愀琀椀挀开愀猀猀攀爀琀⠀猀椀稀攀漀昀⠀圀䜀倀唀䤀渀猀琀愀渀挀攀⤀ 㴀㴀 猀椀稀攀漀昀⠀猀椀稀攀开琀⤀Ⰰ ∀圀䜀倀唀䤀渀猀琀愀渀挀攀 猀椀稀攀 洀椀猀洀愀琀挀栀∀⤀㬀ഀഀ
    size_t webgpu_instance = 0;਍    伀刀吀开䔀一䘀伀刀䌀䔀⠀猀琀搀㨀㨀攀爀爀挀笀紀 㴀㴀ഀഀ
                std::from_chars(webgpu_instance_str.data(), webgpu_instance_str.data() + webgpu_instance_str.size(), webgpu_instance).ec);਍    挀漀渀昀椀最⸀椀渀猀琀愀渀挀攀 㴀 爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀圀䜀倀唀䤀渀猀琀愀渀挀攀㸀⠀眀攀戀最瀀甀开椀渀猀琀愀渀挀攀⤀㬀ഀഀ
  }਍ഀഀ
  if (std::string webgpu_device_str;਍      挀漀渀昀椀最开漀瀀琀椀漀渀猀⸀吀爀礀䜀攀琀䌀漀渀昀椀最䔀渀琀爀礀⠀欀圀攀戀䜀瀀甀䐀攀瘀椀挀攀Ⰰ 眀攀戀最瀀甀开搀攀瘀椀挀攀开猀琀爀⤀⤀ 笀ഀഀ
    static_assert(sizeof(WGPUDevice) == sizeof(size_t), "WGPUDevice size mismatch");਍    猀椀稀攀开琀 眀攀戀最瀀甀开搀攀瘀椀挀攀 㴀 　㬀ഀഀ
    ORT_ENFORCE(std::errc{} ==਍                猀琀搀㨀㨀昀爀漀洀开挀栀愀爀猀⠀眀攀戀最瀀甀开搀攀瘀椀挀攀开猀琀爀⸀搀愀琀愀⠀⤀Ⰰ 眀攀戀最瀀甀开搀攀瘀椀挀攀开猀琀爀⸀搀愀琀愀⠀⤀ ⬀ 眀攀戀最瀀甀开搀攀瘀椀挀攀开猀琀爀⸀猀椀稀攀⠀⤀Ⰰ 眀攀戀最瀀甀开搀攀瘀椀挀攀⤀⸀攀挀⤀㬀ഀഀ
    config.device = reinterpret_cast<WGPUDevice>(webgpu_device);਍  紀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kDawnProcTable, dawn_proc_table_str)) {਍    猀椀稀攀开琀 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀 㴀 　㬀ഀഀ
    ORT_ENFORCE(std::errc{} ==਍                猀琀搀㨀㨀昀爀漀洀开挀栀愀爀猀⠀搀愀眀渀开瀀爀漀挀开琀愀戀氀攀开猀琀爀⸀搀愀琀愀⠀⤀Ⰰ 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀开猀琀爀⸀搀愀琀愀⠀⤀ ⬀ 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀开猀琀爀⸀猀椀稀攀⠀⤀Ⰰ 搀愀眀渀开瀀爀漀挀开琀愀戀氀攀⤀⸀攀挀⤀㬀ഀഀ
    config.dawn_proc_table = reinterpret_cast<const void*>(dawn_proc_table);਍  紀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kValidationMode, validation_mode_str)) {਍    挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开攀砀瀀氀椀挀椀琀氀礀开猀攀琀 㴀 琀爀甀攀㬀ഀഀ
    if (validation_mode_str == kValidationMode_Disabled) {਍      挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀 㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䐀椀猀愀戀氀攀搀㬀ഀഀ
    } else if (validation_mode_str == kValidationMode_wgpuOnly) {਍      挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀 㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀圀䜀倀唀伀渀氀礀㬀ഀഀ
    } else if (validation_mode_str == kValidationMode_basic) {਍      挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀 㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䈀愀猀椀挀㬀ഀഀ
    } else if (validation_mode_str == kValidationMode_full) {਍      挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀 㴀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀㨀䘀甀氀氀㬀ഀഀ
    } else {਍      伀刀吀开吀䠀刀伀圀⠀∀䤀渀瘀愀氀椀搀 瘀愀氀椀搀愀琀椀漀渀 洀漀搀攀㨀 ∀Ⰰ 瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀开猀琀爀⤀㬀ഀഀ
    }਍  紀ഀഀ
਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kPreserveDevice, preserve_device_str)) {਍    椀昀 ⠀瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀开猀琀爀 㴀㴀 欀倀爀攀猀攀爀瘀攀䐀攀瘀椀挀攀开伀一⤀ 笀ഀഀ
      config.preserve_device = true;਍    紀 攀氀猀攀 椀昀 ⠀瀀爀攀猀攀爀瘀攀开搀攀瘀椀挀攀开猀琀爀 㴀㴀 欀倀爀攀猀攀爀瘀攀䐀攀瘀椀挀攀开伀䘀䘀⤀ 笀ഀഀ
      config.preserve_device = false;਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid preserve device: ", preserve_device_str);਍    紀ഀഀ
  }਍ഀഀ
  std::string max_storage_buffer_binding_size_str;਍  椀昀 ⠀挀漀渀昀椀最开漀瀀琀椀漀渀猀⸀吀爀礀䜀攀琀䌀漀渀昀椀最䔀渀琀爀礀⠀欀䴀愀砀匀琀漀爀愀最攀䈀甀昀昀攀爀䈀椀渀搀椀渀最匀椀稀攀Ⰰ 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开猀琀爀⤀⤀ 笀ഀഀ
    ORT_ENFORCE(਍        猀琀搀㨀㨀攀爀爀挀笀紀 㴀㴀 猀琀搀㨀㨀昀爀漀洀开挀栀愀爀猀⠀ഀഀ
                           max_storage_buffer_binding_size_str.data(),਍                           洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开猀琀爀⸀搀愀琀愀⠀⤀ ⬀ 洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀开猀琀爀⸀猀椀稀攀⠀⤀Ⰰഀഀ
                           config.max_storage_buffer_binding_size)਍                           ⸀攀挀Ⰰഀഀ
        "Invalid maxStorageBufferBindingSize value: ", max_storage_buffer_binding_size_str);਍  紀ഀഀ
਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 䐀攀瘀椀挀攀 䤀䐀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀挀漀渀琀攀砀琀开椀搀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP WGPUInstance: " << reinterpret_cast<size_t>(config.instance);਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 圀䜀倀唀䐀攀瘀椀挀攀㨀 ∀ 㰀㰀 爀攀椀渀琀攀爀瀀爀攀琀开挀愀猀琀㰀猀椀稀攀开琀㸀⠀挀漀渀昀椀最⸀搀攀瘀椀挀攀⤀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP DawnProcTable: " << reinterpret_cast<size_t>(config.dawn_proc_table);਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 嘀愀氀椀搀愀琀椀漀渀䴀漀搀攀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀瘀愀氀椀搀愀琀椀漀渀开洀漀搀攀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP PreserveDevice: " << config.preserve_device;਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 洀愀砀 猀琀漀爀愀最攀 戀甀昀昀攀爀 戀椀渀搀椀渀最 猀椀稀攀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀洀愀砀开猀琀漀爀愀最攀开戀甀昀昀攀爀开戀椀渀搀椀渀最开猀椀稀攀㬀ഀഀ
਍  ⼀⼀ 戀甀昀昀攀爀 挀愀挀栀攀 洀漀搀攀猀ഀഀ
  auto parse_buffer_cache_mode = [&config_options](const std::string& config_entry_str,਍                                                   䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀☀ 瘀愀氀甀攀⤀ ⴀ㸀 瘀漀椀搀 笀ഀഀ
    std::string buffer_cache_mode_str;਍    椀昀 ⠀挀漀渀昀椀最开漀瀀琀椀漀渀猀⸀吀爀礀䜀攀琀䌀漀渀昀椀最䔀渀琀爀礀⠀挀漀渀昀椀最开攀渀琀爀礀开猀琀爀Ⰰ 戀甀昀昀攀爀开挀愀挀栀攀开洀漀搀攀开猀琀爀⤀⤀ 笀ഀഀ
      if (buffer_cache_mode_str == kBufferCacheMode_Disabled) {਍        瘀愀氀甀攀 㴀 䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀䐀椀猀愀戀氀攀搀㬀ഀഀ
      } else if (buffer_cache_mode_str == kBufferCacheMode_LazyRelease) {਍        瘀愀氀甀攀 㴀 䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀䰀愀稀礀刀攀氀攀愀猀攀㬀ഀഀ
      } else if (buffer_cache_mode_str == kBufferCacheMode_Simple) {਍        瘀愀氀甀攀 㴀 䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀匀椀洀瀀氀攀㬀ഀഀ
      } else if (buffer_cache_mode_str == kBufferCacheMode_Bucket) {਍        瘀愀氀甀攀 㴀 䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀㨀㨀䈀甀挀欀攀琀㬀ഀഀ
      } else {਍        伀刀吀开吀䠀刀伀圀⠀∀䤀渀瘀愀氀椀搀 戀甀昀昀攀爀 挀愀挀栀攀 洀漀搀攀㨀 ∀Ⰰ 戀甀昀昀攀爀开挀愀挀栀攀开洀漀搀攀开猀琀爀⤀㬀ഀഀ
      }਍    紀ഀഀ
  };਍ഀഀ
  WebGpuBufferCacheConfig& buffer_cache_config = config.buffer_cache_config;਍  瀀愀爀猀攀开戀甀昀昀攀爀开挀愀挀栀攀开洀漀搀攀⠀欀匀琀漀爀愀最攀䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀Ⰰ 戀甀昀昀攀爀开挀愀挀栀攀开挀漀渀昀椀最⸀猀琀漀爀愀最攀⸀洀漀搀攀⤀㬀ഀഀ
  parse_buffer_cache_mode(kUniformBufferCacheMode, buffer_cache_config.uniform.mode);਍  瀀愀爀猀攀开戀甀昀昀攀爀开挀愀挀栀攀开洀漀搀攀⠀欀儀甀攀爀礀刀攀猀漀氀瘀攀䈀甀昀昀攀爀䌀愀挀栀攀䴀漀搀攀Ⰰ 戀甀昀昀攀爀开挀愀挀栀攀开挀漀渀昀椀最⸀焀甀攀爀礀开爀攀猀漀氀瘀攀⸀洀漀搀攀⤀㬀ഀഀ
  parse_buffer_cache_mode(kDefaultBufferCacheMode, buffer_cache_config.default_entry.mode);਍ഀഀ
  // power preference਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kPowerPreference, power_preference_str)) {਍    椀昀 ⠀瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀开猀琀爀 㴀㴀 欀倀漀眀攀爀倀爀攀昀攀爀攀渀挀攀开䠀椀最栀倀攀爀昀漀爀洀愀渀挀攀⤀ 笀ഀഀ
      config.power_preference = static_cast<int>(WGPUPowerPreference_HighPerformance);਍    紀 攀氀猀攀 椀昀 ⠀瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀开猀琀爀 㴀㴀 欀倀漀眀攀爀倀爀攀昀攀爀攀渀挀攀开䰀漀眀倀漀眀攀爀⤀ 笀ഀഀ
      config.power_preference = static_cast<int>(WGPUPowerPreference_LowPower);਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid power preference: ", power_preference_str);਍    紀ഀഀ
  }਍ഀഀ
  // backend type਍  椀昀 ⠀猀琀搀㨀㨀猀琀爀椀渀最 戀愀挀欀攀渀搀开琀礀瀀攀开猀琀爀㬀ഀഀ
      config_options.TryGetConfigEntry(kDawnBackendType, backend_type_str)) {਍    椀昀 ⠀戀愀挀欀攀渀搀开琀礀瀀攀开猀琀爀 㴀㴀 欀䐀愀眀渀䈀愀挀欀攀渀搀吀礀瀀攀开䐀㌀䐀㄀㈀⤀ 笀ഀഀ
      config.backend_type = static_cast<int>(WGPUBackendType_D3D12);਍    紀 攀氀猀攀 椀昀 ⠀戀愀挀欀攀渀搀开琀礀瀀攀开猀琀爀 㴀㴀 欀䐀愀眀渀䈀愀挀欀攀渀搀吀礀瀀攀开嘀甀氀欀愀渀⤀ 笀ഀഀ
      config.backend_type = static_cast<int>(WGPUBackendType_Vulkan);਍    紀 攀氀猀攀 笀ഀഀ
      ORT_THROW("Invalid Dawn backend type: ", backend_type_str);਍    紀ഀഀ
  }਍ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP storage buffer cache mode: " << config.buffer_cache_config.storage.mode;਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 甀渀椀昀漀爀洀 戀甀昀昀攀爀 挀愀挀栀攀 洀漀搀攀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀戀甀昀昀攀爀开挀愀挀栀攀开挀漀渀昀椀最⸀甀渀椀昀漀爀洀⸀洀漀搀攀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP query resolve buffer cache mode: " << config.buffer_cache_config.query_resolve.mode;਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 搀攀昀愀甀氀琀 戀甀昀昀攀爀 挀愀挀栀攀 洀漀搀攀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀戀甀昀昀攀爀开挀愀挀栀攀开挀漀渀昀椀最⸀搀攀昀愀甀氀琀开攀渀琀爀礀⸀洀漀搀攀㬀ഀഀ
਍  䰀伀䜀匀开䐀䔀䘀䄀唀䰀吀⠀嘀䔀刀䈀伀匀䔀⤀ 㰀㰀 ∀圀攀戀䜀倀唀 䔀倀 瀀漀眀攀爀 瀀爀攀昀攀爀攀渀挀攀㨀 ∀ 㰀㰀 挀漀渀昀椀最⸀瀀漀眀攀爀开瀀爀攀昀攀爀攀渀挀攀㬀ഀഀ
  LOGS_DEFAULT(VERBOSE) << "WebGPU EP Dawn backend type: " << config.backend_type;਍ഀഀ
  return config;਍紀ഀഀ
਍紀  ⼀⼀ 渀愀洀攀猀瀀愀挀攀ഀഀ
਍猀琀搀㨀㨀猀栀愀爀攀搀开瀀琀爀㰀䤀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀䘀愀挀琀漀爀礀㸀 圀攀戀䜀瀀甀倀爀漀瘀椀搀攀爀䘀愀挀琀漀爀礀䌀爀攀愀琀漀爀㨀㨀䌀爀攀愀琀攀⠀挀漀渀猀琀 䌀漀渀昀椀最伀瀀琀椀漀渀猀☀ 挀漀渀昀椀最开漀瀀琀椀漀渀猀⤀ 笀ഀഀ
  // prepare WebGpuExecutionProviderConfig਍  圀攀戀䜀瀀甀䔀砀攀挀甀琀椀漀渀倀爀漀瘀椀搀攀爀䌀漀渀昀椀最 眀攀戀最瀀甀开攀瀀开挀漀渀昀椀最 㴀 倀愀爀猀攀䔀瀀䌀漀渀昀椀最⠀挀漀渀昀椀最开漀瀀琀椀漀渀猀⤀㬀ഀഀ
਍  ⼀⼀ 瀀爀攀瀀愀爀攀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䌀漀渀昀椀最ഀഀ
  WebGpuContextConfig config = ParseWebGpuContextConfig(config_options);਍ഀഀ
  // Load the Dawn library and create the WebGPU instance.਍  愀甀琀漀☀ 挀漀渀琀攀砀琀 㴀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䌀爀攀愀琀攀䌀漀渀琀攀砀琀⠀挀漀渀昀椀最⤀㬀ഀഀ
਍  ⼀⼀ 䌀爀攀愀琀攀 圀攀戀䜀倀唀 䔀倀 昀愀挀琀漀爀礀⸀ഀഀ
  return std::make_shared<WebGpuProviderFactory>(config.context_id, context, std::move(webgpu_ep_config));਍紀ഀഀ
਍⼀⼀ 圀攀戀䜀倀唀 䐀愀琀愀吀爀愀渀猀昀攀爀 椀洀瀀氀攀洀攀渀琀愀琀椀漀渀 眀爀愀瀀瀀攀爀 昀漀爀 琀栀攀 䌀 䄀倀䤀 眀椀琀栀 氀愀稀礀 椀渀椀琀椀愀氀椀稀愀琀椀漀渀ഀഀ
struct WebGpuDataTransferImpl : OrtDataTransferImpl {਍  圀攀戀䜀瀀甀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⠀挀漀渀猀琀 伀爀琀䄀瀀椀☀ 漀爀琀开愀瀀椀开椀渀Ⰰ 椀渀琀 挀漀渀琀攀砀琀开椀搀⤀ഀഀ
      : ort_api{ort_api_in},਍        攀瀀开愀瀀椀笀⨀漀爀琀开愀瀀椀开椀渀⸀䜀攀琀䔀瀀䄀瀀椀⠀⤀紀Ⰰഀഀ
        data_transfer_{nullptr},਍        挀漀渀琀攀砀琀开椀搀开笀挀漀渀琀攀砀琀开椀搀紀Ⰰഀഀ
        init_mutex_{} {਍    漀爀琀开瘀攀爀猀椀漀渀开猀甀瀀瀀漀爀琀攀搀 㴀 伀刀吀开䄀倀䤀开嘀䔀刀匀䤀伀一㬀ഀഀ
    CanCopy = CanCopyImpl;          // OrtDataTransferImpl::CanCopy callback਍    䌀漀瀀礀吀攀渀猀漀爀猀 㴀 䌀漀瀀礀吀攀渀猀漀爀猀䤀洀瀀氀㬀  ⼀⼀ 伀爀琀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀㨀㨀䌀漀瀀礀吀攀渀猀漀爀猀 挀愀氀氀戀愀挀欀ഀഀ
    Release = ReleaseImpl;          // OrtDataTransferImpl::Release callback਍  紀ഀഀ
਍  猀琀愀琀椀挀 戀漀漀氀 䌀愀渀䌀漀瀀礀䤀洀瀀氀⠀挀漀渀猀琀 伀爀琀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⨀ 琀栀椀猀开瀀琀爀Ⰰഀഀ
                          const OrtMemoryDevice* src_memory_device,਍                          挀漀渀猀琀 伀爀琀䴀攀洀漀爀礀䐀攀瘀椀挀攀⨀ 搀猀琀开洀攀洀漀爀礀开搀攀瘀椀挀攀⤀ 渀漀攀砀挀攀瀀琀 笀ഀഀ
    const auto& impl = *static_cast<const WebGpuDataTransferImpl*>(this_ptr);਍    伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀 猀爀挀开琀礀瀀攀 㴀 椀洀瀀氀⸀攀瀀开愀瀀椀⸀䴀攀洀漀爀礀䐀攀瘀椀挀攀开䜀攀琀䐀攀瘀椀挀攀吀礀瀀攀⠀猀爀挀开洀攀洀漀爀礀开搀攀瘀椀挀攀⤀㬀ഀഀ
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);਍ഀഀ
    // Check if at least one device is GPU਍    戀漀漀氀 栀愀猀开最瀀甀 㴀 ⠀猀爀挀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀⤀ 簀簀 ⠀搀猀琀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀⤀㬀ഀഀ
    if (!has_gpu) {਍      爀攀琀甀爀渀 昀愀氀猀攀㬀ഀഀ
    }਍ഀഀ
    // WebGPU uses vendor ID 0 (VendorIds::NONE). Only handle GPU devices with vendor ID 0.਍    ⼀⼀ 吀栀椀猀 瀀爀攀瘀攀渀琀猀 愀琀琀攀洀瀀琀椀渀最 琀漀 挀漀瀀礀 搀愀琀愀 昀漀爀 漀琀栀攀爀 䔀倀猀✀ 昀愀欀攀 䜀倀唀 搀攀瘀椀挀攀猀 ⠀攀⸀最⸀Ⰰ 攀砀愀洀瀀氀攀 䔀倀 眀椀琀栀 瘀攀渀搀漀爀 　砀䈀䔀㔀㜀⤀ഀഀ
    if (src_type == OrtMemoryInfoDeviceType_GPU) {਍      甀椀渀琀㌀㈀开琀 猀爀挀开瘀攀渀搀漀爀 㴀 椀洀瀀氀⸀攀瀀开愀瀀椀⸀䴀攀洀漀爀礀䐀攀瘀椀挀攀开䜀攀琀嘀攀渀搀漀爀䤀搀⠀猀爀挀开洀攀洀漀爀礀开搀攀瘀椀挀攀⤀㬀ഀഀ
      if (src_vendor != 0) {਍        爀攀琀甀爀渀 昀愀氀猀攀㬀  ⼀⼀ 一漀琀 愀 圀攀戀䜀倀唀 搀攀瘀椀挀攀ഀഀ
      }਍    紀ഀഀ
਍    椀昀 ⠀搀猀琀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀⤀ 笀ഀഀ
      uint32_t dst_vendor = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);਍      椀昀 ⠀搀猀琀开瘀攀渀搀漀爀 ℀㴀 　⤀ 笀ഀഀ
        return false;  // Not a WebGPU device਍      紀ഀഀ
    }਍ഀഀ
    // If both are GPU, they must have the same device ID਍    椀昀 ⠀猀爀挀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀 ☀☀ 搀猀琀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀⤀ 笀ഀഀ
      int src_device_id = impl.ep_api.MemoryDevice_GetDeviceId(src_memory_device);਍      椀渀琀 搀猀琀开搀攀瘀椀挀攀开椀搀 㴀 椀洀瀀氀⸀攀瀀开愀瀀椀⸀䴀攀洀漀爀礀䐀攀瘀椀挀攀开䜀攀琀䐀攀瘀椀挀攀䤀搀⠀搀猀琀开洀攀洀漀爀礀开搀攀瘀椀挀攀⤀㬀ഀഀ
      if (src_device_id != impl.context_id_ || dst_device_id != impl.context_id_) {਍        爀攀琀甀爀渀 昀愀氀猀攀㬀  ⼀⼀ 䌀愀渀渀漀琀 挀漀瀀礀 戀攀琀眀攀攀渀 搀椀昀昀攀爀攀渀琀 搀攀瘀椀挀攀猀ഀഀ
      }਍    紀ഀഀ
਍    ⼀⼀ 圀攀戀䜀倀唀 猀甀瀀瀀漀爀琀猀 䜀倀唀㰀ⴀ㸀䜀倀唀Ⰰ 䜀倀唀㰀ⴀ㸀䌀倀唀 挀漀瀀椀攀猀 ⠀眀栀攀爀攀 䜀倀唀 栀愀猀 瘀攀渀搀漀爀 䤀䐀 　⤀ഀഀ
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||਍           ⠀猀爀挀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀 ☀☀ 搀猀琀开琀礀瀀攀 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䌀倀唀⤀ 簀簀ഀഀ
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);਍  紀ഀഀ
਍  猀琀愀琀椀挀 伀爀琀匀琀愀琀甀猀⨀ 䌀漀瀀礀吀攀渀猀漀爀猀䤀洀瀀氀⠀伀爀琀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⨀ 琀栀椀猀开瀀琀爀Ⰰഀഀ
                                    const OrtValue** src_tensors,਍                                    伀爀琀嘀愀氀甀攀⨀⨀ 搀猀琀开琀攀渀猀漀爀猀Ⰰഀഀ
                                    OrtSyncStream** /*streams*/,਍                                    猀椀稀攀开琀 渀甀洀开琀攀渀猀漀爀猀⤀ 渀漀攀砀挀攀瀀琀 笀ഀഀ
    auto& impl = *static_cast<WebGpuDataTransferImpl*>(this_ptr);਍ഀഀ
    if (num_tensors == 0) {਍      爀攀琀甀爀渀 渀甀氀氀瀀琀爀㬀ഀഀ
    }਍ഀഀ
    // Lazy initialization: Use double-checked locking to avoid unnecessary lock operations਍    椀昀 ⠀椀洀瀀氀⸀搀愀琀愀开琀爀愀渀猀昀攀爀开 㴀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
      std::lock_guard<std::mutex> lock(impl.init_mutex_);਍      椀昀 ⠀椀洀瀀氀⸀搀愀琀愀开琀爀愀渀猀昀攀爀开 㴀㴀 渀甀氀氀瀀琀爀⤀ 笀ഀഀ
        // Always create a new context with context_id 0਍        椀昀 ⠀椀洀瀀氀⸀挀漀渀琀攀砀琀开椀搀开 ℀㴀 　⤀ 笀ഀഀ
          return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, "Shared data transfer can only be created for the default device (0).");਍        紀ഀഀ
਍        愀甀琀漀☀ 挀漀渀琀攀砀琀 㴀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀㨀㨀䐀攀昀愀甀氀琀䌀漀渀琀攀砀琀⠀⤀㬀ഀഀ
਍        ⼀⼀ 䌀爀攀愀琀攀 琀栀攀 䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀 椀渀猀琀愀渀挀攀ഀഀ
        // Note: The DataTransferImpl holds a const reference to BufferManager. The BufferManager's lifecycle਍        ⼀⼀ 椀猀 洀愀渀愀最攀搀 戀礀 琀栀攀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀Ⰰ 眀栀椀挀栀 椀猀 猀琀漀爀攀搀 椀渀 愀 猀琀愀琀椀挀 圀攀戀䜀瀀甀䌀漀渀琀攀砀琀䘀愀挀琀漀爀礀 愀渀搀 瀀攀爀猀椀猀琀猀ഀഀ
        // for the lifetime of the application, ensuring the reference remains valid.਍        椀洀瀀氀⸀搀愀琀愀开琀爀愀渀猀昀攀爀开 㴀 猀琀搀㨀㨀洀愀欀攀开甀渀椀焀甀攀㰀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀㸀⠀挀漀渀琀攀砀琀⸀䈀甀昀昀攀爀䴀愀渀愀最攀爀⠀⤀⤀㬀ഀഀ
      }਍    紀ഀഀ
਍    ⼀⼀ 一漀眀 瀀攀爀昀漀爀洀 琀栀攀 愀挀琀甀愀氀 琀攀渀猀漀爀 挀漀瀀礀ഀഀ
    for (size_t idx = 0; idx < num_tensors; ++idx) {਍⌀椀昀 搀攀昀椀渀攀搀⠀伀刀吀开唀匀䔀开䔀倀开䄀倀䤀开䄀䐀䄀倀吀䔀刀匀⤀ഀഀ
      Ort::ConstValue src_value{src_tensors[idx]};਍      挀漀渀猀琀 瘀漀椀搀⨀ 猀爀挀开搀愀琀愀 㴀 猀爀挀开瘀愀氀甀攀⸀䜀攀琀吀攀渀猀漀爀刀愀眀䐀愀琀愀⠀⤀㬀ഀഀ
      size_t size = src_value.GetTensorSizeInBytes();਍      戀漀漀氀 猀爀挀开椀猀开最瀀甀 㴀 猀爀挀开瘀愀氀甀攀⸀䜀攀琀吀攀渀猀漀爀䴀攀洀漀爀礀䤀渀昀漀⠀⤀⸀䜀攀琀䐀攀瘀椀挀攀吀礀瀀攀⠀⤀ 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀㬀ഀഀ
਍      伀爀琀㨀㨀唀渀漀眀渀攀搀嘀愀氀甀攀 搀猀琀开瘀愀氀甀攀笀搀猀琀开琀攀渀猀漀爀猀嬀椀搀砀崀紀㬀ഀഀ
      void* dst_data = dst_value.GetTensorMutableRawData();਍      戀漀漀氀 搀猀琀开椀猀开最瀀甀 㴀 搀猀琀开瘀愀氀甀攀⸀䜀攀琀吀攀渀猀漀爀䴀攀洀漀爀礀䤀渀昀漀⠀⤀⸀䜀攀琀䐀攀瘀椀挀攀吀礀瀀攀⠀⤀ 㴀㴀 伀爀琀䴀攀洀漀爀礀䤀渀昀漀䐀攀瘀椀挀攀吀礀瀀攀开䜀倀唀㬀ഀഀ
#else਍      挀漀渀猀琀 吀攀渀猀漀爀☀ 猀爀挀开琀攀渀猀漀爀 㴀 猀爀挀开琀攀渀猀漀爀猀嬀椀搀砀崀ⴀ㸀䜀攀琀㰀吀攀渀猀漀爀㸀⠀⤀㬀ഀഀ
      const void* src_data = src_tensor.DataRaw();਍      猀椀稀攀开琀 猀椀稀攀 㴀 猀爀挀开琀攀渀猀漀爀⸀匀椀稀攀䤀渀䈀礀琀攀猀⠀⤀㬀ഀഀ
      bool src_is_gpu = src_tensor.Location().device.Type() == OrtDevice::GPU;਍ഀഀ
      Tensor& dst_tensor = *dst_tensors[idx]->GetMutable<Tensor>();਍      瘀漀椀搀⨀ 搀猀琀开搀愀琀愀 㴀 搀猀琀开琀攀渀猀漀爀⸀䴀甀琀愀戀氀攀䐀愀琀愀刀愀眀⠀⤀㬀ഀഀ
      bool dst_is_gpu = dst_tensor.Location().device.Type() == OrtDevice::GPU;਍⌀攀渀搀椀昀ഀഀ
      auto status = impl.data_transfer_->CopyTensor(src_data,਍                                                    猀爀挀开椀猀开最瀀甀Ⰰഀഀ
                                                    dst_data,਍                                                    搀猀琀开椀猀开最瀀甀Ⰰഀഀ
                                                    size);਍      椀昀 ⠀℀猀琀愀琀甀猀⸀䤀猀伀䬀⠀⤀⤀ 笀ഀഀ
        return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, status.ErrorMessage().c_str());਍      紀ഀഀ
    }਍    爀攀琀甀爀渀 渀甀氀氀瀀琀爀㬀ഀഀ
  }਍ഀഀ
  static void ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {਍    愀甀琀漀⨀ 瀀开椀洀瀀氀 㴀 猀琀愀琀椀挀开挀愀猀琀㰀圀攀戀䜀瀀甀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⨀㸀⠀琀栀椀猀开瀀琀爀⤀㬀ഀഀ
    int context_id = p_impl->context_id_;਍    戀漀漀氀 搀愀琀愀开琀爀愀渀猀昀攀爀开椀渀椀琀椀愀氀椀稀攀搀 㴀 昀愀氀猀攀㬀ഀഀ
    {਍      猀琀搀㨀㨀氀漀挀欀开最甀愀爀搀㰀猀琀搀㨀㨀洀甀琀攀砀㸀 氀漀挀欀⠀瀀开椀洀瀀氀ⴀ㸀椀渀椀琀开洀甀琀攀砀开⤀㬀ഀഀ
      data_transfer_initialized = (p_impl->data_transfer_ != nullptr);਍    紀ഀഀ
    delete p_impl;਍    椀昀 ⠀搀愀琀愀开琀爀愀渀猀昀攀爀开椀渀椀琀椀愀氀椀稀攀搀⤀ 笀ഀഀ
      WebGpuContextFactory::ReleaseContext(context_id);਍    紀ഀഀ
  }਍ഀഀ
  const OrtApi& ort_api;਍  挀漀渀猀琀 伀爀琀䔀瀀䄀瀀椀☀ 攀瀀开愀瀀椀㬀ഀഀ
  std::unique_ptr<DataTransferImpl> data_transfer_;  // Lazy-initialized਍  椀渀琀 挀漀渀琀攀砀琀开椀搀开㬀                                   ⼀⼀ 吀爀愀挀欀 眀栀椀挀栀 挀漀渀琀攀砀琀 眀攀✀爀攀 甀猀椀渀最ഀഀ
  std::mutex init_mutex_;                            // Protects lazy initialization਍紀㬀ഀഀ
਍伀爀琀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⨀ 伀爀琀圀攀戀䜀瀀甀䌀爀攀愀琀攀䐀愀琀愀吀爀愀渀猀昀攀爀⠀椀渀琀 挀漀渀琀攀砀琀开椀搀 ⼀⨀ 㴀 　 ⨀⼀⤀ 笀ഀഀ
#if defined(ORT_USE_EP_API_ADAPTERS)਍  爀攀琀甀爀渀 渀攀眀 圀攀戀䜀瀀甀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⠀漀渀渀砀爀甀渀琀椀洀攀㨀㨀攀瀀㨀㨀䄀瀀椀⠀⤀⸀漀爀琀Ⰰ 挀漀渀琀攀砀琀开椀搀⤀㬀ഀഀ
#else਍  ⼀⼀ 嘀愀氀椀搀愀琀攀 䄀倀䤀 瘀攀爀猀椀漀渀 椀猀 猀甀瀀瀀漀爀琀攀搀ഀഀ
  const OrtApi* api = OrtApis::GetApi(ORT_API_VERSION);਍  椀昀 ⠀℀愀瀀椀⤀ 笀ഀഀ
    // API version not supported - return nullptr to indicate failure਍    爀攀琀甀爀渀 渀甀氀氀瀀琀爀㬀ഀഀ
  }਍  爀攀琀甀爀渀 渀攀眀 圀攀戀䜀瀀甀䐀愀琀愀吀爀愀渀猀昀攀爀䤀洀瀀氀⠀⨀愀瀀椀Ⰰ 挀漀渀琀攀砀琀开椀搀⤀㬀ഀഀ
#endif਍紀ഀഀ
਍紀  ⼀⼀ 渀愀洀攀猀瀀愀挀攀 漀渀渀砀爀甀渀琀椀洀攀ഀഀ
