// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/providers/cann/cann_provider_options.h"

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
struct CANNExecutionProviderInfo;
enum class ArenaExtendStrategy : int32_t;

struct ProviderInfo_CANN {
  virtual int cannGetDeviceCount() = 0;

  virtual void cannMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void cannMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual void CANNExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options,
                                                              onnxruntime::CANNExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
  CreateExecutionProviderFactory(const onnxruntime::CANNExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator>
  CreateCannAllocator(int16_t device_id, size_t npu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy,
                      OrtArenaCfg* default_memory_arena_cfg) = 0;
};

}  // namespace onnxruntime
