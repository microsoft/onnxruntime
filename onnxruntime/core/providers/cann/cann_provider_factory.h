// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/providers/cann/cann_provider_options.h"

namespace onnxruntime {
struct IExecutionProviderFactory;
struct CANNExecutionProviderInfo;

struct ProviderInfo_CANN {
  virtual void CANNExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options,
                                                              onnxruntime::CANNExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
  CreateExecutionProviderFactory(const onnxruntime::CANNExecutionProviderInfo& info) = 0;
};

}  // namespace onnxruntime
