// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/providers/dnnl/dnnl_provider_options.h"

namespace onnxruntime {
struct IExecutionProviderFactory;
struct DnnlExecutionProviderInfo;

struct ProviderInfo_Dnnl {
  virtual void DnnlExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options,
                                                              onnxruntime::DnnlExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
  CreateExecutionProviderFactory(const onnxruntime::DnnlExecutionProviderInfo& info) = 0;
};

}  // namespace onnxruntime
