// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

struct DnnlExecutionProviderInfo {
  int use_arena{true};             // If arena is used, use_arena 0 = not used, nonzero = used
  void* threadpool_args{nullptr};  // Used to enable ORT threadpool when using the test runner

  static DnnlExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const DnnlExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtDnnlProviderOptions& info);
};

}  // namespace onnxruntime
