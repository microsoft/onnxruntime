// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#include "dnnl_execution_provider_info.h"

#include "core/providers/dnnl/dnnl_provider_options.h"
#include "core/framework/provider_options_utils.h"
#include "core/common/common.h"

namespace onnxruntime::dnnl::provider_option_names {
constexpr const char* kUseArena = "use_arena";
constexpr const char* kThreadpoolArgs = "threadpool_args";
}  // namespace onnxruntime::dnnl::provider_option_names

namespace onnxruntime {

DnnlExecutionProviderInfo DnnlExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  DnnlExecutionProviderInfo info{};
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              dnnl::provider_option_names::kThreadpoolArgs,
              [&info](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                info.threadpool_args = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(dnnl::provider_option_names::kUseArena, info.use_arena)
          .Parse(options));
  return info;
}

ProviderOptions DnnlExecutionProviderInfo::ToProviderOptions(const DnnlExecutionProviderInfo& info) {
  const ProviderOptions options{
      {dnnl::provider_option_names::kUseArena, MakeStringWithClassicLocale(info.use_arena)},
      {dnnl::provider_option_names::kThreadpoolArgs, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.threadpool_args))},
  };

  return options;
}

ProviderOptions DnnlExecutionProviderInfo::ToProviderOptions(const OrtDnnlProviderOptions& info) {
  const ProviderOptions options{
      {dnnl::provider_option_names::kUseArena, MakeStringWithClassicLocale(info.use_arena)},
  };

  return options;
}

}  // namespace onnxruntime