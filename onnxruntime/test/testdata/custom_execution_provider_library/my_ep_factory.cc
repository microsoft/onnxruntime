// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "my_ep_factory.h"
#include "my_execution_provider.h"
#include "core/common/gsl.h"
#include "core/providers/shared/common.h"
#include <iostream>
#include "core/framework/provider_options_utils.h"

using namespace onnxruntime;

namespace onnxruntime {

struct MyProviderFactory : IExecutionProviderFactory {
  MyProviderFactory(const MyProviderInfo& info) : info_{info} {}
  ~MyProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  MyProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MyProviderFactory::CreateProvider() {
  std::cout << "My EP provider created, with device id: " << info_.device_id << ", some_option: " << info_.some_config << std::endl;
  return std::make_unique<MyExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MyEP(const MyProviderInfo& info) {
  return std::make_shared<onnxruntime::MyProviderFactory>(info);
}

struct MyEP_Provider : Provider {
  GSL_SUPPRESS(c .35)
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    ProviderOptions* options = (ProviderOptions*)(provider_options);
    MyProviderInfo info;
    ORT_THROW_IF_ERROR(
        ProviderOptionsParser{}
            .AddValueParser(
                "device_id",
                [&info](const std::string& value_str) -> Status {
                  ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                  return Status::OK();
                })
            .AddValueParser(
                "some_config",
                [&info](const std::string& value_str) -> Status {
                  ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.some_config));
                  return Status::OK();
                })
            .Parse(*options));
    return std::make_shared<MyProviderFactory>(info);
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}

ORT_API(size_t, ProviderHashFunc, const void* provider_options) {
  ProviderOptions* options = (ProviderOptions*)(provider_options);
  MyProviderInfo info;
  ORT_IGNORE_RETURN_VALUE(ProviderOptionsParser{}
                              .AddValueParser(
                                  "device_id",
                                  [&info](const std::string& value_str) -> Status {
                                    ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                                    return Status::OK();
                                  })
                              .AddValueParser(
                                  "some_config",
                                  [&info](const std::string& value_str) -> Status {
                                    ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.some_config));
                                    return Status::OK();
                                  })
                              .Parse(*options));
  // use device id as hash key
  return info.device_id;
}
}
