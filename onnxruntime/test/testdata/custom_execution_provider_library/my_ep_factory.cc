// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "my_ep_factory.h"
#include "my_execution_provider.h"
#include "core/common/gsl.h"
#include "core/providers/shared/common.h"
#include <iostream>
#include "core/framework/provider_options_utils.h"
#include "onnxruntime_lite_custom_op.h"
#include <cmath>

using namespace onnxruntime;

namespace onnxruntime {

void KernelTwo(const Ort::Custom::Tensor<float>& X,
               Ort::Custom::Tensor<int32_t>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = static_cast<int32_t>(round(X_raw[i]));
  }
}

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

//ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MyEP, _In_ OrtSessionOptions* options, int device_id) {
//  ORT_UNUSED_PARAMETER(device_id);
//  using LiteOp = Ort::Custom::OrtLiteCustomOp;
//  static const std::unique_ptr<LiteOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", onnxruntime::kMyProvider, KernelTwo)};
//  Ort::CustomOpDomain domain{"test.customop"};
//  domain.Add(c_CustomOpTwo.get());
//
//  Ort::UnownedSessionOptions session_options(options);
//  session_options.Add(domain);
//  return nullptr;
//}

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

ORT_API_STATUS_IMPL(RegisterCustomOp, _In_ OrtSessionOptions* options) {
  using LiteOp = Ort::Custom::OrtLiteCustomOp;
  static const std::unique_ptr<LiteOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", onnxruntime::kMyProvider, KernelTwo)};
  Ort::CustomOpDomain domain{"test.customop"};
  domain.Add(c_CustomOpTwo.get());

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(domain);
  return nullptr;
}
}
