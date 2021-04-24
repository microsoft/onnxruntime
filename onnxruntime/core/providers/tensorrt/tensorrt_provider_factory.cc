// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

void Shutdown_DeleteRegistry();

struct TensorrtProviderFactory : IExecutionProviderFactory {
  TensorrtProviderFactory(const TensorrtExecutionProviderInfo& info) : info_{info} {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  TensorrtExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<TensorrtExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id;
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const TensorrtExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}

struct Tensorrt_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    TensorrtExecutionProviderInfo info;
    info.device_id = device_id;
    return std::make_shared<TensorrtProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtTensorRTProviderOptions*>(provider_options);
    TensorrtExecutionProviderInfo info;
    info.device_id = options.device_id;
    info.has_user_compute_stream = options.has_user_compute_stream;
    info.user_compute_stream = options.user_compute_stream;
    info.has_trt_options = options.has_trt_options;
    info.max_workspace_size = options.trt_max_workspace_size;
    info.fp16_enable = options.trt_fp16_enable;
    info.int8_enable = options.trt_int8_enable;
    info.int8_calibration_table_name = options.trt_int8_calibration_table_name == nullptr ? "" : options.trt_int8_calibration_table_name;
    info.int8_use_native_calibration_table = options.trt_int8_use_native_calibration_table;
    info.force_sequential_engine_build = options.force_sequential_engine_build;
    return std::make_shared<TensorrtProviderFactory>(info);
  }

  void Shutdown() override {
    Shutdown_DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
