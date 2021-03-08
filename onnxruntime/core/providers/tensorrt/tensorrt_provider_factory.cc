// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
#include <iostream> //slx

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
  std::cout << "TensorrtProviderFactory::CreateProvider(): info_.device_id: " << info_.device_id << ", info_.fp16_enable: " << info_.fp16_enable << std::endl;//slx
  return onnxruntime::make_unique<TensorrtExecutionProvider>(info_);
}

/* //slx new
struct TensorrtProviderFactory : IExecutionProviderFactory {
  //TensorrtProviderFactory(const TensorrtExecutionProviderInfo& info) : info_{info} {}//slx ??
  
//
  TensorrtProviderFactory(bool trt_fp16_enable, bool trt_int8_enable,
                          const char* trt_int8_calibration_table_name, bool trt_int8_use_native_calibration_table)
      : fp16_enable_(trt_fp16_enable), int8_enable_(trt_int8_enable) {
    int8_calibration_table_name_ = (trt_int8_calibration_table_name == nullptr) ? "" : trt_int8_calibration_table_name;
    use_native_calibration_table_(trt_int8_use_native_calibration_table);
  }

  ~TensorrtProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  //TensorrtExecutionProviderInfo info_;//slx
  bool fp16_enable_;
  bool int8_enable_,
  const char* int8_calibration_table_name_;
  bool int8_use_native_calibration_table_;
};

std::unique_ptr<IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  TensorrtExecutionProviderInfo info(fp16_enable_, int8_enable_, int8_calibration_table_name_, int8_use_native_calibration_table_);
  return onnxruntime::make_unique<TensorrtExecutionProvider>(info_);
}
*/


std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {//slx: still keep this??
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id;
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}
/*
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(//slx: openvino's way, not necessary??
    bool fp16_enable, bool int8_enable, const char* int8_calibration_table_name, bool int8_use_native_calibration_table) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(fp16_enable, int8_enable, int8_calibration_table_name, int8_use_native_calibration_table);
}
*/
/* //slx ?? still need it??
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const TensorrtExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}
*/
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions& provider_options) {
    std::cout << "tensorrt_provider_factory.cc: CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions& provider_options)" << std::endl; //slx
    TensorrtExecutionProviderInfo info;
    info.device_id = provider_options.device_id;
    info.has_user_compute_stream = provider_options.has_user_compute_stream;///int to bool
    info.user_compute_stream = provider_options.user_compute_stream;
	
    //slx
    info.max_workspace_size = provider_options.trt_max_workspace_size == nullptr ? "" : provider_options.trt_max_workspace_size;///char* to string
    info.fp16_enable = provider_options.trt_fp16_enable == nullptr ? "" : provider_options.trt_fp16_enable;//slx ??? int to bool ???
    info.int8_enable = provider_options.trt_int8_enable == nullptr ? "" : provider_options.trt_int8_enable;
    info.int8_calibration_table_name = provider_options.trt_int8_calibration_table_name == nullptr ? "" : provider_options.trt_int8_calibration_table_name;
    info.int8_use_native_calibration_table = provider_options.trt_int8_use_native_calibration_table == nullptr ? "" : provider_options.trt_int8_use_native_calibration_table;

    std::cout << "tensorrt_provider_factory.cc: CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions& provider_options): info.device_id: " << info.device_id << ", info.max_workspace_size: " << info.max_workspace_size << ", info.fp16_enable: " << info.fp16_enable << ", info.int8_enable: " << info.int8_enable << ", info.int8_calibration_table_name: " << info.int8_calibration_table_name << ", info.int8_use_native_calibration_table: " << info.int8_use_native_calibration_table << std::endl;//slx
    return std::make_shared<TensorrtProviderFactory>(info);
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(info);
}

struct Tensorrt_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    TensorrtExecutionProviderInfo info;
    info.device_id = device_id;
    return std::make_shared<TensorrtProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    std::cout << "!!!!!tensorrt_provider_factory.cc: CreateExecutionProviderFactory(const void* provider_options)" << std::endl;//slx
    auto& options = *reinterpret_cast<const OrtTensorRTProviderOptions*>(provider_options);
    std::cout << "options.device_id: " << options.device_id << std::endl;//slx
    std::cout << "options.has_user_compute_stream: " << options.has_user_compute_stream << std::endl;//slx
    std::cout << "options.user_compute_stream: " << options.user_compute_stream << std::endl;//slx: !!!!!ok because it's void*	
    //std::cout << "options.trt_fp16_enable: " << options.trt_fp16_enable << std::endl;//slx	!!!!!!wrong because it's char*

    TensorrtExecutionProviderInfo info;
    info.device_id = options.device_id;
    info.has_user_compute_stream = options.has_user_compute_stream;
    info.user_compute_stream = options.user_compute_stream;
	
    //slx
    info.max_workspace_size = options.trt_max_workspace_size == nullptr ? "" : options.trt_max_workspace_size;//char * to string
    info.fp16_enable = options.trt_fp16_enable == nullptr ? "" : options.trt_fp16_enable;
    info.int8_enable = options.trt_int8_enable == nullptr ? "" : options.trt_int8_enable;
    info.int8_calibration_table_name = options.trt_int8_calibration_table_name == nullptr ? "" : options.trt_int8_calibration_table_name;
    info.int8_use_native_calibration_table = options.trt_int8_use_native_calibration_table == nullptr ? "" : options.trt_int8_use_native_calibration_table;

    std::cout << "info.device_id: " << info.device_id << ", info.max_workspace_size: " << info.max_workspace_size << ", info.fp16_enable: " << info.fp16_enable << ", info.int8_enable: " << info.int8_enable <<std::endl;//slx
    std::cout << "info.int8_calibration_table_name: " << info.int8_calibration_table_name << ", info.int8_use_native_calibration_table: " << info.int8_use_native_calibration_table << std::endl;//slx	
    return std::make_shared<TensorrtProviderFactory>(info);
  }

/* //slx: openvino's way, not necessary???
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto& params = *reinterpret_cast<const OrtTensorRTProviderOptions*>(void_params);
    return std::make_shared<TensorrtProviderFactory>(params.fp16_enable, params.int8_enable, params.int8_calibration_table_name, params.int8_use_native_calibration_table);
  }
*/
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
