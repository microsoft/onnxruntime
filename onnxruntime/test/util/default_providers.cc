// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "default_providers.h"
#include "providers.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CPUProviderFactoryCreator::Create(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  OrtTensorRTProviderOptions params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0};
  if (auto factory = TensorrtProviderFactoryCreator::Create(&params))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptionsV2* params) {
#ifdef USE_TENSORRT
  if (auto factory = TensorrtProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider() {
#ifdef USE_MIGRAPHX
  OrtMIGraphXProviderOptions params{
      0,
      0,
      0,
      0,
      nullptr};
  return MIGraphXProviderFactoryCreator::Create(&params)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions* params) {
#ifdef USE_MIGRAPHX
  if (auto factory = MIGraphXProviderFactoryCreator::Create(params))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> OpenVINOExecutionProviderWithOptions(const OrtOpenVINOProviderOptions* params) {
#ifdef USE_OPENVINO
  return OpenVINOProviderFactoryCreator::Create(params)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(params);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  ProviderOptions provider_options_map;
  return OpenVINOProviderFactoryCreator::Create(&provider_options_map)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  OrtCUDAProviderOptions provider_options{};
  provider_options.do_copy_in_default_stream = true;
  if (auto factory = CudaProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> CudaExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options) {
#ifdef USE_CUDA
  if (auto factory = CudaProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(provider_options);
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider() {
#ifdef USE_DNNL
  OrtDnnlProviderOptions dnnl_options;
  dnnl_options.use_arena = 1;
  dnnl_options.threadpool_args = nullptr;
  if (auto factory = DnnlProviderFactoryCreator::Create(&dnnl_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DnnlExecutionProviderWithOptions(const OrtDnnlProviderOptions* provider_options) {
#ifdef USE_DNNL
  if (auto factory = DnnlProviderFactoryCreator::Create(provider_options))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(provider_options);
#endif
  return nullptr;
}

// std::unique_ptr<IExecutionProvider> DefaultTvmExecutionProvider() {
// #ifdef USE_TVM
//   return TVMProviderFactoryCreator::Create("")->CreateProvider();
// #else
//   return nullptr;
// #endif
// }

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
// The NNAPI EP uses a stub implementation on non-Android platforms so cannot be used to execute a model.
// Manually append an NNAPI EP instance to the session to unit test the GetCapability and Compile implementation.
#if defined(USE_NNAPI) && defined(__ANDROID__)
  return NnapiProviderFactoryCreator::Create(0, {})->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider() {
#ifdef USE_RKNPU
  return RknpuProviderFactoryCreator::Create()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena) {
#ifdef USE_ACL
  return ACLProviderFactoryCreator::Create(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena) {
#ifdef USE_ARMNN
  return ArmNNProviderFactoryCreator::Create(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider(bool test_tunable_op) {
#ifdef USE_ROCM
  OrtROCMProviderOptions provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.tunable_op_enable = test_tunable_op ? 1 : 0;
  provider_options.tunable_op_tuning_enable = test_tunable_op ? 1 : 0;
  provider_options.tunable_op_max_tuning_duration_ms = 0;
  if (auto factory = RocmProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  ORT_UNUSED_PARAMETER(test_tunable_op);
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider() {
// For any non - macOS system, CoreML will only be used for ort model converter
// Make it unavailable here, you can still manually append CoreML EP to session for model conversion
#if defined(USE_COREML) && defined(__APPLE__)
  // We want to run UT on CPU only to get output value without losing precision
  uint32_t coreml_flags = 0;
  coreml_flags |= COREML_FLAG_USE_CPU_ONLY;
  return CoreMLProviderFactoryCreator::Create(coreml_flags)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultSnpeExecutionProvider() {
#if defined(USE_SNPE)
  ProviderOptions provider_options_map;
  return SNPEProviderFactoryCreator::Create(provider_options_map)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultQnnExecutionProvider() {
#ifdef USE_QNN
  ProviderOptions provider_options_map;
  // Limit to CPU backend for now. TODO: Enable HTP emulator
  std::string backend_path = "./libQnnCpu.so";
#if defined(_WIN32) || defined(_WIN64)
  backend_path = "./QnnCpu.dll";
#endif
  provider_options_map["backend_path"] = backend_path;
  return QNNProviderFactoryCreator::Create(provider_options_map, nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> QnnExecutionProviderWithOptions(const ProviderOptions& options) {
#ifdef USE_QNN
  return QNNProviderFactoryCreator::Create(options, nullptr)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(options);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultXnnpackExecutionProvider() {
#ifdef USE_XNNPACK
  return XnnpackProviderFactoryCreator::Create(ProviderOptions(), nullptr)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCannExecutionProvider() {
#ifdef USE_CANN
  OrtCANNProviderOptions provider_options{};
  if (auto factory = CannProviderFactoryCreator::Create(&provider_options))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultDmlExecutionProvider() {
#ifdef USE_DML
  if (auto factory = DMLProviderFactoryCreator::Create(0))
    return factory->CreateProvider();
#endif
  return nullptr;
}

}  // namespace test
}  // namespace onnxruntime
