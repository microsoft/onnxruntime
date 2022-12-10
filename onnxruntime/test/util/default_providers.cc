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
  auto ret = CPUProviderFactoryCreator::Create(enable_arena)->CreateProvider();
  // The factory created CPU provider doesn't create/reg allocators; something that is expected by
  // clients of DefaultCpuExecutionProvider; hence the need to call RegisterAllocator explicitly.
  AllocatorManager mgr;  // needed only to call RegisterAllocator
  ret->RegisterAllocator(mgr);
  return ret;
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
      0};
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

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  OrtOpenVINOProviderOptions params;
  return OpenVINOProviderFactoryCreator::Create(&params)->CreateProvider();
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

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena) {
#ifdef USE_DNNL
  if (auto factory = DnnlProviderFactoryCreator::Create(enable_arena ? 1 : 0))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
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
  provider_options.tunable_op_enabled = test_tunable_op ? 1 : 0;
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
