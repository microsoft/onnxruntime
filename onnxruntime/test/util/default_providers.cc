// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#define FACTORY_PTR_HOLDER \
  std::unique_ptr<ONNXRuntimeProviderFactoryPtr> ptr_holder_(f);

namespace onnxruntime {
namespace test {
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  ONNXRuntimeProviderFactoryPtr* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateCpuExecutionProviderFactory(enable_arena ? 1 : 0, &f));
  FACTORY_PTR_HOLDER;
  ONNXRuntimeProviderPtr out;
  ONNXRUNTIME_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  ONNXRuntimeProviderFactoryPtr* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateCUDAExecutionProviderFactory(0, &f));
  FACTORY_PTR_HOLDER;
  ONNXRuntimeProviderPtr out;
  ONNXRUNTIME_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena) {
#ifdef USE_MKLDNN
  ONNXRuntimeProviderFactoryPtr* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateMkldnnExecutionProviderFactory(enable_arena ? 1 : 0, &f));
  FACTORY_PTR_HOLDER;
  ONNXRuntimeProviderPtr out;
  ONNXRUNTIME_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  ONNXRUNTIME_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider() {
#ifdef USE_NUPHAR
  ONNXRuntimeProviderFactoryPtr* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateNupharExecutionProviderFactory(0, "", &f));
  FACTORY_PTR_HOLDER;
  ONNXRuntimeProviderPtr out;
  ONNXRUNTIME_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultBrainSliceExecutionProvider() {
#ifdef USE_BRAINSLICE
  ONNXRuntimeProviderFactoryPtr* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateBrainSliceExecutionProviderFactory(0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin", & f));
  FACTORY_PTR_HOLDER;
  ONNXRuntimeProviderPtr out;
  ONNXRUNTIME_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
