// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#define FACTORY_PTR_HOLDER \
  std::unique_ptr<OrtProviderFactoryInterface*> ptr_holder_(f);

namespace onnxruntime {
namespace test {
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateCpuExecutionProviderFactory(enable_arena ? 1 : 0, &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateCUDAExecutionProviderFactory(0, &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena) {
#ifdef USE_MKLDNN
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateMkldnnExecutionProviderFactory(enable_arena ? 1 : 0, &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider() {
#ifdef USE_NUPHAR
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateNupharExecutionProviderFactory(0, "", &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultBrainSliceExecutionProvider() {
#ifdef USE_BRAINSLICE
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateBrainSliceExecutionProviderFactory(0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin", &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultTRTExecutionProvider() {
#ifdef USE_TRT
  OrtProviderFactoryInterface** f;
  ORT_THROW_ON_ERROR(OrtCreateTRTExecutionProviderFactory(0, &f));
  FACTORY_PTR_HOLDER;
  OrtProvider* out;
  ORT_THROW_ON_ERROR((*f)->CreateProvider(f, &out));
  return std::unique_ptr<IExecutionProvider>((IExecutionProvider*)out);
#else
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
