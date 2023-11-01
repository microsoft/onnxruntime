// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(ENABLE_TRAINING)

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/providers/cuda/cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

// #include <cuda.h>
// #include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Cuda {

void KernelOne(const Ort::Custom::CudaContext& cuda_ctx,
               const Ort::Custom::Tensor<float>& X,
               const Ort::Custom::Tensor<float>& Y,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  CUSTOM_ENFORCE(cuda_ctx.cuda_stream, "failed to fetch cuda stream");
  CUSTOM_ENFORCE(cuda_ctx.cudnn_handle, "failed to fetch cudnn handle");
  CUSTOM_ENFORCE(cuda_ctx.cublas_handle, "failed to fetch cublas handle");
  void* deferred_cpu_mem = cuda_ctx.AllocDeferredCpuMem(sizeof(int32_t));
  CUSTOM_ENFORCE(deferred_cpu_mem, "failed to allocate deferred cpu allocator");
  cuda_ctx.FreeDeferredCpuMem(deferred_cpu_mem);
  auto z_raw = Z.Allocate(input_shape);
  cuda_add(Z.NumberOfElement(), z_raw, X.Data(), Y.Data(), cuda_ctx.cuda_stream);
}

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CUDAExecutionProvider", KernelOne)};
  domain.Add(c_CustomOpOne.get());
}

}  // namespace Cuda

#endif