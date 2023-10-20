// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_ROCM

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/providers/rocm/rocm_context.h"
#include "onnxruntime_lite_custom_op.h"

void rocm_add(int64_t, float*, const float*, const float*, hipStream_t compute_stream);

using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Rocm {

void KernelOne(const Ort::Custom::RocmContext& rocm_ctx,
               const Ort::Custom::Tensor<float>& X,
               const Ort::Custom::Tensor<float>& Y,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  CUSTOM_ENFORCE(rocm_ctx.hip_stream, "failed to fetch hip stream");
  CUSTOM_ENFORCE(rocm_ctx.miopen_handle, "failed to fetch miopen handle");
  CUSTOM_ENFORCE(rocm_ctx.rblas_handle, "failed to fetch rocblas handle");
  auto z_raw = Z.Allocate(input_shape);
  rocm_add(Z.NumberOfElement(), z_raw, X.Data(), Y.Data(), rocm_ctx.hip_stream);
}

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "ROCMExecutionProvider", KernelOne)};
  domain.Add(c_CustomOpOne.get());
}

}  // namespace Rocm

#endif