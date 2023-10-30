// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "core/providers/cuda/math/gemm.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "contrib_ops/cuda/math/gemm_float8.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if !defined(DISABLE_FLOAT8_TYPES)
#define GEMM_FLOAT8_CONSTRAINTS BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>()
#else
#define GEMM_FLOAT8_CONSTRAINTS BuildKernelDefConstraints<MLFloat16, BFloat16, float>()
#endif

#define REGISTER_KERNEL()                                            \
  ONNX_OPERATOR_KERNEL_EX(                                           \
      GemmFloat8,                                                    \
      kMSDomain,                                                     \
      1,                                                             \
      kCudaExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("TA", GEMM_FLOAT8_CONSTRAINTS)             \
          .TypeConstraint("TB", GEMM_FLOAT8_CONSTRAINTS)             \
          .TypeConstraint("TR", GEMM_FLOAT8_CONSTRAINTS)             \
          .TypeConstraint("TS", BuildKernelDefConstraints<float>()), \
      GemmFloat8);

REGISTER_KERNEL()

GemmFloat8::GemmFloat8(const OpKernelInfo& info) : CudaKernel(info) {
  transA_ = info.GetAttrOrDefault<int64_t>("transA", 0);
  transB_ = info.GetAttrOrDefault<int64_t>("transB", 0);
  dtype_ = info.GetAttrOrDefault<int64_t>("dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto& device_prop = GetDeviceProp();
  sm_count_ = device_prop.multiProcessorCount;
  alpha_ = info.GetAttrOrDefault<float>("alpha", 1);
  beta_ = info.GetAttrOrDefault<float>("beta", 0);

#if (CUDA_VERSION <= 12000)
  ORT_ENFORCE(beta_ == 0, "CUDA < 12.0 does not support bias, beta must be 0.");
#endif

  std::string stemp = info.GetAttrOrDefault<std::string>("activation", "NONE");
  if (stemp == "NONE") {
    epilogue_ = CUBLASLT_EPILOGUE_DEFAULT;
  } else if (stemp == "RELU") {
    epilogue_ = CUBLASLT_EPILOGUE_RELU;
  } else if (stemp == "GELU") {
    epilogue_ = CUBLASLT_EPILOGUE_GELU;
  } else {
    ORT_THROW("Unexpected value for activation: '", stemp, "'.");
  }
}

Status GemmFloat8::SetCheck(const TensorShape& a_shape, const TensorShape& b_shape, int& M, int& N, int& K) const {
  GemmHelper helper(a_shape, transA_, b_shape, transB_, TensorShape({}));
  if (!helper.State().IsOK())
    return helper.State();

  M = gsl::narrow_cast<int>(helper.M());
  N = gsl::narrow_cast<int>(helper.N());
  K = gsl::narrow_cast<int>(helper.K());
  return helper.State();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
