// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove GetDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/math/gemm.h"
#include "contrib_ops/rocm/function_op/dummy_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    DummyGemm,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create()),
    DummyGemm);

using namespace ONNX_NAMESPACE;

Status DummyGemm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  auto dtype = input->DataType();
  if (dtype == DataTypeImpl::GetType<float>()) {
    std::cout << "here run into float gemm" << std::endl;
    return gemm_float_.Compute(context);
  } else if (dtype == DataTypeImpl::GetType<MLFloat16>()) {
    std::cout << "here run into half gemm" << std::endl;
    return gemm_half_.Compute(context);
  } else {
    return ORT_MAKE_STATUS(NONE, INVALID_ARGUMENT, "not support input datatype.");
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
