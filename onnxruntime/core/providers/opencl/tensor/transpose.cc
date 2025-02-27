// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME transpose_kernel_src
#include "opencl_generated/tensor/kernels/transpose.cl.inc"
}  // namespace

class Transpose : public OpenCLKernel, public TransposeBase {
 public:
  explicit Transpose(const OpKernelInfo& info)
      : OpenCLKernel(info), TransposeBase(info) {
    LoadProgram(transpose_kernel_src, transpose_kernel_src_len);
    LoadKernel("Transpose");
  };

  Status Compute(OpKernelContext* context) const override;
};

Status Transpose::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& X = *input_tensor_ptr;
  const TensorShape& input_shape = X.Shape();
  auto input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  TensorShapeVector output_dims(rank);
  const InlinedVector<size_t>* p_perm;
  InlinedVector<size_t> default_perm(rank);
  Status status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);

  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  size_t tensor_size = input_shape.NumDimensions() * sizeof(int64_t);

  cl_mem P_perm = exec_->GetScratchBufferTmp(tensor_size);
  auto Input_shape = exec_->GetScratchBufferTmp(tensor_size);
  auto Output_shape = exec_->GetScratchBufferTmp(tensor_size);
  if (p_perm) {
    exec_->WriteToCLBuffer(P_perm, p_perm->data(), tensor_size);
  }

  exec_->WriteToCLBuffer(Input_shape, X.Shape().GetDims().data(), tensor_size);
  exec_->WriteToCLBuffer(Output_shape, output_shape.GetDims().data(), tensor_size);

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Transpose")}
          .SetBuffers(P_perm, X, Y)
          .SetBuffers(Input_shape, Output_shape)
          .SetArg<cl_long>(input_shape.NumDimensions())
          .Launch(*exec_, {Y.SizeInBytes() / 4, 1, 1}));

  exec_->ReleaseCLBuffer(Input_shape);
  exec_->ReleaseCLBuffer(Output_shape);
  exec_->ReleaseCLBuffer(P_perm);

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13, 20,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Transpose)

}  // namespace opencl
}  // namespace onnxruntime
