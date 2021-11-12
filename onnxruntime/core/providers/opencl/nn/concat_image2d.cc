// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "concat.h"

#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME conncat_kernel_src
#include "opencl_generated/nn/kernels/concat_image2d.cl.inc"
}  // namespace

class Concat final : public OpenCLKernel, public ConcatBase {
 public:
  explicit Concat(const OpKernelInfo& info)
      : OpenCLKernel(info), ConcatBase(info) {
    LoadProgram(conncat_kernel_src, conncat_kernel_src_len);
    LoadKernel("ConcatChannel4X");
    LoadKernel("ConcatChannel");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Concat::Compute");
    VLOG_CL_NODE();

    // Number of input tensors to concatenate
    auto input_count = Node().InputArgCount().front();

    // Hold pointers to the input tensors to be used in the PrepareForCompute() step
    InlinedTensorsVector input_tensors;
    input_tensors.reserve(input_count);
    for (int i = 0; i < input_count; ++i) {
      input_tensors.push_back(context->Input<Tensor>(i));
    }

    // Validate inputs and prepare some metadata used during actual compute
    Prepare p;
    auto status = PrepareForCompute(context, input_tensors, p);
    if (!status.IsOK())
      return status;

    // Return at this point if output tensor is going to be empty
    if (p.output_num_elements == 0)
      return Status::OK();
    if (is_stack_ || p.axis != 1) ORT_THROW("[CL] Concat does not support stack mode and only support channel-concat now");
    // Compute values to be placed in the output tensor

    const auto& X_shape = input_tensors[0]->Shape();
    ORT_RETURN_IF(X_shape.NumDimensions() != 4, "Input dimension must be 4, aka, NCHW tensor");

    auto Y_shape = p.output_tensor->Shape();
    const auto* Y = p.output_tensor;
    VLOG_CL_IMAGE2D("Input[1]", input_tensors[0]);
    VLOG_CL_IMAGE2D("Input[0]", input_tensors[1]);
    VLOG_CL_IMAGE2D("Output", Y);

    const auto& N = Y_shape[0];
    const auto& C_out = Y_shape[1];
    const auto& H_out = Y_shape[2];
    const auto& W_out = Y_shape[3];

    auto C_in = X_shape[1];

    // int64_t cin_blocks = CeilDiv(C_in, 4);
    int64_t cout_blocks = CeilDiv(C_out, 4);
    ZoneNamedN(_tracy_Concat, "Concat (kernel launch)", true);
    if (input_tensors[0]->Shape()[1] % 4) {
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("ConcatChannel")}
              .SetInt3(cout_blocks, W_out, N * H_out)
              .SetImage2Ds(*input_tensors[0], *input_tensors[1], *Y)
              .SetInt2(C_in, C_out)
              .Launch(*exec_, {cout_blocks, W_out, N * H_out}));
    } else {
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("ConcatChannel4X")}
              .SetInt3(cout_blocks, W_out, N * H_out)
              .SetImage2Ds(*input_tensors[0], *input_tensors[1], *Y)
              .SetInt2(C_in, C_out)
              .Launch(*exec_, {cout_blocks, W_out, N * H_out}));
    }
    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    4, 10,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    11, 12,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat)

ONNX_OPENCL_OPERATOR_KERNEL(
    Concat,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat)

}  // namespace opencl
}  // namespace onnxruntime
