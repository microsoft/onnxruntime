// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/matmul.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {


ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    MatMul);



Status MatMulNativeProgram::GenerateShaderCode(ShaderHelper& sh) const {

    return Status::OK();
}


Status MatMulProgram::GenerateShaderCode(ShaderHelper& sh) const {
    return Status::OK();
}

Status MatMul::ComputeInternal(ComputeContext& context) const {
    // calculate output shape
    MatMulComputeHelper helper;
    const auto* a = context.Input(0);
    const auto* b = context.Input(1);
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));

    if (helper.N() < 8 && helper.K() < 8) {
        // call MatMulNativeProgram
        MatMulNativeProgram program{helper.OutputShape()};


    } else {

        // const batchA = ShapeUtil.size(context.inputs[0].dims.slice(0, -2));
        // const batchB = ShapeUtil.size(context.inputs[1].dims.slice(0, -2));
        int64_t batchA = a->Shape().SizeToDimension(a->Shape().NumDimensions() - 2);
        int64_t batchB = b->Shape().SizeToDimension(b->Shape().NumDimensions() - 2);

        // check if A is  batch of vector (bach is not 1, M is 1) and B is a matrix (batch is 1)
        if (batchA != 1 && m == 1 && batchB == 1) {
            // optimization for batched vector matrix multiplication
            // const reshapedA = context.inputs[0].reshape([1, batchA, K]);
            // const reshapedB = context.inputs[1].reshape([1, K, N]);
            // const matmulOutputShape = [1, batchA, N];
            // const matmulInputs = [reshapedA, reshapedB];

            // dimensions of A: [1,`batchA`,K]
            const gsl::span<const int64_t>& dims_a = {1, batchA, helper.K()};
            // dimensions of B: [1,K,N]
            const gsl::span<const int64_t>& dims_b = {1, helper.K(), helper.N()};

            a.Reshape(TensorShape(dims_a));
            b.Reshape(TensorShape(dims_b));

            TensorShape output_shape = {1, batchA, helper.N()};


            MatMulProgram program;
        } else {
            // call MatMulProgram
            MatMulProgram program;
        }


    }

    return Status::OK();

}


}  // namespace webgpu
}  // namespace onnxruntime
