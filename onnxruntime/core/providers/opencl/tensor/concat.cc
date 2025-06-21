// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/concat.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/concatbase.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME concat_kernel_src
#include "opencl_generated/tensor/kernels/concat.cl.inc"
}  // namespace

class Concat : public OpenCLKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : OpenCLKernel(info), ConcatBase(info) {
    LoadProgram(concat_kernel_src, concat_kernel_src_len);
    LoadKernel("Concat");
  }

  Status Compute(OpKernelContext* context) const override;
};

Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  cl_mem In = exec_->GetScratchBufferTmp(p.output_tensor->SizeInBytes());
  size_t offset = 0;
  for (size_t i = 0; i < p.inputs.size(); ++i) {
    size_t size = p.inputs[i].tensor->SizeInBytes();
    clEnqueueCopyBuffer(exec_->GetCommandQueue(), CL_BUFFER_FROM_TENSOR(*(p.inputs[i].tensor)), In, 0, offset, size, 0, NULL, NULL);
    offset += size;
  }

  auto element_bytes = p.output_tensor->DataType()->Size();
  size_t input_shape_size = p.output_tensor->Shape().NumDimensions() * sizeof(int64_t);
  cl_mem out_shape = exec_->GetScratchBufferTmp(input_shape_size);
  exec_->WriteToCLBuffer(out_shape, p.output_tensor->Shape().GetDims().data(), input_shape_size);

  cl_mem In_shape = exec_->GetScratchBufferTmp(input_shape_size * p.inputs.size());
  offset = 0;
  for (size_t i = 0; i < p.inputs.size(); ++i) {
    clEnqueueWriteBuffer(exec_->GetCommandQueue(), In_shape, CL_TRUE, offset,
                         input_shape_size, p.inputs[i].tensor->Shape().GetDims().data(), 0, nullptr, nullptr);
    offset += input_shape_size;
  }

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Concat")}
          .SetBuffer(*(p.output_tensor))
          .SetBuffers(In, out_shape, In_shape)
          .SetArg<cl_long>(p.inputs.size())
          .SetArg<cl_long>(p.axis)
          .SetArg<cl_long>(p.output_tensor->Shape().NumDimensions())
          .SetArg<cl_long>(element_bytes)
          .Launch(*exec_, {p.output_num_elements, 1, 1}));

  exec_->ReleaseCLBuffer(In_shape);
  exec_->ReleaseCLBuffer(out_shape);
  exec_->ReleaseCLBuffer(In);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(Concat,
                        kOnnxDomain,
                        13,
                        kOpenCLExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Concat);

}  // namespace opencl
}  // namespace onnxruntime
