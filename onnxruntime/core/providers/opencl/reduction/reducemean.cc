
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reducemean.h"
#include "core/providers/common.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME reducemean_kernel_src
#include "opencl_generated/reduction/kernels/reducemean.cl.inc"
}  // namespace

template <typename T>
class ReduceMean final : public OpenCLKernel, public ReduceKernelBase<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : OpenCLKernel(info), ReduceKernelBase<true>(info) {
    LoadProgram(reducemean_kernel_src, reducemean_kernel_src_len);
    LoadKernel("ReduceMean_float");
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
Status ReduceMean<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const TensorShape& input_shape = input->Shape();
  int64_t num_dims = input_shape.NumDimensions();

  // Get the axes and keepdims flag from the base class
  TensorShapeVector axes;
  for (auto id : axes_) {
    axes.push_back(HandleNegativeAxis(id, num_dims));
  }

  bool keepdims = keepdims_;
  // Prepare the output shape
  TensorShapeVector keepdimsoutshape;
  TensorShapeVector Oshape(5);
  // If noop_with_empty_axes_ is true and axes is empty, copy input to output
  if (this->noop_with_empty_axes_ && axes.empty()) {
    Tensor* Y = context->Output(0, input_shape);
    CopyCpuTensor(input, Y);
    return Status::OK();
  }
  if (axes.empty() && noop_with_empty_axes_ == false) {
    for (int i = 0; i < num_dims; i++) {
      axes.push_back(i);
    }
  }
  std::vector<bool> is_reduced_dim(num_dims, false);
  for (auto axis : axes) {
    is_reduced_dim[axis] = true;
  }
  for (int64_t i = 0; i < num_dims; ++i) {
    if (!is_reduced_dim[i]) {
      Oshape[i] = input_shape.GetDims().data()[i];
      keepdimsoutshape.push_back(input_shape.GetDims().data()[i]);
    } else {
      if (keepdims) {
        keepdimsoutshape.push_back(1);
      }
      Oshape[i] = 1;
    }
  }
  if (keepdimsoutshape.empty()) {
    keepdimsoutshape.push_back(1);
  }
  TensorShape output_shape(keepdimsoutshape);
  // Allocate the output tensor
  Tensor* output = context->Output(0, output_shape);

  size_t tensor_size = 40;
  auto Input_shape = exec_->GetScratchBufferTmp(tensor_size);
  auto Output_shape = exec_->GetScratchBufferTmp(tensor_size);
  auto axi = exec_->GetScratchBufferTmp(tensor_size);
  exec_->WriteToCLBuffer(Input_shape, input->Shape().GetDims().data(), tensor_size);
  exec_->WriteToCLBuffer(Output_shape, Oshape.data(), tensor_size);
  exec_->WriteToCLBuffer(axi, axes.data(), sizeof(int64_t) * axes.size());
  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("ReduceMean_float")}
          .SetBuffers(*input, *output)
          .SetBuffers(Input_shape, Output_shape, axi)
          .SetArg<cl_int>(axes.size())
          .SetArg<cl_int>(input_shape.NumDimensions())
          .Launch(*exec_, {output->SizeInBytes() / 4, 1, 1}));

  exec_->ReleaseCLBuffer(Input_shape);
  exec_->ReleaseCLBuffer(Output_shape);
  exec_->ReleaseCLBuffer(axi);
  return Status::OK();
}
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceMean,
    kOnnxDomain,
    13, 17,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReduceMean<float>)

}  // namespace opencl
}  // namespace onnxruntime
