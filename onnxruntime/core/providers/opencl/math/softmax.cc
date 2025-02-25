// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME softmax_kernel_src
#include "opencl_generated/math/kernels/softmax.cl.inc"
}  // namespace

template <typename T>
class Softmax final : public OpenCLKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpenCLKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }

    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
    LoadProgram(softmax_kernel_src, softmax_kernel_src_len);
    LoadKernel("softmax_float_opset13");
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  // Status ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
  //                    concurrency::ThreadPool* thread_pool) const;

  Status ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis, OpKernelContext* ctx) const;

  int axis_;
  int opset_;
  bool log_softmax_;
};

template <typename T>
Status Softmax<T>::Compute(OpKernelContext* context) const {
  VLOG_CL_NODE();
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  size_t rank = X_shape.NumDimensions();
  auto* Y = context->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  if (opset_ < 13) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Softmax: Unsupported opset < 13 now.");
  } else {
    return ComputeImplOpset13(*X, *Y, axis, context);
  }
}

// opset-13 and above
template <typename T>
Status Softmax<T>::ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                                      OpKernelContext* ctx) const {
  const auto& X_shape = input.Shape();
  size_t rank = X_shape.NumDimensions();
  // Softmax:Unsupported pre-transpose input tensor

  int Shape_sIze = 40;
  auto Input_shape = exec_->GetScratchBufferTmp(Shape_sIze);
  exec_->WriteToCLBuffer(Input_shape, input.Shape().GetDims().data(), Shape_sIze);

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("softmax_float_opset13")}
          .SetBuffers(input, Input_shape, output)
          .SetArg<int>((int)(input.Shape().NumDimensions()))
          .SetArg<int>((int)(axis))
          .Launch(*exec_, {output.SizeInBytes() / (4 * input.Shape().GetDims().data()[axis]), 1, 1}));

  exec_->ReleaseCLBuffer(Input_shape);
  return Status::OK();

  // bool is_transpose_required = false;
  // Tensor transposed_input;
  // std::vector<int64_t> transposed_input_dims;
  // Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
  // std::vector<size_t> permutation(rank);

  // // The "semantic" meaning of axis has changed in opset-13.
  // // Please compare: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
  // // with https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-11 for detailed explanations
  // // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
  // // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
  // // the innermost dim
  // if (axis != (rank - 1)) {
  //   is_transpose_required = true;
  // }

  // if (is_transpose_required) {
  //   return Status(common::ONNXRUNTIME, common::FAIL, "Softmax:Unsupported pre-transpose input tensor.");
  // }else{

  //   auto Input = exec_->GetScratchBufferTmp(input.SizeInBytes());
  //   auto Output = exec_->GetScratchBufferTmp(output.SizeInBytes());
  //   exec_->WriteToCLBuffer(Input,input.Data<T>() , input.SizeInBytes());
  //   int Shape_sIze = 40;
  //   auto Input_shape = exec_->GetScratchBufferTmp(Shape_sIze);
  //   exec_->WriteToCLBuffer(Input_shape, input.Shape().GetDims().data(), Shape_sIze);

  //   ORT_RETURN_IF_ERROR(
  //       KernelLauncher{GetKernel("softmax_float_opset13")}
  //           .SetBuffers(Input, Input_shape, Output)
  //           .SetArg<int>((int)(input.Shape().NumDimensions()))
  //           .SetArg<int>((int)(axis))
  //           .Launch(*exec_, {output.SizeInBytes()/4, 1, 1}));

  //   exec_->ReadFromCLBuffer(Output,output.MutableData<T>(),output.SizeInBytes());
  //   exec_->ReleaseCLBuffer(Input);
  //   exec_->ReleaseCLBuffer(Input_shape);
  //   exec_->ReleaseCLBuffer(Output);
  //   return Status::OK();
  // }

  return Status::OK();
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Softmax,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);
}  // namespace opencl
}  // namespace onnxruntime
