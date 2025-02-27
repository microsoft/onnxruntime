// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "trilu.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME trilu_kernel_src
#include "opencl_generated/tensor/kernels/trilu.cl.inc"
}  // namespace

class Trilu : public OpenCLKernel {
 public:
  explicit Trilu(const OpKernelInfo& info) : OpenCLKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("upper", &temp).IsOK());
    upper = (int)temp;
    LoadProgram(trilu_kernel_src, trilu_kernel_src_len);
    LoadKernel("Trilu_float");
    LoadKernel("Trilu_double");
  };

  Status Compute(OpKernelContext* context) const override;

  template <typename T>
  Status TriluImpl(const Tensor* X, Tensor* Y, const int64_t k_, int up) const;

 private:
  int upper;
};

template <>
Status Trilu::TriluImpl<float>(const Tensor* X, Tensor* Y, const int64_t k_, int up) const {
  int64_t shape_size = sizeof(int64_t) * X->Shape().NumDimensions();
  auto TensorShape = exec_->GetScratchBufferTmp(shape_size);
  exec_->WriteToCLBuffer(TensorShape, X->Shape().GetDims().data(), shape_size);
  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Trilu_float")}
          .SetBuffers(*X, *Y)
          .SetBuffer(TensorShape)
          .SetArg<cl_long>(X->Shape().NumDimensions())
          .SetArg<cl_long>(k_)
          .SetArg<cl_int>(up)
          .Launch(*exec_, {X->SizeInBytes() / 4, 1, 1}));

  exec_->ReleaseCLBuffer(TensorShape);
  return Status::OK();
}

template <>
Status Trilu::TriluImpl<double>(const Tensor* X, Tensor* Y, const int64_t k_, int up) const {
  int64_t shape_size = sizeof(int64_t) * X->Shape().NumDimensions();
  auto TensorShape = exec_->GetScratchBufferTmp(shape_size);
  exec_->WriteToCLBuffer(TensorShape, X->Shape().GetDims().data(), shape_size);
  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Trilu_double")}
          .SetBuffers(*X, *Y)
          .SetBuffer(TensorShape)
          .SetArg<cl_long>(X->Shape().NumDimensions())
          .SetArg<cl_long>(k_)
          .SetArg<cl_int>(up)
          .Launch(*exec_, {X->SizeInBytes() / 4, 1, 1}));

  exec_->ReleaseCLBuffer(TensorShape);
  return Status::OK();
}

Status Trilu::Compute(OpKernelContext* context) const {
  Status status;
  const auto* X = context->Input<Tensor>(0);
  const auto* k = context->Input<Tensor>(1);
  int64_t k_val = 0;

  int up = upper;

  if (k) {
    ORT_ENFORCE(IsScalarOr1ElementVector(k), "k should be a 1-D or 0-D tensor.");
    k_val = *(k->Data<int64_t>());
  }

  const auto& X_shape = X->Shape();
  auto* Y = context->Output(0, X_shape);

  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  // input validation
  if (X_num_dims < 2) {  // this is getting capture by shape inference code as well
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }

  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      status = TriluImpl<float>(X, Y, k_val, up);
      break;
    case sizeof(double):
      status = TriluImpl<double>(X, Y, k_val, up);
      break;
    // case sizeof(bool):
    //   status = TriluImpl<bool>(X, Y, k, up);
    //   break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kOnnxDomain,
    14,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", BuildKernelDefConstraints<float, double, int32_t, int64_t>()),
    Trilu);

}  // namespace opencl
}  // namespace onnxruntime
