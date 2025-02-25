// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/where.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME where_kernel_src
#include "opencl_generated/tensor/kernels/where.cl.inc"
}  // namespace

// Compute where operator output shape based upon three way broad-casting.
Status OutputShape(const std::string& node_name, const TensorShape& cond_shape,
                   const TensorShape& x_shape, const TensorShape& y_shape, TensorShape& out_shape) {
  size_t cond_rank = cond_shape.NumDimensions();
  size_t x_rank = x_shape.NumDimensions();
  size_t y_rank = y_shape.NumDimensions();
  size_t out_rank = std::max(std::max(cond_rank, x_rank), y_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t cond_dim = 1;
    if (i < cond_rank)
      cond_dim = cond_shape[cond_rank - 1 - i];

    int64_t x_dim = 1;
    if (i < x_rank)
      x_dim = x_shape[x_rank - 1 - i];

    int64_t y_dim = 1;
    if (i < y_rank)
      y_dim = y_shape[y_rank - 1 - i];

    int64_t out_dim = std::max(std::max(cond_dim, x_dim), y_dim);
    // special case to handle a dim of 0 which can be broadcast with a 1
    if (out_dim == 1)
      out_dim = std::min(std::min(cond_dim, x_dim), y_dim);

    if (cond_dim != out_dim && cond_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": condition operand cannot broadcast on dim ", cond_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (x_dim != out_dim && x_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": X operand cannot broadcast on dim ", x_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (y_dim != out_dim && y_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": Y operand cannot broadcast on dim ", y_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }

  out_shape = TensorShape(output_dims);
  return Status::OK();
}

#define WHERE_OP_IMPL(CLASS_NAME, type)                                                                   \
  class CLASS_NAME : public OpenCLKernel {                                                                \
   public:                                                                                                \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {                                  \
      LoadProgram(where_kernel_src, where_kernel_src_len);                                                \
      LoadKernel(#CLASS_NAME);                                                                            \
    };                                                                                                    \
                                                                                                          \
    Status Compute(OpKernelContext* context) const override {                                             \
      const auto* const condition = context->Input<Tensor>(0);                                            \
      const auto* const X = context->Input<Tensor>(1);                                                    \
      const auto* const Y = context->Input<Tensor>(2);                                                    \
      ORT_ENFORCE(condition&& X&& Y, "condition, X, and Y inputs are required!");                         \
      auto const& condition_shape = condition->Shape();                                                   \
      auto const& X_shape = X->Shape();                                                                   \
      auto const& Y_shape = Y->Shape();                                                                   \
                                                                                                          \
      TensorShape output_shape;                                                                           \
      ORT_RETURN_IF_ERROR(OutputShape(Node().Name(), condition_shape, X_shape, Y_shape, output_shape));   \
      auto output_tensor = context->Output(0, output_shape);                                              \
                                                                                                          \
      if (output_shape.Size() == 0)                                                                       \
        return Status::OK();                                                                              \
      int64_t Ndims = output_shape.NumDimensions();                                                       \
      int64_t Shape_Size = sizeof(int64_t) * Ndims;                                                       \
      std::vector<int64_t> cond_shape(Ndims, 1);                                                          \
      for (int64_t i = Ndims - condition_shape.NumDimensions(); i < Ndims; i++) {                         \
        cond_shape[i] = condition_shape.GetDims().data()[i - Ndims + condition_shape.NumDimensions()];    \
      }                                                                                                   \
      auto condition_Shape = exec_->GetScratchBufferTmp(Shape_Size);                                      \
      exec_->WriteToCLBuffer(condition_Shape, cond_shape.data(), Shape_Size);                             \
      auto condition_stride = exec_->GetScratchBufferTmp(Shape_Size);                                     \
      std::vector<int64_t> cond_strides(Ndims, 1);                                                        \
      for (int i = Ndims - 2; i >= 0; --i) {                                                              \
        cond_strides[i] = cond_strides[i + 1] * cond_shape[i + 1];                                        \
      }                                                                                                   \
      exec_->WriteToCLBuffer(condition_stride, cond_strides.data(), Shape_Size);                          \
      std::vector<int64_t> x_shape(Ndims, 1);                                                             \
      for (int64_t i = Ndims - X_shape.NumDimensions(); i < Ndims; i++) {                                 \
        x_shape[i] = X_shape.GetDims().data()[i - Ndims + X_shape.NumDimensions()];                       \
      }                                                                                                   \
      auto inputx_Shape = exec_->GetScratchBufferTmp(Shape_Size);                                         \
      exec_->WriteToCLBuffer(inputx_Shape, x_shape.data(), Shape_Size);                                   \
      auto inputx_stride = exec_->GetScratchBufferTmp(Shape_Size);                                        \
      std::vector<int64_t> x_strides(Ndims, 1);                                                           \
      for (int i = Ndims - 2; i >= 0; --i) {                                                              \
        x_strides[i] = x_strides[i + 1] * x_shape[i + 1];                                                 \
      }                                                                                                   \
      exec_->WriteToCLBuffer(inputx_stride, x_strides.data(), Shape_Size);                                \
      std::vector<int64_t> y_shape(Ndims, 1);                                                             \
      for (int64_t i = Ndims - Y_shape.NumDimensions(); i < Ndims; i++) {                                 \
        y_shape[i] = Y_shape.GetDims().data()[i - Ndims + Y_shape.NumDimensions()];                       \
      }                                                                                                   \
      auto inputy_Shape = exec_->GetScratchBufferTmp(Shape_Size);                                         \
      exec_->WriteToCLBuffer(inputy_Shape, y_shape.data(), Shape_Size);                                   \
      auto inputy_stride = exec_->GetScratchBufferTmp(Shape_Size);                                        \
      std::vector<int64_t> y_strides(Ndims, 1);                                                           \
      for (int i = Ndims - 2; i >= 0; --i) {                                                              \
        y_strides[i] = y_strides[i + 1] * y_shape[i + 1];                                                 \
      }                                                                                                   \
      exec_->WriteToCLBuffer(inputy_stride, y_strides.data(), Shape_Size);                                \
      auto output_Shape = exec_->GetScratchBufferTmp(Shape_Size);                                         \
      exec_->WriteToCLBuffer(output_Shape, output_shape.GetDims().data(), Shape_Size);                    \
      auto output_stride = exec_->GetScratchBufferTmp(Shape_Size);                                        \
      std::vector<int64_t> o_strides(Ndims, 1);                                                           \
      for (int i = Ndims - 2; i >= 0; --i) {                                                              \
        o_strides[i] = o_strides[i + 1] * output_shape.GetDims().data()[i + 1];                           \
      }                                                                                                   \
      exec_->WriteToCLBuffer(output_stride, o_strides.data(), Shape_Size);                                \
      ORT_RETURN_IF_ERROR(                                                                                \
          KernelLauncher{GetKernel(#CLASS_NAME)}                                                          \
              .SetBuffers(*condition, *X, *Y)                                                             \
              .SetBuffer(*output_tensor)                                                                  \
              .SetBuffers(condition_Shape, condition_stride)                                              \
              .SetBuffers(inputx_Shape, inputx_stride)                                                    \
              .SetBuffers(inputy_Shape, inputy_stride)                                                    \
              .SetBuffers(output_Shape, output_stride)                                                    \
              .SetArg<cl_long>((cl_long)Ndims)                                                            \
              .Launch(*exec_, {output_tensor->SizeInBytes() / output_tensor->DataType()->Size(), 1, 1})); \
                                                                                                          \
      exec_->ReleaseCLBuffer(condition_Shape);                                                            \
      exec_->ReleaseCLBuffer(inputx_Shape);                                                               \
      exec_->ReleaseCLBuffer(inputy_Shape);                                                               \
      exec_->ReleaseCLBuffer(condition_stride);                                                           \
      exec_->ReleaseCLBuffer(inputx_stride);                                                              \
      exec_->ReleaseCLBuffer(inputy_stride);                                                              \
      exec_->ReleaseCLBuffer(output_Shape);                                                               \
      exec_->ReleaseCLBuffer(output_stride);                                                              \
      return Status::OK();                                                                                \
    }                                                                                                     \
  };

WHERE_OP_IMPL(where_float, float);
WHERE_OP_IMPL(where_double, double);
WHERE_OP_IMPL(where_int, int32_t);
WHERE_OP_IMPL(where_long, int64_t);

#define ONNX_OPENCL_OPERATOR_VERSIONED_TYPED_KERNEL(name, startver, endver, type, builder, ...)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kOnnxDomain, startver, endver, type, kOpenCLExecutionProvider, builder, \
                                          __VA_ARGS__)

#define OPENCL_WHERE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_OPENCL_OPERATOR_VERSIONED_TYPED_KERNEL(                                                     \
      OP_TYPE,                                                                                     \
      VERSION_FROM, VERSION_TO,                                                                    \
      TYPE,                                                                                        \
      (*KernelDefBuilder::Create())                                                                \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),                               \
      KERNEL_CLASS);

OPENCL_WHERE_VERSIONED_TYPED_KERNEL(Where, 9, 15, float, where_float);
OPENCL_WHERE_VERSIONED_TYPED_KERNEL(Where, 9, 15, double, where_double);
OPENCL_WHERE_VERSIONED_TYPED_KERNEL(Where, 9, 15, int32_t, where_int);
OPENCL_WHERE_VERSIONED_TYPED_KERNEL(Where, 9, 15, int64_t, where_long);

}  // namespace opencl
}  // namespace onnxruntime
