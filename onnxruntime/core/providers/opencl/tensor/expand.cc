// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/expand.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME expand_kernel_src
#include "opencl_generated/tensor/kernels/expand.cl.inc"
}  // namespace

Status ExpandComputeOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}

#define EXPAND_OP_IMPL(CLASS_NAME, type)                                                                                  \
  class CLASS_NAME : public OpenCLKernel {                                                                                \
   public:                                                                                                                \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {                                                  \
      LoadProgram(expand_kernel_src, expand_kernel_src_len);                                                              \
      LoadKernel(#CLASS_NAME);                                                                                            \
    };                                                                                                                    \
                                                                                                                          \
    Status Compute(OpKernelContext* context) const override {                                                             \
      const auto& input_data_tensor = *context->Input<Tensor>(0);                                                         \
      const auto& input_shape_tensor = *context->Input<Tensor>(1);                                                        \
                                                                                                                          \
      const auto* p_shape = input_shape_tensor.Data<int64_t>();                                                           \
      TensorShapeVector output_dims{p_shape, p_shape + input_shape_tensor.Shape().Size()};                                \
      TensorShape output_shape(output_dims);                                                                              \
                                                                                                                          \
      ORT_RETURN_IF_ERROR(ExpandComputeOutputShape(Node().Name(), input_data_tensor.Shape(), output_dims, output_shape)); \
      auto& output_tensor = *context->Output(0, output_shape);                                                            \
      if (0 == output_shape.Size()) {                                                                                     \
        return Status::OK();                                                                                              \
      }                                                                                                                   \
      int64_t Ndims = output_shape.NumDimensions();                                                                       \
      const auto& input_shape = input_data_tensor.Shape();                                                                \
      int64_t input_shape_dims = input_shape.NumDimensions();                                                             \
      std::vector<int64_t> input_shape_data(Ndims, 1);                                                                    \
      for (int64_t i = Ndims - input_shape.NumDimensions(); i < Ndims; i++) {                                             \
        input_shape_data[i] = input_shape.GetDims().data()[i - Ndims + input_shape.NumDimensions()];                      \
      }                                                                                                                   \
      size_t Shape_Size = Ndims * sizeof(int64_t);                                                                        \
      auto input_Shape_cl = exec_->GetScratchBufferTmp(Shape_Size);                                                       \
      exec_->WriteToCLBuffer(input_Shape_cl, input_shape_data.data(), Shape_Size);                                        \
      auto output_Shape_cl = exec_->GetScratchBufferTmp(Shape_Size);                                                      \
      exec_->WriteToCLBuffer(output_Shape_cl, output_shape.GetDims().data(), Shape_Size);                                 \
      ORT_RETURN_IF_ERROR(                                                                                                \
          KernelLauncher{GetKernel(#CLASS_NAME)}                                                                          \
              .SetBuffers(input_data_tensor, output_tensor)                                                               \
              .SetBuffers(input_Shape_cl, output_Shape_cl)                                                                \
              .SetArg<cl_long>((cl_long)Ndims)                                                                            \
              .Launch(*exec_, {output_tensor.SizeInBytes() / output_tensor.DataType()->Size(), 1, 1}));                   \
                                                                                                                          \
      exec_->ReleaseCLBuffer(input_Shape_cl);                                                                             \
      exec_->ReleaseCLBuffer(output_Shape_cl);                                                                            \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
  };

EXPAND_OP_IMPL(expand_float, float);
EXPAND_OP_IMPL(expand_double, double);
EXPAND_OP_IMPL(expand_int, int32_t);
EXPAND_OP_IMPL(expand_long, int64_t);
EXPAND_OP_IMPL(expand_uint, uint32_t);
EXPAND_OP_IMPL(expand_ulong, uint64_t);
EXPAND_OP_IMPL(expand_uchar, bool);

#define ONNX_OPENCL_OPERATOR_TYPED_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kOnnxDomain, ver, type, kOpenCLExecutionProvider, builder, __VA_ARGS__)

#define REG_EXPAND_LOGICALOP_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS) \
  ONNX_OPENCL_OPERATOR_TYPED_KERNEL(                                            \
      OP_TYPE,                                                                  \
      VERSION,                                                                  \
      TYPE,                                                                     \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())             \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                              \
      KERNEL_CLASS);

REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, double, expand_double);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, float, expand_float);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, int32_t, expand_int);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, int64_t, expand_long);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, uint32_t, expand_uint);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, uint64_t, expand_ulong);
REG_EXPAND_LOGICALOP_TYPED_KERNEL(Expand, 13, bool, expand_uchar);

}  // namespace opencl
}  // namespace onnxruntime
