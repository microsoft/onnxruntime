// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace {

#define CONTENT_NAME elementwise_kernel_src
#include "opencl_generated/math/kernels/elementwise.cl.inc"

}  // namespace

namespace onnxruntime {
namespace opencl {
TensorShape broadcast_shape(const TensorShape& shape1, const TensorShape& shape2) {
  size_t dims1 = shape1.NumDimensions();
  size_t dims2 = shape2.NumDimensions();
  size_t max_dims = dims1 > dims2 ? dims1 : dims2;

  std::vector<int64_t> result_shape(max_dims);

  for (size_t i = 0; i < max_dims; ++i) {
    int64_t dim1 = (i < (max_dims - dims1)) ? 1 : shape1[i - (max_dims - dims1)];
    int64_t dim2 = (i < (max_dims - dims2)) ? 1 : shape2[i - (max_dims - dims2)];
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      throw std::invalid_argument("Shapes are not compatible for broadcasting");
    }
    result_shape[i] = dim1 > dim2 ? dim1 : dim2;
  }

  return TensorShape(result_shape);
}

#define ELEMENT_WISE_OP_IMPL(CLASS_NAME, type)                                                                          \
  class CLASS_NAME : public OpenCLKernel {                                                                              \
   public:                                                                                                              \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {                                                \
      LoadProgram(elementwise_kernel_src, elementwise_kernel_src_len);                                                  \
      LoadKernel(#CLASS_NAME);                                                                                          \
    };                                                                                                                  \
                                                                                                                        \
    Status Compute(OpKernelContext* context) const override {                                                           \
      VLOG_CL_NODE();                                                                                                   \
      auto* a = context->Input<Tensor>(0);                                                                              \
      auto* b = context->Input<Tensor>(1);                                                                              \
      TensorShape out_shape = broadcast_shape(a->Shape(), b->Shape());                                                  \
      auto* c = context->Output(0, out_shape);                                                                          \
      int Shape_sIze = 40;                                                                                              \
      auto A_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      auto B_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      auto C_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      int64_t constant_shape[] = {1, 1, 1, 1, 1};                                                                       \
      const int64_t* a_Shape = (a->Shape().GetDims().data() == nullptr) ? constant_shape : a->Shape().GetDims().data(); \
      const int64_t* b_Shape = (b->Shape().GetDims().data() == nullptr) ? constant_shape : b->Shape().GetDims().data(); \
      const int64_t* c_Shape = (c->Shape().GetDims().data() == nullptr) ? constant_shape : c->Shape().GetDims().data(); \
      exec_->WriteToCLBuffer(A_Shape, a_Shape, Shape_sIze);                                                             \
      exec_->WriteToCLBuffer(B_Shape, b_Shape, Shape_sIze);                                                             \
      exec_->WriteToCLBuffer(C_Shape, c_Shape, Shape_sIze);                                                             \
                                                                                                                        \
      ORT_RETURN_IF_ERROR(                                                                                              \
          KernelLauncher{GetKernel(#CLASS_NAME)}                                                                        \
              .SetBuffers(*a, *b, *c)                                                                                   \
              .SetBuffers(A_Shape, B_Shape, C_Shape)                                                                    \
              .SetArg<cl_int>((a->Shape().GetDims().data() == nullptr) ? 1 : a->Shape().NumDimensions())                \
              .SetArg<cl_int>((b->Shape().GetDims().data() == nullptr) ? 1 : b->Shape().NumDimensions())                \
              .SetArg<cl_int>((c->Shape().GetDims().data() == nullptr) ? 1 : c->Shape().NumDimensions())                \
              .Launch(*exec_, {c->SizeInBytes() / c->DataType()->Size(), 1, 1}));                                       \
                                                                                                                        \
      exec_->ReleaseCLBuffer(A_Shape);                                                                                  \
      exec_->ReleaseCLBuffer(B_Shape);                                                                                  \
      exec_->ReleaseCLBuffer(C_Shape);                                                                                  \
      return Status::OK();                                                                                              \
    }                                                                                                                   \
  };

ELEMENT_WISE_OP_IMPL(add_float, float);
ELEMENT_WISE_OP_IMPL(add_double, double);
ELEMENT_WISE_OP_IMPL(add_int, int32_t);
ELEMENT_WISE_OP_IMPL(add_long, int64_t);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Add, 14, float, add_float);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Add, 14, double, add_double);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Add, 14, int32_t, add_int);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Add, 14, int64_t, add_long);

ELEMENT_WISE_OP_IMPL(div_float, float);
ELEMENT_WISE_OP_IMPL(div_double, double);
ELEMENT_WISE_OP_IMPL(div_int, int32_t);
ELEMENT_WISE_OP_IMPL(div_long, int64_t);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Div, 14, float, div_float);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Div, 14, double, div_double);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Div, 14, int32_t, div_int);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Div, 14, int64_t, div_long);

ELEMENT_WISE_OP_IMPL(mul_float, float);
ELEMENT_WISE_OP_IMPL(mul_double, double);
ELEMENT_WISE_OP_IMPL(mul_int, int32_t);
ELEMENT_WISE_OP_IMPL(mul_long, int64_t);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Mul, 14, float, mul_float);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Mul, 14, double, mul_double);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Mul, 14, int32_t, mul_int);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Mul, 14, int64_t, mul_long);

ELEMENT_WISE_OP_IMPL(sub_float, float);
ELEMENT_WISE_OP_IMPL(sub_double, double);
ELEMENT_WISE_OP_IMPL(sub_int, int32_t);
ELEMENT_WISE_OP_IMPL(sub_long, int64_t);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Sub, 14, float, sub_float);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Sub, 14, double, sub_double);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Sub, 14, int32_t, sub_int);
REG_ELEMENTWISE_TYPED_KERNEL_CLASS(Sub, 14, int64_t, sub_long);

ELEMENT_WISE_OP_IMPL(pow_float, float);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow,
    kOnnxDomain,
    13, 14,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    pow_float)

#define ELEMENT_LOGICAL_OP_IMPL(CLASS_NAME, type)                                                                       \
  class CLASS_NAME : public OpenCLKernel {                                                                              \
   public:                                                                                                              \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {                                                \
      LoadProgram(elementwise_kernel_src, elementwise_kernel_src_len);                                                  \
      LoadKernel(#CLASS_NAME);                                                                                          \
    };                                                                                                                  \
                                                                                                                        \
    Status Compute(OpKernelContext* context) const override {                                                           \
      VLOG_CL_NODE();                                                                                                   \
      auto* a = context->Input<Tensor>(0);                                                                              \
      auto* b = context->Input<Tensor>(1);                                                                              \
      TensorShape out_shape = broadcast_shape(a->Shape(), b->Shape());                                                  \
      auto* c = context->Output(0, out_shape);                                                                          \
      int Shape_sIze = 40;                                                                                              \
      auto A_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      auto B_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      auto C_Shape = exec_->GetScratchBufferTmp(Shape_sIze);                                                            \
      int64_t constant_shape[] = {1, 1, 1, 1, 1};                                                                       \
      const int64_t* a_Shape = (a->Shape().GetDims().data() == nullptr) ? constant_shape : a->Shape().GetDims().data(); \
      const int64_t* b_Shape = (b->Shape().GetDims().data() == nullptr) ? constant_shape : b->Shape().GetDims().data(); \
      const int64_t* c_Shape = (c->Shape().GetDims().data() == nullptr) ? constant_shape : c->Shape().GetDims().data(); \
      exec_->WriteToCLBuffer(A_Shape, a_Shape, Shape_sIze);                                                             \
      exec_->WriteToCLBuffer(B_Shape, b_Shape, Shape_sIze);                                                             \
      exec_->WriteToCLBuffer(C_Shape, c_Shape, Shape_sIze);                                                             \
                                                                                                                        \
      ORT_RETURN_IF_ERROR(                                                                                              \
          KernelLauncher{GetKernel(#CLASS_NAME)}                                                                        \
              .SetBuffers(*a, *b, *c)                                                                                   \
              .SetBuffers(A_Shape, B_Shape, C_Shape)                                                                    \
              .SetArg<cl_int>((a->Shape().GetDims().data() == nullptr) ? 1 : a->Shape().NumDimensions())                \
              .SetArg<cl_int>((b->Shape().GetDims().data() == nullptr) ? 1 : b->Shape().NumDimensions())                \
              .SetArg<cl_int>((c->Shape().GetDims().data() == nullptr) ? 1 : c->Shape().NumDimensions())                \
              .Launch(*exec_, {c->SizeInBytes() / c->DataType()->Size(), 1, 1}));                                       \
                                                                                                                        \
      exec_->ReleaseCLBuffer(A_Shape);                                                                                  \
      exec_->ReleaseCLBuffer(B_Shape);                                                                                  \
      exec_->ReleaseCLBuffer(C_Shape);                                                                                  \
      return Status::OK();                                                                                              \
    }                                                                                                                   \
  };

ELEMENT_LOGICAL_OP_IMPL(equal_bool, bool)
ELEMENT_LOGICAL_OP_IMPL(equal_int, int)
ELEMENT_LOGICAL_OP_IMPL(equal_long, long)
ELEMENT_LOGICAL_OP_IMPL(equal_float, float)
ELEMENT_LOGICAL_OP_IMPL(equal_double, double)

REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, bool, equal_bool);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int32_t, equal_int);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, int64_t, equal_long);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, float, equal_float);
REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(Equal, 13, 18, double, equal_double);

ELEMENT_LOGICAL_OP_IMPL(greater_int, int)
ELEMENT_LOGICAL_OP_IMPL(greater_long, long)
ELEMENT_LOGICAL_OP_IMPL(greater_float, float)
ELEMENT_LOGICAL_OP_IMPL(greater_double, double)

REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, float, greater_int);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, double, greater_long);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int32_t, greater_float);
REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(Greater, 13, int64_t, greater_double);

}  // namespace opencl
}  // namespace onnxruntime
