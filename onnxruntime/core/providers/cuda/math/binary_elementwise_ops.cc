// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

template <>
Status BinaryElementwise<ShouldNotBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  p->lhs_tensor = context->Input<Tensor>(0);
  p->rhs_tensor = context->Input<Tensor>(1);
  if (!(p->lhs_tensor->Shape() == p->rhs_tensor->Shape()))
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, Node().Name(), ": mismatching input shapes: ",
                           p->lhs_tensor->Shape().ToString(), " != ", p->rhs_tensor->Shape().ToString());
  p->output_tensor = context->Output(0, p->lhs_tensor->Shape());
  p->output_rank_or_simple_broadcast = static_cast<int32_t>(SimpleBroadcast::NoBroadcast);
  return Status::OK();
}

Status ComputeOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
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

Status BinaryElementwiseBroadcastPrepare(
    const Tensor* lhs_tensor,
    const Tensor* rhs_tensor,
    Tensor* output_tensor,
    BinaryElementwisePreparation* p,
    const TensorShape* override_lhs_shape,
    const TensorShape* override_rhs_shape) {
  p->lhs_tensor = lhs_tensor;
  p->rhs_tensor = rhs_tensor;
  const auto& lhs_shape = override_lhs_shape ? *override_lhs_shape : lhs_tensor->Shape();
  const auto& rhs_shape = override_rhs_shape ? *override_rhs_shape : rhs_tensor->Shape();

  p->output_tensor = output_tensor;
  const auto& output_shape = output_tensor->Shape();

  ORT_RETURN_IF_ERROR(p->BinaryElementwiseBroadcastPrepareHelper(lhs_shape, rhs_shape, output_shape));

  return Status::OK();
}

template <>
Status BinaryElementwise<ShouldBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  auto lhs_tensor = context->Input<Tensor>(0);
  auto rhs_tensor = context->Input<Tensor>(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), lhs_shape, rhs_shape, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, p));

  return Status::OK();
}

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, class_name, ver, T)       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      x,                                                                        \
      kOnnxDomain,                                                              \
      ver,                                                                      \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      class_name<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, ver, T) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, x, ver, T)

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_NONTEMP(x, class_name, ver, ...)              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                         \
      x,                                                                                 \
      kOnnxDomain,                                                                       \
      ver,                                                                               \
      kCudaExecutionProvider,                                                            \
      KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<>(__VAR_ARGS__)), \
      class_name);

#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(x, ver, T)                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                          \
      x,                                                                                                                                  \
      kOnnxDomain,                                                                                                                        \
      ver,                                                                                                                                \
      T,                                                                                                                                  \
      kCudaExecutionProvider,                                                                                                             \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      x<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                         \
      x,                                                                           \
      kOnnxDomain,                                                                 \
      startver,                                                                    \
      endver,                                                                      \
      T,                                                                           \
      kCudaExecutionProvider,                                                      \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),    \
      x<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_CLASS(x, class_name, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                           \
      x,                                                                                             \
      kOnnxDomain,                                                                                   \
      startver,                                                                                      \
      endver,                                                                                        \
      T,                                                                                             \
      kCudaExecutionProvider,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                      \
      class_name<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                         \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare;                                                                        \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                             \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                \
        Stream(),                                                                                                \
        prepare.output_rank_or_simple_broadcast,                                                                 \
        &prepare.lhs_padded_strides,                                                                             \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        &prepare.rhs_padded_strides,                                                                             \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        &prepare.fdm_output_strides,                                                                             \
        prepare.fdm_H,                                                                                           \
        prepare.fdm_C,                                                                                           \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        prepare.output_tensor->Shape().Size());                                                                  \
    return Status::OK();                                                                                         \
  }

#define BINARY_OP_VERSIONED_TYPED(name, startver, endver, T) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, T)

#define BINARY_OP_TYPED(name, ver, T)                    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

#define BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, T)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_CLASS(name, class_name, startver, endver, T) \
  BINARY_ELEMENTWISE_COMPUTE(class_name, T)

#define BINARY_LOGICALOP_TYPED(name, ver, T)                       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

// since different ops has different types, we cannot use BINARY_OPS() directly
// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define BINARY_OP_TYPED_BF16(name, ver) BINARY_OP_TYPED(name, ver, BFloat16)
#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_BF16(name, ver) BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, BFloat16)
#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED_BF16(name, ver) BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, BFloat16)
#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_BF16(name, startver, endver) BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, BFloat16)
#define BINARY_OP_TYPED_VERSIONED_V_BF16(name, class_name, startver, endver) BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, BFloat16)
#else
#define BINARY_OP_TYPED_BF16(name, ver)
#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_BF16(name, ver)
#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED_BF16(name, ver)
#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_BF16(name, startver, endver)
#define BINARY_OP_TYPED_VERSIONED_V_BF16(name, class_name, startver, endver)
#endif

#define BINARY_OP_VERSIONED_HFD(name, startver, endver)         \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, MLFloat16)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, float)      \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, double)

#define BINARY_OP_VERSIONED_UZILHFD(name, startver, endver)   \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint32_t) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint64_t) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_VERSIONED_HFD(name, startver, endver)

#define BINARY_OP_HFD(name, ver)        \
  BINARY_OP_TYPED(name, ver, MLFloat16) \
  BINARY_OP_TYPED_BF16(name, ver)       \
  BINARY_OP_TYPED(name, ver, float)     \
  BINARY_OP_TYPED(name, ver, double)

#define BINARY_OP_UZILHFD(name, ver)   \
  BINARY_OP_TYPED(name, ver, uint32_t) \
  BINARY_OP_TYPED(name, ver, uint64_t) \
  BINARY_OP_TYPED(name, ver, int32_t)  \
  BINARY_OP_TYPED(name, ver, int64_t)  \
  BINARY_OP_HFD(name, ver)

#define BINARY_OP_REGISTER_VERSIONED_OIL(name, startver, endver)                      \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, bool)    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)

#define BINARY_LOGICALOP_REGISTER_OIL(name, ver)                         \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, bool)    \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int32_t) \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int64_t)

#define BINARY_OP_REGISTER_HFD(name, ver)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, MLFloat16) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_BF16(name, ver)       \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, float)     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, double)

#define BINARY_OP_REGISTER_UZILHFD(name, ver)                   \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, uint32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, uint64_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int32_t)  \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int64_t)  \
  BINARY_OP_REGISTER_HFD(name, ver)

#define BINARY_LOGICALOP_REGISTER_UZILHFD(name, ver)                       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, uint32_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, uint64_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int32_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int64_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, MLFloat16) \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED_BF16(name, ver)       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, float)     \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, double)

#define BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_BF16(name, startver, endver)       \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, float)     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, double)

#define BINARY_OP_REGISTER_VERSIONED_CLASS_HFD(name, class_name, startver, endver) \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, MLFloat16)       \
  BINARY_OP_TYPED_VERSIONED_V_BF16(name, class_name, startver, endver)             \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, float)           \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, double)

#define BINARY_OP_REGISTER_VERSIONED_UZILHFD(name, startver, endver)                   \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint64_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)

BINARY_OP_VERSIONED_UZILHFD(Add, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Sub, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Mul, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Div, 7, 12)

BINARY_OP_UZILHFD(Add, 13)
BINARY_OP_UZILHFD(Sub, 13)
BINARY_OP_UZILHFD(Mul, 13)
BINARY_OP_UZILHFD(Div, 13)

BINARY_OP_REGISTER_VERSIONED_CLASS_HFD(Pow, Pow_7, 7, 11)
BINARY_LOGICALOP_TYPED(And, 7, bool)
BINARY_LOGICALOP_TYPED(Or, 7, bool)
BINARY_LOGICALOP_TYPED(Xor, 7, bool)
BINARY_OP_VERSIONED_HFD(PRelu, 7, 8)
BINARY_OP_HFD(PRelu, 9)

// Pow since version 12
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow,
    kOnnxDomain,
    12, 12,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()).TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()),
    Pow);

ONNX_OPERATOR_KERNEL_EX(
    Pow,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()).TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()),
    Pow);

namespace pow12_internal {
template <class T>
Status DispatchOnFirstArg(cudaStream_t stream, const BinaryElementwisePreparation& prepare) {
  namespace on = ONNX_NAMESPACE;
  Status s;
  switch (prepare.rhs_tensor->GetElementType()) {
    case on::TensorProto_DataType_INT32:
      ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<int32_t>::MappedType>(
          stream,
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
          &prepare.rhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<int32_t>::MappedType*>(prepare.rhs_tensor->template Data<int32_t>()),
          &prepare.fdm_output_strides,
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
      break;
    case on::TensorProto_DataType_INT64:
      ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<int64_t>::MappedType>(
          stream,
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
          &prepare.rhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<int64_t>::MappedType*>(prepare.rhs_tensor->template Data<int64_t>()),
          &prepare.fdm_output_strides,
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
      break;
    case on::TensorProto_DataType_FLOAT:
      ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<float>::MappedType>(
          stream,
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
          &prepare.rhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<float>::MappedType*>(prepare.rhs_tensor->template Data<float>()),
          &prepare.fdm_output_strides,
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
      break;
    case on::TensorProto_DataType_DOUBLE:
      ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<double>::MappedType>(
          stream,
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
          &prepare.rhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<double>::MappedType*>(prepare.rhs_tensor->template Data<double>()),
          &prepare.fdm_output_strides,
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
      break;
    case on::TensorProto_DataType_FLOAT16:
      ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<MLFloat16>::MappedType>(
          stream,
          prepare.output_rank_or_simple_broadcast,
          &prepare.lhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
          &prepare.rhs_padded_strides,
          reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType*>(prepare.rhs_tensor->template Data<MLFloat16>()),
          &prepare.fdm_output_strides,
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
      break;
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported Y type: ",
                          DataTypeImpl::ToString(prepare.rhs_tensor->DataType()));
  }
  return s;
}
}  // namespace pow12_internal

Status Pow::ComputeInternal(OpKernelContext* context) const {
  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));
  namespace on = ONNX_NAMESPACE;
  using namespace pow12_internal;

  Status s;

  switch (prepare.lhs_tensor->GetElementType()) {
    case on::TensorProto_DataType_INT32:
      s = DispatchOnFirstArg<int32_t>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_INT64:
      s = DispatchOnFirstArg<int64_t>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_FLOAT:
      s = DispatchOnFirstArg<float>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_DOUBLE:
      s = DispatchOnFirstArg<double>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_FLOAT16:
      s = DispatchOnFirstArg<MLFloat16>(Stream(), prepare);
      break;
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported X type: ",
                          DataTypeImpl::ToString(prepare.lhs_tensor->DataType()));
  }
  return s;
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T, typename CudaT>
Status CompareFunction<T, CudaT>::CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const {
  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));

  Impl_Compare(
      Stream(),
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<ToCudaType<bool>::MappedType*>(prepare.output_tensor->template MutableData<bool>()),
      prepare.output_tensor->Shape().Size());

  return Status::OK();
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Greater<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &ImplT2_Greater);

  return Status::OK();
}

template <typename T>
Status Equal<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &ImplT2_Equal);

  return Status::OK();
}

//Less op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Less<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &ImplT2_Less);

  return Status::OK();
}

//GreaterOrEqual op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status GreaterOrEqual<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &ImplT2_GreaterOrEqual);

  return Status::OK();
}

//LessOrEqual op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status LessOrEqual<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &ImplT2_LessOrEqual);

  return Status::OK();
}

BINARY_LOGICALOP_REGISTER_UZILHFD(Equal, 13)
BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(Equal, 13, bool)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Equal, 11, 12)
BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(Equal, 11, 12, bool)
BINARY_OP_REGISTER_VERSIONED_OIL(Equal, 7, 10)
BINARY_LOGICALOP_REGISTER_UZILHFD(Greater, 13)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Greater, 9, 12)
BINARY_OP_REGISTER_VERSIONED_HFD(Greater, 7, 8)
BINARY_LOGICALOP_REGISTER_UZILHFD(Less, 13)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Less, 9, 12)
BINARY_OP_REGISTER_VERSIONED_HFD(Less, 7, 8)
BINARY_LOGICALOP_REGISTER_UZILHFD(GreaterOrEqual, 12)
BINARY_LOGICALOP_REGISTER_UZILHFD(LessOrEqual, 12)


}  // namespace cuda
}  // namespace onnxruntime
