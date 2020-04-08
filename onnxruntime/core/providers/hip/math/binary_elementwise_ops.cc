// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "binary_elementwise_ops.h"
#include "binary_elementwise_ops_impl.h"
#include "unary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace hip {

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

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, ver, T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      x,                                                                        \
      kOnnxDomain,                                                              \
      ver,                                                                      \
      T,                                                                        \
      kHipExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(x, ver, T)                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                          \
      x,                                                                                                                                  \
      kOnnxDomain,                                                                                                                        \
      ver,                                                                                                                                \
      T,                                                                                                                                  \
      kHipExecutionProvider,                                                                                                             \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      x<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                         \
      x,                                                                           \
      kOnnxDomain,                                                                 \
      startver,                                                                    \
      endver,                                                                      \
      T,                                                                           \
      kHipExecutionProvider,                                                      \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),    \
      x<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                         \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare(this);                                                                  \
    Prepare(context,  &prepare);                                                          \
    ORT_RETURN_IF_ERROR(prepare.CopyToGpu());                                                                    \
    Impl_##x<typename ToHipType<T>::MappedType>(                                                                \
        prepare.output_rank_or_simple_broadcast,                                                                 \
        prepare.lhs_padded_strides.GpuPtr(),                                                                     \
        reinterpret_cast<const typename ToHipType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        prepare.rhs_padded_strides.GpuPtr(),                                                                     \
        reinterpret_cast<const typename ToHipType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        prepare.fdm_output_strides.GpuPtr(),                                                                     \
        prepare.fdm_H,                                                                                           \
        prepare.fdm_C,                                                                                           \
        reinterpret_cast<typename ToHipType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        prepare.output_tensor->Shape().Size());                                                                 \
    return Status::OK();                                                                                         \
  }

#define BINARY_OP_TYPED(name, ver, T)                    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

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

#define BINARY_OP_HFD(name, ver)        \
  BINARY_OP_TYPED(name, ver, MLFloat16) \
  BINARY_OP_TYPED(name, ver, float)     \
  BINARY_OP_TYPED(name, ver, double)

#define BINARY_OP_UZILHFD(name, ver)   \
  BINARY_OP_TYPED(name, ver, uint32_t) \
  BINARY_OP_TYPED(name, ver, uint64_t) \
  BINARY_OP_TYPED(name, ver, int32_t)  \
  BINARY_OP_TYPED(name, ver, int64_t)  \
  BINARY_OP_HFD(name, ver)

#define BINARY_OP_REGISTER_OIL(name, ver)                      \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, bool)    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int64_t)

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
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, float)     \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, double)

#define BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, float)     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, double)

#define BINARY_OP_REGISTER_VERSIONED_UZILHFD(name, startver, endver)                   \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint64_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)

BINARY_OP_UZILHFD(Add, 7)
BINARY_OP_UZILHFD(Sub, 7)
BINARY_OP_UZILHFD(Mul, 7)
BINARY_OP_UZILHFD(Div, 7)
BINARY_OP_HFD(Pow, 7)
BINARY_LOGICALOP_TYPED(And, 7, bool)
BINARY_LOGICALOP_TYPED(Or, 7, bool)
BINARY_LOGICALOP_TYPED(Xor, 7, bool)
BINARY_OP_HFD(PRelu, 7)

template <typename T, typename HipT>
Status VariadicInputBase<T, HipT>::ComputeMethod(OpKernelContext* context, ImplCompute Impl_Compute) const {
  const auto& node = Node();
  const auto& node_name = node.Name();
  auto input_count = node.InputArgCount().front();
  ORT_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");

  auto lhs_tensor = context->Input<Tensor>(0);
  if (input_count == 1) {
    const auto& input_shape = lhs_tensor->Shape();
    auto output_tensor = context->Output(0, input_shape);
    if (lhs_tensor->DataRaw() != output_tensor->DataRaw()) {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(output_tensor->MutableDataRaw(), lhs_tensor->DataRaw(), sizeof(HipT) * input_shape.Size(), hipMemcpyDeviceToDevice));
    }
  } else {
    // compute output shape first, using broadcast rule
    TensorShape output_shape;
    TensorShape previous_output_shape = lhs_tensor->Shape();
    for (int index = 1; index < input_count; index++) {
      ORT_RETURN_IF_ERROR(ComputeOutputShape(node_name, previous_output_shape, context->Input<Tensor>(index)->Shape(), output_shape));
      previous_output_shape = output_shape;
    }
    Tensor* output_tensor = context->Output(0, output_shape);
    BinaryElementwisePreparation prepare(this);

    auto rhs_tensor = context->Input<Tensor>(1);
    if (input_count == 2) {
      // special case for 2 tensors to avoid memset zero
      ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, &prepare));
      Impl_Compute(
          prepare.output_rank_or_simple_broadcast,
          prepare.lhs_padded_strides.GpuPtr(),
          reinterpret_cast<const HipT*>(prepare.lhs_tensor->template Data<T>()),
          prepare.rhs_padded_strides.GpuPtr(),
          reinterpret_cast<const HipT*>(prepare.rhs_tensor->template Data<T>()),
          prepare.fdm_output_strides.GpuPtr(),
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<HipT*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());
    } else {
      // for more than 2 inputs, we need to accumulate into output tensor, as the shape from input0 + input1 might be different from output shape
      HIP_RETURN_IF_ERROR(hipMemset(output_tensor->MutableDataRaw(), 0, output_shape.Size() * sizeof(HipT)));

      ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(output_tensor, lhs_tensor, output_tensor, &prepare));
      Impl_Add(
          prepare.output_rank_or_simple_broadcast,
          prepare.lhs_padded_strides.GpuPtr(),
          reinterpret_cast<const HipT*>(prepare.lhs_tensor->template Data<T>()),
          prepare.rhs_padded_strides.GpuPtr(),
          reinterpret_cast<const HipT*>(prepare.rhs_tensor->template Data<T>()),
          prepare.fdm_output_strides.GpuPtr(),
          prepare.fdm_H,
          prepare.fdm_C,
          reinterpret_cast<HipT*>(prepare.output_tensor->template MutableData<T>()),
          prepare.output_tensor->Shape().Size());

      for (int index = 1; index < input_count; index++) {
        ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(output_tensor, context->Input<Tensor>(index), output_tensor, &prepare));
        Impl_Compute(
            prepare.output_rank_or_simple_broadcast,
            prepare.lhs_padded_strides.GpuPtr(),
            reinterpret_cast<const HipT*>(prepare.lhs_tensor->template Data<T>()),
            prepare.rhs_padded_strides.GpuPtr(),
            reinterpret_cast<const HipT*>(prepare.rhs_tensor->template Data<T>()),
            prepare.fdm_output_strides.GpuPtr(),
            prepare.fdm_H,
            prepare.fdm_C,
            reinterpret_cast<HipT*>(prepare.output_tensor->template MutableData<T>()),
            prepare.output_tensor->Shape().Size());
      }
    }
  }
  return Status::OK();
}

template <typename T>
Status Sum<T>::ComputeInternal(OpKernelContext* context) const {
  this->ComputeMethod(context, &Impl_Add);

  return Status::OK();
}

template <typename T>
Status Max<T>::ComputeInternal(OpKernelContext* context) const {
  this->ComputeMethod(context, &Impl_Max);

  return Status::OK();
}

template <typename T>
Status Min<T>::ComputeInternal(OpKernelContext* context) const {
  this->ComputeMethod(context, &Impl_Min);

  return Status::OK();
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T, typename HipT>
Status CompareFunction<T, HipT>::CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const {
  BinaryElementwisePreparation prepare(this);
  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));
  size_t output_size = prepare.output_tensor->Shape().Size();
  IAllocatorUniquePtr<T> output_buffer = GetScratchBuffer<T>(output_size);
  Impl_Compare(
      prepare.output_rank_or_simple_broadcast,
      prepare.lhs_padded_strides.GpuPtr(),
      reinterpret_cast<const HipT*>(prepare.lhs_tensor->template Data<T>()),
      prepare.rhs_padded_strides.GpuPtr(),
      reinterpret_cast<const HipT*>(prepare.rhs_tensor->template Data<T>()),
      prepare.fdm_output_strides.GpuPtr(),
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<HipT*>(output_buffer.get()),
      prepare.output_tensor->Shape().Size());

  Impl_Cast<HipT, ToHipType<bool>::MappedType>(
      reinterpret_cast<HipT*>(output_buffer.get()),
      reinterpret_cast<ToHipType<bool>::MappedType*>(prepare.output_tensor->template MutableData<bool>()),
      output_size);

  return Status::OK();
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Greater<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &Impl_Greater);

  return Status::OK();
}

template <typename T>
Status Equal<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &Impl_Equal);

  return Status::OK();
}

//Less op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Less<T>::ComputeInternal(OpKernelContext* context) const {
  this->CompareMethod(context, &Impl_Less);

  return Status::OK();
}

BINARY_OP_REGISTER_UZILHFD(Sum, 8)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Sum, 6, 7)
BINARY_LOGICALOP_REGISTER_UZILHFD(Greater, 9)
BINARY_OP_REGISTER_VERSIONED_OIL(Equal, 7, 10)
BINARY_OP_REGISTER_OIL(Equal, 11)
BINARY_OP_REGISTER_VERSIONED_HFD(Greater, 7, 8)
BINARY_OP_REGISTER_HFD(Max, 8)
BINARY_OP_REGISTER_VERSIONED_HFD(Max, 6, 7)
BINARY_OP_REGISTER_HFD(Min, 8)
BINARY_OP_REGISTER_VERSIONED_HFD(Min, 6, 7)
BINARY_LOGICALOP_REGISTER_UZILHFD(Less, 9)
BINARY_OP_REGISTER_VERSIONED_HFD(Less, 7, 8)

}  // namespace hip
}  // namespace onnxruntime
