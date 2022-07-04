// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/ternary_elementwise_ops.h"
#include "core/providers/cuda/math/ternary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

// Compute output shape based upon three way broad-casting.
Status ComputeOutputShape(const std::string& node_name, const TensorShape& x_shape, const TensorShape& y_shape,
                          const TensorShape& z_shape, TensorShape& out_shape) {
  size_t cond_rank = x_shape.NumDimensions();
  size_t x_rank = y_shape.NumDimensions();
  size_t y_rank = z_shape.NumDimensions();
  size_t out_rank = std::max(std::max(cond_rank, x_rank), y_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t cond_dim = 1;
    if (i < cond_rank)
      cond_dim = x_shape[cond_rank - 1 - i];

    int64_t x_dim = 1;
    if (i < x_rank)
      x_dim = y_shape[x_rank - 1 - i];

    int64_t y_dim = 1;
    if (i < y_rank)
      y_dim = z_shape[y_rank - 1 - i];

    int64_t out_dim = std::max(std::max(cond_dim, x_dim), y_dim);
    // special case to handle a dim of 0 which can be broadcast with a 1
    if (out_dim == 1)
      out_dim = std::min(std::min(cond_dim, x_dim), y_dim);

    if (cond_dim != out_dim && cond_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": condition operand cannot broadcast on dim ",
                             cond_rank - 1 - i, " Condition Shape: ", x_shape.ToString(), ", X Shape: ",
                             y_shape.ToString(), ", Y Shape: ", z_shape.ToString());
    if (x_dim != out_dim && x_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": X operand cannot broadcast on dim ", x_rank - 1 - i,
                             " Condition Shape: ", x_shape.ToString(), ", X Shape: ", y_shape.ToString(),
                             ", Y Shape: ", z_shape.ToString());
    if (y_dim != out_dim && y_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": Y operand cannot broadcast on dim ", y_rank - 1 - i,
                             " Condition Shape: ", x_shape.ToString(), ", X Shape: ", y_shape.ToString(),
                             ", Y Shape: ", z_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }

  out_shape = TensorShape(output_dims);
  return Status::OK();
}

Status TernaryElementwiseBroadcastPrepare(
    const Tensor* x_tensor,
    const Tensor* y_tensor,
    const Tensor* z_tensor,
    Tensor* output_tensor,
    TernaryElementwisePreparation* p) {
  p->x_tensor = x_tensor;
  p->y_tensor = y_tensor;
  p->z_tensor = z_tensor;
  p->output_tensor = output_tensor;

  const auto& output_shape = output_tensor->Shape();
  ORT_RETURN_IF_ERROR(p->TernaryElementwiseBroadcastPrepareHelper(x_tensor->Shape(), y_tensor->Shape(),
                                                                  z_tensor->Shape(), output_shape));

  return Status::OK();
}

Status TernaryElementwise::Prepare(OpKernelContext* context, TernaryElementwisePreparation* p) const {
  const auto* const X = context->Input<Tensor>(0);
  const auto* const Y = context->Input<Tensor>(1);
  const auto* const Z = context->Input<Tensor>(2);
  ORT_ENFORCE(X && Y && Z, "X, Y and Z inputs are required!");

  auto const& X_shape = X->Shape();
  auto const& Y_shape = Y->Shape();
  auto const& Z_shape = Z->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), X_shape, Y_shape, Z_shape, output_shape));

  if (output_shape.Size() == 0)
    return Status::OK();

  auto output_tensor = context->Output(0, output_shape);
  ORT_RETURN_IF_ERROR(TernaryElementwiseBroadcastPrepare(X, Y, Z, output_tensor, p));
  return Status::OK();
}

#define TERNARY_ELEMENTWISE_COMPUTE(x, T)                                                                        \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    TernaryElementwisePreparation prepare;                                                                       \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                             \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                \
        Stream(),                                                                                                \
        prepare.output_rank_or_simple_broadcast,                                                                 \
        prepare.x_index_type,                                                                                    \
        prepare.x_padded_strides,                                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.x_tensor->template Data<T>()),       \
        prepare.y_index_type,                                                                                    \
        prepare.y_padded_strides,                                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.y_tensor->template Data<T>()),       \
        prepare.z_index_type,                                                                                    \
        prepare.z_padded_strides,                                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.z_tensor->template Data<T>()),       \
        prepare.fdm_output_strides,                                                                              \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        prepare.output_tensor->Shape().Size(),                                                                   \
        prepare.only_last_dim_broadcast,                                                                         \
        prepare.last_dim_fdm);                                                                                   \
                                                                                                                 \
    return Status::OK();                                                                                         \
  }

#define TERNARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, class_name, domain, ver, T)         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      domain,                                                                              \
      ver,                                                                                 \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      class_name<T>);

#define TERNARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, domain, ver, T) \
  TERNARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, x, domain, ver, T)

#define TERNARY_OP_TYPED(name, domain, ver, T)                    \
  TERNARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, domain, ver, T) \
  TERNARY_ELEMENTWISE_COMPUTE(name, T)

#define TERNARY_OP_HFD(name, domain, ver)        \
  TERNARY_OP_TYPED(name, domain, ver, MLFloat16) \
  TERNARY_OP_TYPED(name, domain, ver, float)     \
  TERNARY_OP_TYPED(name, domain, ver, double)    \
  TERNARY_OP_TYPED(name, domain, ver, BFloat16)

// TERNARY_OP_HFD(BiasGeluGrad_dX, kMSDomain, 1)

}  // namespace cuda
}  // namespace onnxruntime
