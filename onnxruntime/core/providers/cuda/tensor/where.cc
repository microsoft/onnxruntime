// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "where.h"
#include "where_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

// kernel builder functions
#define WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      Where,                                                        \
      kOnnxDomain,                                                  \
      9,                                                            \
      TName,                                                        \
      kCudaExecutionProvider,                                       \
      KernelDefBuilder()                                            \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Where<T>);

// Compute where operator output shape based upon three way broad-casting.
Status ComputeOutputShape(const std::string& node_name, const TensorShape& cond_shape,
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

struct TernaryElementwisePreparation {
  const Tensor* a_tensor = nullptr;
  const Tensor* b_tensor = nullptr;
  const Tensor* c_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;             // for no_broadcast cases, output_rank uses SimpleBroadcast enums
  CudaKernel::CudaAsyncBuffer<int64_t> a_padded_strides;  // for a shape == output shape, this is nullptr
  CudaKernel::CudaAsyncBuffer<int64_t> b_padded_strides;  // for b shape == output shape, this is nullptr
  CudaKernel::CudaAsyncBuffer<int64_t> c_padded_strides;  // for c shape == output shape, this is nullptr
  CudaKernel::CudaAsyncBuffer<fast_divmod> fdm_output_strides;

  TernaryElementwisePreparation(const CudaKernel* op_kernel, const Tensor* a,
                                const Tensor* b, const Tensor* c)
      : a_padded_strides(op_kernel),
        b_padded_strides(op_kernel),
        c_padded_strides(op_kernel),
        fdm_output_strides(op_kernel),
        a_tensor(a),
        b_tensor(b),
        c_tensor(c) {}

  Status CopyToGpu() {
    ORT_RETURN_IF_ERROR(a_padded_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(b_padded_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(c_padded_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());
    return Status::OK();
  }

  Status TernaryElementwiseBroadcastPrepareHelper(const TensorShape& a_shape,
                                                  const TensorShape& b_shape,
                                                  const TensorShape& c_shape,
                                                  const TensorShape& output_shape) {
    size_t a_rank = a_shape.NumDimensions();
    size_t b_rank = b_shape.NumDimensions();
    size_t c_rank = c_shape.NumDimensions();
    size_t out_rank = std::max(std::max(a_rank, b_rank), c_rank);

    // early return when shapes match
    if (a_shape == b_shape && b_shape == c_shape) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    output_rank_or_simple_broadcast = out_rank;

    if (a_shape != output_shape) {
      // compute strides with 1 more dim than out_rank, and use strides[0] == strides[1]
      // to decide if dim0 needs broadcast
      a_padded_strides.AllocCpuPtr(out_rank + 1);
      ORT_RETURN_IF_NOT(TensorPitches::Calculate(a_padded_strides.CpuSpan(), a_shape.GetDims()));
      if (a_shape[0] > 1 && a_rank == out_rank)
        a_padded_strides.CpuPtr()[0] = 0;
    }

    if (b_shape != output_shape) {
      b_padded_strides.AllocCpuPtr(out_rank + 1);
      ORT_RETURN_IF_NOT(TensorPitches::Calculate(b_padded_strides.CpuSpan(), b_shape.GetDims()));
      if (b_shape[0] > 1 && b_rank == out_rank)
        b_padded_strides.CpuPtr()[0] = 0;
    }

    if (c_shape != output_shape) {
      c_padded_strides.AllocCpuPtr(out_rank + 1);
      ORT_RETURN_IF_NOT(TensorPitches::Calculate(c_padded_strides.CpuSpan(), c_shape.GetDims()));
      if (c_shape[0] > 1 && c_rank == out_rank)
        c_padded_strides.CpuPtr()[0] = 0;
    }

    fdm_output_strides.AllocCpuPtr(out_rank);
    ORT_RETURN_IF_NOT(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_shape.GetDims()));
    return Status::OK();
  }
};

template <typename T>
Status Where<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto* const condition = context->Input<Tensor>(0);
  const auto* const X = context->Input<Tensor>(1);
  const auto* const Y = context->Input<Tensor>(2);
  ORT_ENFORCE(condition && X && Y, "condition, X, and Y inputs are required!");

  auto const& condition_shape = condition->Shape();
  auto const& X_shape = X->Shape();
  auto const& Y_shape = Y->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), condition_shape, X_shape, Y_shape, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  TernaryElementwisePreparation prepare(this, condition, X, Y);
  ORT_RETURN_IF_ERROR(prepare.TernaryElementwiseBroadcastPrepareHelper(condition_shape, X_shape, Y_shape, output_shape));
  ORT_RETURN_IF_ERROR(prepare.CopyToGpu());

  WhereImpl<CudaT>(
      prepare.output_rank_or_simple_broadcast,
      prepare.a_padded_strides.GpuPtr(),
      reinterpret_cast<const bool*>(prepare.a_tensor->template Data<bool>()),
      prepare.b_padded_strides.GpuPtr(),
      reinterpret_cast<const CudaT*>(prepare.b_tensor->template Data<T>()),
      prepare.c_padded_strides.GpuPtr(),
      reinterpret_cast<const CudaT*>(prepare.c_tensor->template Data<T>()),
      prepare.fdm_output_strides.GpuPtr(),
      reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>()),
      output_tensor->Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE_WITH_NAME(T, TName) \
  WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)   \
  template Status Where<T>::ComputeInternal(OpKernelContext* context) const;

#define SPECIALIZED_COMPUTE(T) \
  SPECIALIZED_COMPUTE_WITH_NAME(T, T)

SPECIALIZED_COMPUTE(int32_t)
SPECIALIZED_COMPUTE(int64_t)
SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(MLFloat16)
}  // namespace cuda
}  // namespace onnxruntime
