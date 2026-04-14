// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <numeric>

#include "cumprod.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

namespace cumprod_op {
Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) {
  if (!axis_tensor)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor must be provided to the CumProd op");

  if (axis_tensor->Shape().NumDimensions() > 1 || axis_tensor->Shape().Size() != 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Axis tensor must be a scalar (0-D) or 1-D tensor with exactly one element");

  if (axis_tensor->IsDataType<int32_t>()) {
    axis_out = static_cast<int64_t>(axis_tensor->Data<int32_t>()[0]);
  } else if (axis_tensor->IsDataType<int64_t>()) {
    axis_out = axis_tensor->Data<int64_t>()[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be of type `int32_t` or `int64_t`");
  }

  axis_out = HandleNegativeAxis(axis_out, input_rank);

  return Status::OK();
}

}  // namespace cumprod_op

// Opset 26 kernels
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    uint32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint32_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<uint32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumProd,
    26,
    uint64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumProd<uint64_t>);

template <typename T>
CumProd<T>::CumProd(const OpKernelInfo& info) : OpKernel(info), exclusive_(), reverse_() {
  int64_t exclusive = 0;
  auto status = info.GetAttr("exclusive", &exclusive);
  if (status.IsOK()) {
    ORT_ENFORCE(exclusive == 0 || exclusive == 1, "exclusive attribute must be 0 or 1, got: ", exclusive);
    exclusive_ = exclusive;
  }
  int64_t reverse = 0;
  status = info.GetAttr("reverse", &reverse);
  if (status.IsOK()) {
    ORT_ENFORCE(reverse == 0 || reverse == 1, "reverse attribute must be 0 or 1, got: ", reverse);
    reverse_ = reverse;
  }
}

template <typename T>
Status CumProd<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  int64_t rank = static_cast<int64_t>(input->Shape().NumDimensions());
  if (rank == 0)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot apply CumProd operator on a scalar");

  const Tensor* axis_tensor = ctx->Input<Tensor>(1);

  TensorShape output_shape(input->Shape());
  auto& output_tensor = *ctx->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  int64_t axis_input = 0;
  ORT_THROW_IF_ERROR(cumprod_op::GetAxis(axis_tensor, rank, axis_input));

  // We solve the problem by using the identity that (in the case of exclusive)
  // 1) out[upper_dims...][0][lower_dims...] = 1
  // 2) out[upper_dims...][i][lower_dims...] =
  //      in[upper_dims...][i-1][lower_dims...] * out[upper_dims...][i-1][lower_dims...]
  // We loop through the [upper_dims...] and start applying the identity in each slice.
  // Since the [lower_dims...] are adjacent in memory, we can multiply them like vectors.

  const auto input_shape = input->Shape().GetDims();
  const size_t axis = onnxruntime::narrow<size_t>(axis_input);
  const int64_t dim = input->Shape()[axis];  // dimension size for the axis
  const int64_t upper_dim_count =            // number of slices we can walk through iteratively
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, static_cast<int64_t>(1), std::multiplies<int64_t>());
  const int64_t lower_dim_size =  // sizes of the slices we can treat as 1D arrays
      std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  const T* input_data = input->Data<T>();
  T* output_data = output_tensor.MutableData<T>();
  const int64_t slice_size = dim * lower_dim_size;
  auto* tp = ctx->GetOperatorThreadPool();

  if (!reverse_) {
    if (exclusive_) {
      concurrency::ThreadPool::TryBatchParallelFor(
          tp, static_cast<int32_t>(upper_dim_count),
          [&](ptrdiff_t outer) {
            const int64_t base = outer * slice_size;
            const T* in = input_data + base;
            T* out = output_data + base;

            for (int64_t inner = 0; inner < lower_dim_size; inner++) {
              out[inner] = static_cast<T>(1);
            }
            for (int64_t cum_axis = 1; cum_axis < dim; cum_axis++) {
              const int64_t curr_offset = cum_axis * lower_dim_size;
              const int64_t prev_offset = (cum_axis - 1) * lower_dim_size;
              for (int64_t inner = 0; inner < lower_dim_size; inner++) {
                out[curr_offset + inner] = out[prev_offset + inner] * in[prev_offset + inner];
              }
            }
          },
          0);
    } else {
      concurrency::ThreadPool::TryBatchParallelFor(
          tp, static_cast<int32_t>(upper_dim_count),
          [&](ptrdiff_t outer) {
            const int64_t base = outer * slice_size;
            const T* in = input_data + base;
            T* out = output_data + base;

            for (int64_t inner = 0; inner < lower_dim_size; inner++) {
              out[inner] = in[inner];
            }
            for (int64_t cum_axis = 1; cum_axis < dim; cum_axis++) {
              const int64_t curr_offset = cum_axis * lower_dim_size;
              const int64_t prev_offset = (cum_axis - 1) * lower_dim_size;
              for (int64_t inner = 0; inner < lower_dim_size; inner++) {
                out[curr_offset + inner] = out[prev_offset + inner] * in[curr_offset + inner];
              }
            }
          },
          0);
    }
  } else {
    if (exclusive_) {
      concurrency::ThreadPool::TryBatchParallelFor(
          tp, static_cast<int32_t>(upper_dim_count),
          [&](ptrdiff_t outer) {
            const int64_t base = outer * slice_size;
            const T* in = input_data + base;
            T* out = output_data + base;

            const int64_t last_offset = (dim - 1) * lower_dim_size;
            for (int64_t inner = 0; inner < lower_dim_size; inner++) {
              out[last_offset + inner] = static_cast<T>(1);
            }
            for (int64_t cum_axis = dim - 2; cum_axis >= 0; cum_axis--) {
              const int64_t curr_offset = cum_axis * lower_dim_size;
              const int64_t next_offset = (cum_axis + 1) * lower_dim_size;
              for (int64_t inner = 0; inner < lower_dim_size; inner++) {
                out[curr_offset + inner] = out[next_offset + inner] * in[next_offset + inner];
              }
            }
          },
          0);
    } else {
      concurrency::ThreadPool::TryBatchParallelFor(
          tp, static_cast<int32_t>(upper_dim_count),
          [&](ptrdiff_t outer) {
            const int64_t base = outer * slice_size;
            const T* in = input_data + base;
            T* out = output_data + base;

            const int64_t last_offset = (dim - 1) * lower_dim_size;
            for (int64_t inner = 0; inner < lower_dim_size; inner++) {
              out[last_offset + inner] = in[last_offset + inner];
            }
            for (int64_t cum_axis = dim - 2; cum_axis >= 0; cum_axis--) {
              const int64_t curr_offset = cum_axis * lower_dim_size;
              const int64_t next_offset = (cum_axis + 1) * lower_dim_size;
              for (int64_t inner = 0; inner < lower_dim_size; inner++) {
                out[curr_offset + inner] = out[next_offset + inner] * in[curr_offset + inner];
              }
            }
          },
          0);
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
