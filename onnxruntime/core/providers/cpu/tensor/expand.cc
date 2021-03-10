// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand.h"
#include <cmath>

namespace onnxruntime {

#define REG_EXPAND_KERNEL(TYPE)                                                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                        \
      Expand,                                                                      \
      8,                                                                           \
      12,                                                                          \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Expand<TYPE>);                                                               \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Expand,                                                                      \
      13,                                                                          \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Expand<TYPE>);

REG_EXPAND_KERNEL(float)
REG_EXPAND_KERNEL(double)
REG_EXPAND_KERNEL(int8_t)
REG_EXPAND_KERNEL(int16_t)
REG_EXPAND_KERNEL(int32_t)
REG_EXPAND_KERNEL(int64_t)
REG_EXPAND_KERNEL(uint8_t)
REG_EXPAND_KERNEL(uint16_t)
REG_EXPAND_KERNEL(uint32_t)
REG_EXPAND_KERNEL(uint64_t)
REG_EXPAND_KERNEL(bool)
REG_EXPAND_KERNEL(MLFloat16)

template <typename T>
Status Expand<T>::Compute(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* input_data = input_tensor->template Data<T>();
  const auto& input_shape = input_tensor->Shape().GetDims();

  const auto* input_dims = input_shape.data();
  auto input_dims_size = static_cast<int64_t>(input_shape.size());

  const auto* shape_tensor = context->Input<Tensor>(1);
  const auto* shape_dims = shape_tensor->Data<int64_t>();
  std::vector<int64_t> output_shape{shape_dims, shape_dims + shape_tensor->Shape().Size()};

  if (input_shape.size() > output_shape.size()) {
    output_shape.insert(output_shape.begin(), input_shape.size() - output_shape.size(), 1);
  }

  auto input_shape_iter = input_shape.rbegin();
  auto output_shape_iter = output_shape.rbegin();
  while (input_shape_iter != input_shape.rend() &&
         output_shape_iter != output_shape.rend()) {
    if (*input_shape_iter != *output_shape_iter) {
      if (1 == *output_shape_iter) {
        *output_shape_iter = *input_shape_iter;
      } else if (1 != *input_shape_iter) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid expand shape");
      }
    }
    input_shape_iter++;
    output_shape_iter++;
  }

  TensorShape output_tensor_shape(output_shape);
  auto* output_tensor = context->Output(0, output_tensor_shape);
  auto* output_data = output_tensor->template MutableData<T>();
  auto* output_dims = output_shape.data();
  auto output_dims_size = static_cast<int64_t>(output_shape.size());
  auto max_dims_size = std::max(input_dims_size, output_dims_size);

  if (0 == max_dims_size) {
    *output_data = *input_data;
    return Status::OK();
  }

  std::unique_ptr<int64_t[]> input_dim_group{new int64_t[max_dims_size]};
  std::unique_ptr<int64_t[]> output_dim_group{new int64_t[max_dims_size]};
  std::unique_ptr<int64_t[]> expand_dim_size{new int64_t[max_dims_size]};
  auto dim_group_start = max_dims_size;

  for (int64_t input_dims_iter = input_dims_size - 1,
               output_dims_iter = output_dims_size - 1,
               last_dim_size = 1,
               input_count = 1,
               output_count = 1;
       output_dims_iter > -1;
       input_dims_iter--, output_dims_iter--) {
    auto input_dim = input_dims_iter > -1 ? input_dims[input_dims_iter] : 1;
    auto output_dim = output_dims[output_dims_iter];

    input_count *= input_dim;
    output_count *= output_dim;

    if (0 == input_count || 0 == output_count) {
      return Status::OK();
    }

    if (input_dim == 1 && output_dim > 1 || output_dims_iter == 0) {
      --dim_group_start;
      input_dim_group[dim_group_start] = input_count;
      output_dim_group[dim_group_start] = output_count;
      expand_dim_size[dim_group_start] = output_count / input_count / last_dim_size;
      last_dim_size *= expand_dim_size[dim_group_start];
    }
  }

  auto distribute_count = input_dim_group[dim_group_start] / input_dim_group[max_dims_size - 1];
  std::vector<int64_t> output_offsets(distribute_count, 0);
  int64_t copy_len = input_dim_group[max_dims_size - 1];
  auto copy_byte = copy_len * sizeof(T);

  auto distribute_fn =
    [&](ptrdiff_t i_start, ptrdiff_t i_end) {
    for (auto i = i_start; i < i_end; i++) {
      auto input_offset = i * copy_len;
      int64_t output_offset = 0;
      for (auto j = dim_group_start + 1, remains = input_offset; j < max_dims_size; ++j) {
        auto current_count = remains / input_dim_group[j];
        output_offset += current_count * output_dim_group[j];
        remains = remains % input_dim_group[j];
      }  //for j
      memcpy(output_data + output_offset, input_data + input_offset, copy_byte);
      output_offsets[i] = output_offset;
    } //for i
  };  //distribute_fn

  auto per_thread_tasks =
    distribute_count / concurrency::ThreadPool::DegreeOfParallelism(context->GetOperatorThreadPool());

  if (per_thread_tasks > 4) {
    concurrency::ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(),
      distribute_count,
      static_cast<double>(copy_byte),
      std::move(distribute_fn));
  } else {
    distribute_fn(0, distribute_count);
  }  //else

  for (auto i = max_dims_size - 1; i >= dim_group_start; --i) {
    auto copy_fn =
      [&](ptrdiff_t j_start, ptrdiff_t j_end) {
      for (auto j = j_start; j < j_end; j++) {
	auto output_offset = output_offsets[j];
	if (output_offset % output_dim_group[i] == 0) {
	  auto copy_len = output_dim_group[i] / expand_dim_size[i];
	  auto copy_byte = copy_len * sizeof(T);
	  auto output_from = output_data + output_offset;
	  auto output_at = output_from + copy_len;
	  auto output_end = output_from + output_dim_group[i];
	  while (output_at + copy_len <= output_end) {
	    memcpy(output_at, output_from, copy_byte);
	    output_at += copy_len;
	    copy_len <<= 1;
	    copy_byte <<= 1;
	  }  //while
	  while (output_at < output_end) {
	    if (output_at + copy_len <= output_end) {
	      memcpy(output_at, output_from, copy_byte);
	      output_at += copy_len;
	    } else {
	      copy_len >>= 1;
	      copy_byte >>= 1;
	    }
	  }  //while
	}  //if
      } // for
    };  //copy_fn
    if (per_thread_tasks > 20) {
      concurrency::ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(),
        distribute_count,
        static_cast<double>(copy_byte),
        std::move(copy_fn));
    } else {
      copy_fn(0, distribute_count);
    }  //else
  }  //for
  return Status::OK();
}  //Expand::compute

}  //namespace onnxruntime
