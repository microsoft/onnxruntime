// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
using namespace std;
namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion)                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      float,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      x<float>);                                                                      \
                                                                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      int32_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      x<int32_t>);

REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL1, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL2, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSum, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSumExp, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMax, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMean, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMin, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceProd, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSum, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSumSquare, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMax, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMin, 1);

// When all reduce axises located at the tail of the dims, quite general cases, transpose and extra
// copy could be skiped to improve performance, if required by check_no_transpose = true;
// return value: true means transposedInputData is not created/copied, input tensor data could
//               be direct use as row major matrix [block_size, blocks], where blocks is the
//               size of each reduce.
template <typename T>
bool PrepareForReduce(OpKernelContext* ctx,
                      std::vector<T>& transposedInputData,
                      Tensor** reducedTensor,
                      int64_t& block_size,
                      int64_t& blocks,
                      const std::vector<int64_t>& axes_,
                      bool keepdims_,
                      bool check_no_transpose = false) {
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  const Tensor& input = *input_tensor_ptr;

  size_t ndim = input.Shape().GetDims().size();
  std::vector<int64_t> axes;
  for (int64_t axis : axes_) {
    axes.push_back(HandleNegativeAxis(axis, static_cast<int64_t>(ndim)));
  }

  if (axes.empty()) {
    // This is the default case for non-arg kind reductions. Reduce on all dimensions.
    for (size_t i = 0; i < ndim; i++)
      axes.push_back(i);
  }

  std::sort(axes.begin(), axes.end());

  // If all reduced axes are located at the tail of the input shape, then copy could be skipped is required
  bool need_copy = true;
  if (axes.size() <= ndim && axes.front() == static_cast<int64_t>(ndim - axes.size()) && axes.back() == static_cast<int64_t>(ndim) - 1) {
    need_copy = false;
  }

  vector<bool> keep_axis(ndim, true);
  for (auto i : axes) {
    keep_axis[i] = false;
  }

  //transpose the input so that all to-be-reduced axes are at the head
  vector<int64_t> transposed_axes(axes.begin(), axes.end());
  for (int i = 0; i < ndim; ++i) {
    if (keep_axis[i]) {
      transposed_axes.push_back(i);
    }
  }

  vector<int64_t> new_dims_(transposed_axes.size());
  for (int i = 0; i < transposed_axes.size(); ++i) {
    new_dims_[i] = input.Shape().GetDims().at(transposed_axes[i]);
  }

  int num_axes = static_cast<int>(transposed_axes.size());
  auto in_dims = input.Shape().GetDims();

  // Measure amount of contiguous data we can copy at once
  int64_t blocksize = 1;
  int n_shared_idxs = 0;
  for (int i = num_axes - 1; i >= 0; --i) {
    if (transposed_axes[i] == i) {
      blocksize *= new_dims_[i];
      ++n_shared_idxs;
    } else {
      break;
    }
  }

  const T* from_data = input.template Data<T>();
  size_t count = input.Shape().Size();

  //set to-be-reduced axes to one. squeeze is keepdims_ is false
  int64_t first_dim = 1;
  std::vector<int64_t> reduced_dims;
  for (int i = 0; i < in_dims.size(); i++) {
    if (keep_axis[i]) {
      reduced_dims.push_back(in_dims[i]);
    } else {
      first_dim *= in_dims[i];
      if (keepdims_) {
        reduced_dims.push_back(1);
      }
    }
  }

  *reducedTensor = ctx->Output(0, reduced_dims);
  block_size = input.Shape().Size() / first_dim;
  blocks = first_dim;

  if (!need_copy && check_no_transpose) {
    return true;
  }

  transposedInputData.resize(input.Shape().Size(), 0);
  T* to_data = &transposedInputData[0];
  if (num_axes < 2 || n_shared_idxs == num_axes) {
    memcpy(to_data, from_data, count * sizeof(T));
    return false;
  }

  int itr_axes = num_axes - n_shared_idxs;

  // Calculate strides
  std::vector<int64_t> stride_x(itr_axes, 0);
  for (size_t i = 0; i < itr_axes; i++) {
    stride_x[i] = 1;
    for (size_t j = transposed_axes[i] + 1; j < itr_axes; j++) {
      stride_x[i] *= in_dims[j];
    }
  }

  std::vector<int64_t> itr_idxs(itr_axes, 0);

  // Branch here to avoid branching within the loop
  if (blocksize > 1) {
    for (size_t index = 0; index < (count / blocksize); index++) {
      int64_t from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      memcpy(
          to_data + blocksize * index,
          from_data + blocksize * from_index,
          blocksize * sizeof(T));

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = new_dims_[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  } else {
    for (size_t index = 0; index < count; index++) {
      int64_t from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      *(to_data + index) = *(from_data + from_index);

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = new_dims_[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  }
  return false;
}

template <typename T>
Status ReduceL1<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).cwiseAbs().rowwise().sum();

  return Status::OK();
}

template <typename T>
Status ReduceL2<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().norm();

  return Status::OK();
}

template <typename T>
Status ReduceLogSum<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().sum();
  for (int j = 0; j < block_size; ++j) {
    *(output_data) = static_cast<T>(std::log(*(output_data)));
    ++output_data;
  }

  return Status::OK();
}

template <typename T>
Status ReduceLogSumExp<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  for (int j = 0; j < block_size; ++j) {
    T max_value = std::numeric_limits<T>::lowest();
    for (int i = 0; i < blocks; ++i) {
      max_value = std::max(max_value, transposedInputData[i * block_size + j]);
    }
    T scaled_exp_sum = 0;
    for (int i = 0; i < blocks; ++i) {
      scaled_exp_sum += static_cast<T>(std::exp(transposedInputData[i * block_size + j] - max_value));
    }
    *(output_data++) = static_cast<T>(std::log(scaled_exp_sum) + max_value);
  }
  return Status::OK();
}

template <typename T>
Status ReduceMax<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().maxCoeff();

  return Status::OK();
}

template <typename T>
ConstEigenVectorMap<T> GenMap(const T* input, int32_t index, int64_t blocks) {
  return ConstEigenVectorMap<T>(input + (index * blocks), blocks);
}

template <typename T>
Status ReduceMean<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  bool no_transpose = PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_, true);

  T* output_data = reduced->template MutableData<T>();

  if (no_transpose) {
    const T* input_data = ctx->Input<Tensor>(0)->template Data<T>();

    // Get access to the internal threadpool
    auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
    auto thread_pool = ctx_internal->GetOperatorThreadPool();

    std::function<void(int32_t)> work_object = [&](int32_t i) {
      output_data[i] = GenMap<T>(input_data, i, blocks).mean();
    };
    const_cast<onnxruntime::concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)block_size, work_object);
  } else {
    EigenVectorMap<T> out_vec(output_data, block_size);
    out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().mean();
  }

  return Status::OK();
}

template <typename T>
Status ReduceMin<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().minCoeff();

  return Status::OK();
}

template <typename T>
Status ReduceProd<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().prod();

  return Status::OK();
}

template <typename T>
Status ReduceSum<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  bool no_transpose = PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_, true);

  T* output_data = reduced->template MutableData<T>();

  if (no_transpose) {
    const T* input_data = ctx->Input<Tensor>(0)->template Data<T>();

    // Get access to the internal threadpool
    auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
    auto thread_pool = ctx_internal->GetOperatorThreadPool();

    std::function<void(int32_t)> work_object = [&](int32_t i) {
      output_data[i] = GenMap<T>(input_data, i, blocks).sum();
    };
    const_cast<onnxruntime::concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)block_size, work_object);
  } else {
    EigenVectorMap<T> out_vec(output_data, block_size);
    out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().sum();
  }

  return Status::OK();
}

template <typename T>
Status ReduceSumSquare<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  T* output_data = reduced->template MutableData<T>();

  EigenVectorMap<T> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks).rowwise().squaredNorm();

  return Status::OK();
}

template <typename T>
Status ArgMax<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  int64_t* output_data = reduced->template MutableData<int64_t>();

  Eigen::MatrixXf::Index maxIndex;
  auto matrixData = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks);
  for (int i = 0; i < block_size; ++i) {
    matrixData.row(i).maxCoeff(&maxIndex);
    *(output_data++) = maxIndex;
  }

  return Status::OK();
}

template <typename T>
Status ArgMin<T>::Compute(OpKernelContext* ctx) const {
  std::vector<T> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce<T>(ctx, transposedInputData, &reduced, block_size, blocks, axes_, keepdims_);

  int64_t* output_data = reduced->template MutableData<int64_t>();

  Eigen::MatrixXf::Index minIndex;
  auto matrixData = ConstEigenMatrixMap<T>(&transposedInputData[0], block_size, blocks);
  for (int i = 0; i < block_size; ++i) {
    matrixData.row(i).minCoeff(&minIndex);
    *(output_data++) = minIndex;
  }

  return Status::OK();
}

}  // namespace onnxruntime
