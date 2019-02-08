/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "core/providers/cpu/math/top_k.h"
#include "core/providers/common.h"
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include <queue>
using namespace std;
namespace onnxruntime {
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
ONNX_CPU_OPERATOR_KERNEL(
    TopK,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    TopK<float>);

static int64_t SizeToDim(size_t k, const vector<int64_t>& dims) {
  ORT_ENFORCE(k <= dims.size());
  int64_t r = 1;
  for (size_t i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Define these two names to allow lookup into the 2d tensors like
// mytensor(i, j)
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
struct ValueCmp {
  bool operator()(
      const pair<T, int64_t>& lhs,
      const pair<T, int64_t>& rhs) {
    return (
        lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

template <>
Status TopK<float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const vector<int64_t>& in_dims = X->Shape().GetDims();
  // Will return axis_ as is if positive or fixes it in case it is negative
  auto axis_fixed = HandleNegativeAxis(axis_, in_dims.size());	
  // Check to ensure k_ is within the bounds of what is available in that specific axis 	
  if (in_dims.at(axis_fixed) < k_) {
    ostringstream err_msg;
    err_msg << "k argment [" << k_ << "] should not be greater than specified axis dim [" << in_dims.at(axis_fixed) << "]";
    return Status(common::ONNXRUNTIME, common::FAIL, err_msg.str());
  }

  vector<int64_t> linear_shape = {SizeToDim(in_dims.size() - 1, in_dims),
                                  in_dims[in_dims.size() - 1]};
  auto input_map = ConstEigenMatrixMapRowMajor<float>(
      static_cast<const float*>(X->template Data<float>()),
      linear_shape[0],
      linear_shape[1]);

  // Resize output tensors to be the same shape as the linearized input except
  // for the specified dimension (axis_), which will be of size k. E.x. for an input tensor
  // of shape [3, 4, 5] and k=2 with axis=2, both of these will be shape [3, 4, 2]
  vector<int64_t> output_linear_shape = in_dims;
  output_linear_shape[axis_fixed] = k_;
  auto* Values = p_op_kernel_context->Output(0, output_linear_shape);
  auto* Indices = p_op_kernel_context->Output(1, output_linear_shape);

  // Use Eigen maps to allow indexing into the 2d tensors like Values_map(i,j)
  auto Values_map = EigenMatrixMapRowMajor<float>(
      Values->template MutableData<float>(), linear_shape[0], k_);
  auto Indices_map = EigenMatrixMapRowMajor<int64_t>(
      Indices->template MutableData<int64_t>(), linear_shape[0], k_);

  // Sort preserving Indices
  for (int64_t i = 0; i < linear_shape[0]; ++i) {
    // Build a min-heap, the heap element is pair of (value, idx)
    // the top of the heap is the smallest value
    priority_queue<
        pair<float, int64_t>,
        vector<pair<float, int64_t>>,
        ValueCmp<float>>
        min_heap;

    // Maintain the size of heap to be less or equal to k_, so the
    // heap will hold the k_ largest Values
    for (int64_t j = 0; j < linear_shape[1]; ++j) {
      const auto value = input_map(i, j);
      if (min_heap.size() < k_ || value > min_heap.top().first) {
        min_heap.push({value, j});
      }
      if (min_heap.size() > k_) {
        min_heap.pop();
      }
    }
    for (int64_t j = 0; j < k_; ++j) {
      auto& pqElem = min_heap.top();
      Values_map(i, k_ - j - 1) = pqElem.first;
      Indices_map(i, k_ - j - 1) = pqElem.second;
      min_heap.pop();
    }
  }

  // Reshape output tensors to [a_1, a_2, ..., a_n, k]
  auto out_dims = in_dims;
  out_dims[out_dims.size() - 1] = k_;
  Values->Reshape(out_dims);
  Indices->Reshape(out_dims);
  return Status::OK();
}
}  // namespace onnxruntime
