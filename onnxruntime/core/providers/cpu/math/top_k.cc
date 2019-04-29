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

// Helper methods
static int64_t SizeToDim(size_t k, const vector<int64_t>& dims) {
  ORT_ENFORCE(k <= dims.size());
  int64_t r = 1;
  for (size_t i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

static int64_t SizeFromDim(size_t k, const vector<int64_t>& dims) {
  ORT_ENFORCE(k <= dims.size());
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

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

// Core TopK implementation
Status TopKImpl(OpKernelContext* p_op_kernel_context, const Tensor* X, const int axis, const unsigned k) {

  const vector<int64_t>& in_dims = X->Shape().GetDims();
  // Will return axis_ as is if positive or fixes it in case it is negative
  auto axis_parsed = HandleNegativeAxis(axis, in_dims.size());
  // Check to ensure k is within the bounds of what is available in that specific axis
  if (in_dims.at(axis_parsed) < k) {
    ostringstream err_msg;
    err_msg << "k argment [" << k << "] should not be greater than specified axis dim value [" << in_dims.at(axis_parsed) << "]";
    return Status(common::ONNXRUNTIME, common::FAIL, err_msg.str());
  }

  if (k == 0) {
    vector<int64_t> out_dims = in_dims;
    out_dims[axis_parsed] = 0;
    p_op_kernel_context->Output(0, out_dims);
    p_op_kernel_context->Output(1, out_dims);
    return Status::OK();
  }

  const int64_t rows = SizeToDim(axis_parsed, in_dims);
  const int64_t cols = X->Shape().Size() / rows;
  auto input_map = ConstEigenMatrixMapRowMajor<float>(
      static_cast<const float*>(X->template Data<float>()),
      rows,
      cols);

  // Resize output tensors to be the same shape as the input except
  // for the specified dimension ((i.e.) axis_parsed), which will be of size k. E.x. for an input tensor
  // of shape [3, 4, 5] and k=2 with axis_parsed=1, both of these will be shape [3, 2, 5]
  vector<int64_t> output_linear_shape = in_dims;
  output_linear_shape[axis_parsed] = k;
  auto* Values = p_op_kernel_context->Output(0, output_linear_shape);
  auto* Indices = p_op_kernel_context->Output(1, output_linear_shape);

  // Use Eigen maps to allow indexing into the 2d tensors like Values_map(i,j)
  const int64_t reduced_cols = SizeFromDim(axis_parsed, output_linear_shape);
  auto Values_map = EigenMatrixMapRowMajor<float>(
      Values->template MutableData<float>(), rows, reduced_cols);
  auto Indices_map = EigenMatrixMapRowMajor<int64_t>(
      Indices->template MutableData<int64_t>(), rows, reduced_cols);

  // This is basically the number of elements within each of the "k" rows
  const int64_t block_slice = reduced_cols / k;
  // Sort preserving Indices
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < block_slice; ++j) {
      // Build a min-heap, the heap element is pair of (value, idx)
      // the top of the heap is the smallest value
      priority_queue<
          pair<float, int64_t>,
          vector<pair<float, int64_t>>,
          ValueCmp<float>>
          min_heap;
      // Maintain the size of heap to be less or equal to k_, so the
      // heap will hold the k largest Values
      for (int64_t l = 0; l < in_dims[axis_parsed]; ++l) {
        const auto value = input_map(i, l * block_slice + j);
        if (min_heap.size() < k || value > min_heap.top().first) {
          min_heap.push({value, l});
        }
        if (min_heap.size() > k) {
          min_heap.pop();
        }
      }
      // Extract these k elements and place them in the results placeholder
      for (int64_t l = 0; l < k; ++l) {
        auto& pqElem = min_heap.top();
        auto col_index = (k - l - 1) * block_slice + j;
        Values_map(i, col_index) = pqElem.first;
        Indices_map(i, col_index) = pqElem.second;
        min_heap.pop();
      }
    }
  }

  return Status::OK();
}

// Opset ver - 1 to 9
template <>
TopK<9, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t k_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_temp).IsOK());
  ORT_ENFORCE(k_temp > 0);
  k_ = gsl::narrow_cast<unsigned>(k_temp);

  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 1 to 9
template <>
Status TopK<9, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL,
                                  "input count mismatch, expected 1 input - the tensor to be processed");
  return TopKImpl(p_op_kernel_context, X, axis_, k_);
}

// Opset ver - 10
template <>
TopK<10, float>::TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  int64_t axis_temp;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
  axis_ = gsl::narrow_cast<int>(axis_temp);
}

// Opset ver - 10
template <>
Status TopK<10, float>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* Y = p_op_kernel_context->Input<Tensor>(1);
  if (X == nullptr || Y == nullptr) return Status(common::ONNXRUNTIME, common::FAIL,
                                                  "input count mismatch, expected 2 inputs - "
                                                  "the tensor to be processed and a tensor containing k value");
  const vector<int64_t>& y_shape = Y->Shape().GetDims();
  if (y_shape.size() != 1 || y_shape[0] != 1) return Status(common::ONNXRUNTIME, common::FAIL, "k tensor should be a 1D tensor of size 1");
  auto parsed_input_k = Y->template Data<int64_t>()[0];
  if (parsed_input_k < 0) return Status(common::ONNXRUNTIME, common::FAIL, "value of k must not be negative");
  return TopKImpl(p_op_kernel_context, X, axis_, gsl::narrow_cast<unsigned>(parsed_input_k));
}

// Register necessary kernels
// spec https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    TopK,
    1, 9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    TopK<9, float>);

ONNX_CPU_OPERATOR_KERNEL(
    TopK,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    TopK<10, float>);

}  // namespace onnxruntime