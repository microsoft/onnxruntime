// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
template <typename T>
inline void TensorShapeCopyDims(const TensorShape& shape, T* dims, size_t num_dims) {
  size_t n = std::min(num_dims, shape.NumDimensions());
  for (size_t i = 0; i != n; ++i)
    dims[i] = static_cast<ptrdiff_t>(shape[i]);
}

template <>
inline void TensorShapeCopyDims(const TensorShape& shape, int64_t* dims, size_t num_dims) {
  shape.CopyDims(dims, num_dims);
}

class MatMulComputeHelper {
 public:
  Status Compute(const TensorShape& left_shape, const TensorShape& right_shape,
                 bool transa = false, bool transb = false) {
    // Following numpy.matmul for shape inference:
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
    // The behavior depends on the arguments in the following way.
    // * If both arguments are 2 - D they are multiplied like conventional matrices.
    // * If either argument is N - D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    // * If the first argument is 1 - D, it is promoted to a matrix by prepending a 1 to its dimensions.After matrix multiplication the prepended 1 is removed.
    // * If the second argument is 1 - D, it is promoted to a matrix by appending a 1 to its dimensions.After matrix multiplication the appended 1 is removed.

    size_t left_num_dims = left_shape.NumDimensions();
    size_t right_num_dims = right_shape.NumDimensions();
    ORT_RETURN_IF_NOT(left_num_dims >= 1 && right_num_dims >= 1, "left_num_dims and right_num_dims must be >= 1");

    // Special cases below for right_shape being 2D and left_shape > 2D by flattening left_shape to 2D
    // Note that padding 1s in front of the right_shape can be flattened too
    // A: [M1, M2, ... K], B: [K, N]
    // A: [M1, M2, ... K], B: [N, K]^T
    // A: [M1, M2, ... K], B: [1, ..., 1, K, N]
    // A: [M1, M2, ... K], B: [1, ..., 1, N, K]^T
    if (!transa && left_num_dims >= 2 && right_num_dims >= 2 && left_num_dims >= right_num_dims &&
        right_shape.SizeToDimension(right_num_dims - 1) == right_shape[right_num_dims - 2]) {
      M_ = static_cast<ptrdiff_t>(left_shape.SizeToDimension(left_num_dims - 1));
      K_ = static_cast<ptrdiff_t>(left_shape[left_num_dims - 1]);
      N_ = static_cast<ptrdiff_t>(transb ? right_shape[right_num_dims - 2] : right_shape[right_num_dims - 1]);
      output_shape_ = left_shape;
      output_shape_[left_num_dims - 1] = N_;
      output_offsets_ = {0};
      left_offsets_ = {0};
      right_offsets_ = {0};
      ORT_RETURN_IF_NOT(K_ == right_shape[right_num_dims - 2] ||
                            transb && K_ == right_shape[right_num_dims - 1],
                        "MatMul dimension mismatch");
      return Status::OK();
    }

    bool has_1D_input = (left_num_dims == 1 || right_num_dims == 1);

    size_t num_input_dims = std::max(left_num_dims, right_num_dims);

    // use padded dims to compute matrix offsets, right 1D would be padded
    size_t num_dims_with_pad = num_input_dims + (right_num_dims == 1 ? 1 : 0);

    // output shape would squeeze the reduced 1D dimension
    size_t num_output_dims = num_input_dims - (has_1D_input ? 1 : 0);

    left_padded_dims_ = std::vector<ptrdiff_t>(num_dims_with_pad, 1);
    right_padded_dims_ = std::vector<ptrdiff_t>(num_dims_with_pad, 1);

    if (right_num_dims == 1) {
      // right padded to (1,...,K,1)
      right_padded_dims_[num_dims_with_pad - 2] = static_cast<ptrdiff_t>(right_shape[0]);

      if (num_input_dims >= 2) {
        // left padded to (...,1,K)
        TensorShapeCopyDims(left_shape, &left_padded_dims_[0], left_num_dims - 2);
        left_padded_dims_[num_dims_with_pad - 3] = static_cast<ptrdiff_t>(left_shape[transa ? left_num_dims - 1 : left_num_dims - 2]);
        left_padded_dims_[num_dims_with_pad - 1] = static_cast<ptrdiff_t>(left_shape[transa ? left_num_dims - 2 : left_num_dims - 1]);
      } else {
        // pad 1 in the front
        TensorShapeCopyDims(left_shape, &left_padded_dims_[num_dims_with_pad - left_num_dims], left_num_dims);
      }
    } else {
      // pad 1 in the front for left
      TensorShapeCopyDims(left_shape, &left_padded_dims_[num_dims_with_pad - left_num_dims], left_num_dims);
      // pad 1 in the front for right
      TensorShapeCopyDims(right_shape, &right_padded_dims_[num_dims_with_pad - right_num_dims], right_num_dims);
    }

    // validate input shape and generate output shape
    std::vector<int64_t> output_dims(num_output_dims);

    // broadcasting for all output dims except last two
    for (size_t idx_dim = 0; idx_dim < num_dims_with_pad - 2; ++idx_dim) {
      output_dims[idx_dim] = std::max(left_padded_dims_[idx_dim], right_padded_dims_[idx_dim]);
      if (left_padded_dims_[idx_dim] != output_dims[idx_dim])
        ORT_RETURN_IF_NOT(left_padded_dims_[idx_dim] == 1, "left operand cannot broadcast on dim ", idx_dim);
      if (right_padded_dims_[idx_dim] != output_dims[idx_dim])
        ORT_RETURN_IF_NOT(right_padded_dims_[idx_dim] == 1, "right operand cannot broadcast on dim ", idx_dim);
    }
    if (transa) {
      M_ = static_cast<ptrdiff_t>(has_1D_input ? 1 : left_shape[left_num_dims - 1]);
      K_ = static_cast<ptrdiff_t>(left_shape[left_num_dims - 2]);
    } else {
      M_ = static_cast<ptrdiff_t>(has_1D_input ? 1 : left_shape[left_num_dims - 2]);
      K_ = static_cast<ptrdiff_t>(left_shape[left_num_dims - 1]);
    }

    if (transb) {
      N_ = static_cast<ptrdiff_t>((right_num_dims == 1) ? 1 : right_shape[right_num_dims - 2]);
    } else {
      N_ = static_cast<ptrdiff_t>((right_num_dims == 1) ? 1 : right_shape[right_num_dims - 1]);
    }

    if (!has_1D_input) {
      ORT_RETURN_IF_NOT(K_ == right_shape[transb ? right_num_dims - 1 : right_num_dims - 2],
                        "MatMul dimension mismatch");
      // left (...M x K), right (...K x N), output (...M x N)
      ORT_RETURN_IF_NOT(num_dims_with_pad == num_output_dims, "num_dims_with_pad != num_output_dims");
      output_dims[num_output_dims - 2] = M_;
      output_dims[num_output_dims - 1] = N_;
    } else {
      if (num_output_dims == 0) {
        // for left and right being both vector, output is scalar thus no shape
        ORT_RETURN_IF_NOT(M_ == 1 && N_ == 1, "M_ == 1 && N_ == 1 was false");
      } else {
        if (left_num_dims == 1) {
          ORT_RETURN_IF_NOT(num_dims_with_pad - 1 == num_output_dims, "num_dims_with_pad - 1 != num_output_dims");
          ORT_RETURN_IF_NOT(K_ == right_shape[transb ? right_num_dims - 1 : right_num_dims - 2],
                            "MatMul dimension mismatch");
          // left (K), right (...K,N), output (...N)
          output_dims[num_output_dims - 1] = N_;
        } else {
          ORT_RETURN_IF_NOT(num_dims_with_pad - 2 == num_output_dims, "num_dims_with_pad - 2 != num_output_dims");
          ORT_RETURN_IF_NOT(K_ == right_shape[0], "MatMul dimension mismatch");
          // left(...K), right (K), output (...), already assigned
        }
      }
    }

    // assign shape
    output_shape_ = TensorShape(output_dims);

    // compute broadcast offsets
    ComputeBroadcastOffsets();

    return Status::OK();
  }

  Status Compute(const TensorShape& left_shape, const TensorShape& right_shape,
                 const TensorShape* right_scale_shape, const TensorShape* right_zp_shape,
                 bool transa = false, bool transb = false) {
    ORT_RETURN_IF_ERROR(Compute(left_shape, right_shape, transa, transb));
    right_zp_offsets_.clear();
    right_scale_offsets_.clear();
    right_zp_offsets_.resize(right_offsets_.size());
    right_scale_offsets_.resize(right_offsets_.size());

    auto set_right_param = [this, &right_shape](const TensorShape* param_shape, std::vector<size_t>& param_offsets) {
      if (nullptr != param_shape && param_shape->NumDimensions() > 1) {
        ORT_RETURN_IF_NOT(param_shape->NumDimensions() == right_shape.NumDimensions() && param_shape->Size() * K_ == right_shape.Size(),
                          "Per-column quantization parameter of batched matrix should have same dimension as the matrix,"
                          "and its size by K should be equal to the matrix's size.");
        for (size_t batch_id = 0; batch_id < param_offsets.size(); batch_id++) {
          param_offsets[batch_id] = right_offsets_[batch_id] / K_;
        }
      }
      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(set_right_param(right_zp_shape, right_zp_offsets_));
    ORT_RETURN_IF_ERROR(set_right_param(right_scale_shape, right_scale_offsets_));

    return Status::OK();
  }

 private:
  void ComputeBroadcastOffsets() {
    num_broadcasted_dims_ = left_padded_dims_.size() - 2;

    if (num_broadcasted_dims_ == 0) {
      left_offsets_ = {0};
      right_offsets_ = {0};
      output_offsets_ = {0};
      return;
    }

    left_mat_size_ = M_ * K_;
    right_mat_size_ = K_ * N_;
    output_mat_size_ = M_ * N_;

    // stride in mats and dims for broadcasting
    left_padded_strides_.resize(num_broadcasted_dims_);
    right_padded_strides_.resize(num_broadcasted_dims_);
    output_broadcast_strides_.resize(num_broadcasted_dims_);
    output_broadcast_dims_.resize(num_broadcasted_dims_);
    for (size_t i = num_broadcasted_dims_; i > 0; --i) {
      size_t idx = i - 1;
      output_broadcast_dims_[idx] = std::max(left_padded_dims_[idx], right_padded_dims_[idx]);
      output_broadcast_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : output_broadcast_strides_[idx + 1] * output_broadcast_dims_[idx + 1]);
      left_padded_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : left_padded_strides_[idx + 1] * left_padded_dims_[idx + 1]);
      right_padded_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : right_padded_strides_[idx + 1] * right_padded_dims_[idx + 1]);
    }

    size_t num_offsets = output_broadcast_dims_[0] * output_broadcast_strides_[0];
    left_offsets_.resize(num_offsets);
    right_offsets_.resize(num_offsets);
    output_offsets_.resize(num_offsets);

    RecursiveFill(0, 0, 0, 0);
  }

  void RecursiveFill(size_t idx_dim, size_t idx_left, size_t idx_right, size_t idx_out) {
    if (idx_dim == num_broadcasted_dims_) {
      left_offsets_[idx_out] = idx_left * left_mat_size_;
      right_offsets_[idx_out] = idx_right * right_mat_size_;
      output_offsets_[idx_out] = idx_out * output_mat_size_;
    } else {
      auto left_dim = left_padded_dims_[idx_dim];
      auto right_dim = right_padded_dims_[idx_dim];
      auto output_dim = output_broadcast_dims_[idx_dim];
      for (int i = 0; i < output_dim; ++i) {
        RecursiveFill(idx_dim + 1,
                      idx_left + i * (left_dim == 1 ? 0 : left_padded_strides_[idx_dim]),
                      idx_right + i * (right_dim == 1 ? 0 : right_padded_strides_[idx_dim]),
                      idx_out + i * output_broadcast_strides_[idx_dim]);
      }
    }
  }

 private:
  size_t left_mat_size_ = 0;
  size_t right_mat_size_ = 0;
  size_t output_mat_size_ = 0;

  size_t num_broadcasted_dims_ = 0;

  std::vector<ptrdiff_t> left_padded_dims_;
  std::vector<ptrdiff_t> right_padded_dims_;
  std::vector<ptrdiff_t> output_broadcast_dims_;

  std::vector<size_t> left_padded_strides_;
  std::vector<size_t> right_padded_strides_;
  std::vector<size_t> output_broadcast_strides_;

  TensorShape output_shape_;

  ptrdiff_t M_ = 0;
  ptrdiff_t N_ = 0;
  ptrdiff_t K_ = 0;

  std::vector<size_t> left_offsets_;
  std::vector<size_t> right_offsets_;
  std::vector<size_t> output_offsets_;

  std::vector<size_t> right_zp_offsets_;
  std::vector<size_t> right_scale_offsets_;

 public:
  // output shape
  const TensorShape& OutputShape() const {
    return output_shape_;
  }

  // left and output matrices' first dim
  ptrdiff_t M() const {
    return M_;
  }

  // right and output matrices' second dim
  ptrdiff_t N() const {
    return N_;
  }

  // left matrices' second dim, and right matrices' first dim
  ptrdiff_t K() const {
    return K_;
  }

  // Batched Gemm offsets in left matrices
  const std::vector<size_t>& LeftOffsets() const {
    return left_offsets_;
  }

  // Batched Gemm offsets in right matrices
  const std::vector<size_t>& RightOffsets() const {
    return right_offsets_;
  }

  // Batched Gemm offsets in output matrices
  const std::vector<size_t>& OutputOffsets() const {
    return output_offsets_;
  }

  // Batched Scale Offset for right matrices
  const std::vector<size_t>& RightScaleOffsets() const {
    return right_scale_offsets_;
  }

  // Batched Zero Point Offset for right matrices
  const std::vector<size_t>& RightZeroPointOffsets() const {
    return right_zp_offsets_;
  }

  template <typename T>
  static void OffsetToArrays(T* p, const std::vector<size_t>& offsets, gsl::span<T*> arrays) {
    auto len = offsets.size();
    ORT_ENFORCE(arrays.size() == len);
    for (size_t i = 0; i < len; i++) {
      arrays[i] = p + offsets[i];
    }
  }

  template <typename T>
  static void OffsetToArrays(const T* p, const std::vector<size_t>& offsets, gsl::span<const T*> arrays) {
    auto len = offsets.size();
    ORT_ENFORCE(arrays.size() == len);
    for (size_t i = 0; i < len; i++) {
      arrays[i] = p + offsets[i];
    }
  }
};

}  // namespace onnxruntime
