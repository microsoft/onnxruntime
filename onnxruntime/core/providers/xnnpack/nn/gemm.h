// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/math/gemm_base.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

#include "core/common/common.h"
#include "core/util/math.h"
//#include "core/providers/cpu/activation/activations.h"
//#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Gemm : protected GemmBase, public OpKernel {
 public:
  Gemm(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // Required for checking XNNpack restrictions on ORT side
  static bool IsOnnxNodeSupported(const onnxruntime::Node& nchw_node, const GraphViewer& graph);

  // Not sure if it's needed yet
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  /* this is from the CPU Gemm operator
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, //output
                 PrePackedWeights* prepacked_weights) override; //output

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   bool& used_shared_buffers) override; //output
  */
  Status CreateXnnpackOpp(
      size_t input_channels,
      size_t output_channels,
      size_t input_stride,
      size_t output_stride,
      const float* kernel,
      const float* bias,
      float output_min,
      float output_max,
      uint32_t flags);

 private:

  TensorShape b_shape_;
  BufferUniquePtr packed_b_=nullptr;
  std::unique_ptr<Tensor> B_;

  // For fused gemm + activation
  // std::unique_ptr<functors::ElementWiseRangedTransform<T>> activation_;

  //void ComputeActivation(T* y_data, size_t y_size, concurrency::ThreadPool* thread_pool) const;

  // from Conv XNNpack opp
  int64_t M=-1;
  int64_t K=-1;
  int64_t N=-1;

  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
};

class GemmHelper {
 public:
  GemmHelper(const TensorShape& left, bool trans_left, const TensorShape& right, bool trans_right, const TensorShape& bias) {
    //dimension check
    ORT_ENFORCE(left.NumDimensions() == 2 || left.NumDimensions() == 1);
    ORT_ENFORCE(right.NumDimensions() == 2);

    if (trans_left) {
      M_ = left.NumDimensions() == 2 ? left[1] : left[0];
      K_ = left.NumDimensions() == 2 ? left[0] : 1;
    } else {
      M_ = left.NumDimensions() == 2 ? left[0] : 1;
      K_ = left.NumDimensions() == 2 ? left[1] : left[0];
    }

    int k_dim;
    if (trans_right) {
      N_ = right[0];
      k_dim = 1;
    } else {
      N_ = right[1];
      k_dim = 0;
    }

    if (right[k_dim] != K_)
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "GEMM: Dimension mismatch, W: ",
                                right.ToString(),
                                " K: " + std::to_string(K_),
                                " N:" + std::to_string(N_));

    if (!IsValidBroadcast(bias, M_, N_))
      status_ = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: Invalid bias shape for broadcast");

    // it is possible the input is empty tensor, for example the output of roipool in fast rcnn.
    ORT_ENFORCE(M_ >= 0 && K_ > 0 && N_ >= 0);
  }

  int64_t M() const { return M_; }
  int64_t N() const { return N_; }
  int64_t K() const { return K_; }
  Status State() const { return status_; }

 private:
  bool IsValidBroadcast(const TensorShape& bias_shape, int64_t M, int64_t N) {
    // valid shapes are (,) , (1, N) , (M, 1) , (M, N)
    if (bias_shape.NumDimensions() > 2)
      return false;
    // shape is (1,) or (1, 1), or (,)
    if (bias_shape.Size() == 1)
      return true;
    // valid bias_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
    // In last case no broadcasting needed, so don't fail it
    return ((bias_shape.NumDimensions() == 1 && bias_shape[0] == N) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == M && (bias_shape[1] == 1 || bias_shape[1] == N)) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == 1 && bias_shape[1] == N));
  }

 private:
  int64_t M_;
  int64_t K_;
  int64_t N_;
  Status status_;
};

template <typename T>
void GemmBroadcastBias(int64_t M, int64_t N, float beta,
                       const T* c_data, const TensorShape* c_shape,
                       T* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<T>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<T>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<T>(c_data, M, N);
    }
  }
}

}  // namespace xnnpack
}  // namespace onnxruntime
