// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"

#include <cudnn.h>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

enum RNN_Input_Index {
  X = 0,
  W = 1,
  R = 2,
  B = 3,
  sequence_lens = 4,
  initial_h = 5,
  initial_c = 6
};

// Onnx RNN/GRU/LSTM only support 1 layer
constexpr int RNN_NUM_LAYERS = 1;

class CudnnRNN {
 public:
  CudnnRNN() : cudnn_rnn_desc_(nullptr) {
  }

  ~CudnnRNN() {
    if (cudnn_rnn_desc_ != nullptr) {
      cudnnDestroyRNNDescriptor(cudnn_rnn_desc_);
      cudnn_rnn_desc_ = nullptr;
    }
  }

  Status Set(int64_t input_size, int64_t hidden_size, int64_t proj_size, int num_layers,
             cudnnDropoutDescriptor_t cudnn_dropout_desc, cudnnDirectionMode_t cudnn_direction_model,
             cudnnRNNMode_t rnn_mode, bool has_bias, cudnnDataType_t dataType) {
    if (!cudnn_rnn_desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreateRNNDescriptor(&cudnn_rnn_desc_));

    CUDNN_RETURN_IF_ERROR(cudnnSetRNNDescriptor_v8(cudnn_rnn_desc_,
                                                   CUDNN_RNN_ALGO_STANDARD,  // CUDNN_RNN_ALGO_PERSIST_STATIC, CUDNN_RNN_ALGO_PERSIST_DYNAMIC
                                                   rnn_mode,
                                                   has_bias ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS,
                                                   cudnn_direction_model,
                                                   CUDNN_LINEAR_INPUT,
                                                   dataType,
                                                   dataType,
                                                   dataType == CUDNN_DATA_HALF ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH,
                                                   gsl::narrow_cast<int>(input_size),
                                                   gsl::narrow_cast<int>(hidden_size),
                                                   gsl::narrow_cast<int>(proj_size),  // projected size
                                                   num_layers,
                                                   cudnn_dropout_desc,
                                                   // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED works with CUDNN_RNN_PADDED_IO_ENABLED, so that it will auto fill 0 for the shorter sequences
                                                   CUDNN_RNN_PADDED_IO_ENABLED));

    return Status::OK();
  }

  operator cudnnRNNDescriptor_t() const {
    return cudnn_rnn_desc_;
  }

 private:
  cudnnRNNDescriptor_t cudnn_rnn_desc_;
};

template <typename T>
class CudnnRnnBase : public CudaKernel {
  const std::set<std::string> allowed_directions{"forward", "reverse", "bidirectional"};

 public:
  CudnnRnnBase(const OpKernelInfo& info) : CudaKernel{info} {
    reverse_ = false;
    std::string direction = "forward";
    direction = info.GetAttrOrDefault<std::string>("direction", "forward");
    cudnn_direction_mode_ = CUDNN_UNIDIRECTIONAL;
    if (direction == "bidirectional") {
      cudnn_direction_mode_ = CUDNN_BIDIRECTIONAL;
    } else if (direction == "forward") {
      cudnn_direction_mode_ = CUDNN_UNIDIRECTIONAL;
    } else if (direction == "reverse") {
      cudnn_direction_mode_ = CUDNN_UNIDIRECTIONAL;
      // need to reverse data
      reverse_ = true;
    }

    num_directions_ = cudnn_direction_mode_ == CUDNN_BIDIRECTIONAL ? 2 : 1;
    ORT_ENFORCE(allowed_directions.find(direction) != allowed_directions.end());

    ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK() && hidden_size_ > 0);
    rnn_mode_ = CUDNN_LSTM;
    weight_cached_ = false;
    w_data_cache_ = nullptr;

    size_t state_size;
    auto default_cudnn_handle = DefaultCudnnHandle();
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.CreateDescriptorIfNeeded());
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.GetCudnnDropoutStatesSize(default_cudnn_handle, state_size));
    state_buffer_ = GetScratchBuffer<void>(state_size, nullptr);
    ORT_THROW_IF_ERROR(cudnn_dropout_desc_.Set(default_cudnn_handle, state_buffer_.get(), state_size));

    layout_ = info.GetAttrOrDefault("layout", static_cast<int64_t>(0));
    ORT_ENFORCE(layout_ == 0,
                "Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification.");
  }

  Status CacheCudnnRnnWeights(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

  void SetRNNMode(cudnnRNNMode_t rnn_mode) { rnn_mode_ = rnn_mode; }

 private:
  Status SetCudnnRnnWeightBias(const cudnnHandle_t cudnn_handle,
                               const cudnnRNNDescriptor_t rnn_desc,
                               size_t w_data_size,
                               void* w_data,
                               const T* W_data,
                               const T* R_data,
                               const T* B_data,
                               cudaStream_t cuda_stream) const;

  Status ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                           size_t& target_w_data_size_in_bytes,
                           IAllocatorUniquePtr<void>& target_w_data,
                           CudnnFilterDescriptor& target_w_desc,
                           CudnnRNN& rnn_desc,
                           onnxruntime::Stream* ort_stream) const;

  Status SetWeightBias(const cudnnHandle_t handle,
                       const cudnnRNNDescriptor_t rnn_desc,
                       const int pseudo_layer,
                       size_t w_data_size,
                       const void* w_data,
                       const int lin_layer_id,
                       const T* pos,
                       int& offset,
                       bool is_matrix,
                       cudaStream_t cuda_stream) const;

  void SetZeroSequences(const int64_t zero_seq_index_cache_size,
                        const std::vector<int32_t> zero_seq_index_cache,
                        T* y_data,
                        T* y_h_data,
                        T* y_c_data,
                        onnxruntime::Stream* cuda_stream) const;

 protected:
  // W_lin_layer_id_ & R_lin_layer_id_ are set in Constructor
  std::vector<int> W_lin_layer_id_;
  std::vector<int> R_lin_layer_id_;

 private:
  cudnnDirectionMode_t cudnn_direction_mode_;
  bool reverse_;
  int64_t num_directions_;
  // hidden_size_ from attribute
  int64_t hidden_size_;
  cudnnRNNMode_t rnn_mode_;
  // w_desc_cache_ & w_data_cache_ are changed in Constructor if we can get the weights as constant input
  CudnnFilterDescriptor w_desc_cache_;
  size_t w_data_cache_size_in_bytes_;
  IAllocatorUniquePtr<void> w_data_cache_;
  bool weight_cached_;
  int64_t layout_;

  // cudnn_dropout_desc_ is a cache, never to be changed
  IAllocatorUniquePtr<void> state_buffer_;
  CudnnDropout cudnn_dropout_desc_;

  enum Output_Index {
    Y = 0,
    Y_h = 1,
    Y_c = 2
  };
};

}  // namespace cuda
}  // namespace onnxruntime
