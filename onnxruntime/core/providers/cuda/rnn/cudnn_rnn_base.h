// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

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
const int RNN_NUM_LAYERS = 1;

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

  Status Set(const cudnnHandle_t& cudnnHandle, int64_t hidden_size, int num_layers,
             cudnnDropoutDescriptor_t cudnn_dropout_desc, cudnnDirectionMode_t cudnn_direction_model,
             cudnnRNNMode_t rnn_mode, cudnnDataType_t dataType, const cudaDeviceProp& prop) {
    if (!cudnn_rnn_desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreateRNNDescriptor(&cudnn_rnn_desc_));

    CUDNN_RETURN_IF_ERROR(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                                cudnn_rnn_desc_,
                                                gsl::narrow_cast<int>(hidden_size),
                                                num_layers,
                                                cudnn_dropout_desc,
                                                CUDNN_LINEAR_INPUT,  // We can also skip the input matrix transformation
                                                cudnn_direction_model,
                                                rnn_mode,
                                                CUDNN_RNN_ALGO_STANDARD,  //CUDNN_RNN_ALGO_PERSIST_STATIC, CUDNN_RNN_ALGO_PERSIST_DYNAMIC
                                                dataType));

    if (prop.major >= 7 && dataType == CUDNN_DATA_HALF) {
      cudnnSetRNNMatrixMathType(cudnn_rnn_desc_, CUDNN_TENSOR_OP_MATH);
    }

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
    cudnn_dropout_desc_.CreateDescriptorIfNeeded();
    cudnn_dropout_desc_.GetCudnnDropoutStatesSize(CudnnHandle(), state_size);
    state_buffer_ = GetScratchBuffer<void>(state_size);
    cudnn_dropout_desc_.Set(CudnnHandle(), state_buffer_.get(), state_size);

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
                               const cudnnTensorDescriptor_t x_desc,
                               const cudnnFilterDescriptor_t w_desc,
                               void* w_data,
                               const T* W_data,
                               const T* R_data,
                               const T* B_data) const;

  Status ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                           IAllocatorUniquePtr<void>& target_w_data,
                           CudnnFilterDescriptor& target_w_desc,
                           CudnnRNN& rnn_desc) const;

  void SetWeightBias(const cudnnHandle_t handle,
                     const cudnnRNNDescriptor_t rnn_desc,
                     const int pseudo_layer,
                     const cudnnTensorDescriptor_t x_desc,
                     const cudnnFilterDescriptor_t w_desc,
                     const cudnnFilterDescriptor_t filter_desc,
                     const void* w_data,
                     const int lin_layer_id,
                     const T* pos,
                     int& offset,
                     bool is_matrix) const;

  void SetZeroSequences(const int64_t zero_seq_index_cache_size,
                        const std::vector<int32_t> zero_seq_index_cache,
                        T* y_data,
                        T* y_h_data,
                        T* y_c_data) const;

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
