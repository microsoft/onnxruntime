// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn_rnn_base.h"
#include "rnn_impl.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void CudnnRnnBase<T>::SetWeightBias(const cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnn_desc,
                                    const int pseudo_layer,
                                    const cudnnTensorDescriptor_t x_desc,
                                    const cudnnFilterDescriptor_t w_desc,
                                    const cudnnFilterDescriptor_t filter_desc,
                                    const void* w_data,
                                    const int lin_layer_id,
                                    const T* pos,
                                    int& offset,
                                    bool is_matrix) const {
  int numDims;
  std::vector<int> matDims(3);
  cudnnDataType_t dt;
  cudnnTensorFormat_t tf;
  T* mem_offset;

  if (is_matrix) {
    cudnnGetRNNLinLayerMatrixParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w_data, lin_layer_id, filter_desc, (void**)&mem_offset);
  } else {
    cudnnGetRNNLinLayerBiasParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w_data, lin_layer_id, filter_desc, (void**)&mem_offset);
  }

  cudnnGetFilterNdDescriptor(filter_desc, 3, &dt, &tf, &numDims, matDims.data());
  int count = matDims[0] * matDims[1] * matDims[2];
  cudaMemcpyAsync(mem_offset, pos + offset, count * sizeof(T), cudaMemcpyDeviceToDevice);
  offset += count;
}
template <typename T>
Status CudnnRnnBase<T>::SetCudnnRnnWeightBias(const cudnnHandle_t cudnn_handle,
                                              const cudnnRNNDescriptor_t rnn_desc,
                                              const cudnnTensorDescriptor_t x_desc,
                                              const cudnnFilterDescriptor_t w_desc,
                                              void* w_data,
                                              const T* W_data,
                                              const T* R_data,
                                              const T* B_data) const {
  //Onnx only support 1 layer
  int w_offset = 0;
  int r_offset = 0;
  int bias_offset = 0;
  for (int layer = 0; layer < num_layers_ * num_directions_; ++layer) {
    for (int idx = 0; idx < W_lin_layer_id_.size(); ++idx) {
      SetWeightBias(cudnn_handle, rnn_desc, layer, x_desc, w_desc, filter_desc_, w_data, W_lin_layer_id_[idx], W_data, w_offset, true);
      if (B_data != nullptr) {
        SetWeightBias(cudnn_handle, rnn_desc, layer, x_desc, w_desc, filter_desc_, w_data, W_lin_layer_id_[idx], B_data, bias_offset, false);
      }
    }
    for (int idx = 0; idx < R_lin_layer_id_.size(); ++idx) {
      SetWeightBias(cudnn_handle, rnn_desc, layer, x_desc, w_desc, filter_desc_, w_data, R_lin_layer_id_[idx], R_data, r_offset, true);
      if (B_data != nullptr) {
        SetWeightBias(cudnn_handle, rnn_desc, layer, x_desc, w_desc, filter_desc_, w_data, R_lin_layer_id_[idx], B_data, bias_offset, false);
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::SetCudnnRnnDesc() {
  typedef typename ToCudaType<T>::MappedType CudaT;

  cudnnDirectionMode_t cudnn_direction = CUDNN_UNIDIRECTIONAL;
  if (direction_ == "bidirectional") {
    cudnn_direction = CUDNN_BIDIRECTIONAL;
  } else if (direction_ == "forward") {
    cudnn_direction = CUDNN_UNIDIRECTIONAL;
  } else if (direction_ == "reverse") {
    cudnn_direction = CUDNN_UNIDIRECTIONAL;
    // need to reverse data
    reverse_ = true;
  }

  cudnn_dropout_desc_.Set(CudnnHandle());
  ORT_RETURN_IF_ERROR(rnn_desc_.Set(CudnnHandle(), hidden_size_, num_layers_, cudnn_dropout_desc_,
                                            cudnn_direction, rnn_mode_, CudnnTensor::GetDataType<CudaT>()));

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                                          IAllocatorUniquePtr<void>& target_w_data,
                                          CudnnFilterDescriptor& target_w_desc) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  int64_t input_size = W->Shape()[2];
  // RNN W[num_directions_, hidden_size_, input_size]
  // RNN R[num_directions_, hidden_size_, hidden_size_]
  // RNN B[num_directions_, 2*hidden_size_]
  // GRU W[num_directions_, 3*hidden_size_, input_size]
  // GRU R[num_directions_, 3*hidden_size_, hidden_size_]
  // GRU B[num_directions_, 6*hidden_size_]
  // LSTM W[num_directions_, 4*hidden_size_, input_size]
  // LSTM R[num_directions_, 4*hidden_size_, hidden_size_]
  // LSTM B[num_directions_, 8*hidden_size_]
  size_t number = W_lin_layer_id_.size();
  int64_t w_size = num_directions_ * (number * hidden_size_ * (input_size + hidden_size_ + 2));
  std::vector<int64_t> dims_w({w_size, 1, 1});
  ORT_RETURN_IF_ERROR(target_w_desc.Set(dims_w, CudnnTensor::GetDataType<CudaT>()));

  std::vector<int64_t> fake_dims_x({1, input_size, 1});
  CudnnTensor fake_x_desc;
  fake_x_desc.Set(fake_dims_x, CudnnTensor::GetDataType<CudaT>());

  // Prepare the weight data
  target_w_data = GetScratchBuffer<void>(w_size * sizeof(T));

  const T* W_data = W->template Data<T>();
  const T* R_data = R->template Data<T>();
  const T* B_data = B == nullptr ? nullptr : B->template Data<T>();

  ORT_RETURN_IF_ERROR(SetCudnnRnnWeightBias(CudnnHandle(), rnn_desc_, fake_x_desc, target_w_desc,
                                                    target_w_data.get(), W_data, R_data, B_data));

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::CacheCudnnRnnWeights(const OpKernelInfo& info) {
  // Cache the weight
  const Tensor* W;
  const Tensor* R;
  const Tensor* B;
  bool get_W = info.TryGetConstantInput(Input_Index::W, &W);
  bool get_R = info.TryGetConstantInput(Input_Index::R, &R);
  bool get_B = info.TryGetConstantInput(Input_Index::B, &B);

  if (get_W && get_R) {
    if (get_B) {
      ORT_RETURN_IF_ERROR(ReorganizeWeights(W, R, B, w_data_cache_, w_desc_cache_));
    } else {
      ORT_RETURN_IF_ERROR(ReorganizeWeights(W, R, nullptr, w_data_cache_, w_desc_cache_));
    }
    weight_cached_ = true;
  }

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  // inputs
  const Tensor* X = ctx->Input<Tensor>(Input_Index::X);  // inputs. [seq_length, batch_size, input_size]
  ORT_ENFORCE(nullptr != X);

  // optional inputs
  const Tensor* sequence_lens = ctx->Input<Tensor>(Input_Index::sequence_lens);  // [batch_size]
  const Tensor* initial_h = ctx->Input<Tensor>(Input_Index::initial_h);          // initial hidden. [num_directions_, batch_size, hidden_size_]
  const Tensor* initial_c(nullptr);
  if (rnn_mode_ == CUDNN_LSTM) {
    initial_c = ctx->Input<Tensor>(Input_Index::initial_c);  // initial cell. [num_directions_, batch_size, hidden_size_]
  }

  int64_t seq_length = X->Shape()[0];
  int64_t batch_size = X->Shape()[1];
  int64_t input_size = X->Shape()[2];

  // optional outputs
  std::vector<int64_t> dims_Y({seq_length, num_directions_, batch_size, hidden_size_});
  std::vector<int64_t> dims_hxy({num_layers_ * num_directions_, batch_size, hidden_size_});
  std::vector<int64_t> dims_yc{num_directions_, batch_size, hidden_size_};
  Tensor* Y = ctx->Output(Output_Index::Y, dims_Y);
  Tensor* Y_h = ctx->Output(Output_Index::Y_h, dims_hxy);
  Tensor* Y_c = ctx->Output(Output_Index::Y_c, dims_yc);

  std::vector<int64_t> dims_x({batch_size, input_size, 1});
  std::vector<int64_t> dims_y({batch_size, hidden_size_ * num_directions_, 1});

  CudnnTensor x_desc_temp;
  x_desc_temp.Set(dims_x, CudnnTensor::GetDataType<CudaT>());
  CudnnTensor y_desc_temp;
  y_desc_temp.Set(dims_y, CudnnTensor::GetDataType<CudaT>());
  std::vector<cudnnTensorDescriptor_t> x_desc(seq_length, x_desc_temp);
  std::vector<cudnnTensorDescriptor_t> y_desc(seq_length, y_desc_temp);

  CudnnTensor hx_desc;
  CudnnTensor cx_desc;
  CudnnTensor y_h_desc;
  CudnnTensor y_c_desc;
  ORT_RETURN_IF_ERROR(hx_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(cx_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(y_h_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(y_c_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));

  // Prepare the weight data
  IAllocatorUniquePtr<void> w_data;
  CudnnFilterDescriptor w_desc;
  if (!weight_cached_) {
    const Tensor& W = *ctx->Input<Tensor>(Input_Index::W);
    const Tensor& R = *ctx->Input<Tensor>(Input_Index::R);
    const Tensor* B = ctx->Input<Tensor>(Input_Index::B);
    ReorganizeWeights(&W, &R, B, w_data, w_desc);
  }

  IAllocatorUniquePtr<T> x_reversed_data;
  T* x_data = const_cast<T*>(X->template Data<T>());
  if (reverse_) {
    // reverse input data
    x_reversed_data = GetScratchBuffer<T>(seq_length * batch_size * input_size);
    ReverseBySequence(gsl::narrow_cast<int32_t>(seq_length),
                      gsl::narrow_cast<int32_t>(batch_size),
                      gsl::narrow_cast<int32_t>(input_size),
                      reinterpret_cast<CudaT*>(x_data),
                      reinterpret_cast<CudaT*>(x_reversed_data.get()),
                      seq_length * batch_size * input_size);
    x_data = x_reversed_data.get();
  }

  const T* hx_data = (initial_h == nullptr) ? nullptr : initial_h->template Data<T>();
  const T* cx_data = (initial_c == nullptr) ? nullptr : initial_c->template Data<T>();
  T* y_h_data = (Y_h == nullptr) ? nullptr : Y_h->template MutableData<T>();
  T* y_c_data = (Y_c == nullptr) ? nullptr : Y_c->template MutableData<T>();
  int64_t output_size = seq_length * num_directions_ * batch_size * hidden_size_;
  T* y_data = nullptr;
  IAllocatorUniquePtr<T> y_alloc_data;
  if (Y != nullptr) {
    y_data = Y->template MutableData<T>();
  } else {
    y_alloc_data = GetScratchBuffer<T>(output_size);
    y_data = y_alloc_data.get();
  }
  const int32_t* sequence_lens_data = (sequence_lens == nullptr) ? nullptr : sequence_lens->template Data<int32_t>();

  size_t workspace_bytes;
  CUDNN_RETURN_IF_ERROR(cudnnGetRNNWorkspaceSize(CudnnHandle(), rnn_desc_, gsl::narrow_cast<int>(seq_length), x_desc.data(), &workspace_bytes));
  workspace_bytes *= num_directions_;
  auto workspace_cuda = GetScratchBuffer<void>(workspace_bytes);

  CUDNN_RETURN_IF_ERROR(cudnnRNNForwardInference(CudnnHandle(),
                                                 rnn_desc_,
                                                 gsl::narrow_cast<int>(seq_length),
                                                 x_desc.data(),
                                                 x_data,
                                                 hx_desc,
                                                 hx_data,
                                                 cx_desc,
                                                 cx_data,
                                                 weight_cached_ ? w_desc_cache_ : w_desc,
                                                 weight_cached_ ? w_data_cache_.get() : w_data.get(),
                                                 y_desc.data(),
                                                 y_data,
                                                 y_h_desc,
                                                 y_h_data,
                                                 y_c_desc,
                                                 y_c_data,
                                                 workspace_cuda.get(),
                                                 workspace_bytes));
  IAllocatorUniquePtr<T> y_reorganized_data;
  if (reverse_ || num_directions_ == 2) {
    //reverse output
    y_reorganized_data = GetScratchBuffer<T>(output_size);
    if (reverse_) {
      //reverse output data
      ReverseBySequence(gsl::narrow_cast<int32_t>(seq_length),
                        gsl::narrow_cast<int32_t>(batch_size),
                        gsl::narrow_cast<int32_t>(hidden_size_),
                        reinterpret_cast<CudaT*>(y_data),
                        reinterpret_cast<CudaT*>(y_reorganized_data.get()),
                        output_size);
    } else {
      ReorderBidirectionalDataInSequence(gsl::narrow_cast<int32_t>(seq_length),
                                         gsl::narrow_cast<int32_t>(batch_size),
                                         gsl::narrow_cast<int32_t>(hidden_size_),
                                         reinterpret_cast<CudaT*>(y_data),
                                         reinterpret_cast<CudaT*>(y_reorganized_data.get()),
                                         output_size);
    }

    if (Y != nullptr) {
      // User specified this optional output, so need to copy the reversed data to orignial place
      cudaMemcpyAsync(y_data, y_reorganized_data.get(), output_size * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      y_data = y_reorganized_data.get();
    }
  }

  if (sequence_lens_data != nullptr && y_h_data != nullptr && y_data != nullptr) {
    RnnMaskImpl(gsl::narrow_cast<int32_t>(num_directions_),
                gsl::narrow_cast<int32_t>(seq_length),
                gsl::narrow_cast<int32_t>(batch_size),
                gsl::narrow_cast<int32_t>(hidden_size_),
                sequence_lens_data,
                reinterpret_cast<CudaT*>(y_data),
                reinterpret_cast<CudaT*>(y_h_data),
                output_size);
  }

  return Status::OK();
}

template class CudnnRnnBase<float>;
template class CudnnRnnBase<double>;
template class CudnnRnnBase<MLFloat16>;

}  // namespace cuda
}  // namespace onnxruntime
