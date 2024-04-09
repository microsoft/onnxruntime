// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "cudnn_rnn_base.h"
#include "rnn_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status CudnnRnnBase<T>::SetWeightBias(const cudnnHandle_t handle,
                                      const cudnnRNNDescriptor_t rnn_desc,
                                      const int pseudo_layer,
                                      size_t reorganized_w_data_size,
                                      const void* reorganized_w_data,
                                      const int lin_layer_id,
                                      const T* pos,
                                      int& offset,
                                      bool is_matrix,
                                      cudaStream_t cuda_stream) const {
  int numDims;
  std::array<int, 3> matDims;
  std::array<int, 3> strideA;
  cudnnDataType_t dt;
  T* mem_offset;

  CudnnTensor tensor_desc_matrix, tensor_desc_bias;
  ORT_RETURN_IF_ERROR(tensor_desc_bias.CreateTensorIfNeeded());
  ORT_RETURN_IF_ERROR(tensor_desc_matrix.CreateTensorIfNeeded());

  T *mem_offset_matrix, *mem_offset_bias;
  CUDNN_RETURN_IF_ERROR(cudnnGetRNNWeightParams(
      handle, rnn_desc, pseudo_layer, reorganized_w_data_size, reorganized_w_data,
      lin_layer_id, tensor_desc_matrix, (void**)&mem_offset_matrix, tensor_desc_bias, (void**)&mem_offset_bias));
  CUDNN_RETURN_IF_ERROR(cudnnGetTensorNdDescriptor(
      is_matrix ? tensor_desc_matrix : tensor_desc_bias, 3, &dt, &numDims, matDims.data(), strideA.data()));

  mem_offset = is_matrix ? mem_offset_matrix : mem_offset_bias;
  int count = matDims[0] * matDims[1] * matDims[2];

  if (strideA[0] != count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::INVALID_ARGUMENT, "Stride is not packed");
  }
  CUDA_CALL_THROW(cudaMemcpyAsync(mem_offset, pos + offset, count * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream));

  offset += count;

  return Status::OK();
}
template <typename T>
Status CudnnRnnBase<T>::SetCudnnRnnWeightBias(const cudnnHandle_t cudnn_handle,
                                              const cudnnRNNDescriptor_t rnn_desc,
                                              size_t reorganized_w_data_size,
                                              void* reorganized_w_data,
                                              const T* W_data,
                                              const T* R_data,
                                              const T* B_data,
                                              cudaStream_t cuda_stream) const {
  int w_offset = 0;
  int r_offset = 0;
  int bias_offset = 0;
  for (int layer = 0; layer < RNN_NUM_LAYERS * num_directions_; ++layer) {
    for (size_t idx = 0; idx < W_lin_layer_id_.size(); ++idx) {
      ORT_RETURN_IF_ERROR(SetWeightBias(
          cudnn_handle, rnn_desc, layer, reorganized_w_data_size, reorganized_w_data,
          W_lin_layer_id_[idx], W_data, w_offset, true, cuda_stream));
      if (B_data != nullptr) {
        ORT_RETURN_IF_ERROR(SetWeightBias(cudnn_handle, rnn_desc, layer, reorganized_w_data_size, reorganized_w_data,
                                          W_lin_layer_id_[idx], B_data, bias_offset, false, cuda_stream));
      }
    }
    for (size_t idx = 0; idx < R_lin_layer_id_.size(); ++idx) {
      ORT_RETURN_IF_ERROR(SetWeightBias(cudnn_handle, rnn_desc, layer, reorganized_w_data_size, reorganized_w_data,
                                        R_lin_layer_id_[idx], R_data, r_offset, true, cuda_stream));
      if (B_data != nullptr) {
        ORT_RETURN_IF_ERROR(SetWeightBias(cudnn_handle, rnn_desc, layer, reorganized_w_data_size, reorganized_w_data,
                                          R_lin_layer_id_[idx], B_data, bias_offset, false, cuda_stream));
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::ReorganizeWeights(const Tensor* W, const Tensor* R, const Tensor* B,
                                          size_t& reorganized_w_data_size_in_bytes,
                                          IAllocatorUniquePtr<void>& reorganized_w_data,
                                          CudnnFilterDescriptor& target_w_desc,
                                          CudnnRNN& rnn_desc, onnxruntime::Stream* ort_stream) const {
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
  TensorShapeVector dims_w({w_size, 1, 1});
  ORT_RETURN_IF_ERROR(target_w_desc.Set(dims_w, CudnnTensor::GetDataType<CudaT>()));

  // Prepare the weight data
  reorganized_w_data_size_in_bytes = w_size * sizeof(T);
  reorganized_w_data = GetScratchBuffer<void>(reorganized_w_data_size_in_bytes, ort_stream);

  // In many cases, this allocation is bigger than needed, leaving part of
  // the buffer uninitialized. non-zero garbage data leads to wrong result
  // in call to cudnnRNNForwardInference()
  // TODO! refine allocation size for each case.
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(reorganized_w_data.get(), 0, reorganized_w_data_size_in_bytes, cuda_stream));

  const T* W_data = W->Data<T>();
  const T* R_data = R->Data<T>();
  const T* B_data = B == nullptr ? nullptr : B->Data<T>();

  auto* ort_cuda_stream = dynamic_cast<CudaStream*>(ort_stream);
  cudnnHandle_t cudnn_handle = ort_cuda_stream ? ort_cuda_stream->cudnn_handle_ : DefaultCudnnHandle();
  ORT_RETURN_IF_ERROR(SetCudnnRnnWeightBias(cudnn_handle, rnn_desc,
                                            reorganized_w_data_size_in_bytes, reorganized_w_data.get(),
                                            W_data, R_data, B_data, cuda_stream));

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::CacheCudnnRnnWeights(const OpKernelInfo& info) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  // Cache the weight
  const Tensor* W;
  const Tensor* R;
  const Tensor* B;
  bool get_W = info.TryGetConstantInput(RNN_Input_Index::W, &W);
  bool get_R = info.TryGetConstantInput(RNN_Input_Index::R, &R);
  bool get_B = info.TryGetConstantInput(RNN_Input_Index::B, &B);

  bool has_bias = B != nullptr;

  if (get_W && get_R) {
    CudnnRNN tmp_rnn_desc;
    auto proj_size = hidden_size_;
    ORT_RETURN_IF_ERROR(tmp_rnn_desc.Set(W->Shape()[2],  // input_size
                                         hidden_size_,
                                         proj_size,
                                         RNN_NUM_LAYERS,
                                         cudnn_dropout_desc_,
                                         cudnn_direction_mode_,
                                         rnn_mode_,
                                         has_bias,
                                         CudnnTensor::GetDataType<CudaT>()));
    if (get_B) {
      ORT_RETURN_IF_ERROR(ReorganizeWeights(W, R, B,
                                            w_data_cache_size_in_bytes_, w_data_cache_, w_desc_cache_,
                                            tmp_rnn_desc, nullptr));
    } else {
      ORT_RETURN_IF_ERROR(ReorganizeWeights(W, R, nullptr,
                                            w_data_cache_size_in_bytes_, w_data_cache_, w_desc_cache_,
                                            tmp_rnn_desc, nullptr));
    }
    cudaStreamSynchronize(nullptr);

    weight_cached_ = true;
  }

  return Status::OK();
}

template <typename T>
Status CudnnRnnBase<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  // inputs
  const Tensor* X = ctx->Input<Tensor>(RNN_Input_Index::X);  // inputs. [seq_length, batch_size, input_size]
  ORT_ENFORCE(nullptr != X);

  // optional inputs
  // [batch_size]
  const Tensor* sequence_lens = ctx->Input<Tensor>(RNN_Input_Index::sequence_lens);
  // initial hidden. [num_directions_, batch_size, hidden_size_]
  const Tensor* initial_h = ctx->Input<Tensor>(RNN_Input_Index::initial_h);
  const Tensor* initial_c(nullptr);
  if (rnn_mode_ == CUDNN_LSTM) {
    // initial cell. [num_directions_, batch_size, hidden_size_]
    initial_c = ctx->Input<Tensor>(RNN_Input_Index::initial_c);
  }

  size_t proj_size = hidden_size_;
  int64_t seq_length = X->Shape()[0];
  int64_t batch_size = X->Shape()[1];
  int64_t input_size = X->Shape()[2];

  // we thread a single input as sequence_lens of length 1, require to expand to [batch_size]?
  std::vector<int32_t> sequence_lengths_temp;
  if (!sequence_lens) {
    sequence_lengths_temp.resize(batch_size, gsl::narrow_cast<int32_t>(seq_length));
  }

  const int32_t* sequence_lens_data = (sequence_lens == nullptr)
                                          ? sequence_lengths_temp.data()
                                          : sequence_lens->Data<int32_t>();

  // cuDNN doesn't support 0 sequence inside the batch, find the 0 sequence and set it to 1
  // there's a ZeroMask kernel to reset the result to 0 for the 0 sequence
  int64_t zero_seq_count = 0;
  std::vector<int32_t> zero_seq_index_cache(batch_size, 0);

  CudaAsyncBuffer<int32_t> sequence_lens_buffer(this, batch_size);
  int32_t* seq_len_array = sequence_lens_buffer.CpuPtr();

  // 0-len sequences are not supported by cuDNN.
  // Replace them by sequences of len 1 and mask them out with SetZeroSequences
  for (int i = 0; i < batch_size; ++i) {
    if (0 == sequence_lens_data[i]) {
      seq_len_array[i] = 1;
      zero_seq_index_cache[zero_seq_count] = i;
      ++zero_seq_count;
    } else {
      seq_len_array[i] = sequence_lens_data[i];
    }
  }

  // Calculate the zero position cache for reverse direction if it's bidirectional
  // The cache is for Y_h or Y_c, and the 1st sequence for Y, no need to do it for other sequence in Y since
  // we hacked the 0 sequence to 1
  if (zero_seq_count && num_directions_ > 1) {
    zero_seq_index_cache.resize(zero_seq_count * num_directions_);
    for (int64_t i = 0; i < zero_seq_count; ++i) {
      zero_seq_index_cache[static_cast<size_t>(zero_seq_count) + i] =
          static_cast<int32_t>(batch_size + zero_seq_index_cache[i]);
    }
    zero_seq_count *= num_directions_;
  }

  // Prior to cuDNN 8.9.1 the sequence lens buffer must be passed to cudnnRNNForward and thus is must
  // be copied to the GPU always.
  ORT_RETURN_IF_ERROR(sequence_lens_buffer.CopyToGpu(ctx->GetComputeStream()));
  // Starting with cuDNN 8.9.1 the sequence lens buffer is ignored by cudnnRNNForward and thus it must
  // be copied to the GPU only for the ReverseBySequence kernels.
  // if (reverse_) {
  //  ORT_RETURN_IF_ERROR(sequence_lens_buffer.CopyToGpu(ctx->GetComputeStream()));
  // }

  // optional outputs
  TensorShapeVector dims_Y({seq_length, num_directions_, batch_size, hidden_size_});
  TensorShapeVector dims_hxy({RNN_NUM_LAYERS * num_directions_, batch_size, hidden_size_});
  TensorShapeVector dims_yc{num_directions_, batch_size, hidden_size_};
  Tensor* Y = ctx->Output(Output_Index::Y, dims_Y);
  Tensor* Y_h = ctx->Output(Output_Index::Y_h, dims_hxy);
  Tensor* Y_c = ctx->Output(Output_Index::Y_c, dims_yc);

  IAllocatorUniquePtr<T> x_reversed_data;
  const T* x_data = X->Data<T>();
  if (reverse_) {
    // reverse input data
    x_reversed_data = GetScratchBuffer<T>(seq_length * batch_size * input_size, ctx->GetComputeStream());
    ReverseBySequence(Stream(ctx),
                      gsl::narrow_cast<int32_t>(seq_length),
                      sequence_lens_buffer.GpuPtr(),
                      gsl::narrow_cast<int32_t>(batch_size),
                      gsl::narrow_cast<int32_t>(input_size),
                      reinterpret_cast<const CudaT*>(x_data),
                      reinterpret_cast<CudaT*>(x_reversed_data.get()),
                      seq_length * batch_size * input_size);
  }

  const T* x_data_input = reverse_ ? x_reversed_data.get() : x_data;

  const T* hx_data = (initial_h == nullptr) ? nullptr : initial_h->Data<T>();
  const T* cx_data = (initial_c == nullptr) ? nullptr : initial_c->Data<T>();
  T* y_h_data = (Y_h == nullptr) ? nullptr : Y_h->MutableData<T>();
  T* y_c_data = (Y_c == nullptr) ? nullptr : Y_c->MutableData<T>();
  int64_t output_size = seq_length * num_directions_ * batch_size * hidden_size_;
  T* y_data = nullptr;
  IAllocatorUniquePtr<T> y_alloc_data;
  if (Y != nullptr) {
    y_data = Y->MutableData<T>();
  } else {
    y_alloc_data = GetScratchBuffer<T>(output_size, ctx->GetComputeStream());
    y_data = y_alloc_data.get();
  }

  const Tensor* B = ctx->Input<Tensor>(RNN_Input_Index::B);
  bool has_bias = B != nullptr;

  CudnnRNN rnn_desc;
  ORT_RETURN_IF_ERROR(rnn_desc.Set(input_size,
                                   hidden_size_,
                                   proj_size,
                                   RNN_NUM_LAYERS,
                                   cudnn_dropout_desc_,
                                   cudnn_direction_mode_,
                                   rnn_mode_,
                                   has_bias,
                                   CudnnTensor::GetDataType<CudaT>()));

  // Prepare the weight data
  size_t w_data_size_in_bytes = 0;
  IAllocatorUniquePtr<void> w_data;
  CudnnFilterDescriptor w_desc;
  if (!weight_cached_) {
    const Tensor& W = *ctx->Input<Tensor>(RNN_Input_Index::W);
    const Tensor& R = *ctx->Input<Tensor>(RNN_Input_Index::R);
    ORT_RETURN_IF_ERROR(ReorganizeWeights(&W, &R, B, w_data_size_in_bytes, w_data, w_desc,
                                          rnn_desc, ctx->GetComputeStream()));
  }

  CudnnDataTensor x_desc1;
  ORT_RETURN_IF_ERROR(x_desc1.Set(CudnnTensor::GetDataType<CudaT>(), seq_length, batch_size,
                                  input_size, seq_len_array));
  CudnnDataTensor y_desc1;
  ORT_RETURN_IF_ERROR(y_desc1.Set(CudnnTensor::GetDataType<CudaT>(), seq_length, batch_size,
                                  ((rnn_mode_ == CUDNN_LSTM) ? proj_size : hidden_size_) * num_directions_,
                                  seq_len_array));

  CudnnTensor cx_desc;
  ORT_RETURN_IF_ERROR(cx_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));

  CudnnTensor hx_desc;
  ORT_RETURN_IF_ERROR(hx_desc.Set(dims_hxy, CudnnTensor::GetDataType<CudaT>()));

  // reserveSpaceSize is not required cudnnRNNForward, but returned by cudnnGetRNNTempSpaceSizes
  size_t workspace_bytes, reservespace_bytes;

  CUDNN_RETURN_IF_ERROR(cudnnGetRNNTempSpaceSizes(GetCudnnHandle(ctx), rnn_desc, CUDNN_FWD_MODE_INFERENCE,
                                                  x_desc1, &workspace_bytes, &reservespace_bytes));
  auto workspace_cuda = GetScratchBuffer<void>(workspace_bytes, ctx->GetComputeStream());
  auto reservespace_cuda = GetScratchBuffer<void>(reservespace_bytes, ctx->GetComputeStream());

  CUDNN_RETURN_IF_ERROR(cudnnRNNForward(GetCudnnHandle(ctx),
                                        rnn_desc,
                                        CUDNN_FWD_MODE_INFERENCE,
                                        sequence_lens_buffer.GpuPtr(),  // should be zero starting with cudnn 8.9.1
                                        x_desc1,
                                        x_data_input,
                                        y_desc1,
                                        y_data,  // output
                                        hx_desc,
                                        hx_data,   // input
                                        y_h_data,  // output
                                        cx_desc, cx_data, y_c_data,
                                        weight_cached_ ? w_data_cache_size_in_bytes_ : w_data_size_in_bytes,
                                        weight_cached_ ? w_data_cache_.get() : w_data.get(),
                                        workspace_bytes,
                                        workspace_cuda.get(),
                                        reservespace_bytes,
                                        reservespace_cuda.get()));

  // Early terminate for this case since Y data is not required, and Y_h is obtained correctly,
  // no need the following code to retrieve Y_h from Y data.
  if (nullptr == Y) {
    // Mask on output for 0 sequence batches
    if (zero_seq_count > 0) {
      // Mask on output for 0 sequence batches
      SetZeroSequences(zero_seq_count, zero_seq_index_cache, y_data, y_h_data, y_c_data, ctx->GetComputeStream());
    }
    return Status::OK();
  }

  IAllocatorUniquePtr<T> y_reorganized_data;
  if (reverse_ || num_directions_ == 2) {
    // reverse output
    y_reorganized_data = GetScratchBuffer<T>(output_size, ctx->GetComputeStream());
    if (reverse_) {
      // reverse output data
      ReverseBySequence(Stream(ctx),
                        gsl::narrow_cast<int32_t>(seq_length),
                        sequence_lens_buffer.GpuPtr(),
                        gsl::narrow_cast<int32_t>(batch_size),
                        gsl::narrow_cast<int32_t>(hidden_size_),
                        reinterpret_cast<CudaT*>(y_data),
                        reinterpret_cast<CudaT*>(y_reorganized_data.get()),
                        output_size);
    } else {
      ReorderBidirectionalDataInSequence(Stream(ctx),
                                         gsl::narrow_cast<int32_t>(seq_length),
                                         gsl::narrow_cast<int32_t>(batch_size),
                                         gsl::narrow_cast<int32_t>(hidden_size_),
                                         reinterpret_cast<CudaT*>(y_data),
                                         reinterpret_cast<CudaT*>(y_reorganized_data.get()),
                                         output_size);
    }

    if (Y != nullptr) {
      // User specified this optional output, so need to copy the reversed data to original place
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(y_data, y_reorganized_data.get(), output_size * sizeof(T),
                                           cudaMemcpyDeviceToDevice, Stream(ctx)));
    } else {
      y_data = y_reorganized_data.get();
    }
  }

  // Mask on output for 0 sequence batches
  if (zero_seq_count > 0) {
    SetZeroSequences(zero_seq_count, zero_seq_index_cache, y_data, y_h_data, y_c_data, ctx->GetComputeStream());
  }

  return Status::OK();
}

template <typename T>
void CudnnRnnBase<T>::SetZeroSequences(const int64_t zero_seq_index_cache_size,
                                       const std::vector<int32_t> zero_seq_index_cache,
                                       T* y_data,
                                       T* y_h_data,
                                       T* y_c_data,
                                       onnxruntime::Stream* ort_stream) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaAsyncBuffer<int32_t> zero_seq_index_cache_async_buffer(this, zero_seq_index_cache_size);
  memcpy(zero_seq_index_cache_async_buffer.CpuPtr(), zero_seq_index_cache.data(),
         zero_seq_index_cache_size * sizeof(int32_t));
  ORT_THROW_IF_ERROR(zero_seq_index_cache_async_buffer.CopyToGpu(ort_stream));
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  MaskZeroSequences(cuda_stream,
                    gsl::narrow_cast<int32_t>(hidden_size_),
                    reinterpret_cast<CudaT*>(y_data),
                    reinterpret_cast<CudaT*>(y_h_data),
                    reinterpret_cast<CudaT*>(y_c_data),
                    zero_seq_index_cache_async_buffer.GpuPtr(),
                    static_cast<int64_t>(zero_seq_index_cache_size));
}

template class CudnnRnnBase<float>;
template class CudnnRnnBase<double>;
template class CudnnRnnBase<MLFloat16>;

}  // namespace cuda
}  // namespace onnxruntime
