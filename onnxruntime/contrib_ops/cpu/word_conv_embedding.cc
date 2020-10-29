// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "word_conv_embedding.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

void WordConvEmbedding::CharEmbeddingLookup(
    const int* seq_ptr,
    const float* char_embedding_weight_p,
    size_t seq_len,
    size_t word_len,
    size_t char_embedding_size,
    size_t filter_width,
    const int* words_len_ptr,
    float* dst) const {
  for (size_t word_inx = 0; word_inx < seq_len; word_inx++) {
    if (words_len_ptr[word_inx] > 0) {
      const int* cur_seq_ptr = seq_ptr + word_inx * word_len;
      float* cur_dst_ptr = dst + word_inx * word_len * char_embedding_size;
      size_t char_length_to_lookup = std::max<size_t>(words_len_ptr[word_inx], filter_width);
      for (size_t char_inx = 0; char_inx < char_length_to_lookup; char_inx++) {
        memcpy(cur_dst_ptr, char_embedding_weight_p + (*cur_seq_ptr) * char_embedding_size, sizeof(float) * char_embedding_size);
        cur_dst_ptr += char_embedding_size;
        cur_seq_ptr++;
      }
    }
  }
}

//input : [sequence_length, word_length, char_embedding_size]
void WordConvEmbedding::ComputeConvMaxPoolWithActivation(
    AllocatorPtr allocator,
    const float* input,
    const float* weights,
    const float* bias,
    const int* words_len_ptr,
    int64_t seq_len,
    int64_t word_len,
    int64_t char_embedding_size,
    int64_t filter_width,
    int64_t num_filters,
    float* output, concurrency::ThreadPool* tp) const {
  int64_t input_word_size = word_len * char_embedding_size;
  int64_t unfolded_width = word_len - filter_width + 1;
  int64_t unfolded_kernal_size = filter_width * char_embedding_size;
  int64_t unfolded_segment_size = unfolded_width * unfolded_kernal_size;
  int64_t conv_res_segment_size = unfolded_width * num_filters;
  int64_t memcpy_size = unfolded_kernal_size * sizeof(float);

  auto unfolded_buffer_p = IAllocator::MakeUniquePtr<float>(allocator, seq_len * unfolded_segment_size);
  auto conv_result_p = IAllocator::MakeUniquePtr<float>(allocator, seq_len * conv_res_segment_size);
  auto conv_activation_result_p = IAllocator::MakeUniquePtr<float>(allocator, seq_len * conv_res_segment_size);

  int64_t word_inx = 0;
  while (word_inx < seq_len) {
    if (words_len_ptr[word_inx] <= 0) {
      word_inx++;
      continue;
    }

    float* words_unfolded_buffer_p = unfolded_buffer_p.get();
    int64_t words_unfolded_width = 0;
    int64_t tmp_word_inx = word_inx;
    float* conv_buf_p = conv_result_p.get();
    float* pactivationbuf = conv_activation_result_p.get();

    // unfolding buffer
    while (tmp_word_inx < seq_len && words_len_ptr[tmp_word_inx] > 0) {
      const float* current_word_input = input + tmp_word_inx * input_word_size;
      int64_t word_unfolded_width = std::max<int64_t>(words_len_ptr[tmp_word_inx], filter_width) - filter_width + 1;
      words_unfolded_width += word_unfolded_width;
      for (int64_t unfolded_inx = 0; unfolded_inx < word_unfolded_width; unfolded_inx++) {
        memcpy(words_unfolded_buffer_p, current_word_input, memcpy_size);
        current_word_input += char_embedding_size;
        words_unfolded_buffer_p += unfolded_kernal_size;
      }
      tmp_word_inx++;
    }

    math::GemmEx<float>(
        CblasNoTrans, CblasTrans,
        static_cast<int>(words_unfolded_width), static_cast<int>(num_filters), static_cast<int>(unfolded_kernal_size), 1.0f,
        unfolded_buffer_p.get(), static_cast<int>(unfolded_kernal_size),
        weights, static_cast<int>(unfolded_kernal_size), 0.0f,
        conv_buf_p, static_cast<int>(num_filters), tp);

    for (int64_t unfolded_inx = 0; unfolded_inx < words_unfolded_width; unfolded_inx++)
      for (int64_t filter_inx = 0; filter_inx < num_filters; filter_inx++) {
        conv_buf_p[unfolded_inx * num_filters + filter_inx] += bias[filter_inx];
      }
    MlasComputeTanh(conv_buf_p, pactivationbuf, words_unfolded_width * num_filters);

    float* activationbuf_cur_ptr = pactivationbuf;
    for (int64_t pool_word_inx = word_inx; pool_word_inx < tmp_word_inx; pool_word_inx++) {
      float* result_ptr = output + pool_word_inx * num_filters;
      for (int64_t filter_inx = 0; filter_inx < num_filters; filter_inx++) {
        result_ptr[filter_inx] = -1.0f * 1e12f;
      }

      int64_t word_unfolded_width = std::max<int64_t>(words_len_ptr[pool_word_inx], filter_width) - filter_width + 1;
      for (int64_t unfolded_inx = 0; unfolded_inx < word_unfolded_width; unfolded_inx++) {
        for (int64_t filter_inx = 0; filter_inx < num_filters; filter_inx++) {
          result_ptr[filter_inx] = std::max(activationbuf_cur_ptr[filter_inx], result_ptr[filter_inx]);
        }
        activationbuf_cur_ptr += num_filters;
      }
    }
    word_inx = tmp_word_inx;
  }
}
void WordConvEmbedding::CalculateLengthOfEachWordInSequence(
    const int* seq_ptr,
    int* words_len_ptr,
    size_t seq_len,
    size_t word_len) const {
  for (size_t seq_inx = 0; seq_inx < seq_len; seq_inx++) {
    size_t w_off = seq_inx * word_len;
    int word_length = 0;
    if (seq_ptr[w_off] > 0) {
      for (size_t char_inx = 0; char_inx < word_len; char_inx++) {
        if (seq_ptr[w_off + char_inx] > 0) word_length++;
      }
    }
    words_len_ptr[seq_inx] = word_length;
  }
}

Status WordConvEmbedding::ValidateInputShape(const TensorShape& w_conv_shape, const TensorShape& w_char_embedding_shape) const {
  if (embedding_size_ != -1 && w_conv_shape[0] != embedding_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Conv filter size does not match embedding_size attribute.",
                           " embedding_size attribute: ", embedding_size_,
                           " conv filter size: ", w_conv_shape[0]);
  }

  if (conv_window_size_ != -1 && w_conv_shape[2] != conv_window_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Conv kernal size 1 does not match conv_window_size attribute .",
                           " conv_window_size attribute: ", conv_window_size_,
                           " conv kernal size 1: ", w_conv_shape[2]);
  }

  if (char_embedding_size_ != -1 && w_char_embedding_shape[1] != char_embedding_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Char embedding size does not match char_embedding_size attribute.",
                           " char_embedding_size attribute: ", conv_window_size_,
                           " Char embedding size: ", w_conv_shape[1]);
  }

  if (w_char_embedding_shape[1] != w_conv_shape[3]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Char embedding size does not match conv kernal size 2.",
                           " Char embedding size: ", conv_window_size_,
                           " Conv kernal size 2 : ", w_conv_shape[3]);
  }

  return Status::OK();
}

Status WordConvEmbedding::Compute(OpKernelContext* ctx) const {
  // original lstm processing
  const Tensor& sequence = *(ctx->Input<Tensor>(0));          // sequence: [sequence_length, word_length]
  const Tensor& w_conv = *(ctx->Input<Tensor>(1));            // conv weight: [M, C/group, kH, kW]
  const Tensor& b_conv = *(ctx->Input<Tensor>(2));            // conv bias: [M]
  const Tensor& w_char_embedding = *(ctx->Input<Tensor>(3));  // conv weights. [index, char_embedding_size]

  const TensorShape& sequence_shape = sequence.Shape();
  const TensorShape& w_conv_shape = w_conv.Shape();
  const TensorShape& w_char_embedding_shape = w_char_embedding.Shape();

  ORT_RETURN_IF_ERROR(ValidateInputShape(w_conv_shape, w_char_embedding_shape));

  int64_t seq_len = sequence_shape[0];
  int64_t word_len = sequence_shape[1];
  int64_t char_embedding_size = w_char_embedding_shape[1];
  int64_t filter_size = w_conv_shape[0];
  int64_t filter_width = w_conv_shape[2];

  TensorShape Y_dims{seq_len, filter_size};
  Tensor* Y = ctx->Output(/*index*/ 0, Y_dims);

  const int* seq_ptr = sequence.Data<int>();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  // allocate memory for char look up
  // seq_len * word_len * char_embedding_size
  size_t chars_embeddings_size = seq_len * word_len * char_embedding_size;
  auto chars_embeddings_ptr = IAllocator::MakeUniquePtr<float>(alloc, chars_embeddings_size);
  auto words_length_ptr = IAllocator::MakeUniquePtr<int>(alloc, seq_len);
  std::memset(chars_embeddings_ptr.get(), 0, chars_embeddings_size * sizeof(float));
  std::memset(words_length_ptr.get(), 0, seq_len * sizeof(int));

  CalculateLengthOfEachWordInSequence(seq_ptr, words_length_ptr.get(), seq_len, word_len);

  CharEmbeddingLookup(seq_ptr,
                      w_char_embedding.Data<float>(),
                      seq_len,
                      word_len,
                      char_embedding_size,
                      filter_width,
                      words_length_ptr.get(),
                      chars_embeddings_ptr.get());

  ComputeConvMaxPoolWithActivation(
      alloc,
      chars_embeddings_ptr.get(),
      w_conv.Data<float>(),
      b_conv.Data<float>(),
      words_length_ptr.get(),
      seq_len,
      word_len,
      char_embedding_size,
      filter_width,
      filter_size,
      Y->MutableData<float>(),
      ctx->GetOperatorThreadPool());

  return Status::OK();
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    WordConvEmbedding,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    WordConvEmbedding);

}  // namespace contrib
}  // namespace onnxruntime
