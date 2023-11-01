// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/topk_impl.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/framework/ort_value.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include <cuda_runtime.h>
#include "contrib_ops/cuda/transformers/generation_cuda_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cuda/transformers/beam_search_topk.h"
#include "contrib_ops/cuda/transformers/greedy_search_top_one.h"
#include "core/providers/cuda/tensor/transpose.h"

// the includes would be dummy for ROCm, we will ignore them for now
#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif

#include "sampling_cuda_helper.h"

#ifdef DEBUG_GENERATION
#include <iostream>
#endif

using onnxruntime::cuda::TArray;
using onnxruntime::cuda::ToCudaType;
using onnxruntime::cuda::TopKImpl;

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

#include "generation_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace GenerationCudaDeviceHelper {

// This function assumes the Attention type is same as graph's past state type.
// e.g In the case of past(fp32) -> cast to fp16 -> Attention(fp16), the reorder
// function will use the fp32 chunk size and cause the model silently generates
// the incorrect results.
// TODO: Fix this issue. Either retrive the Attention op type from the graph or
// check the type of past state as graph input should be same as Attention op type.
// It might be better to forcefully require the same type since cast node generates
// extra overhead.
Status ReorderPastState(
    const void*,
    Tensor& past_state,
    Tensor& past_state_staging,
    Stream* stream) {
  ORT_ENFORCE(stream);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream->GetHandle());

  const auto& past_state_shape = past_state.Shape();

  const auto& past_state_dims = past_state_shape.GetDims();
  const bool packed_past = past_state_dims.size() == 5;

  size_t batch_size = packed_past ? past_state_dims[1] : past_state_dims[0];
  size_t num_heads = packed_past ? past_state_dims[2] : past_state_dims[1];
  size_t max_length = packed_past ? past_state_dims[3] : past_state_dims[2];
  size_t head_size = packed_past ? past_state_dims[4] : past_state_dims[3];

  // Copy the 'K' values into the temp staging buffer
  size_t past_state_size = packed_past ? past_state.SizeInBytes() / 2 : past_state.SizeInBytes();
  void* past_state_staging_buffer = past_state_staging.MutableDataRaw();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_state_staging_buffer, past_state.DataRaw(), past_state_size,
                                       cudaMemcpyDeviceToDevice, cuda_stream));

  // Now consider the original 'K' values to be of shape [B, N, max_length, head_size / x, x] and transpose it into
  // [B, N, head_size / x, max_length, x], where x = 16 / sizeof(T)
  int64_t chunk_size = static_cast<int64_t>(16 / past_state.DataType()->Size());

  cuda::ReorderPastStatesKernelLauncher(past_state.MutableDataRaw(),
                                        past_state_staging_buffer,
                                        static_cast<int>(batch_size),
                                        static_cast<int>(num_heads),
                                        static_cast<int>(max_length),
                                        static_cast<int>(head_size),
                                        static_cast<int>(chunk_size),
                                        cuda_stream);

  return Status::OK();
}

Status InitCacheIndir(Tensor& cache_indir, Stream* stream) {
  ORT_ENFORCE(stream);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream->GetHandle());

  // Initialize the cache_indir tensor to all 0s
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(cache_indir.MutableDataRaw(), 0, cache_indir.SizeInBytes(), cuda_stream));

  return Status::OK();
}

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            Stream* stream,
            onnxruntime::concurrency::ThreadPool* /*threadpool*/,
            Tensor& output_values,
            Tensor& output_indices) {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator topkRange("TopK", profile::Color::Green);
  topkRange.Begin();
#endif

  ORT_ENFORCE(nullptr != input);
  int32_t rank = static_cast<int32_t>(input->Shape().NumDimensions());

  ORT_ENFORCE(axis >= 0 && axis < rank);
  ORT_ENFORCE(k > 0 && k <= input->Shape().GetDims()[axis]);

  auto input_shape = input->Shape();
  auto output_shape = input_shape;
  output_shape[axis] = k;

  TArray<int64_t> elem_nums_cuda(input->Shape().GetDims());
  for (int32_t i = elem_nums_cuda.Size() - 2; i >= 0; --i) {
    elem_nums_cuda[i] *= elem_nums_cuda[i + 1];
  }

  int64_t dimension = input_shape[axis];
  int64_t N = elem_nums_cuda[0] / dimension;

  output_values = std::move(*Tensor::Create(input->DataType(), output_shape, allocator));
  output_indices = std::move(*Tensor::Create(DataTypeImpl::GetType<int64_t>(), output_shape, std::move(allocator)));

  Status result;
  if (input->IsDataType<float>()) {
    result = TopKImpl<float>(nullptr,  // We limit number of beams in BeamSearchParameters, so K <= 256 and use NULL here
                             false /*use_deterministic_compute*/,
                             stream,
                             input->Data<float>(),
                             static_cast<float*>(output_values.MutableDataRaw()),
                             static_cast<int64_t*>(output_indices.MutableDataRaw()),
                             elem_nums_cuda,
                             static_cast<size_t>(elem_nums_cuda.Size()),
                             static_cast<int32_t>(axis),
                             static_cast<int64_t>(k),
                             static_cast<int64_t>(largest),
                             static_cast<int64_t>(sorted),
                             N,
                             dimension);
  } else if (input->IsDataType<MLFloat16>()) {
    result = TopKImpl<MLFloat16>(nullptr,
                                 false /*use_deterministic_compute*/,
                                 stream,
                                 input->Data<MLFloat16>(),
                                 static_cast<MLFloat16*>(output_values.MutableDataRaw()),
                                 static_cast<int64_t*>(output_indices.MutableDataRaw()),
                                 elem_nums_cuda,
                                 static_cast<size_t>(elem_nums_cuda.Size()),
                                 static_cast<int32_t>(axis),
                                 static_cast<int64_t>(k),
                                 static_cast<int64_t>(largest),
                                 static_cast<int64_t>(sorted),
                                 N,
                                 dimension);
  } else {
    result = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "BeamSearch op: An implementation for the input type ",
                             input->DataType(), " is not supported yet");
  }

#ifdef ENABLE_NVTX_PROFILE
  topkRange.End();
#endif
  return result;
}

Status AddToFeeds(Stream* ort_stream,
                  std::initializer_list<OrtValue> inputs,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& buffer,
                  AllocatorPtr device_allocator,
                  AllocatorPtr host_allocator,
                  const OrtMemoryInfo& location) {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator addToFeedsRange("AddToFeeds", profile::Color::Blue);
  addToFeedsRange.Begin();
#endif

  // Copy tensors to GPU, then add to feeds
  size_t total_bytes = 0;
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      total_bytes += input.Get<Tensor>().SizeInBytes();
    }
  }

  ORT_ENFORCE(total_bytes > 0);

  cudaStream_t stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  auto pinned_buffer = IAllocator::MakeUniquePtr<void>(host_allocator, total_bytes);
  char* pinned_data = static_cast<char*>(pinned_buffer.get());
  // Copy tensors to one pinned memory buffer (so that we only need copy to GPU once)
  char* destination = pinned_data;
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      const Tensor& tensor = input.Get<Tensor>();
      const size_t bytes = tensor.SizeInBytes();
      MLDataType dataType = tensor.DataType();
      if (dataType == DataTypeImpl::GetType<int32_t>()) {
        memcpy(destination, input.Get<Tensor>().Data<int32_t>(), bytes);
      } else if (dataType == DataTypeImpl::GetType<int64_t>()) {
        memcpy(destination, input.Get<Tensor>().Data<int64_t>(), bytes);
      } else if (dataType == DataTypeImpl::GetType<float>()) {
        memcpy(destination, input.Get<Tensor>().Data<float>(), bytes);
      } else if (dataType == DataTypeImpl::GetType<MLFloat16>()) {
        memcpy(destination, input.Get<Tensor>().Data<MLFloat16>(), bytes);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "AddToFeeds: An implementation for the input type ",
                               dataType, " is not supported yet");
      }
      // Do not need alignment because GPT has int32 inputs (past is empty) and T5 encoder has int64 inputs.
      destination += bytes;
    }
  }
  if (!buffer) {
    buffer = IAllocator::MakeUniquePtr<char>(device_allocator, total_bytes, false, ort_stream, WaitCudaNotificationOnDevice);
  }
  char* gpu_data = buffer.get();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_data, pinned_data, total_bytes, cudaMemcpyHostToDevice, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  onnxruntime::contrib::cuda::AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));
  // TODO(tianleiwu): allocate a buffer for subgraph inputs so that we can reuse the buffer in each subgraph call.
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      const Tensor& tensor = input.Get<Tensor>();
      const TensorShape& shape = tensor.Shape();
      const size_t bytes = tensor.SizeInBytes();
      MLDataType dataType = tensor.DataType();

      OrtValue device_input;
      Tensor::InitOrtValue(dataType, shape, gpu_data, location, device_input);
      gpu_data += bytes;
      feeds.push_back(device_input);
    }
  }

#ifdef ENABLE_NVTX_PROFILE
  addToFeedsRange.End();
#endif

  return Status::OK();
}

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   Stream* ort_stream) {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator initStateRange("InitBeamState", profile::Color::Red);
  initStateRange.Begin();
#endif

  // TODO(tianleiwu): we can use another stream to avoid blocking subgraph execution.
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->next_scores.data(), 0, beam_state->next_scores.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(beam_state->topk_buffer.data(), 0, beam_state->topk_buffer.size_bytes(), cuda_stream));

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  cuda::LaunchInitKernel(beam_state->beam_scores.data(), batch_size, num_beams, cuda_stream);

  // copy sequence lengths to GPU
  // since next_positions is only needed to update feeds after subgraph execution, so it is fine to use Async here.
  if (!beam_state->next_positions.empty()) {  // next_positions is empty for T5
    CUDA_CALL_THROW(cudaMemcpyAsync(beam_state->next_positions.data(), sequence_lengths.data(), sequence_lengths.size_bytes(),
                                    cudaMemcpyHostToDevice, cuda_stream));
  }

#ifdef ENABLE_NVTX_PROFILE
  initStateRange.End();
#endif
}

template <typename T>
void InitGreedyState(transformers::IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     Stream* ort_stream) {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator initStateRange("InitGreedyState", profile::Color::Red);
  initStateRange.Begin();
#endif

  cudaStream_t cuda_stream = ort_stream ? reinterpret_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  CUDA_CALL_THROW(cudaMemsetAsync(greedy_state->next_token_scores.data(), 0, greedy_state->next_token_scores.size_bytes(), cuda_stream));
  CUDA_CALL_THROW(cudaMemsetAsync(greedy_state->next_positions.data(), 0, greedy_state->next_positions.size_bytes(), cuda_stream));

  CUDA_CALL_THROW(cudaMemcpyAsync(greedy_state->next_positions.data(), sequence_lengths.data(), sequence_lengths.size_bytes(),
                                  cudaMemcpyHostToDevice, cuda_stream));

#ifdef ENABLE_NVTX_PROFILE
  initStateRange.End();
#endif
}

template <typename T>
Status ProcessLogits(const OrtValue& logits,                                 // logits output of subgraph
                     transformers::IBeamSearchState<T>* beam_state,          // state
                     transformers::ISequences* sequences,                    // sequences
                     AllocatorPtr& allocator,                                // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
                     transformers::ILogitsProcessorList* logits_processors,  // logits processors
                     transformers::IBeamScorer* beam_scorer,                 // beam scorer
                     const transformers::IGenerationParameters* parameters,  // parameters
                     int step,                                               // iteration counter
                     Stream* ort_stream,                                     // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper) {           // tensor dumper

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator processLogitsRange("ProcessLogits", profile::Color::Red);
  processLogitsRange.Begin();
#endif

  ORT_UNUSED_PARAMETER(logits_processors);
  ORT_UNUSED_PARAMETER(thread_pool);

#ifndef DEBUG_GENERATION
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int num_beams = parameters->num_beams;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size * num_beams;

  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* logits_data = reinterpret_cast<const CudaT*>(logits.Get<Tensor>().Data<T>());

  // Logits has shape (batch_size * num_beams, input_length, padded_vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];
  auto logits_batch_size = logits_shape[0];

  // NOTE: `padded_vocab_size` MAY be different from `vocab_size`.
  // But the following implementation should work correctly if they are the same
  // or different.
  auto padded_vocab_size = static_cast<int>(logits_shape[2]);

  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_logits = beam_state->next_token_logits;
  // TODO(tianleiwu): use one kernel to replace a loop of memory copy.
  if (input_length > 1 || logits_batch_size == batch_size) {
    // Move the pointer in increments of padded_vocab_size to account for any padding
    // if any in the logits weight of the MatMul.
    const CudaT* current_logits = logits_data + (input_length - 1) * padded_vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      // We only copy what is relevant (i.e.) vocab_size as padded_vocab_size will contain
      // some logits corresponding to the "padded" vocab size which we will ignore
      // for token generation.
      gsl::span<const T> source(reinterpret_cast<const T*>(current_logits), vocab_size);
      gsl::span<T> target = next_token_logits.subspan(static_cast<size_t>(i) * vocab_size, vocab_size);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), sizeof(T) * vocab_size,
                                           cudaMemcpyDeviceToDevice, cuda_stream));
      if (logits_batch_size == batch_beam_size) {
        current_logits += input_length * padded_vocab_size;
      } else if (logits_batch_size == batch_size && i % num_beams == num_beams - 1) {
        current_logits += input_length * padded_vocab_size;
      }
    }
  }

#ifdef DEBUG_GENERATION
  dumper->Print("logits", logits);
  if (input_length > 1 || logits_batch_size == batch_size) {
    dumper->Print("next_token_logits", next_token_logits.data(), batch_size, num_beams, vocab_size);
  }
#endif

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  gsl::span<float>& next_token_scores = beam_state->next_token_scores;

  // The output will be float for consideration of precision and easy integration with remaining parts.
  float* Y_data = next_token_scores.data();
  bool is_reuse_logits_buffer = (input_length == 1 && logits_batch_size == batch_beam_size);

  const CudaT* X_data = is_reuse_logits_buffer ? logits_data : reinterpret_cast<const CudaT*>(next_token_logits.data());

  ORT_RETURN_IF_ERROR((dispatch_blockwise_softmax_forward<CudaT, float, float, true>(
      ort_stream, Y_data, X_data, vocab_size,
      is_reuse_logits_buffer ? padded_vocab_size : vocab_size,
      vocab_size,
      batch_size * num_beams)));

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  cuda::LaunchLogitsProcessKernel<float>(
      next_token_scores.data(),
      parameters->vocab_mask.data(),
      step > 1 ? nullptr : parameters->prefix_vocab_mask.data(),  // prefix vocab mask is applied to first step only.
      nullptr,                                                    // parameters->presence_mask.data(),
      parameters->presence_penalty,
      parameters->temperature,
      parameters->batch_size,
      parameters->num_beams,
      vocab_size,
      vocab_size,
      (parameters->min_length > 0 && sequences->GetSequenceLength() < parameters->min_length) ? parameters->eos_token_id : -1,
      sequences->GetCurrentDeviceSequences().data(),
      parameters->max_length,
      sequences->GetSequenceLength(),
      parameters->repetition_penalty,
      parameters->no_repeat_ngram_size,
      cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after logits process", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif
  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  cuda::LaunchAddProbsKernel(next_token_scores.data(), beam_state->beam_scores.data(),
                             batch_size, num_beams, vocab_size, cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  if (output_scores) {
    // Append next token scores to the scores output.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(beam_state->remaining_scores.data(),
                                         next_token_scores.data(),
                                         next_token_scores.size_bytes(),
                                         cudaMemcpyDeviceToDevice,
                                         cuda_stream));
    beam_state->remaining_scores = beam_state->remaining_scores.subspan(next_token_scores.size());
  }

  if (num_beams <= 32) {
    constexpr size_t max_parts_of_vocab = 128;
    size_t candidate_count = SafeInt<size_t>(batch_beam_size) * 2 * num_beams;
    float* topk_tmp_buffer = beam_state->topk_buffer.data();
    float* topk_scores_1st_stage = topk_tmp_buffer;
    int32_t* topk_tokens_1st_stage = reinterpret_cast<int32_t*>(topk_scores_1st_stage + candidate_count * max_parts_of_vocab);
    float* topk_scores_2nd_stage = reinterpret_cast<float*>(topk_tokens_1st_stage + candidate_count * max_parts_of_vocab);
    int32_t* topk_tokens_2nd_stage = reinterpret_cast<int32_t*>(topk_scores_2nd_stage + candidate_count);

    cuda::BeamSearchTopK(next_token_scores.data(),
                         batch_size,
                         num_beams,
                         vocab_size,
                         2 * num_beams,
                         topk_scores_1st_stage,
                         topk_tokens_1st_stage,
                         topk_scores_2nd_stage,
                         topk_tokens_2nd_stage,
                         beam_state->next_scores.data(),
                         beam_state->next_tokens.data(),
                         beam_state->next_indices.data(),
                         cuda_stream);

    // Select [batch_size, 2 * num_beams] from [batch_size * num_beams, 2 * num_beams]
#ifdef DEBUG_GENERATION
    dumper->Print("next_tokens before scorer", beam_state->next_tokens.data(), batch_size, 2 * num_beams);
    dumper->Print("next_indices before scorer", beam_state->next_indices.data(), batch_size, 2 * num_beams);
    dumper->Print("next_scores before scorer", beam_state->next_scores.data(), batch_size, 2 * num_beams);
#endif
  } else {
    // Apply top-k selection like the following:
    //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
    //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
    int64_t next_token_scores_dims[] = {batch_size, static_cast<int64_t>(num_beams) * static_cast<int64_t>(vocab_size)};

    TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
    auto element_type = DataTypeImpl::GetType<float>();
    OrtValue next_token_scores_value;
    Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(),
                         next_token_scores_value);
    const Tensor& input = next_token_scores_value.Get<Tensor>();

    constexpr int axis = 1;
    const unsigned top_k = static_cast<unsigned>(2 * num_beams);
    constexpr bool largest = true;
    constexpr bool sorted = true;  // results returned in sorted order.

    std::unique_ptr<Tensor> topk_scores = Tensor::CreateDefault();
    std::unique_ptr<Tensor> topk_indices = Tensor::CreateDefault();
    ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, ort_stream, thread_pool,
                             *topk_scores, *topk_indices));

#ifdef DEBUG_GENERATION
    dumper->Print("topk_scores", *(topk_scores.get()));
    dumper->Print("topk_indices", *(topk_indices.get()));
#endif

    // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
    //   next_indices = (next_tokens / vocab_size).long()
    //   next_tokens = next_tokens % vocab_size
    const int64_t* next_token_indices = topk_indices->Data<int64_t>();
    cuda::LaunchNextTokenKernel(next_token_indices, beam_state->next_indices.data(), beam_state->next_tokens.data(),
                                batch_size, top_k, vocab_size, cuda_stream);

#ifdef DEBUG_GENERATION
    dumper->Print("next_scores before scorer", topk_scores->Data<float>(), batch_size, top_k);
    dumper->Print("next_tokens before scorer", beam_state->next_tokens.data(), batch_size, top_k);
    dumper->Print("next_indices before scorer", beam_state->next_indices.data(), batch_size, top_k);
#endif
  }

  // gsl::span doesn't convert from non const to const, so all we're doing here is making each const.
  gsl::span<const float> next_scores(beam_state->next_scores.data(), beam_state->next_scores.size());
  gsl::span<const int32_t> next_tokens(beam_state->next_tokens.data(), beam_state->next_tokens.size());
  gsl::span<const int32_t> next_indices(beam_state->next_indices.data(), beam_state->next_indices.size());

  beam_scorer->Process(
      *sequences,
      next_scores,
      next_tokens,
      next_indices);

#ifdef ENABLE_NVTX_PROFILE
  processLogitsRange.End();
#endif

  return Status::OK();
}

struct CudaBeamSearchScorer : transformers::IBeamScorer {
  CudaBeamSearchScorer(const transformers::IGenerationParameters& parameters,
                       AllocatorPtr& allocator, AllocatorPtr& allocator_cpu,
                       Stream* stream);

  void Process(transformers::ISequences& sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Finalize(transformers::ISequences& sequences,
                gsl::span<const float>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  bool IsDone() const override { return false; }  // For CUDA we speculatively run the next step while we wait for the GPU to report status. We use 'IsDoneLater()' for this
  bool IsDoneLater() const override;

  gsl::span<float> GetNextScores() override { return next_beam_scores_; }
  gsl::span<int32_t> GetNextTokens() override { return next_beam_tokens_; }
  gsl::span<int32_t> GetNextIndicesCPU() override {
    CUDA_CALL_THROW(cudaMemcpyAsync(next_beam_indices_cpu_.data(), next_beam_indices_.data(), next_beam_indices_.size_bytes(), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    return next_beam_indices_cpu_;
  }
  gsl::span<int32_t> GetNextIndicesGPU() override { return next_beam_indices_; }

 private:
  mutable cuda::AutoDestoryCudaEvent event_process_complete_;
  IAllocatorUniquePtr<cuda::BeamScorerState> state_cpu_;
  IAllocatorUniquePtr<cuda::BeamScorerState> state_gpu_;
  cudaStream_t stream_;

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_cpu_ptr_;
  gsl::span<int32_t> next_beam_indices_cpu_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_used_{};                     // Offset of available buffer, or length of used buffer.

  IAllocatorUniquePtr<cuda::HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  IAllocatorUniquePtr<cuda::BeamHypotheses> beam_hyps_ptr_;
  gsl::span<cuda::BeamHypotheses> beam_hyps_;  // Shape is batch_size_
};

template <typename TAlloc>
gsl::span<TAlloc> Allocate(std::shared_ptr<IAllocator> allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr) {
  unique_ptr = IAllocator::MakeUniquePtr<TAlloc>(std::move(allocator), size);
  return gsl::make_span(unique_ptr.get(), size);
}

template <typename T>
IAllocatorUniquePtr<T> AllocateCPUPinned() {
  T* p;
  CUDA_CALL_THROW(cudaMallocHost(&p, sizeof(T)));
  return IAllocatorUniquePtr<T>{p, [](cuda::BeamScorerState* p) { ORT_IGNORE_RETURN_VALUE(cudaFreeHost(p)); }};
}

CudaBeamSearchScorer::CudaBeamSearchScorer(const transformers::IGenerationParameters& parameters,
                                           AllocatorPtr& allocator, AllocatorPtr& allocator_cpu, Stream* stream)
    : stream_{stream ? reinterpret_cast<cudaStream_t>(stream->GetHandle()) : nullptr} {
  CUDA_CALL_THROW(cudaEventCreate(&event_process_complete_.Get()));

  state_cpu_ = AllocateCPUPinned<cuda::BeamScorerState>();
  state_cpu_->batch_size_ = static_cast<size_t>(parameters.batch_size);
  state_cpu_->num_beams_ = static_cast<size_t>(parameters.num_beams);
  state_cpu_->max_length_ = static_cast<size_t>(parameters.max_length);
  state_cpu_->num_return_sequences_ = static_cast<size_t>(parameters.num_return_sequences);
  state_cpu_->pad_token_id_ = parameters.pad_token_id;
  state_cpu_->eos_token_id_ = parameters.eos_token_id;
  state_cpu_->early_stopping_ = parameters.early_stopping;
  state_cpu_->not_done_count_ = parameters.batch_size;
  state_cpu_->hypothesis_buffer_used_ = 0;
  state_gpu_ = IAllocator::MakeUniquePtr<cuda::BeamScorerState>(allocator, 1);
  CUDA_CALL_THROW(cudaMemcpyAsync(state_gpu_.get(), state_cpu_.get(), sizeof(cuda::BeamScorerState), ::cudaMemcpyHostToDevice, stream_));

  size_t batch_beam_size = state_cpu_->batch_size_ * state_cpu_->num_beams_;

  auto beams = Allocate<cuda::HypothesisScore>(allocator, batch_beam_size, hypothesis_scores_ptr_);
  beam_hyps_ = Allocate<cuda::BeamHypotheses>(allocator, state_cpu_->batch_size_, beam_hyps_ptr_);

  cuda::LaunchInitializeBeamHypotheses(beam_hyps_, parameters.length_penalty, beams, parameters.num_beams, stream_);

  next_beam_scores_ = Allocate<float>(allocator, batch_beam_size, next_beam_scores_ptr_);
  next_beam_tokens_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_tokens_ptr_);
  next_beam_indices_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_indices_ptr_);
  next_beam_indices_cpu_ = Allocate<int32_t>(allocator_cpu, batch_beam_size, next_beam_indices_cpu_ptr_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t per_beam = (SafeInt<size_t>(state_cpu_->max_length_) * (state_cpu_->max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ = Allocate<int32_t>(allocator, batch_beam_size * per_beam, hypothesis_buffer_ptr_);
}

void CudaBeamSearchScorer::Process(transformers::ISequences& sequences,
                                   gsl::span<const float>& next_scores,
                                   gsl::span<const int32_t>& next_tokens,
                                   gsl::span<const int32_t>& next_indices) {
  cuda::LaunchBeamSearchScorer_Process(*state_cpu_,
                                       *state_gpu_,
                                       sequences.GetCurrentDeviceSequences(),
                                       sequences.GetSequenceLength(),
                                       beam_hyps_,
                                       next_beam_scores_,
                                       next_beam_tokens_,
                                       next_beam_indices_,
                                       hypothesis_buffer_,
                                       next_scores,
                                       next_tokens,
                                       next_indices,
                                       stream_);
  CUDA_CALL_THROW(cudaEventRecord(event_process_complete_.Get(), stream_));

  cuda::LaunchBeamSearchScorer_AppendNextTokenToSequences(*state_cpu_,
                                                          *state_gpu_,
                                                          sequences.GetCurrentDeviceSequences(),
                                                          sequences.GetNextDeviceSequences(),
                                                          sequences.GetSequenceLength(),
                                                          next_beam_tokens_,
                                                          next_beam_indices_,
                                                          stream_);
}

bool CudaBeamSearchScorer::IsDoneLater() const {
  CUDA_CALL_THROW(cudaEventSynchronize(event_process_complete_.Get()));
  return state_cpu_->not_done_count_ == 0;
}

void CudaBeamSearchScorer::Finalize(transformers::ISequences& sequences,
                                    gsl::span<const float>& final_beam_scores,
                                    Tensor* output_sequences,
                                    Tensor* output_sequence_scores) {
  ORT_ENFORCE(output_sequences != nullptr);

  // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length).
  gsl::span<int32_t> output{output_sequences->MutableData<int32_t>(), static_cast<size_t>(output_sequences->Shape().Size())};

  // Score of each sequence, with shape (batch_size * num_return_sequences).
  gsl::span<float> sequence_scores;
  if (output_sequence_scores) {
    sequence_scores = gsl::span<float>{output_sequence_scores->MutableData<float>(), static_cast<size_t>(output_sequence_scores->Shape().Size())};
  }

  cuda::LaunchBeamSearchScorer_Finalize(state_cpu_->batch_size_, *state_gpu_, sequences.GetCurrentDeviceSequences(), sequences.GetSequenceLength(), beam_hyps_, final_beam_scores, output, sequence_scores, stream_);
}

std::unique_ptr<transformers::IBeamScorer> CreateBeamScorer(const transformers::IGenerationParameters& parameters,
                                                            AllocatorPtr& allocator, AllocatorPtr& allocator_cpu, Stream* stream) {
  return std::make_unique<CudaBeamSearchScorer>(parameters, allocator, allocator_cpu, stream);
}

template <typename T>
Status GreedySearchProcessLogits(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IGreedySearchState<T>* greedy_state,      // state
    transformers::ISamplingState<T>* sampling_state,        // buffers
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    const transformers::IGenerationParameters* parameters,  // parameters
    bool do_sampling,                                       // whether to do sampling
    int step,                                               // iteration counter
    Stream* stream,                                         // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper) {           // tensor dumper

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator processLogitsRange("ProcessLogits", profile::Color::Red);
  processLogitsRange.Begin();
#endif

  ORT_UNUSED_PARAMETER(logits_processors);
  ORT_UNUSED_PARAMETER(thread_pool);
#ifndef DEBUG_GENERATION
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size;

  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* logits_data = reinterpret_cast<const CudaT*>(logits.Get<Tensor>().Data<T>());

  // Logits has shape (batch_size, input_length, padded_vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  // NOTE: `padded_vocab_size` MAY be different from `vocab_size`.
  // But the following implementation should work correctly if they are the same
  // or different.
  auto padded_vocab_size = static_cast<int>(logits_shape[2]);

  cudaStream_t cuda_stream = stream ? reinterpret_cast<cudaStream_t>(stream->GetHandle()) : nullptr;

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // In greedy search, next_token_scores is next_token_logits.
  gsl::span<T>& next_token_scores = greedy_state->next_token_scores;

  // TODO(hasesh/wy): Support re-using logits buffer for the sampling case.
  // Currently, we cannot re-use the logits because the sampling logic expects
  // `next_token_scores` to be populated.
  auto is_reuse_logits_buffer = !do_sampling && (input_length == 1);

  // Copy over the logits data into the staging buffer, only if
  // we do not plan to re-use the logits buffer directly
  if (!is_reuse_logits_buffer) {
    // TODO(tianleiwu): use one kernel to replace a loop of memory copy.

    // Move the pointer in increments of padded_vocab_size to account for any padding
    // if any in the logits weight of the MatMul.
    const CudaT* current_logits = logits_data + (input_length - 1) * padded_vocab_size;
    for (ptrdiff_t i = 0; i < batch_beam_size; i++) {
      // We only copy what is relevant (i.e.) vocab_size as padded_vocab_size will contain
      // some logits corresponding to the "padded" vocab size which we will ignore
      // for token generation.
      gsl::span<const T> source(reinterpret_cast<const T*>(current_logits), vocab_size);
      gsl::span<T> target = next_token_scores.subspan(i * vocab_size, vocab_size);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), sizeof(T) * vocab_size,
                                           cudaMemcpyDeviceToDevice, cuda_stream));
      current_logits += input_length * padded_vocab_size;
    }
  }

#ifdef DEBUG_GENERATION
  dumper->Print("logits", logits);
  if (is_reuse_logits_buffer) {
    // TODO: Handle padded logits in the logits buffer before printing its contents
    ORT_THROW("Dumping contents of logits buffer is not implemented yet");
  } else {
    dumper->Print("next_token_scores", next_token_scores.data(), batch_size, vocab_size);
  }
#endif

  // Sequences generated by beam scorer is currently stored in CPU.
  // Copy sequences to device only when repetition penalty or no repeat ngram is used in kernel
  BufferUniquePtr sequences_buffer;
  int current_sequence_length = sequences->GetSequenceLength();
  if (parameters->repetition_penalty != 1.0f) {
    size_t bytes = SafeInt<size_t>(sizeof(int32_t)) * batch_beam_size * parameters->max_length;
    void* data = allocator->Alloc(bytes);
    BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
    sequences_buffer = std::move(temp_buffer);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(sequences_buffer.get(), sequences->GetSequence(0).data(), bytes,
                                         cudaMemcpyHostToDevice, cuda_stream));
  }

  // Copy parameters->presence_mask to sampling_state->presence_mask
  gsl::span<int>& presence_mask = sampling_state->d_presence_mask;
  if (step == 1 && parameters->presence_mask.data() != nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(presence_mask.data(), parameters->presence_mask.data(),
                                         sizeof(int) * batch_size * vocab_size, cudaMemcpyDeviceToDevice, cuda_stream));
  }

  // TODO(hasesh): Can we avoid the const_cast by changing the interface of
  // GreedySearchProcessLogits() to take in a non-const OrtValue for logits
  // as this is the only place we will ever use the logits and it may be reasonable
  // to allow this method to mutate/process the logits in-place
  cuda::LaunchLogitsProcessKernel<CudaT>(
      is_reuse_logits_buffer ? const_cast<CudaT*>(logits_data)
                             : reinterpret_cast<CudaT*>(next_token_scores.data()),
      parameters->vocab_mask.data(),
      step > 1 ? nullptr : parameters->prefix_vocab_mask.data(),  // prefix vocab mask is applied to first step only.
      parameters->presence_mask.data() ? presence_mask.data() : nullptr,
      parameters->presence_penalty,
      parameters->temperature,
      parameters->batch_size,
      parameters->num_beams,
      vocab_size,
      is_reuse_logits_buffer ? padded_vocab_size : vocab_size,
      (parameters->min_length > 0 && current_sequence_length < parameters->sequence_length + parameters->min_length)
          ? parameters->eos_token_id
          : -1,
      reinterpret_cast<int32_t*>(sequences_buffer.get()),
      parameters->max_length,
      current_sequence_length,
      parameters->repetition_penalty,
      parameters->no_repeat_ngram_size,
      cuda_stream);

#ifdef DEBUG_GENERATION
  if (is_reuse_logits_buffer) {
    // TODO: Handle padded logits in the logits buffer before printing its contents
    ORT_THROW("Dumping contents of logits buffer is not implemented yet");
  } else {
    dumper->Print("next_token_scores after logits process", next_token_scores.data(), batch_size, vocab_size);
  }
#endif

  // TODO(wy): support output_scores in greedy search
  ORT_UNUSED_PARAMETER(output_scores);

  if (do_sampling) {
    ORT_RETURN_IF_ERROR(SamplingCudaHelper::Sample(allocator,
                                                   stream,
                                                   next_token_scores,
                                                   sampling_state,
                                                   greedy_state,
                                                   parameters,
                                                   step,
                                                   dumper));

    return Status::OK();
  }

  const CudaT* top_one_input = is_reuse_logits_buffer ? logits_data
                                                      : reinterpret_cast<const CudaT*>(next_token_scores.data());
  cuda::GreedySearchTopOne(
      top_one_input,
      batch_size,
      is_reuse_logits_buffer ? padded_vocab_size : vocab_size,
      reinterpret_cast<CudaT*>(greedy_state->temp_topk_scores_buffer.data()),
      greedy_state->temp_topk_tokens_buffer.data(),
      reinterpret_cast<CudaT*>(greedy_state->topk_scores_buffer.data()),
      greedy_state->topk_tokens_buffer.data(),
      cuda_stream);

#ifdef DEBUG_GENERATION
  dumper->Print("topk_scores", greedy_state->topk_scores_buffer.data(), batch_size, 1);
  dumper->Print("topk_indices", greedy_state->topk_tokens_buffer.data(), batch_size, 1);
#endif

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(greedy_state->next_tokens.data(),
                                       greedy_state->topk_tokens_buffer.data(),
                                       greedy_state->next_tokens.size_bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));

#ifdef DEBUG_GENERATION
  dumper->Print("greedy_state->next_tokens", greedy_state->next_tokens.data(), batch_size, 1);
#endif

#ifdef ENABLE_NVTX_PROFILE
  processLogitsRange.End();
#endif

  return Status::OK();
}

template <typename T>
Status DeviceCopy(gsl::span<T> target, gsl::span<const T> source, Stream* ort_stream, int copyDirection) {
  assert(copyDirection >= 0 && copyDirection <= 3);
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  if (cuda_stream == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(target.data(), source.data(), source.size_bytes(),
                                    static_cast<cudaMemcpyKind>(copyDirection)));
  } else {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(),
                                         static_cast<cudaMemcpyKind>(copyDirection), cuda_stream));
    if (copyDirection != DeviceCopyDirection::deviceToDevice)
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
  }
  return Status::OK();
}

template <typename T>
Status PickGptPastState(const std::vector<OrtValue>& last_outputs,
                        std::vector<OrtValue>& next_inputs,
                        gsl::span<const int32_t>& beam_indices,
                        AllocatorPtr allocator,
                        ptrdiff_t gpt_subgraph_first_past_input_idx,
                        ptrdiff_t gpt_subgraph_first_present_output_idx,
                        Stream* ort_stream) {
  ptrdiff_t num_present_tensors = static_cast<ptrdiff_t>(last_outputs.size()) - gpt_subgraph_first_present_output_idx;
  for (int i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[gpt_subgraph_first_present_output_idx + i];

    // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers, and use a CUDA kernel to copy key/value cache data.
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<T>();
    Tensor::InitOrtValue(past_type, past_shape, allocator, past);
    cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), past_shape.Size());
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), past_shape.Size());

    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const T> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam,
                                                              block_size_per_beam);

      gsl::span<T> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<T> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_key.data(), present_key.data(), present_key.size_bytes(),
                                           cudaMemcpyDeviceToDevice, cuda_stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_value.data(), present_value.data(), present_value.size_bytes(),
                                           cudaMemcpyDeviceToDevice, cuda_stream));
    }

    next_inputs[gpt_subgraph_first_past_input_idx + i] = past;
  }

  return Status::OK();
}

// Copy present state to past state for T5 model
template <typename T>
Status PickT5PastState(const std::vector<OrtValue>& last_outputs,
                       std::vector<OrtValue>& next_inputs,
                       int num_present_tensors,
                       gsl::span<const int32_t>& beam_indices,
                       AllocatorPtr allocator,
                       ptrdiff_t t5_decoder_first_past_input_idx,
                       ptrdiff_t t5_decoder_first_present_output_idx,
                       Stream* ort_stream) {
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  for (int i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[t5_decoder_first_present_output_idx + i];

    // shape is like (batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[1] * past_shape[2] * past_shape[3];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers, and use a CUDA kernel to copy key/value cache data.
    OrtValue past;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), past_shape, allocator, past);

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), past_shape.Size());
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), past_shape.Size());
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_beam = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<T> past_beam = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(past_beam.data(), present_beam.data(), present_beam.size_bytes(),
                                           cudaMemcpyDeviceToDevice, cuda_stream));
    }

    next_inputs[t5_decoder_first_past_input_idx + i] = past;
  }

  return Status::OK();
}

template <typename T>
Status UpdateGptFeeds(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir) {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxNestedRangeCreator updateFeedsRange("UpdateGptFeeds", profile::Color::Yellow);
  updateFeedsRange.Begin();
#endif

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());
  TensorShape input_ids_shape{batch_beam_size, 1};
  auto element_type = DataTypeImpl::GetType<int32_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  // num_beams == 1 using cudaMemcpyHostToDevice is because GreedySearch still uses CPU, BeamSearch is fully GPU
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_ids_data, beam_next_tokens.data(), beam_next_tokens.size_bytes(),
                                       num_beams == 1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice, cuda_stream));

  next_inputs[0] = input_ids;

  // Update position IDs
  int32_t* position_data = increase_position ? position_ids.GetMutable<Tensor>()->MutableData<int32_t>() : nullptr;
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const int32_t* old_mask_data = old_mask.Get<Tensor>().Data<int32_t>();
  TensorShape mask_shape{batch_beam_size, current_length};
  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<int32_t>();
  Tensor::InitOrtValue(mask_type, mask_shape, allocator, attention_mask);
  int32_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();

  // Launch kernel to update position_ids and attention_mask for next iteration
  cuda::LaunchUpdateGptKernel(old_mask_data, mask_data, position_data, batch_beam_size, current_length,
                              cuda_stream);

  next_inputs[2] = attention_mask;

  if (past_present_share_buffer) {
    // Update past sequence length input
    const ptrdiff_t past_sequence_length_idx = (static_cast<ptrdiff_t>(last_outputs.size()) - gpt_subgraph_first_present_output_idx) + gpt_subgraph_first_past_input_idx;
    *(next_inputs[past_sequence_length_idx].GetMutable<Tensor>()->MutableData<int32_t>()) = past_sequence_len;

    // Update beam search specific input for DecoderMaskedSelfAttention (cache indirection) if present

    // If the last input is not `past_sequence_length`, then the beam search specific inputs
    // for `DecoderMaskedSelfAttention` is present
    if (need_cache_indir) {
      ORT_ENFORCE(!beam_indices_gpu.empty(), "Beam indices must be present on CUDA while using DecoderMaskedSelfAttention with BeamSearch");

      // The cache indirection feed comes 2 feeds after the `past_sequence_length` feed
      const OrtValue& old_cache_indirection = next_inputs[past_sequence_length_idx + 2];

      // New cache indirection updated for next decoding run
      OrtValue cache_indirection;

      Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), old_cache_indirection.Get<Tensor>().Shape(), allocator, cache_indirection);

      // The third index of the past/present tensor is the max_sequence_length
      int max_sequence_length = static_cast<int>(last_outputs[gpt_subgraph_first_present_output_idx].Get<Tensor>().Shape()[3]);

      // Launch kernel to update the cache indirection buffer
      cuda::UpdateDecoderMaskedMultiHeadAttentionCacheIndirection(cache_indirection.GetMutable<Tensor>()->MutableData<int32_t>(),
                                                                  old_cache_indirection.Get<Tensor>().Data<int32_t>(),
                                                                  reinterpret_cast<const int32_t*>(beam_indices_gpu.data()),
                                                                  batch_beam_size / num_beams,
                                                                  num_beams,
                                                                  input_sequence_len,
                                                                  max_sequence_length,
                                                                  current_length,
                                                                  cuda_stream);
      // Update cache indirection for next decoding run
      next_inputs[past_sequence_length_idx + 2] = cache_indirection;
    }
  } else {
    if (num_beams == 1) {
      const int k = gpt_subgraph_first_past_input_idx - gpt_subgraph_first_present_output_idx;
      // feed present_* output to past_* inputs one by one
      for (size_t i = gpt_subgraph_first_present_output_idx; i < last_outputs.size(); ++i) {
        next_inputs[i + k] = last_outputs[i];
      }
    } else {
      ORT_RETURN_IF_ERROR(PickGptPastState<T>(last_outputs, next_inputs, beam_indices_cpu, allocator,
                                              gpt_subgraph_first_past_input_idx,
                                              gpt_subgraph_first_present_output_idx, ort_stream));
      // Make sure data is ready before next subgraph execution.
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
    }
  }

#ifdef ENABLE_NVTX_PROFILE
  updateFeedsRange.End();
#endif

  return Status::OK();
}

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
Status UpdateDecoderFeeds(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper) {
  // last_outputs: logits, present_key_self_0, present_value_self_0, ...
  // next_inputs: input_ids,
  //              encoder_attention_mask, encoder_hidden_states,
  //              past_key_self_0, past_value_self_0, ...
  //              past_key_cross_0, past_value_cross_0, ...
  // Only need copy beam next tokens to input_ids, and copy present_*_self_* to past_*_self_*,

  // Update input_ids with next tokens.
  int batch_beam_size = gsl::narrow<int>(beam_next_tokens.size());
  int sequence_length = !use_sequence_as_input_ids ? 1 : current_length;
  TensorShape input_ids_shape{batch_beam_size, sequence_length};
  auto element_type = DataTypeImpl::GetType<int32_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  if (!use_sequence_as_input_ids) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_ids_data, beam_next_tokens.data(), beam_next_tokens.size_bytes(),
                                         cudaMemcpyHostToDevice, cuda_stream));
  } else {
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const int32_t> sequence = sequences.GetSequence(i);
      const int32_t* sequence_data = sequence.data();
      CUDA_RETURN_IF_ERROR(
          cudaMemcpyAsync(input_ids_data + static_cast<ptrdiff_t>(i) * current_length,
                          sequence_data,
                          current_length * sizeof(int32_t),
                          cudaMemcpyHostToDevice,
                          cuda_stream));
    }
  }
  next_inputs[0] = input_ids;

#ifdef DEBUG_GENERATION
  dumper->Print("input_ids", input_ids);
#else
  ORT_UNUSED_PARAMETER(dumper);
#endif

  // Update past state
  ORT_ENFORCE(last_outputs.size() >= static_cast<size_t>(num_present_tensors) + 1);

  if (past_present_share_buffer) {
    // Update past sequence length input
    const ptrdiff_t past_sequence_length_idx = 2 * (static_cast<ptrdiff_t>(last_outputs.size()) - t5_decoder_first_present_output_idx) + t5_decoder_first_past_input_idx;
    *(next_inputs[past_sequence_length_idx].GetMutable<Tensor>()->MutableData<int32_t>()) = current_length - 1;

    // Update beam search specific input for DecoderMaskedSelfAttention (cache indirection) if present

    // If the last input is not `past_sequence_length`, then the beam search specific inputs
    // for `DecoderMaskedSelfAttention` is present
    if (need_cache_indir) {
      ORT_ENFORCE(!beam_indices_gpu.empty(), "Beam indices must be present on CUDA while using DecoderMaskedMultiHeadAttention with BeamSearch");

      // The cache indirection feed comes 2 feeds after the `past_sequence_length` feed
      const OrtValue& old_cache_indirection = next_inputs[past_sequence_length_idx + 2];

      // New cache indirection updated for next decoding run
      OrtValue cache_indirection;

      Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), old_cache_indirection.Get<Tensor>().Shape(), allocator, cache_indirection);

      // The third index of the past/present tensor is the max_sequence_length
      int max_sequence_length = static_cast<int>(last_outputs[t5_decoder_first_present_output_idx].Get<Tensor>().Shape()[2]);

      // Launch kernel to update the cache indirection buffer
      cuda::UpdateDecoderMaskedMultiHeadAttentionCacheIndirection(cache_indirection.GetMutable<Tensor>()->MutableData<int32_t>(),
                                                                  old_cache_indirection.Get<Tensor>().Data<int32_t>(),
                                                                  reinterpret_cast<const int32_t*>(beam_indices_gpu.data()),
                                                                  batch_beam_size / num_beams,
                                                                  num_beams,
                                                                  input_sequence_len,
                                                                  max_sequence_length,
                                                                  current_length,
                                                                  cuda_stream);

      // Update cache indirection for next decoding run
      next_inputs[past_sequence_length_idx + 2] = cache_indirection;
    }
  } else {
    // TODO(tianleiwu): remove num_beams==1 once GreedySearch operator is available.
    if (num_beams == 1) {
      // feed present_* output to past_* inputs one by one
      for (ptrdiff_t i = 0; i < num_present_tensors; ++i) {
        next_inputs[t5_decoder_first_past_input_idx + i] =
            last_outputs[t5_decoder_first_present_output_idx + i];
      }
      return Status::OK();
    }

    return PickT5PastState<T>(last_outputs, next_inputs, num_present_tensors, beam_indices, allocator,
                              t5_decoder_first_past_input_idx, t5_decoder_first_present_output_idx, ort_stream);
  }

  // Make sure data is ready before next subgraph execution.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));

  return Status::OK();
}

namespace {
template <typename T>
struct ToCudaTypeWrapper : public ToCudaType<T> {};

template <>
struct ToCudaTypeWrapper<int32_t> {
  using MappedType = int32_t;
};
}  // namespace

template <typename T>
Status ExpandBuffer(Stream* ort_stream,
                    const OrtValue& input,
                    int num_beams,
                    AllocatorPtr allocator,
                    OrtValue& expanded,
                    bool only_copy_shape,
                    int max_sequence_length) {
  // Input shape (batch_size, xxx). The input is required with data type T.
  // Output shape (batch_size * num_beams, xxx)
  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  const int64_t& batch_size = input_shape[0];
  int64_t sequence_length = 0;

  int64_t dims[4] = {0};
  input_shape.CopyDims(dims, input_shape.NumDimensions());
  dims[0] = batch_size * num_beams;
  bool is_kv_cache = input_shape.NumDimensions() == 4;
  if (max_sequence_length > 0 && is_kv_cache) {
    sequence_length = input_shape[2];
    dims[2] = max_sequence_length;
  }
  TensorShape expanded_shape(&dims[0], input_shape.NumDimensions());

  MLDataType element_type = input.Get<Tensor>().DataType();
  ORT_ENFORCE(element_type == DataTypeImpl::GetType<T>());
  Tensor::InitOrtValue(element_type, expanded_shape, allocator, expanded);

  if (only_copy_shape) {
    return Status::OK();
  }

  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  const T* input_data = input.Get<Tensor>().Data<T>();
  T* expanded_data = expanded.GetMutable<Tensor>()->MutableData<T>();

  using CudaT = typename ToCudaTypeWrapper<T>::MappedType;

  if (max_sequence_length == 0) {
    const int64_t& chunk_size = static_cast<int64_t>(input_shape.Size() / batch_size);

    cuda::BufferExpansionKernelLauncher<CudaT>(reinterpret_cast<const CudaT*>(input_data),
                                               reinterpret_cast<CudaT*>(expanded_data),
                                               static_cast<int>(batch_size),
                                               num_beams,
                                               static_cast<int>(chunk_size),
                                               cuda_stream);
    return Status::OK();
  }

  ORT_ENFORCE(is_kv_cache);

  // Expand from [B, N, S, H] to [B*beam, N, S_max, H]
  const int64_t& num_heads = input_shape[1];
  const int64_t& head_size = input_shape[3];

  cuda::KeyCacheExpansionKernelLauncher<CudaT>(reinterpret_cast<const CudaT*>(input_data),
                                               reinterpret_cast<CudaT*>(expanded_data),
                                               static_cast<int>(batch_size),
                                               num_beams,
                                               static_cast<int>(num_heads),
                                               static_cast<int>(sequence_length),
                                               max_sequence_length,
                                               static_cast<int>(head_size),
                                               cuda_stream);

  return Status::OK();
}

// Explicit template instantiations of functions
template void InitBeamState<float>(
    transformers::IBeamSearchState<float>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    Stream* ort_stream);

template void InitGreedyState<float>(
    transformers::IGreedySearchState<float>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    Stream* ort_stream);

template Status ProcessLogits<float>(
    const OrtValue& logits,
    transformers::IBeamSearchState<float>* beam_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    transformers::IBeamScorer* beam_scorer,
    const transformers::IGenerationParameters* parameters,
    int step,
    Stream* ort_stream,
    const transformers::IConsoleDumper* dumper);

template Status GreedySearchProcessLogits<float>(
    const OrtValue& logits,
    transformers::IGreedySearchState<float>* greedy_state,
    transformers::ISamplingState<float>* sampling_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    const transformers::IGenerationParameters* parameters,
    bool do_sampling,
    int step,
    Stream* ort_stream,
    const transformers::IConsoleDumper* dumper);

template Status DeviceCopy<float>(
    gsl::span<float> target,
    gsl::span<const float> source,
    Stream* ort_stream,
    int copyDirection);

template Status DeviceCopy<int32_t>(
    gsl::span<int32_t> target,
    gsl::span<const int32_t> source,
    Stream* ort_stream,
    int copyDirection);

template Status UpdateGptFeeds<float>(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir);

// Float16
template void InitBeamState<MLFloat16>(
    transformers::IBeamSearchState<MLFloat16>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    Stream* ort_stream);

template void InitGreedyState<MLFloat16>(
    transformers::IGreedySearchState<MLFloat16>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    Stream* ort_stream);

template Status ProcessLogits<MLFloat16>(
    const OrtValue& logits,
    transformers::IBeamSearchState<MLFloat16>* beam_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    transformers::IBeamScorer* beam_scorer,
    const transformers::IGenerationParameters* parameters,
    int step,
    Stream* ort_stream,
    const transformers::IConsoleDumper* dumper);

template Status GreedySearchProcessLogits<MLFloat16>(
    const OrtValue& logits,
    transformers::IGreedySearchState<MLFloat16>* greedy_state,
    transformers::ISamplingState<MLFloat16>* sampling_state,
    transformers::ISequences* sequences,
    AllocatorPtr& allocator,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    transformers::ILogitsProcessorList* logits_processors,
    const transformers::IGenerationParameters* parameters,
    bool do_sampling,
    int step,
    Stream* ort_stream,
    const transformers::IConsoleDumper* dumper);

template Status UpdateGptFeeds<MLFloat16>(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    bool increase_position,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices_cpu,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int gpt_subgraph_first_past_input_idx,
    int gpt_subgraph_first_present_output_idx,
    bool past_present_share_buffer,
    int past_sequence_len,
    int input_sequence_len,
    bool need_cache_indir);

template Status UpdateDecoderFeeds<float>(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper);

template Status UpdateDecoderFeeds<MLFloat16>(
    AllocatorPtr allocator,
    Stream* ort_stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int num_present_tensors,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices,
    gsl::span<const int32_t> beam_indices_gpu,
    int num_beams,
    int t5_decoder_first_past_input_idx,
    int t5_decoder_first_present_output_idx,
    bool use_sequence_as_input_ids,
    int current_length,
    int input_sequence_len,
    bool past_present_share_buffer,
    bool need_cache_indir,
    transformers::Sequences& sequences,
    const transformers::IConsoleDumper* dumper);

template Status ExpandBuffer<int32_t>(
    Stream* ort_stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

template Status ExpandBuffer<float>(
    Stream* ort_stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

template Status ExpandBuffer<MLFloat16>(
    Stream* ort_stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);
}  // namespace GenerationCudaDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
