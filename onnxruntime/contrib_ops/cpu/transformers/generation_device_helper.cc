// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <algorithm>
#include <memory>
#include "core/providers/cpu/math/top_k.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/providers/cpu/generator/random.h"
#include "core/common/safeint.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/beam_search_scorer.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"
#include "contrib_ops/cpu/transformers/sampling_cpu_helper.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"

namespace onnxruntime {
namespace contrib {
namespace GenerationCpuDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            Stream* /*stream*/,
            onnxruntime::concurrency::ThreadPool* threadpool,
            Tensor& output_values,
            Tensor& output_indices) {
  if (input->IsDataType<float>()) {
    return GetTopK<float>(input, axis, k, largest, sorted, allocator, threadpool, output_values, output_indices);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "BeamSearch op: An implementation for the input type ",
                         input->DataType(), " is not supported yet");
}

template <typename T>
void ExpandInputs(const OrtValue& input, int num_beams, AllocatorPtr allocator, OrtValue& expanded) {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  const int64_t& batch_size = input_shape[0];
  const int64_t& sequence_length = input_shape[1];

  int64_t dims[] = {batch_size * num_beams, sequence_length};
  TensorShape expanded_shape(&dims[0], 2);

  MLDataType element_type = input.Get<Tensor>().DataType();
  ORT_ENFORCE(element_type == DataTypeImpl::GetType<T>());

  Tensor::InitOrtValue(element_type, expanded_shape, allocator, expanded);

  const T* input_data = input.Get<Tensor>().Data<T>();
  T* expanded_data = expanded.GetMutable<Tensor>()->MutableData<T>();
  T* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      memcpy(target, input_data + i * sequence_length, sizeof(T) * SafeInt<size_t>(sequence_length));
      target += sequence_length;
    }
  }
}

// TODO(wy): Dispatch it to avoid passing multiple functions to interface.
template <typename T>
Status ExpandBuffer(Stream* stream,
                    const OrtValue& input,
                    int num_beams,
                    AllocatorPtr allocator,
                    OrtValue& expanded,
                    bool only_copy_shape,
                    int max_sequence_length) {
  // Input shape (batch_size, xxx). The input is required with data type T.
  // Output shape (batch_size * num_beams, xxx)
  // If max_sequence_length > 0, the output shape will be (batch_size * num_beams, num_heads,
  // max_sequence_length, head_size)
  ORT_UNUSED_PARAMETER(stream);

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

  const T* input_data = input.Get<Tensor>().Data<T>();
  T* expanded_data = expanded.GetMutable<Tensor>()->MutableData<T>();
  T* target = expanded_data;

  if (max_sequence_length == 0) {
    const int64_t& chunk_size = static_cast<int64_t>(input_shape.Size() / batch_size);

    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        memcpy(target, input_data + i * chunk_size, sizeof(T) * SafeInt<size_t>(chunk_size));
        target += chunk_size;
      }
    }
    return Status::OK();
  }

  ORT_ENFORCE(is_kv_cache);

  // Expand from [B, N, S, H] to [B*beam, N, S_max, H]
  const int64_t& num_heads = input_shape[1];
  const int64_t& head_size = input_shape[3];
  const int64_t& input_offset = sequence_length * head_size;
  const int64_t& output_offset = max_sequence_length * head_size;
  const int64_t& NSH = input_offset * num_heads;

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      for (int k = 0; k < num_heads; k++) {
        memcpy(target, input_data + i * NSH + k * input_offset, sizeof(T) * SafeInt<size_t>(input_offset));
        target += output_offset;
      }
    }
  }

  return Status::OK();
}

Status CreateGptInputs(
    const Tensor* original_input_ids,
    const OrtValue* attn_mask_value,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    AllocatorPtr allocator,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask) {
  const TensorShape& input_ids_shape = original_input_ids->Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int32_t>();

  const OrtMemoryInfo& location = allocator->Info();

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape,
                       const_cast<Tensor*>(original_input_ids)->MutableData<int32_t>(), location, input_ids);

  OrtValue position_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, position_ids);

  OrtValue attention_mask;
  if (attn_mask_value != nullptr) {
    const Tensor& attn_mask = attn_mask_value->Get<Tensor>();
    Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(&attn_mask)->MutableData<int32_t>(),
                         allocator->Info(), attention_mask);
  } else {
    auto mask_type = DataTypeImpl::GetType<int32_t>();
    Tensor::InitOrtValue(mask_type, input_ids_shape, allocator, attention_mask);
  }

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int32_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();
  int32_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  const int32_t* word_id = original_input_ids->Data<int32_t>();
  int32_t* mask = mask_data;
  int32_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int32_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == pad_token_id) {
        if (attn_mask_value == nullptr) {
          *mask = 0;
        }
        *position = 0;
      } else {
        if (attn_mask_value == nullptr) {
          *mask = 1;
        }
        *position = abs_position;
        abs_position++;
      }
    }

    for (int k = 0; k < num_beams; k++) {
      sequence_lengths[SafeInt<gsl::index>(i) * num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
  // TODO(tianleiwu): Try expand outputs after first subgraph call instead. That may get better performance.
  if (num_beams == 1) {
    expanded_input_ids = std::move(input_ids);
    expanded_position_ids = std::move(position_ids);
    expanded_attention_mask = std::move(attention_mask);
    return Status::OK();
  }

  ExpandInputs<int32_t>(input_ids, num_beams, allocator, expanded_input_ids);
  ExpandInputs<int32_t>(position_ids, num_beams, allocator, expanded_position_ids);
  ExpandInputs<int32_t>(attention_mask, num_beams, allocator, expanded_attention_mask);

  return Status::OK();
}

Status AddToFeeds(Stream* /*ort_stream*/,
                  std::initializer_list<OrtValue> inputs,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& /*buffer*/,
                  AllocatorPtr /*device_allocator*/,
                  AllocatorPtr /*host_allocator*/,
                  const OrtMemoryInfo& /*location*/) {
  for (auto& input : inputs) {
    if (input.IsAllocated()) {
      feeds.push_back(input);
    }
  }

  return Status::OK();
}

template <typename T>
void InitBeamState(transformers::IBeamSearchState<T>* beam_state,
                   gsl::span<int32_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   Stream* /*stream*/) {
  memset(beam_state->beam_scores.data(), 0, beam_state->beam_scores.size_bytes());
  memset(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes());
  memset(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes());
  memset(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes());
  memset(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes());

  // T5 does not need position, so next_positions is empty for T5.
  if (!beam_state->next_positions.empty()) {
    gsl::copy(sequence_lengths, beam_state->next_positions);
  }

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  // This ensures that the beams in the same group don't produce same tokens every time.
  gsl::span<float>& beam_scores = beam_state->beam_scores;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 1; j < num_beams; j++) {
      beam_scores[SafeInt<gsl::index>(i) * num_beams + j] = -1e9;
    }
  }
}

template <typename T>
void InitGreedyState(transformers::IGreedySearchState<T>* greedy_state,
                     gsl::span<int32_t>& sequence_lengths,
                     Stream* /*stream*/) {
  memset(greedy_state->next_token_scores.data(), 0, greedy_state->next_token_scores.size_bytes());
  memset(greedy_state->next_tokens.data(), 0, greedy_state->next_tokens.size_bytes());
  memset(greedy_state->next_positions.data(), 0, greedy_state->next_positions.size_bytes());

  gsl::copy(sequence_lengths, greedy_state->next_positions);
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
                     Stream* stream,                                         // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper) {           // tensor dumper
#ifndef DEBUG_GENERATION
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int num_beams = parameters->num_beams;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size * num_beams;
  const T* logits_data = logits.Get<Tensor>().Data<T>();

  // Logits has shape (batch_size * num_beams, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];
  auto logits_batch_size = logits_shape[0];

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_logits = beam_state->next_token_logits;

  if (input_length > 1 || logits_batch_size == batch_size) {
    const T* current_logits = logits_data + (input_length - 1) * vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const T> source(current_logits, vocab_size);
      gsl::span<T> target = next_token_logits.subspan(SafeInt<gsl::index>(i) * vocab_size,
                                                      static_cast<gsl::index>(vocab_size));
      gsl::copy(source, target);
      if (logits_batch_size == batch_beam_size) {
        current_logits += input_length * vocab_size;
      } else if (logits_batch_size == batch_size && i % num_beams == num_beams - 1) {
        current_logits += input_length * vocab_size;
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
  gsl::span<T>& next_token_scores = beam_state->next_token_scores;
  ORT_RETURN_IF_ERROR(
      SoftmaxCPU<T>(
          batch_beam_size,  // rows
          vocab_size,       // elements per row
          (input_length == 1 && logits_batch_size == batch_beam_size) ? logits_data : next_token_logits.data(),
          next_token_scores.data(),
          true,
          thread_pool));

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Apply all score processors that updates scores
  logits_processors->Process(sequences, next_token_scores, step);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after logits process", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO(tianleiwu): use thread pool to parallel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < vocab_size; k++, offset++) {
        next_token_scores[offset] += beam_state->beam_scores[batch_beam_index];
      }
    }
  }

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  if (output_scores) {
    // Append next token scores to the scores output.
    gsl::copy(next_token_scores, beam_state->remaining_scores);
    beam_state->remaining_scores = beam_state->remaining_scores.subspan(next_token_scores.size());
  }

  // Apply top-k selection like the following:
  //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
  //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
  int64_t next_token_scores_dims[] = {static_cast<int64_t>(batch_size), SafeInt<int64_t>(num_beams) * vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(),
                       next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr int axis = 1;
  const unsigned top_k = static_cast<unsigned>(2 * num_beams);
  constexpr bool largest = true;
  constexpr bool sorted = true;  // results returned in sorted order.

  Tensor topk_scores;
  Tensor topk_indices;
  ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, stream, thread_pool,
                           topk_scores, topk_indices));

#ifdef DEBUG_GENERATION
  dumper->Print("topk_scores", topk_scores);
  dumper->Print("topk_indices", topk_indices);
#endif

  // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
  //   next_indices = (next_tokens / vocab_size).long()
  //   next_tokens = next_tokens % vocab_size
  gsl::span<const int64_t> next_token_indices = topk_indices.DataAsSpan<int64_t>();
  offset = 0;
  for (int i = 0; i < batch_size; i++) {
    for (unsigned int j = 0; j < top_k; j++, offset++) {
      beam_state->next_indices[offset] = gsl::narrow_cast<int32_t>(next_token_indices[offset] / vocab_size);
      beam_state->next_tokens[offset] = gsl::narrow_cast<int32_t>(next_token_indices[offset] % vocab_size);
    }
  }

  gsl::span<const T> next_scores = topk_scores.DataAsSpan<T>();
  gsl::span<const int32_t> next_tokens(beam_state->next_tokens.data(), beam_state->next_tokens.size());
  gsl::span<const int32_t> next_indices(beam_state->next_indices.data(), beam_state->next_indices.size());

#ifdef DEBUG_GENERATION
  dumper->Print("next_scores before scorer", next_scores.data(), batch_size, top_k);
  dumper->Print("next_tokens before scorer", next_tokens.data(), batch_size, top_k);
  dumper->Print("next_indices before scorer", next_indices.data(), batch_size, top_k);
#endif

  beam_scorer->Process(
      *sequences,
      next_scores,
      next_tokens,
      next_indices);

  return Status::OK();
}

template <typename T>
Status GreedySearchProcessLogits(
    const OrtValue& logits,                                 // logits output of subgraph
    transformers::IGreedySearchState<T>* greedy_state,      // state
    transformers::ISamplingState<T>* sampling_state,        // sampling_state
    transformers::ISequences* sequences,                    // sequences
    AllocatorPtr& allocator,                                // default allocator
    onnxruntime::concurrency::ThreadPool* thread_pool,      // thread pool (for CPU only)
    transformers::ILogitsProcessorList* logits_processors,  // logits processors
    const transformers::IGenerationParameters* parameters,  // parameters
    bool do_sampling,                                       // whether to do sampling
    int step,                                               // iteration counter
    Stream* stream,                                         // cuda stream (for CUDA only)
    const transformers::IConsoleDumper* dumper) {           // tensor dumper

  int batch_size = parameters->batch_size;
  int vocab_size = parameters->vocab_size;

  const T* logits_data = logits.Get<Tensor>().Data<T>();

  // Logits has shape (batch_size, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_scores = greedy_state->next_token_scores;
  const T* current_logits = logits_data + (input_length - 1) * vocab_size;
  for (int i = 0; i < batch_size; i++) {
    gsl::span<const T> source(current_logits, vocab_size);
    gsl::span<T> target = next_token_scores.subspan(SafeInt<gsl::index>(i) * vocab_size,
                                                    static_cast<gsl::index>(vocab_size));
    gsl::copy(source, target);
    current_logits += input_length * vocab_size;
  }

#ifdef DEBUG_GENERATION
  dumper->Print("logits", logits);
  dumper->Print("next_token_logits", next_token_scores.data(), batch_size, 1, vocab_size);
#endif

  // Apply all score processors that updates scores
  logits_processors->Process(sequences, next_token_scores, step);

#ifdef DEBUG_GENERATION
  dumper->Print("next_token_scores after logits processor", next_token_scores.data(), batch_size, 1, vocab_size);
#endif

  if (do_sampling) {
    ORT_RETURN_IF_ERROR(SamplingCpuHelper::Sample(allocator,
                                                  thread_pool,
                                                  next_token_scores,
                                                  sampling_state,
                                                  greedy_state,
                                                  parameters,
                                                  dumper));

    return Status::OK();
  }

  // next_tokens = torch.argmax(scores, dim=-1)
  int64_t next_token_scores_dims[] = {static_cast<int64_t>(batch_size), vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type,
                       next_token_scores_shape,
                       next_token_scores.data(),
                       allocator->Info(),
                       next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr unsigned top_k = 1;
  constexpr int axis = 1;
  constexpr bool largest = true;
  constexpr bool sorted = false;

  Tensor topk_scores;
  Tensor topk_indices;
  ORT_RETURN_IF_ERROR(TopK(&input,
                           axis,
                           top_k,
                           largest,
                           sorted,
                           allocator,
                           stream,
                           thread_pool,
                           topk_scores,
                           topk_indices));

#ifdef DEBUG_GENERATION
  dumper->Print("topk_scores", topk_scores);
  dumper->Print("topk_indices", topk_indices);
#endif

  gsl::span<const int64_t> next_token_indices = topk_indices.DataAsSpan<int64_t>();
  for (size_t i = 0; i < next_token_indices.size(); i++) {
    greedy_state->next_tokens[i] = gsl::narrow_cast<int32_t>(next_token_indices[i]);
  }

#ifdef DEBUG_GENERATION
  gsl::span<const int32_t> next_tokens(greedy_state->next_tokens.data(),
                                       greedy_state->next_tokens.size());
  dumper->Print("next_tokens before scorer", next_tokens.data(), batch_size, top_k);
#endif

  return Status::OK();
}

template <typename T>
Status DeviceCopy(gsl::span<T> target, gsl::span<const T> source, Stream* /*stream*/, int /*copyDirection*/) {
  gsl::copy(source, target);
  return Status::OK();
}

// Copy present state to past state for GPT model
template <typename T>
void PickGptPastState(const std::vector<OrtValue>& last_outputs,
                      std::vector<OrtValue>& next_inputs,
                      gsl::span<const int32_t>& beam_indices,
                      int gpt_subgraph_first_past_input_idx,
                      int gpt_subgraph_first_present_output_idx,
                      AllocatorPtr allocator) {
  int num_present_tensors = static_cast<int>(last_outputs.size()) - gpt_subgraph_first_present_output_idx;
  for (ptrdiff_t i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[gpt_subgraph_first_present_output_idx + i];

    // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<T>();
    Tensor::InitOrtValue(past_type, past_shape, allocator, past);

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), onnxruntime::narrow<size_t>(past_shape.Size()));
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), onnxruntime::narrow<size_t>(past_shape.Size()));
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_key = present_span.subspan(beam_index * SafeInt<size_t>(block_size_per_beam), onnxruntime::narrow<size_t>(block_size_per_beam));
      gsl::span<const T> present_value = present_span.subspan(past_key_size + beam_index * SafeInt<size_t>(block_size_per_beam),
                                                              onnxruntime::narrow<size_t>(block_size_per_beam));

      gsl::span<T> past_key = past_span.subspan(j * SafeInt<size_t>(block_size_per_beam), onnxruntime::narrow<size_t>(block_size_per_beam));
      gsl::span<T> past_value = past_span.subspan(past_key_size + j * SafeInt<size_t>(block_size_per_beam), onnxruntime::narrow<size_t>(block_size_per_beam));
      gsl::copy(present_key, past_key);
      gsl::copy(present_value, past_value);
    }

    next_inputs[gpt_subgraph_first_past_input_idx + i] = past;
  }
}

template <typename T>
Status UpdateGptFeeds(
    AllocatorPtr allocator,
    Stream* stream,
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
  // last_outputs: logits, present_0, present_1, ...
  // next_inputs: input_ids, position_id, attention_mask, past_0, past_1
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(beam_indices_gpu);
  ORT_UNUSED_PARAMETER(input_sequence_len);
  ORT_UNUSED_PARAMETER(need_cache_indir);

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto int32_type = DataTypeImpl::GetType<int32_t>();
  OrtValue input_ids;
  // TODO(tianleiwu): Reuse buffer for input_ids to reduce memory allocation.
  Tensor::InitOrtValue(int32_type, input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = beam_next_tokens[i];
  }
  next_inputs[0] = input_ids;

  if (increase_position) {
    // Update position IDs
    int32_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int32_t>();
    for (int i = 0; i < batch_beam_size; i++) {
      position_data[i]++;
    }
  }
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const int32_t* old_mask_data = old_mask.Get<Tensor>().Data<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  Tensor::InitOrtValue(int32_type, mask_shape, allocator, attention_mask);
  int32_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
  next_inputs[2] = attention_mask;

  if (past_present_share_buffer) {
    int32_t* past_seq_len_data = const_cast<int32_t*>(next_inputs.back().Get<Tensor>().Data<int32_t>());
    *past_seq_len_data = past_sequence_len;
    return Status::OK();
  }

  if (num_beams == 1) {  // Update past state
    // feed present_* output to past_* inputs one by one
    const int k = gpt_subgraph_first_past_input_idx - gpt_subgraph_first_present_output_idx;
    for (size_t i = gpt_subgraph_first_present_output_idx; i < last_outputs.size(); ++i) {
      next_inputs[i + k] = last_outputs[i];
    }
  } else {
    PickGptPastState<T>(last_outputs, next_inputs, beam_indices_cpu,
                        gpt_subgraph_first_past_input_idx,
                        gpt_subgraph_first_present_output_idx, allocator);
  }
  return Status::OK();
}

// ---------------------------------------------------------------
// The following functions are for encoder-decoder model like T5
// ---------------------------------------------------------------
Status CreateEncoderInputs(
    const Tensor* original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    int pad_token_id,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_ids,
    OrtValue& encoder_attention_mask,
    OrtValue& decoder_input_ids) {
  const TensorShape& input_ids_shape = original_encoder_input_ids->Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int32_t>();

  // Use original encoder_input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  Tensor::InitOrtValue(element_type,
                       input_ids_shape,
                       const_cast<Tensor*>(original_encoder_input_ids)->MutableData<int32_t>(),
                       allocator->Info(),
                       encoder_input_ids);

  if (attn_mask_value != nullptr) {
    const Tensor& attention_mask = attn_mask_value->Get<Tensor>();
    Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(&attention_mask)->MutableData<int32_t>(),
                         allocator->Info(), encoder_attention_mask);
  } else {
    auto mask_type = DataTypeImpl::GetType<int32_t>();
    Tensor::InitOrtValue(mask_type, input_ids_shape, allocator, encoder_attention_mask);

    // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
    int32_t* mask_data = encoder_attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();
    const int32_t* word_id = original_encoder_input_ids->Data<int32_t>();
    int32_t* mask = mask_data;
    for (int i = 0; i < batch_size; i++) {
      int32_t abs_position = 0;
      for (int j = 0; j < sequence_length; j++, word_id++, mask++) {
        // T5Tokenizer might add one EOS pad token at the end.
        // That EOS token shall have attention mask 1 even when EOS token is same as pad token.
        // Here we only set attention mask to be 0 for left padding only, so as to be parity with huggingface.
        if (*word_id == pad_token_id && abs_position == 0) {
          *mask = 0;
        } else {
          *mask = 1;
          abs_position++;
        }
      }
    }
  }

  // decoder_input_ids is optional.
  if (start_token_id >= 0) {
    // Filled decoder_input_ids with start token ID
    int64_t dims[] = {batch_size, 1};
    TensorShape decoder_input_ids_shape(&dims[0], 2);
    Tensor::InitOrtValue(element_type, decoder_input_ids_shape, allocator, decoder_input_ids);
    int32_t* data = decoder_input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
    for (int i = 0; i < batch_size; i++, data++) {
      *data = start_token_id;
    }
  }

  return Status::OK();
}

// Copy present state to past state for T5 model
template <typename T>
void PickT5PastState(const std::vector<OrtValue>& last_outputs,
                     std::vector<OrtValue>& next_inputs,
                     int num_present_tensors,
                     gsl::span<const int32_t>& beam_indices,
                     int t5_decoder_first_past_input_idx,
                     int t5_decoder_first_present_output_idx,
                     AllocatorPtr allocator) {
  for (ptrdiff_t i = 0; i < num_present_tensors; ++i) {
    const OrtValue& present = last_outputs[t5_decoder_first_present_output_idx + i];

    // shape is like (batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();
    auto block_size_per_beam = past_shape[1] * past_shape[2] * past_shape[3];

    // Create a tensor with same shape.
    // TODO(tianleiwu): allocate one buffer for all layers
    OrtValue past;
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), past_shape, allocator, past);

    gsl::span<T> past_span = gsl::make_span<T>(past.GetMutable<Tensor>()->MutableData<T>(), onnxruntime::narrow<size_t>(past_shape.Size()));
    gsl::span<const T> present_span = gsl::make_span<const T>(present.Get<Tensor>().Data<T>(), onnxruntime::narrow<size_t>(past_shape.Size()));
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      gsl::span<const T> present_beam = present_span.subspan(beam_index * SafeInt<size_t>(block_size_per_beam), onnxruntime::narrow<size_t>(block_size_per_beam));
      gsl::span<T> past_beam = past_span.subspan(j * SafeInt<size_t>(block_size_per_beam), onnxruntime::narrow<size_t>(block_size_per_beam));
      gsl::copy(present_beam, past_beam);
    }

    next_inputs[t5_decoder_first_past_input_idx + i] = past;
  }
}

// Update decoder inputs given decoder outputs of last iteration.
template <typename T>
Status UpdateDecoderFeeds(
    AllocatorPtr allocator,
    Stream* stream,
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
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(beam_indices_gpu);
  ORT_UNUSED_PARAMETER(input_sequence_len);
  ORT_UNUSED_PARAMETER(past_present_share_buffer);
  ORT_UNUSED_PARAMETER(need_cache_indir);
  // last_outputs: logits, present_key_self_0, present_value_self_0, ...
  // next_inputs: input_ids,
  //              encoder_attention_mask, encoder_hidden_states(optional),
  //              past_key_self_0, past_value_self_0, ...
  //              past_key_cross_0, past_value_cross_0, ...
  // Only need copy beam next tokens to input_ids, and copy present_*_self_* to past_*_self_*,

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());

  // TODO(tianleiwu): Reuse buffer for input_ids to reduce memory allocation.
  OrtValue input_ids;
  int sequence_length = !use_sequence_as_input_ids ? 1 : current_length;
  int64_t dims[] = {batch_beam_size, sequence_length};
  TensorShape input_ids_shape(&dims[0], 2);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), input_ids_shape, allocator, input_ids);

  if (!use_sequence_as_input_ids) {
    gsl::copy(beam_next_tokens, input_ids.GetMutable<Tensor>()->MutableDataAsSpan<int32_t>());
  } else {
    int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const int32_t> sequence = sequences.GetSequence(i);
      const int32_t* sequence_data = sequence.data();
      for (int j = 0; j < current_length; j++) {
        input_ids_data[i * current_length + j] = sequence_data[j];
      }
    }
  }

  next_inputs[0] = input_ids;

#ifdef DEBUG_GENERATION
  dumper->Print("input_ids", input_ids);
#else
  ORT_UNUSED_PARAMETER(dumper);
#endif

  // Update past state
  ORT_ENFORCE(last_outputs.size() >= static_cast<size_t>(1) + num_present_tensors);
  // TODO(tianleiwu): remove num_beams==1 once GreedySearch operator is available.
  if (num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (ptrdiff_t i = 0; i < num_present_tensors; ++i) {
      next_inputs[t5_decoder_first_past_input_idx + i] =
          last_outputs[t5_decoder_first_present_output_idx + i];
    }
  } else {
    PickT5PastState<T>(last_outputs, next_inputs, num_present_tensors, beam_indices,
                       t5_decoder_first_past_input_idx, t5_decoder_first_present_output_idx, allocator);
  }
  return Status::OK();
}

//------------------------------------------------
//  Modified encoder function for Whisper Model
//------------------------------------------------
template <typename T>
Status CreateWhisperEncoderInputs(
    const Tensor* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_features,
    OrtValue& decoder_input_ids) {
  const TensorShape& input_features_shape = original_encoder_input_features->Shape();
  ORT_ENFORCE(input_features_shape.NumDimensions() == 3);
  const int64_t& batch_size = input_features_shape[0];

  // Allocate attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int32_t>();

  // Use original encoder_input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  Tensor::InitOrtValue(DataTypeImpl::GetType<T>(),
                       input_features_shape,
                       const_cast<Tensor*>(original_encoder_input_features)->MutableData<T>(),
                       allocator->Info(),
                       encoder_input_features);

  // decoder_input_ids is optional.
  if (original_decoder_input_ids_value == nullptr) {
    // Filled decoder_input_ids with start token ID
    ORT_ENFORCE(start_token_id >= 0);
    int64_t dims[] = {batch_size, 1};
    TensorShape decoder_input_ids_shape(&dims[0], 2);
    Tensor::InitOrtValue(element_type, decoder_input_ids_shape, allocator, decoder_input_ids);
    int32_t* data = decoder_input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
    for (int i = 0; i < batch_size; i++, data++) {
      *data = start_token_id;
    }
  } else {
    // decoder_input_ids is of shape (batch_size, initial_sequence_length)
    // Example: [[ decoder start token (i.e. start of transcript), language token, task token, timestamp token ]]
    const Tensor* original_decoder_input_ids = &(original_decoder_input_ids_value->Get<Tensor>());
    const TensorShape& original_decoder_input_ids_shape = original_decoder_input_ids->Shape();
    ORT_ENFORCE(original_decoder_input_ids_shape.NumDimensions() == 2);
    Tensor::InitOrtValue(element_type,
                         original_decoder_input_ids_shape,
                         const_cast<Tensor*>(original_decoder_input_ids)->MutableData<int32_t>(),
                         allocator->Info(),
                         decoder_input_ids);
  }

  return Status::OK();
}

//------------------------------------------------
// Explicit template instantiations of functions
//------------------------------------------------

template void InitBeamState<float>(
    transformers::IBeamSearchState<float>* beam_state,
    gsl::span<int32_t>& sequence_lengths,
    int batch_size,
    int num_beams,
    Stream* stream);

template void InitGreedyState<float>(
    transformers::IGreedySearchState<float>* greedy_state,
    gsl::span<int32_t>& sequence_lengths,
    Stream* stream);

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
    Stream* stream,
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
    Stream* stream,
    int copyDirection);

template Status DeviceCopy<int32_t>(
    gsl::span<int32_t> target,
    gsl::span<const int32_t> source,
    Stream* stream,
    int copyDirection);

template Status UpdateGptFeeds<float>(
    AllocatorPtr allocator,
    Stream* stream,
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
    Stream* stream,
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
    Stream* stream,
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

template void ExpandInputs<int32_t>(const OrtValue& input, int num_beams, AllocatorPtr allocator, OrtValue& expanded);

template Status ExpandBuffer<int32_t>(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

template Status ExpandBuffer<float>(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

template Status ExpandBuffer<MLFloat16>(
    Stream* stream,
    const OrtValue& input,
    int num_beams,
    AllocatorPtr allocator,
    OrtValue& expanded,
    bool only_copy_shape,
    int max_sequence_length);

template Status CreateWhisperEncoderInputs<float>(
    const Tensor* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_features,
    OrtValue& decoder_input_ids);

template Status CreateWhisperEncoderInputs<MLFloat16>(
    const Tensor* original_encoder_input_features,
    const OrtValue* original_decoder_input_ids_value,
    int start_token_id,
    AllocatorPtr allocator,
    OrtValue& encoder_input_features,
    OrtValue& decoder_input_ids);

}  // namespace GenerationCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime
