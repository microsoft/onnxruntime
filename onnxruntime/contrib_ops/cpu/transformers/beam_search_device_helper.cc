#include "core/providers/cpu/math/top_k.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/common/safeint.h"
#include "gsl/gsl"
#include "sequences.h"
#include "beam_search_scorer.h"
#include "beam_search_device_helper.h"

namespace onnxruntime {
namespace contrib {
namespace BeamSearchCpuDeviceHelper {

Status TopK(const Tensor* input, const int axis, const unsigned k, bool largest, bool sorted,
            AllocatorPtr allocator,
            void* /*stream*/,
            onnxruntime::concurrency::ThreadPool* threadpool,
            std::unique_ptr<Tensor>& output_values,
            std::unique_ptr<Tensor>& output_indices) {
  if (input->IsDataType<float>()) {
    return GetTopK<float>(input, axis, k, largest, sorted, allocator, threadpool, output_values, output_indices);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "BeamSearch op: An implementation for the input type ",
                         input->DataType(), " is not supported yet");
}

OrtValue ExpandInputs(const OrtValue& input, int num_beams, AllocatorPtr allocator) {
  // Input shape (batch_size, sequence_length)
  // Output shape (batch_size * num_beams, sequence_length)
  if (num_beams == 1)
    return input;

  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  const int64_t& batch_size = input_shape[0];
  const int64_t& sequence_length = input_shape[1];

  int64_t dims[] = {batch_size * num_beams, sequence_length};
  TensorShape expanded_shape(&dims[0], 2);

  OrtValue expanded;
  MLDataType element_type = input.Get<Tensor>().DataType();
  Tensor::InitOrtValue(element_type, expanded_shape, allocator, expanded);

  if (element_type == DataTypeImpl::GetType<int64_t>()) {
    const int64_t* input_data = input.Get<Tensor>().Data<int64_t>();
    int64_t* expanded_data = expanded.GetMutable<Tensor>()->MutableData<int64_t>();
    int64_t* target = expanded_data;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        memcpy(target, input_data + i * sequence_length, sizeof(int64_t) * sequence_length);
        target += sequence_length;
      }
    }
  } else if (element_type == DataTypeImpl::GetType<float>()) {
    const float* input_data = input.Get<Tensor>().Data<float>();
    float* expanded_data = expanded.GetMutable<Tensor>()->MutableData<float>();
    float* target = expanded_data;
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < num_beams; j++) {
        memcpy(target, input_data + i * sequence_length, sizeof(float) * sequence_length);
        target += sequence_length;
      }
    }
  }

  return expanded;
}

Status CreateInputs(
    const Tensor* original_input_ids,
    int num_beams,
    int pad_token_id,
    gsl::span<int64_t>& sequence_lengths,
    AllocatorPtr alloactor,
    OrtValue& expanded_input_ids,
    OrtValue& expanded_position_ids,
    OrtValue& expanded_attention_mask) {
  const TensorShape& input_ids_shape = original_input_ids->Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int64_t>();

  // input_ids for subgraph is int64, so we need Cast input_ids from int32 to int64.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, input_ids);

  const int32_t* source = original_input_ids->Data<int32_t>();
  int64_t* target = input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++, source++, target++) {
      *target = static_cast<int64_t>(*source);
    }
  }

  OrtValue position_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, position_ids);

  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(mask_type, input_ids_shape, alloactor, attention_mask);

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  source = original_input_ids->Data<int32_t>();
  float* mask = mask_data;
  int64_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int64_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, source++, mask++, position++) {
      if (*source == pad_token_id) {
        *mask = 0.0f;
        *position = 0;
      } else {
        *mask = 1.0f;
        *position = abs_position;
        abs_position++;
      }
    }

    for (int k = 0; k < num_beams; k++) {
      sequence_lengths[SafeInt<gsl::index>(i) * num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length) for input_ids, position_ids and attention_mask
  // TODO: Try expand outputs after first subgraph call instead. That may get better performance, but more complex to implement.
  expanded_input_ids = ExpandInputs(input_ids, num_beams, alloactor);
  expanded_position_ids = ExpandInputs(position_ids, num_beams, alloactor);
  expanded_attention_mask = ExpandInputs(attention_mask, num_beams, alloactor);

  return Status::OK();
}

Status AddToFeeds(const IExecutionProvider* /*execution_provider*/,
                  OrtValue& input_ids,
                  OrtValue& position_ids,
                  OrtValue& attention_mask,
                  std::vector<OrtValue>& feeds,
                  IAllocatorUniquePtr<char>& /*buffer*/) {
  feeds.push_back(input_ids);
  feeds.push_back(position_ids);
  feeds.push_back(attention_mask);
  return Status::OK();
}

void InitBeamState(transformers::IBeamSearchState<float>* beam_state,
                   transformers::IBeamSearchCpuState<float>* cpu_state,
                   gsl::span<int64_t>& sequence_lengths,
                   int batch_size,
                   int num_beams,
                   gsl::span<const int64_t> input_ids_in_cpu,
                   int sequence_length,
                   int max_length,
                   void* /*stream*/) {
  memset(beam_state->beam_scores.data(), 0, beam_state->beam_scores.size_bytes());
  memset(beam_state->next_token_logits.data(), 0, beam_state->next_token_logits.size_bytes());
  memset(beam_state->next_token_scores.data(), 0, beam_state->next_token_scores.size_bytes());
  memset(beam_state->next_tokens.data(), 0, beam_state->next_tokens.size_bytes());
  memset(beam_state->next_indices.data(), 0, beam_state->next_indices.size_bytes());
  memset(beam_state->next_positions.data(), 0, beam_state->next_positions.size_bytes());

  // Initialize score of first beam of each group with 0 and the rest with -1e9.
  // This ensures that the beams in the same group don't produce same tokens every time.
  gsl::span<float>& beam_scores = beam_state->beam_scores;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 1; j < num_beams; j++) {
      beam_scores[SafeInt<gsl::index>(i) * num_beams + j] = -1e9;
    }
  }

  gsl::copy(sequence_lengths, beam_state->next_positions);

  memset(cpu_state->sequences_space.data(), 0, cpu_state->sequences_space.size_bytes());

  // Copy input_ids to sequences[0].
  gsl::span<int64_t> sequences_0 = cpu_state->sequences_space;
  int batch_beam_size = batch_size * num_beams;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<const int64_t> source = input_ids_in_cpu.subspan(SafeInt<gsl::index>(i) * sequence_length, static_cast<gsl::index>(sequence_length));
    gsl::span<int64_t> target = sequences_0.subspan(SafeInt<gsl::index>(i) * max_length, static_cast<gsl::index>(sequence_length));
    gsl::copy(source, target);
  }
}

Status ProcessLogits(const OrtValue& logits,                                        // logits output of subgraph
                     transformers::IBeamSearchState<float>* beam_state,             // state in device
                     transformers::IBeamSearchCpuState<float>* /*cpu_state*/,       // state in CPU
                     transformers::ISequences* sequences,                           // sequences
                     AllocatorPtr& allocator,                                       // default allocator
                     onnxruntime::concurrency::ThreadPool* thread_pool,             // thread pool (for CPU only)
                     transformers::ILogitsProcessorList<float>* logits_processors,  // logits processors
                     transformers::IBeamScorer<float>* beam_scorer,                 // beam scorer
                     const transformers::IBeamSearchParameters* parameters,         // parameters
                     int step,                                                      // iteration counter
                     void* stream,                                                  // cuda stream (for CUDA only)
                     const transformers::IConsoleDumper* dumper){                   // tensor dumper
#ifndef DEBUG_BEAM_SEARCH
  ORT_UNUSED_PARAMETER(dumper);
#endif

  int batch_size = parameters->batch_size;
  int num_beams = parameters->num_beams;
  int vocab_size = parameters->vocab_size;
  bool output_scores = parameters->output_scores;

  int batch_beam_size = batch_size * num_beams;
  const float* logits_data = logits.Get<Tensor>().Data<float>();

  // Logits has shape (batch_size * num_beams, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<float>& next_token_logits = beam_state->next_token_logits;
  if (input_length > 1) {
    const float* current_logits = logits_data + (input_length - 1) * vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const float> source(current_logits, vocab_size);
      gsl::span<float> target = next_token_logits.subspan(SafeInt<gsl::index>(i) * vocab_size, static_cast<gsl::index>(vocab_size));
      gsl::copy(source, target);
      current_logits += input_length * vocab_size;
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("logits", logits);
  dumper->Print("next_token_logits", next_token_logits.data(), batch_size, num_beams, vocab_size);
#endif

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  gsl::span<float>& next_token_scores = beam_state->next_token_scores;
  ORT_RETURN_IF_ERROR(SoftmaxCPU<float>(batch_beam_size,  // rows
                                        vocab_size,       // elements per row
                                        input_length > 1 ? next_token_logits.data() : logits_data,
                                        next_token_scores.data(),
                                        true,
                                        thread_pool));

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after softmax", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Apply all score processors that updates scores
  logits_processors->Process(sequences, next_token_scores, step);

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after logits processor", next_token_scores.data(), batch_size, num_beams, vocab_size);
#endif

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO: use thread pool to parrellel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < vocab_size; k++, offset++) {
        next_token_scores[offset] += beam_state->beam_scores[batch_beam_index];
      }
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_token_scores after adding beam_scores", next_token_scores.data(), batch_size, num_beams, vocab_size);
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
  auto element_type = DataTypeImpl::GetType<float>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(), next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr int axis = 1;
  const unsigned top_k = static_cast<unsigned>(2 * num_beams);
  constexpr bool largest = true;
  constexpr bool sorted = true;  // results returned in sorted order.

  std::unique_ptr<Tensor> topk_scores;
  std::unique_ptr<Tensor> topk_indices;
  ORT_RETURN_IF_ERROR(TopK(&input, axis, top_k, largest, sorted, allocator, stream, thread_pool, topk_scores, topk_indices));

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("topk_scores", *(topk_scores.get()));
  dumper->Print("topk_indices", *(topk_indices.get()));
#endif

  // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
  //   next_indices = (next_tokens / vocab_size).long()
  //   next_tokens = next_tokens % vocab_size
  gsl::span<const int64_t> next_token_indices = topk_indices->DataAsSpan<int64_t>();
  offset = 0;
  for (int i = 0; i < batch_size; i++) {
    for (unsigned int j = 0; j < top_k; j++, offset++) {
      beam_state->next_indices[offset] = next_token_indices[offset] / vocab_size;
      beam_state->next_tokens[offset] = next_token_indices[offset] % vocab_size;
    }
  }

  gsl::span<const float> next_scores = topk_scores->DataAsSpan<float>();
  gsl::span<const int64_t> next_tokens(beam_state->next_tokens.data(), beam_state->next_tokens.size());
  gsl::span<const int64_t> next_indices(beam_state->next_indices.data(), beam_state->next_indices.size());

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("next_scores before scorer", next_scores.data(), batch_size, top_k);
  dumper->Print("next_tokens before scorer", next_tokens.data(), batch_size, top_k);
  dumper->Print("next_indices before scorer", next_indices.data(), batch_size, top_k);
#endif

  beam_scorer->Process(
      sequences,
      next_scores,
      next_tokens,
      next_indices);

  return Status::OK();
}

Status DeviceCopy(gsl::span<float> target, gsl::span<const float> source, void* /*stream*/, int /*copyDirection*/) {
  gsl::copy(source, target);
  return Status::OK();
}

void PickPastState(const std::vector<OrtValue>& last_outputs,
                   std::vector<OrtValue>& next_inputs,
                   gsl::span<const int64_t>& beam_indices,
                   AllocatorPtr allocator,
                   void* /*stream*/,
                   const transformers::IConsoleDumper* dumper) {
#ifndef DEBUG_BEAM_SEARCH
  ORT_UNUSED_PARAMETER(dumper);
#endif
                     
  for (size_t i = 1; i < last_outputs.size(); ++i) {
    const OrtValue& present = last_outputs[i];  // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();

    // Create a tensor with same shape.
    // TODO: allocate one buffer for all layers
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<float>();
    Tensor::InitOrtValue(past_type, past_shape, allocator, past);

    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    gsl::span<float> past_span = gsl::make_span<float>(past.GetMutable<Tensor>()->MutableData<float>(), past_shape.Size());
    gsl::span<const float> present_span = gsl::make_span<const float>(present.Get<Tensor>().Data<float>(), past_shape.Size());
    for (gsl::index j = 0; j < beam_indices.length(); j++) {
      int64_t beam_index = beam_indices[j];
      gsl::span<const float> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const float> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      gsl::span<float> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<float> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      gsl::copy(present_key, past_key);
      gsl::copy(present_value, past_value);

#ifdef DEBUG_BEAM_SEARCH
      // if (i == 1)  // only dump past_0
      // {
      //   dumper->Print("past_key of beam", static_cast<int>(j), true);
      //   dumper->Print(nullptr, past_key.data(), 1, static_cast<int>(block_size_per_beam));
      //   dumper->Print("past_value of beam", static_cast<int>(j), true);
      //   dumper->Print(nullptr, past_value.data(), 1, static_cast<int>(block_size_per_beam));
      // }
#endif
    }

    next_inputs[i + 2] = past;
  }
}

Status UpdateFeeds(
    AllocatorPtr allocator,
    void* stream,
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices,
    int num_beams,
    const transformers::IConsoleDumper* dumper) {
  // last_outputs: logits, present_0, present_1, ...
  // next_inputs: input_ids, position_id, attention_mask, past_0, past_1

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.length());
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto element_type = DataTypeImpl::GetType<int64_t>();
  OrtValue input_ids;
  // TODO: Reuse buffer for input_ids to reduce memory allocation.
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator, input_ids);
  int64_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = beam_next_tokens[i];
  }
  next_inputs[0] = input_ids;

  // Update position IDs
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    position_data[i]++;
  }
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const float* old_mask_data = old_mask.Get<Tensor>().Data<float>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(mask_type, mask_shape, allocator, attention_mask);
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1.0f;
  }
  next_inputs[2] = attention_mask;

#ifdef DEBUG_BEAM_SEARCH
  dumper->Print("input_ids", input_ids);
  dumper->Print("position_ids", position_ids);
  dumper->Print("attention_mask", attention_mask);
#endif

  // Update past state
  if (num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (size_t i = 1; i < last_outputs.size(); ++i) {
      next_inputs[i + 2] = last_outputs[i];
    }
  } else {
    PickPastState(last_outputs, next_inputs, beam_indices, allocator, stream, dumper);
  }
  return Status::OK();
}

}  // namespace BeamSearchCpuDeviceHelper
}  // namespace contrib
}  // namespace onnxruntime