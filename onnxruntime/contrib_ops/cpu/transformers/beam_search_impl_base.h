// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <vector>
#include <utility>

namespace onnxruntime {
namespace contrib {

namespace transformers {

template <typename T>
gsl::span<T> AllocateBuffer(AllocatorPtr allocator,
                            BufferUniquePtr& buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {
  size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  void* data = allocator->Alloc(bytes);
  BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  buffer = std::move(temp_buffer);
  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = gsl::make_span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename T>
struct BeamSearchState : public IBeamSearchState<T> {
  void Init(AllocatorPtr allocator,
            int batch_size,
            int num_beams,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool output_scores,
            bool use_position) {
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;
    this->next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size);
    this->next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size);

    this->next_tokens = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    this->next_indices = AllocateBuffer<int32_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    if (use_position) {
      this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_beam_size);
    }

    this->beam_scores = AllocateBuffer<float>(allocator, beam_scores_buffer_, batch_beam_size);

    if (output_scores) {
      size_t elements = SafeInt<size_t>(max_length - sequence_length) * batch_size * num_beams * vocab_size;
      this->scores = AllocateBuffer<float>(allocator, scores_buffer_, elements);
      this->remaining_scores = this->scores;
    }
  }

 private:
  BufferUniquePtr next_token_logits_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_indices_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr beam_scores_buffer_;
  BufferUniquePtr scores_buffer_;
};

struct BeamSearchCpuState : public IBeamSearchCpuState {
  Sequences sequences;

  void Init(AllocatorPtr allocator, size_t batch_beam_size, int max_length, int sequence_length, bool is_cuda) {
    this->sequence_lengths = AllocateBuffer<int32_t>(allocator, sequence_lengths_buffer_, batch_beam_size);

    size_t sequences_bytes = SafeInt<size_t>(2) * batch_beam_size * max_length;
    this->sequences_space = AllocateBuffer<int32_t>(allocator, sequences_space_buffer_, sequences_bytes);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());

    if (is_cuda) {
      // buffers used by CUDA operator but not by CPU operator.
      this->topk_scores = AllocateBuffer<float>(allocator, topk_scores_buffer_, 2 * batch_beam_size);
      this->topk_tokens = AllocateBuffer<int32_t>(allocator, topk_tokens_buffer_, 2 * batch_beam_size);
      this->topk_indices = AllocateBuffer<int32_t>(allocator, topk_indices_buffer_, 2 * batch_beam_size);
      this->final_beam_scores = AllocateBuffer<float>(allocator, final_beam_scores_buffer_, batch_beam_size);
    }

    this->sequences.Init(this->sequences_space, static_cast<int>(batch_beam_size), sequence_length, max_length);
  }

  // Copy input_ids to sequences[0]
  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu,
                   size_t batch_beam_size,
                   int max_length,
                   int sequence_length) {
    gsl::span<int32_t> sequences_0 = sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        const size_t index = SafeInt<gsl::index>(i) * max_length + j;
        sequences_0[index] = input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j];
      }
    }
  }

 private:
  BufferUniquePtr final_beam_scores_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr topk_scores_buffer_;
  BufferUniquePtr topk_tokens_buffer_;
  BufferUniquePtr topk_indices_buffer_;
  BufferUniquePtr sequences_space_buffer_;
};

// Base class of beam search implementation that is common for both GPT-2 and T5.
template <typename T>
class BeamSearchBase {
 public:
  BeamSearchBase(OpKernelContextInternal& context,
                 const SessionState& decoder_session_state,
                 concurrency::ThreadPool* thread_pool,
                 void* cuda_stream,
                 IConsoleDumper* cuda_dumper,
                 BeamSearchParameters& params,
                 const BeamSearchDeviceHelper::TopkFunc& topk_func,
                 const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 const BeamSearchDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func)
      : context_(context),
        decoder_session_state_(decoder_session_state),
        thread_pool_(thread_pool),
        implicit_inputs_(context_.GetImplicitInputs()),
        cuda_stream_(cuda_stream),
        cuda_dumper_(cuda_dumper),
        parameters_(&params),
        cpu_allocator_(nullptr),
        temp_space_allocator_(nullptr),
        topk_func_(topk_func),
        process_logits_func_(process_logits_func),
        device_copy_func_(device_copy_func),
        device_copy_int32_func_(device_copy_int32_func) {
    parameters_->ParseFromInputs(&context);

    cpu_allocator_ = decoder_session_state.GetExecutionProviders()
                         .Get(onnxruntime::kCpuExecutionProvider)
                         ->GetAllocator(0, OrtMemTypeDefault);
  }

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize();

  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context);

 protected:
  // Process logits and append next tokens to sequences.
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int32_t>& beam_next_tokens,
                           gsl::span<int32_t>& beam_indices,
                           BeamSearchState<T>& beam_state,
                           BeamSearchCpuState& cpu_state,
                           int counter);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       BeamSearchCpuState& cpu_state,
                       AllocatorPtr& allocator,
                       int counter);

  bool IsCuda() const { return cuda_stream_ != nullptr; }

  const IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OpKernelContextInternal& context_;

  const SessionState& decoder_session_state_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  void* cuda_stream_;

  IConsoleDumper* cuda_dumper_;
  CpuTensorConsoleDumper cpu_dumper_;

  BeamSearchParameters* parameters_;

  LogitsProcessorList logits_processors_;

  std::unique_ptr<BeamSearchScorer> beam_scorer_;

  AllocatorPtr cpu_allocator_;
  AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<int32_t> device_copy_int32_func_;
};

template <typename T>
Status BeamSearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)
  //   vocab_mask : (vocab_size) or nullptr

  const Tensor* input_ids = context.Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  if (dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input_ids' is expected to have 2 dimensions, got ", dims.size());
  }

  const Tensor* vocab_mask = context.Input<Tensor>(8);
  if (vocab_mask != nullptr) {  // vocab_mask is optional
    const auto& vocab_mask_dims = vocab_mask->Shape().GetDims();
    if (vocab_mask_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'vocab_mask' is expected to have 1 dimension, got ", vocab_mask_dims.size());
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(vocab_mask_dims[0]) != parameters_->vocab_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'vocab_mask' shape does not match with vocab_size, got ", vocab_mask_dims[0]);
    }

    // store vocab mask in parameters.
    parameters_->vocab_mask = vocab_mask->DataAsSpan<int32_t>();
  }

  const Tensor* prefix_vocab_mask = context.Input<Tensor>(9);
  if (prefix_vocab_mask != nullptr) {
    // prefix_vocab_mask is optional
    const auto& vocab_mask_dims = prefix_vocab_mask->Shape().GetDims();
    if (vocab_mask_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'prefix_vocab_mask' is expected to be 2 dimensions, got ", vocab_mask_dims.size());
    }

    // prefix_vocab_mask first dimension should be same as the first dimension of input_ids
    if (static_cast<int>(vocab_mask_dims[0]) != static_cast<int>(dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "input_ids and prefix_vocab_mask must have the same batch_size");
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(vocab_mask_dims[1]) != parameters_->vocab_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'prefix_vocab_mask' shape[1] shall be vocab_size, got ", vocab_mask_dims[1]);
    }

    // store prefix vocab mask in parameters.
    parameters_->prefix_vocab_mask = prefix_vocab_mask->DataAsSpan<int32_t>();
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchBase<T>::Initialize() {
  ORT_RETURN_IF_ERROR(context_.GetTempSpaceAllocator(&temp_space_allocator_));

#define CHECK_SCALAR_INPUT(name, index, required)                                                                 \
  auto* name##_tensor = context_.Input<Tensor>(index);                                                            \
  if (name##_tensor) {                                                                                            \
    if (!name##_tensor->Shape().IsScalar()) {                                                                     \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'BeamSearch' input " #name " should be a scalar. Got shape of ", \
                             name##_tensor->Shape());                                                             \
    }                                                                                                             \
  } else if (required) {                                                                                          \
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'BeamSearch' input " #name " is required");                        \
  }

  CHECK_SCALAR_INPUT(min_length, 1, false);

  CHECK_SCALAR_INPUT(max_length, 2, true);

  CHECK_SCALAR_INPUT(num_beams, 3, true);

  CHECK_SCALAR_INPUT(num_return_sequences, 4, true);

  CHECK_SCALAR_INPUT(temperature, 5, true);

  CHECK_SCALAR_INPUT(length_penalty, 6, true);

  ORT_RETURN_IF(parameters_->num_return_sequences > parameters_->num_beams,
                "'num_return_sequences' has to be smaller or equal to 'num_beams'.");

  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchBase<T>::ProcessLogits(
    const OrtValue& logits,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    AllocatorPtr& allocator,
    int counter) {
  return process_logits_func_(logits, &beam_state, &cpu_state, &(cpu_state.sequences), allocator,
                              thread_pool_, &logits_processors_, beam_scorer_.get(),
                              parameters_, counter, cuda_stream_, GetConsoleDumper());
}

template <typename T>
Status BeamSearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& beam_next_tokens,
    gsl::span<int32_t>& beam_indices,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, cpu_state, temp_space_allocator_, counter));

  gsl::span<float>& beam_scores = beam_scorer_->GetNextScores();
  // It is optional to clone beam_scores. Change it to use same buffer also works for CPU:
  //    beam_state.beam_scores = beam_scores
  // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
  ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores,
                                        beam_scores,
                                        cuda_stream_,
                                        DeviceCopyDirection::hostToDevice));

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("beam_scores from scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_next_tokens", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_indices", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif
  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
