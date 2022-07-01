// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/beam_search_shared.h"

namespace onnxruntime {
namespace contrib {

namespace transformers {

template <typename T>
struct GreedySearchState : public IGreedySearchState<T> {
  Sequences sequences;

  void Init(AllocatorPtr cpu_allocator,
            AllocatorPtr allocator,
            int batch_size,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool is_cuda) {
    // below buffers are on cpu
    this->sequences_space = AllocateBuffer<int32_t>(cpu_allocator, sequences_space_buffer_, SafeInt<size_t>(2) * batch_size * max_length);
    memset(this->sequences_space.data(), 0, this->sequences_space.size_bytes());
    this->sequences.Init(this->sequences_space, static_cast<int>(batch_size), sequence_length, max_length);

    this->sequence_lengths = AllocateBuffer<int32_t>(cpu_allocator, sequence_lengths_buffer_, batch_size);
    this->eos_meet = AllocateBuffer<bool>(cpu_allocator, eos_meet_buffer_, batch_size);
    memset(this->eos_meet.data(), 0, this->eos_meet.size_bytes());

    this->next_positions = AllocateBuffer<int32_t>(cpu_allocator, next_positions_buffer_, batch_size);

    // below buffers are on cpu or cuda
    size_t next_token_size = SafeInt<size_t>(batch_size) * vocab_size;
    this->next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size);
    this->next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size);
    this->next_tokens = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(batch_size));

    if (is_cuda) {
      // buffers used by CUDA operator but not by CPU operator.
      ORT_UNUSED_PARAMETER(is_cuda);
    }
  }

  void SetSequence(gsl::span<const int32_t> input_ids_in_cpu, size_t batch_beam_size, int max_length, int sequence_length) {
    gsl::span<int32_t> sequences_0 = this->sequences_space;
    for (size_t i = 0; i < batch_beam_size; i++) {
      for (int j = 0; j < sequence_length; j++) {
        sequences_0[SafeInt<gsl::index>(i) * max_length + j] = static_cast<int32_t>(input_ids_in_cpu[SafeInt<gsl::index>(i) * sequence_length + j]);
      }
    }
  }

 private:
  BufferUniquePtr sequences_space_buffer_;
  BufferUniquePtr sequence_lengths_buffer_;
  BufferUniquePtr next_token_logits_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr eos_meet_buffer_;
};

// Base class of gready search implementation that is common for both GPT-2 and Bart/T5.
template <typename T>
class GreedySearchBase {
 public:
  GreedySearchBase(OpKernelContextInternal& context,
                   const SessionState& decoder_session_state,
                   concurrency::ThreadPool* thread_pool,
                   void* cuda_stream,
                   IConsoleDumper* cuda_dumper,
                   GreedySearchParameters& params,
                   const BeamSearchDeviceHelper::TopkFunc& topk_func,
                   const BeamSearchDeviceHelper::GreedySearchProcessLogitsFunc<T>& process_logits_func,
                   const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func)
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
        device_copy_func_(device_copy_func) {
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
                           gsl::span<int32_t>& next_tokens,
                           GreedySearchState<T>& greedy_state,
                           int counter,
                           int eos_token_id);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       GreedySearchState<T>& greedy_state,
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

  GreedySearchParameters* parameters_;

  LogitsProcessorList logits_processors_;

  //std::unique_ptr<BeamSearchScorer> beam_scorer_;

  AllocatorPtr cpu_allocator_;
  AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::GreedySearchProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
};

template <typename T>
Status GreedySearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)

  const Tensor* input_ids = context.Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  if (dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input_ids' is expected to have 2 dimensions, got ",
                           dims.size());
  }

  return Status::OK();
}

template <typename T>
Status GreedySearchBase<T>::Initialize() {
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

  CHECK_SCALAR_INPUT(max_length, 1, true);

  CHECK_SCALAR_INPUT(min_length, 2, false);

  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  // Greedy search
  parameters_->no_repeat_ngram_size = 0;

  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T>
Status GreedySearchBase<T>::ProcessLogits(
    const OrtValue& logits,
    GreedySearchState<T>& greedy_state,
    AllocatorPtr& allocator,
    int counter) {
  return process_logits_func_(logits, &greedy_state, &(greedy_state.sequences), allocator,
                              thread_pool_, &logits_processors_,
                              parameters_, counter, cuda_stream_, GetConsoleDumper());
}

template <typename T>
Status GreedySearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& next_tokens,
    GreedySearchState<T>& greedy_state,
    int counter,
    int eos_token_id) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, greedy_state, temp_space_allocator_, counter));

  next_tokens = greedy_state.next_tokens;

  greedy_state.sequences.AppendNextTokenToSequences(next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  greedy_state.sequences.PrintSequences(&cpu_dumper_);
#endif

  gsl::span<int32_t> sequence_lengths = greedy_state.sequence_lengths;
  gsl::span<bool>& eos_meet = greedy_state.eos_meet;

  for (size_t batch_id = 0; batch_id < next_tokens.size(); ++batch_id) {
    if (next_tokens[batch_id] == eos_token_id) {
      eos_meet[batch_id] = true;
      sequence_lengths[batch_id] = greedy_state.sequences.GetSequenceLength();
    }
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
