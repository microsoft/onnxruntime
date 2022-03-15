// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <assert.h>
#include <functional>
#include "core/common/safeint.h"
#include "core/providers/cpu/math/top_k.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/allocator.h"
#include "core/framework/framework_common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/session_options.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "gsl/gsl"
#include "beam_search.h"
#include "logits_processor.h"
#include "sequences.h"
#include "dump_tensor.h"
#include "beam_search_scorer.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BeamSearch,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      transformers::BeamSearch);

REGISTER_KERNEL_TYPED(float)

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
            bool output_scores) {
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;
    this->next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size);
    this->next_token_scores = AllocateBuffer<float>(allocator, next_token_scores_buffer_, next_token_size);

    this->next_tokens = AllocateBuffer<int32_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    this->next_indices = AllocateBuffer<int32_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size);

    this->next_positions = AllocateBuffer<int32_t>(allocator, next_positions_buffer_, batch_beam_size);

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

  void Init(AllocatorPtr allocator, size_t batch_beam_size, int max_length, bool is_cuda) {
    this->sequence_lengths = AllocateBuffer<int32_t>(allocator, sequence_lengths_buffer_, batch_beam_size);
    this->sequences_space = AllocateBuffer<int32_t>(allocator, sequences_space_buffer_, SafeInt<size_t>(2) * batch_beam_size * max_length);

    if (is_cuda) {
      // buffers used by CUDA operator but not by CPU operator.
      this->topk_scores = AllocateBuffer<float>(allocator, topk_scores_buffer_, 2 * batch_beam_size);
      this->topk_tokens = AllocateBuffer<int32_t>(allocator, topk_tokens_buffer_, 2 * batch_beam_size);
      this->topk_indices = AllocateBuffer<int32_t>(allocator, topk_indices_buffer_, 2 * batch_beam_size);
      this->final_beam_scores = AllocateBuffer<float>(allocator, final_beam_scores_buffer_, batch_beam_size);
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

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OpKernelContextInternal& context,
                 const SessionState& session_state,
                 GptSubgraph& gpt_subgraph,
                 concurrency::ThreadPool* thread_pool,
                 void* cuda_stream,
                 IConsoleDumper* cuda_dumper,
                 BeamSearchParameters& params,
                 const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
                 const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
                 const BeamSearchDeviceHelper::TopkFunc& topk_func,
                 const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                 const BeamSearchDeviceHelper::InitBeamStateFunc<T>& init_beam_state_func,
                 const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func,
                 const BeamSearchDeviceHelper::UpdateFeedsFunc<T>& update_feeds_func)
      : context_(context),
        session_state_(session_state),
        gpt_subgraph_(gpt_subgraph),
        thread_pool_(thread_pool),
        implicit_inputs_(context_.GetImplicitInputs()),
        cuda_stream_(cuda_stream),
        cuda_dumper_(cuda_dumper),
        parameters_(&params),
        cpu_allocator_(nullptr),
        temp_space_allocator_(nullptr),
        create_inputs_func_(create_inputs_func),
        add_to_feeds_func_(add_to_feeds_func),
        topk_func_(topk_func),
        process_logits_func_(process_logits_func),
        init_beam_state_func_(init_beam_state_func),
        device_copy_func_(device_copy_func),
        update_feeds_func_(update_feeds_func) {
    parameters_->ParseFromInputs(&context);

    cpu_allocator_ = session_state.GetExecutionProviders()
                         .Get(onnxruntime::kCpuExecutionProvider)
                         ->GetAllocator(0, OrtMemTypeDefault);
  }

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize();

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager& feeds_fetches_manager);

 private:
  bool IsCuda() const { return cuda_stream_ != nullptr; }

  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context);

  // Prepare the inputs for first inference of subgraph
  Status CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer);

  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      OrtValue& position_ids,
      gsl::span<const int32_t> beam_next_tokens,
      gsl::span<const int32_t> beam_indices);

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

  const IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OpKernelContextInternal& context_;

  const SessionState& session_state_;

  GptSubgraph& gpt_subgraph_;

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
  BeamSearchDeviceHelper::CreateInputsFunc create_inputs_func_;
  BeamSearchDeviceHelper::AddToFeedsFunc add_to_feeds_func_;
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::InitBeamStateFunc<T> init_beam_state_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
  BeamSearchDeviceHelper::UpdateFeedsFunc<T> update_feeds_func_;
};

void BeamSearch::Init(const OpKernelInfo& info) {
  // Make sure the decoder attribute was present even though we don't need it here.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("decoder", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);

  parameters_.ParseFromAttributes(info);
}

Status BeamSearch::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                              const std::string& attribute_name,
                                              const SessionState& subgraph_session_state) {
  ORT_ENFORCE(gpt_subgraph_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  // TODO: handle another subgraph with attribute name "encoder_decode_init"
  if (attribute_name == "decoder") {
    const auto& node = Node();
    gpt_subgraph_ = std::make_unique<GptSubgraph>(node, attribute_name, subgraph_session_state.GetGraphViewer());
    ORT_RETURN_IF_ERROR(gpt_subgraph_->Setup(session_state, subgraph_session_state));
    feeds_fetches_manager_ = gpt_subgraph_->GetFeedsFetchesManager();
    parameters_.SetSubgraphParameters(gpt_subgraph_->vocab_size,
                                      gpt_subgraph_->num_heads,
                                      gpt_subgraph_->head_size,
                                      gpt_subgraph_->num_layers);
  }
  return Status::OK();
}

Status BeamSearch::Compute(OpKernelContext* ctx) const {
  if (parameters_.model_type != 0) {
    // TODO: support encoder decoder model like T5
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Support of 'model_type' != 0 is not implemented");
  }

  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("decoder");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'decoder' attribute.");
  ORT_ENFORCE(feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  BeamSearchParameters parameters = parameters_;  // make a copy since we will update the parameters based on inputs later

  // Subgraph has constraint that the output is either float or float16
  if (!gpt_subgraph_->IsOutputFloat16()) {
    BeamSearchImpl<float> impl{*ctx_internal, *session_state, *gpt_subgraph_, thread_pool, cuda_stream_, dumper_, parameters,
                               create_inputs_func_ ? create_inputs_func_ : BeamSearchCpuDeviceHelper::CreateInputs,
                               add_to_feeds_func_ ? add_to_feeds_func_ : BeamSearchCpuDeviceHelper::AddToFeeds,
                               topk_func_ ? topk_func_ : BeamSearchCpuDeviceHelper::TopK,
                               process_logits_func_ ? process_logits_func_ : BeamSearchCpuDeviceHelper::ProcessLogits<float>,
                               init_beam_state_func_ ? init_beam_state_func_ : BeamSearchCpuDeviceHelper::InitBeamState<float>,
                               device_copy_func_ ? device_copy_func_ : BeamSearchCpuDeviceHelper::DeviceCopy<float>,
                               update_feeds_func_ ? update_feeds_func_ : BeamSearchCpuDeviceHelper::UpdateFeeds<float>};
    ORT_RETURN_IF_ERROR(impl.Initialize());

    return impl.Execute(*feeds_fetches_manager_);
  } else {
    BeamSearchImpl<MLFloat16> impl{*ctx_internal, *session_state, *gpt_subgraph_, thread_pool, cuda_stream_, dumper_, parameters,
                                   create_inputs_func_ ? create_inputs_func_ : BeamSearchCpuDeviceHelper::CreateInputs,
                                   add_to_feeds_func_ ? add_to_feeds_func_ : BeamSearchCpuDeviceHelper::AddToFeeds,
                                   topk_func_ ? topk_func_ : BeamSearchCpuDeviceHelper::TopK,
                                   process_logits_fp16_func_,
                                   init_beam_state_fp16_func_,
                                   device_copy_func_,
                                   update_feeds_fp16_func_};
    ORT_RETURN_IF_ERROR(impl.Initialize());

    return impl.Execute(*feeds_fetches_manager_);
  }
}

template <typename T>
Status BeamSearchImpl<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)
  //   vocab_mask : (vocab_size) or nullptr

  const Tensor* input_ids = context.Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  if (dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input_ids' is expected to have 2 dimensions, got ",
                           dims.size());
  }

  const Tensor* vocab_mask = context.Input<Tensor>(8);
  if (vocab_mask != nullptr) {  // vocab_mask is optional
    const auto& vocab_mask_dims = vocab_mask->Shape().GetDims();
    if (vocab_mask_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'vocab_mask' is expected to have 1 dimension, got ",
                             vocab_mask_dims.size());
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(vocab_mask_dims[0]) != parameters_->vocab_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'vocab_mask' shape does not match with vocab_size, got ",
                             vocab_mask_dims[0]);
    }

    // store vocab mask in parameters.
    parameters_->vocab_mask = vocab_mask->DataAsSpan<int32_t>();
  }

  const Tensor* prefix_vocab_mask = context.Input<Tensor>(9);
  if (prefix_vocab_mask != nullptr) {
    // prefix_vocab_mask is optional
    const auto& vocab_mask_dims = prefix_vocab_mask->Shape().GetDims();
    if (vocab_mask_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'prefix_vocab_mask' is expected to have 2 dimensions, got ",
                             vocab_mask_dims.size());
    }

    // prefix_vocab_mask first dimension should be same as the first dimension of input_ids
    if (static_cast<int>(vocab_mask_dims[0]) != static_cast<int>(dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input_ids and prefix_vocab_mask must have the same batch_size");
    }

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
    if (static_cast<int>(vocab_mask_dims[1]) != parameters_->vocab_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'prefix_vocab_mask' shape does not match with vocab_size, got ",
                             vocab_mask_dims[0]);
    }

    // store prefix vocab mask in parameters.
    parameters_->prefix_vocab_mask = prefix_vocab_mask->DataAsSpan<int32_t>();
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::Initialize() {
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

  ORT_RETURN_IF(parameters_->num_return_sequences > parameters_->num_beams, "'num_return_sequences' has to be smaller or equal to 'num_beams'.");

  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::CreateInitialFeeds(gsl::span<int32_t>& sequence_lengths, OrtValue& expanded_input_ids, std::vector<OrtValue>& feeds, IAllocatorUniquePtr<char>& buffer) {
  const OrtValue* input_ids_value = context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  return gpt_subgraph_.CreateInitialFeeds(input_ids, implicit_inputs_, parameters_->num_beams, parameters_->pad_token_id, sequence_lengths, expanded_input_ids, feeds, create_inputs_func_, add_to_feeds_func_, buffer);
}

template <typename T>
Status BeamSearchImpl<T>::ProcessLogits(
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
Status BeamSearchImpl<T>::GenerateNextToken(
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
  ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores, beam_scores, cuda_stream_, DeviceCopyDirection::hostToDevice));

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("beam_scores after scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_next_tokens after scorer", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_indices after scorer", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif
  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    OrtValue& position_ids,
    gsl::span<const int32_t> beam_next_tokens,
    gsl::span<const int32_t> beam_indices) {
  return update_feeds_func_(temp_space_allocator_, cuda_stream_, last_outputs, next_inputs, current_length, position_ids,
                            beam_next_tokens, beam_indices, parameters_->num_beams, GetConsoleDumper());
}

template <typename T>
Status BeamSearchImpl<T>::Execute(const FeedsFetchesManager& feeds_fetches_manager) {
  auto status = Status::OK();
  int64_t sequences_dims[] = {parameters_->batch_size, parameters_->num_return_sequences, parameters_->max_length};
  TensorShape sequences_shape(&sequences_dims[0], sizeof(sequences_dims) / sizeof(sequences_dims[0]));
  Tensor* output_sequences = context_.Output(0, sequences_shape);

  int64_t sequences_scores_dims[] = {parameters_->batch_size, parameters_->num_return_sequences};
  TensorShape sequences_scores_shape(&sequences_scores_dims[0], sizeof(sequences_scores_dims) / sizeof(sequences_scores_dims[0]));
  Tensor* output_sequences_scores = context_.Output(1, sequences_scores_shape);

  int64_t scores_dims[] = {
      static_cast<int64_t>(parameters_->max_length) - static_cast<int64_t>(parameters_->sequence_length),
      parameters_->batch_size, parameters_->num_beams, parameters_->vocab_size};
  TensorShape scores_shape(&scores_dims[0], sizeof(scores_dims) / sizeof(scores_dims[0]));
  Tensor* output_scores = context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  parameters_->output_scores = (output_scores != nullptr);

  std::vector<OrtValue> feeds;
  // TODO: allocate fetches. use ping-pong buffers for past state.
  std::vector<OrtValue> fetches;

  // Initialize resources
  onnxruntime::OrtStlAllocator<HypothesisScore> hypothesis_score_allocator(cpu_allocator_);
  onnxruntime::OrtStlAllocator<BeamHypotheses> beam_hyps_allocator(cpu_allocator_);
  beam_scorer_ = std::make_unique<BeamSearchScorer>(static_cast<size_t>(parameters_->batch_size),
                                                    static_cast<size_t>(parameters_->num_beams),
                                                    static_cast<size_t>(parameters_->max_length),
                                                    parameters_->length_penalty,
                                                    parameters_->early_stopping,
                                                    static_cast<size_t>(parameters_->num_return_sequences),
                                                    parameters_->pad_token_id,
                                                    parameters_->eos_token_id,
                                                    hypothesis_score_allocator,
                                                    beam_hyps_allocator);
  beam_scorer_->Initialize(cpu_allocator_, parameters_->sequence_length);

  BeamSearchCpuState cpu_state;
  cpu_state.Init(cpu_allocator_, static_cast<size_t>(parameters_->BatchBeamSize()), parameters_->max_length, IsCuda());

  // buffer in GPU for input_ids, position_ids and attention_mask
  // size_t buffer_bytes = SafeInt<size_t>(sizeof(int32_t) + sizeof(int32_t) + sizeof(int32_t)) * parameters_->batch_size * parameters_->num_beams * parameters_->sequence_length;
  // IAllocatorUniquePtr<char> buffer = gpt_subgraph_.GetProvider()->GetScratchBuffer<char>(buffer_bytes);
  IAllocatorUniquePtr<char> buffer;
  OrtValue expanded_input_ids_in_cpu;
  ORT_RETURN_IF_ERROR(CreateInitialFeeds(cpu_state.sequence_lengths, expanded_input_ids_in_cpu, feeds, buffer));

  BeamSearchState<T> beam_state;
  beam_state.Init(temp_space_allocator_,
                  parameters_->batch_size,
                  parameters_->num_beams,
                  parameters_->vocab_size,
                  parameters_->sequence_length,
                  parameters_->max_length,
                  parameters_->output_scores);

  cpu_state.sequences.Init(cpu_state.sequences_space,
                           parameters_->BatchBeamSize(),
                           parameters_->sequence_length,
                           parameters_->max_length);

  gsl::span<const int32_t> input_ids = expanded_input_ids_in_cpu.Get<Tensor>().DataAsSpan<int32_t>();
  init_beam_state_func_(&beam_state,
                        &cpu_state,
                        cpu_state.sequence_lengths,
                        parameters_->batch_size,
                        parameters_->num_beams,
                        input_ids,
                        parameters_->sequence_length,
                        parameters_->max_length,
                        cuda_stream_);

#ifdef DEBUG_BEAM_SEARCH
  const IConsoleDumper* dumper = GetConsoleDumper();
  dumper->Print("input_ids", feeds[0]);
  dumper->Print("position_ids", feeds[1]);
  dumper->Print("attention_mask", feeds[2]);
#endif

  // position ids for all iterations except the first. It uses memory buffer owned by next_positions.
  OrtValue position_ids;
  int64_t dims[] = {parameters_->BatchBeamSize(), 1};
  TensorShape shape(&dims[0], 2);
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), shape, beam_state.next_positions.data(), temp_space_allocator_->Info(), position_ids);

  int current_length = parameters_->sequence_length;
  int iteration_counter = 0;
  while (current_length < parameters_->max_length) {
    iteration_counter++;
#ifdef DEBUG_BEAM_SEARCH
    auto cur_len = std::to_string(current_length);
    dumper->Print("***CurrentLength", cur_len, true);
#endif

    status = utils::ExecuteSubgraph(session_state_, feeds_fetches_manager, feeds, fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = fetches[0];
    gsl::span<int32_t> beam_next_tokens;
    gsl::span<int32_t> beam_indices;
    ORT_RETURN_IF_ERROR(GenerateNextToken(logits, beam_next_tokens, beam_indices, beam_state, cpu_state, iteration_counter));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters_->max_length) {
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      position_ids,
                                      beam_next_tokens.as_span<const int32_t>(),
                                      beam_indices.as_span<const int32_t>()));
    }
    fetches.clear();
  }

  gsl::span<const float> final_beam_scores(beam_state.beam_scores.data(), beam_state.beam_scores.size());
  if (IsCuda()) {
    ORT_RETURN_IF_ERROR(device_copy_func_(cpu_state.final_beam_scores, final_beam_scores, nullptr, DeviceCopyDirection::deviceToHost));
    final_beam_scores = gsl::make_span<const float>(cpu_state.final_beam_scores.data(), cpu_state.final_beam_scores.size());
  }

  beam_scorer_->Finalize(&(cpu_state.sequences),
                         final_beam_scores,
                         output_sequences,
                         output_sequences_scores);

  // Output per token scores
  if (output_scores != nullptr) {
    gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
    gsl::span<const float> source = gsl::span<const float>(beam_state.scores.data(), beam_state.scores.size());
    assert(target.length() == source.length());
    ORT_RETURN_IF_ERROR(device_copy_func_(target, source, nullptr, DeviceCopyDirection::deviceToDevice));
  }

  return status;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
