// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/math/top_k.h"
#include "core/framework/allocator.h"
#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/session_options.h"
#include "core/framework/TensorSeq.h"
#include "gsl/gsl"
#include "core/providers/cpu/math/softmax_shared.h"
#include "beam_search.h"
#include "dump_tensor.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

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
      transformers::BeamSearch<T>);

REGISTER_KERNEL_TYPED(float)

namespace transformers {

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OpKernelContextInternal& context,
                 const SessionState& session_state,
                 GptSubgraph& gpt_subgraph,
                 concurrency::ThreadPool* thread_pool,
                 void* stream,
                 BeamSearchParameters& params);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& cached_ffm);

 private:
  Status CheckInputs(const OpKernelContextInternal& context);

  // Prepare the inputs for first inference of subgraph
  void CreateInitialFeeds(std::vector<OrtValue>& feeds);

  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      gsl::span<const int64_t> beam_next_tokens,
      gsl::span<const int64_t> beam_indices);

  // Process logits and append next tokens to sequences
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int64_t>& beam_next_tokens,
                           gsl::span<int64_t>& beam_indices);

  Status ProcessLogits(const OrtValue& logits,
                       BeamSearchState<T>& beam_state,
                       int top_k,
                       AllocatorPtr& allocator);

  // Mask tokens accroding to vocab_mask
  void ApplyVocabMask(gsl::span<T>& next_token_scores);

  // Apply repetion penalty
  void ApplyRepetitionPenalty(const Sequences& sequences, gsl::span<T>& next_token_scores);

  OpKernelContextInternal& context_;
  const SessionState& session_state_;

  GptSubgraph& gpt_subgraph_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  // Not used in CPU. Stream is for CUDA only.
  void* stream_;

  BeamSearchParameters* parameters_;

  std::unique_ptr<BeamSearchScorer<T>> beam_scorer_;

  BeamSearchState<T> beam_state_;

  AllocatorPtr allocator_;
};

template <typename T>
void BeamSearch<T>::Init(const OpKernelInfo& info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;

  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);

  parameters_.ParseFromAttributes(info);

  stream_ = nullptr;
}

template <typename T>
std::unique_ptr<OpKernel> BeamSearch<T>::Create(const OpKernelInfo& info,
                                                void* stream) {
  auto result = std::make_unique<BeamSearch>(info);
  result->SetComputeStream(stream);
  return result;
}

template <typename T>
common::Status BeamSearch<T>::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                                         const std::string& attribute_name,
                                                         const SessionState& subgraph_session_state) {
  ORT_ENFORCE(gpt_subgraph_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  const auto& node = Node();
  gpt_subgraph_ = std::make_unique<GptSubgraph>(node, attribute_name, subgraph_session_state.GetGraphViewer());
  ORT_RETURN_IF_ERROR(gpt_subgraph_->Setup(session_state, subgraph_session_state));
  feeds_fetches_manager_ = gpt_subgraph_->GetFeedsFetchesManager();
  parameters_.SetSubgraphParameters(gpt_subgraph_->vocab_size,
                                    gpt_subgraph_->num_heads,
                                    gpt_subgraph_->head_size,
                                    gpt_subgraph_->num_layers);
  return Status::OK();
}

template <typename T>
Status BeamSearch<T>::Compute(OpKernelContext* ctx) const {
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");
  ORT_ENFORCE(feeds_fetches_manager_, "CreateFeedsFetchesManager must be called prior to execution of graph.");

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  BeamSearchParameters parameters = parameters_;  // make a copy

  BeamSearchImpl<T> impl{*ctx_internal, *session_state, *gpt_subgraph_, thread_pool, stream_, parameters};

  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = impl.Execute(*feeds_fetches_manager_);

  return status;
}

template <typename T>
BeamSearchImpl<T>::BeamSearchImpl(OpKernelContextInternal& context,
                                  const SessionState& session_state,
                                  GptSubgraph& gpt_subgraph,
                                  concurrency::ThreadPool* thread_pool,
                                  void* stream,
                                  BeamSearchParameters& params)
    : context_(context),
      session_state_(session_state),
      gpt_subgraph_(gpt_subgraph),
      thread_pool_(thread_pool),
      implicit_inputs_(context_.GetImplicitInputs()),
      stream_(stream),
      parameters_(&params),
      allocator_(nullptr) {
  parameters_->ParseFromInputs(&context);

  allocator_ = session_state.GetExecutionProviders()
                   .Get(onnxruntime::kCpuExecutionProvider)
                   ->GetAllocator(0, OrtMemTypeDefault);
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
    if (static_cast<int>(vocab_mask_dims[0]) != parameters_->vocab_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'vocab_mask' shape does not match with vocab_size, got ",
                             vocab_mask_dims[0]);
    }

    // store vocab mask in parameters.
    parameters_->vocab_mask = vocab_mask->DataAsSpan<int32_t>();
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::Initialize() {
  auto status = Status::OK();

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

  // CheckInputs shall be after CheckSubgraph due to its dependency on vocab_size
  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  return status;
}

template <typename T>
void BeamSearchImpl<T>::CreateInitialFeeds(std::vector<OrtValue>& feeds) {
  const OrtValue* input_ids_value = context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  gpt_subgraph_.CreateInitialFeeds(input_ids, implicit_inputs_, parameters_->num_beams, parameters_->pad_token_id, feeds);
}

template <typename T>
Status BeamSearchImpl<T>::ProcessLogits(
    const OrtValue& logits,  // logits output of subgraph
    BeamSearchState<T>& beam_state,
    int top_k,
    AllocatorPtr& allocator) {
  const int64_t batch_beam_size = static_cast<int64_t>(parameters_->batch_size * parameters_->num_beams);
  const int& vocab_size = parameters_->vocab_size;

#ifdef DEBUG_BEAM_SEARCH
  //DumpOrtValue("input_ids", input_ids);
  DumpOrtValue("logits", logits);
#endif

  const T* logits_data = logits.Get<Tensor>().Data<T>();

  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);

  // The sequence length of input_ids for the logits.
  // It equals parameters_->sequence_length for first subgraph call, and 1 for the remaining.
  auto input_length = logits_shape[1];

  // Get logits for the last token, where logits has shape (batch_size * num_beams, input_length, vocab_size)
  //    next_token_logits = logits[:, -1, :], where its shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly to avoid copy logits to next_token_logits.
  auto next_token_logits = gsl::make_span(beam_state.next_token_logits);
  if (input_length > 1) {
    const T* current_logits = logits_data + (input_length - 1) * vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const T> source(current_logits, vocab_size);
      gsl::span<T> target = next_token_logits.subspan(i * vocab_size, vocab_size);
      gsl::copy(source, target);
      current_logits += i * (input_length * vocab_size);
    }
  }

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  auto next_token_scores = gsl::make_span(beam_state.next_token_scores);
  Status status = SoftmaxCPU<T>(batch_beam_size,  // rows
                                vocab_size,       // elements per row
                                input_length > 1 ? next_token_logits.data() : logits_data,
                                next_token_scores.data(),
                                true,
                                thread_pool_);
  if (!status.IsOK()) {
    return status;
  }

  // Apply all score processors that updates scores
  ApplyVocabMask(next_token_scores);
  ApplyRepetitionPenalty(beam_state.sequences, next_token_scores);

  // next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  // TODO: use thread pool to parrellel
  int offset = 0;
  int batch_beam_index = 0;
  for (int i = 0; i < parameters_->batch_size; i++) {
    for (int j = 0; j < parameters_->num_beams; j++, batch_beam_index++) {
      for (int k = 0; k < parameters_->vocab_size; k++, offset++) {
        next_token_scores[offset] += beam_state.beam_scores[batch_beam_index];
      }
    }
  }

  if (parameters_->output_scores) {
    beam_state.scores.insert(beam_state.scores.end(), next_token_scores.begin(), next_token_scores.end());
  }

  //next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
  //next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
  int64_t next_token_scores_dims[] = {parameters_->batch_size, parameters_->num_beams * vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(), next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

#ifdef DEBUG_BEAM_SEARCH
  DumpOrtValue("next_token_scores_value", next_token_scores_value);
#endif

  const int axis = 1;
  const unsigned k = static_cast<unsigned>(top_k);
  const bool largest = true;
  const bool sorted = true;  // results returned in sorted order.

  std::unique_ptr<Tensor> topk_scores;
  std::unique_ptr<Tensor> topk_indices;
  status = GetTopK<T>(&input, axis, top_k, largest, sorted, allocator, thread_pool_, topk_scores, topk_indices);
  if (!status.IsOK()) {
    return status;
  }

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor<T>("topk_scores", *(topk_scores.get()));
  DumpTensor<int64_t>("topk_indices", *(topk_indices.get()));
#endif

  //next_indices = (next_tokens / vocab_size).long()
  //next_tokens = next_tokens % vocab_size
  gsl::span<const int64_t> next_token_indices = topk_indices->DataAsSpan<int64_t>();
  beam_state.next_indices.resize(parameters_->batch_size * k);
  beam_state.next_tokens.resize(parameters_->batch_size * k);
  offset = 0;
  for (int i = 0; i < parameters_->batch_size; i++) {
    for (unsigned int j = 0; j < k; j++, offset++) {
      beam_state.next_indices[offset] = next_token_indices[offset] / vocab_size;
      beam_state.next_tokens[offset] = next_token_indices[offset] % vocab_size;
    }
  }

  gsl::span<const T> next_scores = topk_scores->DataAsSpan<T>();
  gsl::span<const int64_t> next_tokens(beam_state.next_tokens.data(), beam_state.next_tokens.size());
  gsl::span<const int64_t> next_indices(beam_state.next_indices.data(), beam_state.next_indices.size());

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor<T>("next_scores before scorer", next_scores.data(), parameters_->batch_size, k);
  DumpTensor<int64_t>("next_tokens before scorer", next_tokens.data(), parameters_->batch_size, k);
  DumpTensor<int64_t>("next_indices before scorer", next_indices.data(), parameters_->batch_size, k);
#endif

  beam_scorer_->Process(
      &(beam_state.sequences),
      next_scores,  //next_token_scores,
      next_tokens,
      next_indices,
      allocator);

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int64_t>& beam_next_tokens,
    gsl::span<int64_t>& beam_indices) {
  // Process logits to get next token scores, and select top_k = 2 * num_beams
  // TODO: we might not need 2 * num_beams when logits processors does not update token scores.
  const int top_k = 2 * parameters_->num_beams;
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state_, top_k, allocator_));

  gsl::span<T>& beam_scores = beam_scorer_->GetNextScores();
  // TODO: may not need clone beam_scores.
  beam_state_.beam_scores.assign(beam_scores.begin(), beam_scores.end());

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor<T>("beam_scores after scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  DumpTensor<int64_t>("beam_next_tokens after scorer", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  DumpTensor<int64_t>("beam_indices after scorer", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  beam_state_.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  beam_state_.sequences.PrintSequences();
#endif
  return Status::OK();
}

template <typename T>
void BeamSearchImpl<T>::ApplyVocabMask(gsl::span<T>& next_token_scores) {
  // Process vocabulary mask and set tokens with mask value 0 to -inf.
  auto& vocab_mask = parameters_->vocab_mask;
  if (!vocab_mask.empty()) {
    T* p = next_token_scores.data();
    // next_token_scores shape (batch_size * num_beams, vocab_size), vocab_mask shape (vocab_size)
    for (int i = 0; i < parameters_->batch_size * parameters_->num_beams; i++) {
      for (int j = 0; j < parameters_->vocab_size; j++, p++) {
        if (vocab_mask[j] == 0) {
          *p = std::numeric_limits<T>::lowest();
        }
      }
    }
  }
  return;
}

template <typename T>
void BeamSearchImpl<T>::ApplyRepetitionPenalty(const Sequences& sequences, gsl::span<T>& next_token_scores) {
  if (parameters_->repetition_penalty == 1.0f) {  // no penalty
    return;
  }

  int batch_beam_size = parameters_->BatchBeamSize();
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.subspan(i * parameters_->vocab_size, parameters_->vocab_size);
    gsl::span<const int64_t> sequence = sequences.GetSequence(i);
    for (const int64_t& word_id : sequence) {
      T score = beam_token_scores[word_id];
      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * parameters_->repetition_penalty : score / parameters_->repetition_penalty);
    }
  }
}

template <typename T>
Status BeamSearchImpl<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices) {
  return gpt_subgraph_.UpdateFeeds(last_outputs, next_inputs, current_length, beam_next_tokens, beam_indices, parameters_->num_beams);
}

template <typename T>
Status BeamSearchImpl<T>::Execute(const FeedsFetchesManager& ffm) {
  auto status = Status::OK();

  std::vector<int64_t> sequences_dims{parameters_->batch_size, parameters_->num_return_sequences, parameters_->max_length};
  TensorShape sequences_shape(sequences_dims);
  Tensor* output_sequences = context_.Output(0, sequences_shape);

  std::vector<int64_t> sequences_scores_dims{parameters_->batch_size, parameters_->num_return_sequences};
  TensorShape sequences_scores_shape(sequences_scores_dims);
  Tensor* output_sequences_scores = context_.Output(1, sequences_scores_shape);

  std::vector<int64_t> scores_dims{
      parameters_->max_length - parameters_->sequence_length,
      parameters_->batch_size, parameters_->num_beams, parameters_->vocab_size};
  TensorShape scores_shape(scores_dims);
  Tensor* output_scores = context_.Output(2, scores_shape);

  // Update the flag to indicate whether scores exists in output
  parameters_->output_scores = (output_scores != nullptr);

  std::vector<OrtValue> feeds;
  std::vector<OrtValue> fetches;

  CreateInitialFeeds(feeds);

  // Initialize resources
  beam_scorer_ = std::make_unique<BeamSearchScorer<T>>(parameters_->batch_size,
                                                       parameters_->num_beams,
                                                       parameters_->max_length,
                                                       parameters_->length_penalty,
                                                       parameters_->early_stopping,
                                                       parameters_->num_return_sequences,
                                                       parameters_->pad_token_id,
                                                       parameters_->eos_token_id);
  const OrtValue& input_ids = feeds[0];
#ifdef DEBUG_BEAM_SEARCH
  DumpOrtValue("input_ids", input_ids);
  DumpOrtValue("position_ids", feeds[1]);
  DumpOrtValue("attention_mask", feeds[2]);
#endif

  beam_state_.Init(input_ids,
                   parameters_->batch_size,
                   parameters_->num_beams,
                   parameters_->vocab_size,
                   parameters_->sequence_length,
                   parameters_->max_length,
                   parameters_->output_scores);

  int current_length = parameters_->sequence_length;
  while (current_length < parameters_->max_length) {
#ifdef DEBUG_BEAM_SEARCH
    DumpString("***CurrentLength", std::to_string(current_length), true);
#endif

    status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, {},
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger());

    ORT_RETURN_IF_ERROR(status);

    const OrtValue& logits = fetches[0];
    gsl::span<int64_t> beam_next_tokens;
    gsl::span<int64_t> beam_indices;
    ORT_RETURN_IF_ERROR(GenerateNextToken(logits, beam_next_tokens, beam_indices));

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters_->max_length) {
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length, beam_next_tokens.as_span<const int64_t>(), beam_indices.as_span<const int64_t>()));
    }
    fetches.clear();

#ifdef DEBUG_BEAM_SEARCH
    if (current_length - parameters_->sequence_length == 3) {  // only dump a few steps.
      DisableTensorDump();
    }
#endif
  }

  gsl::span<const T> beam_scores(beam_state_.beam_scores.data(), beam_state_.beam_scores.size());
  beam_scorer_->Finalize(&(beam_state_.sequences),
                         beam_scores,
                         output_sequences,
                         output_sequences_scores);

  // Output per token scores
  if (output_scores != nullptr) {
    gsl::span<T> target = output_scores->MutableDataAsSpan<T>();
    gsl::span<const T> source = gsl::span<const T>(beam_state_.scores.data(), beam_state_.scores.size());
    gsl::copy(source, target);

    // Fill zeros for the remaining when beam search stopped early
    if (target.length() > source.length()) {
      gsl::span<T> remaining = target.subspan(source.length());
      memset(remaining.data(), 0, remaining.size_bytes());
    }
  }

  return status;
}

// Instantiation
template class BeamSearchImpl<float>;
template class BeamSearch<float>;

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
