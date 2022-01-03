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
#include "logits_processor.h"
#include "sequences.h"
#include "dump_tensor.h"

#ifdef _MSC_VER
#pragma warning(pop)
// Could reduce the chance of arithmetic overflow. TODO: fix it
#pragma warning(disable : 26451)
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
struct BeamSearchState {
  gsl::span<T> beam_scores;           // shape (batch_size, num_beams)
  gsl::span<T> next_token_logits;     // shape (batch_size * num_beams, vocab_size)
  gsl::span<T> next_token_scores;     // shape (batch_size, num_beams * vocab_size)
  gsl::span<int64_t> next_tokens;     // shape (batch_size, 2 * num_beams)
  gsl::span<int64_t> next_indices;    // shape (batch_size, 2 * num_beams)
  gsl::span<int64_t> next_positions;  // shape (batch_size, num_beams). Next position value for position_ids.

  gsl::span<T> scores;            // shape (max_length - sequence_length + 1, batch_size, num_beams * vocab_size)
  gsl::span<T> remaining_scores;  // subspan that is avaiable for appending next token scores.

  Sequences sequences;

  void Init(AllocatorPtr allocator,
            int batch_size,
            int num_beams,
            int vocab_size,
            int sequence_length,
            int max_length,
            bool output_scores) {
    size_t batch_beam_size = SafeInt<size_t>(batch_size) * num_beams;
    beam_scores = AllocateBuffer<T>(allocator, beam_scores_buffer_, batch_beam_size, true, static_cast<T>(0));

    // Initialize score of first beam of each group with 0 and the rest with -1e9.
    // This ensures that the beams in the same group don't produce same tokens every time.
    for (int i = 0; i < batch_size; i++) {
      for (int j = 1; j < num_beams; j++) {
        beam_scores[i * num_beams + j] = -1e9;
      }
    }

    size_t next_token_size = SafeInt<size_t>(batch_beam_size) * vocab_size;
    next_token_logits = AllocateBuffer<T>(allocator, next_token_logits_buffer_, next_token_size, true, static_cast<T>(0));
    next_token_scores = AllocateBuffer<T>(allocator, next_token_scores_buffer_, next_token_size, true, static_cast<T>(0));

    next_tokens = AllocateBuffer<int64_t>(allocator, next_tokens_buffer_, SafeInt<size_t>(2) * batch_beam_size, true, static_cast<int64_t>(0));

    next_indices = AllocateBuffer<int64_t>(allocator, next_indices_buffer_, SafeInt<size_t>(2) * batch_beam_size, true, static_cast<int64_t>(0));

    next_positions = AllocateBuffer<int64_t>(allocator, next_positions_buffer_, batch_beam_size, true, static_cast<int64_t>(0));

    if (output_scores) {
      size_t elements = SafeInt<size_t>(max_length - sequence_length) * batch_size * num_beams * vocab_size;
      scores = AllocateBuffer<T>(allocator, scores_buffer_, elements);
      remaining_scores = scores;
    }

    // sequences will be initialized later since it has dependency on input_ids
  }

 private:
  BufferUniquePtr beam_scores_buffer_;
  BufferUniquePtr next_token_logits_buffer_;
  BufferUniquePtr next_token_scores_buffer_;
  BufferUniquePtr next_tokens_buffer_;
  BufferUniquePtr next_indices_buffer_;
  BufferUniquePtr next_positions_buffer_;
  BufferUniquePtr scores_buffer_;
};

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OpKernelContextInternal& context,
                 const SessionState& session_state,
                 GptSubgraph& gpt_subgraph,
                 concurrency::ThreadPool* thread_pool,
                 void* stream,
                 BeamSearchParameters& params);

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize();

  // Execute beam search in iterations util stopping criteria is reached.
  // In each iteration, GPT subgraph is called, and next token for each sequence is generated.
  Status Execute(const FeedsFetchesManager& cached_ffm);

 private:
  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context);

  // Prepare the inputs for first inference of subgraph
  void CreateInitialFeeds(gsl::span<int64_t>& next_positions, std::vector<OrtValue>& feeds);

  // Update the input for next iteration.
  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      gsl::span<int64_t>& next_positions,
      gsl::span<const int64_t> beam_next_tokens,
      gsl::span<const int64_t> beam_indices);

  // Process logits and append next tokens to sequences.
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int64_t>& beam_next_tokens,
                           gsl::span<int64_t>& beam_indices,
                           BeamSearchState<T>& beam_state);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       AllocatorPtr& allocator);

  OpKernelContextInternal& context_;

  const SessionState& session_state_;

  GptSubgraph& gpt_subgraph_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  // Not used in CPU. Stream is for CUDA only.
  void* stream_;

  BeamSearchParameters* parameters_;

  LogitsProcessorList<T> logits_processors_;

  std::unique_ptr<BeamSearchScorer<T>> beam_scorer_;

  AllocatorPtr allocator_;
};

template <typename T>
void BeamSearch<T>::Init(const OpKernelInfo& info) {
  // Make sure the body attribute was present even though we don't need it here.
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

  BeamSearchParameters parameters = parameters_;  // make a copy since we will update the parameters based on inputs later

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

    // There is dependency on vocab_size parameter, which shall be set before calling this function.
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

  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
  logits_processors_.Init(*parameters_);

  return status;
}

template <typename T>
void BeamSearchImpl<T>::CreateInitialFeeds(gsl::span<int64_t>& next_positions, std::vector<OrtValue>& feeds) {
  const OrtValue* input_ids_value = context_.GetInputOrtValue(0);
  const Tensor& input_ids = input_ids_value->Get<Tensor>();
  gpt_subgraph_.CreateInitialFeeds(input_ids, implicit_inputs_, parameters_->num_beams, parameters_->pad_token_id, next_positions, feeds);
}

template <typename T>
Status BeamSearchImpl<T>::ProcessLogits(
    const OrtValue& logits,
    BeamSearchState<T>& beam_state,
    AllocatorPtr& allocator) {
  const int64_t batch_beam_size = static_cast<int64_t>(parameters_->BatchBeamSize());
  const int& vocab_size = parameters_->vocab_size;

  const T* logits_data = logits.Get<Tensor>().Data<T>();

  // Logits has shape (batch_size * num_beams, input_length, vocab_size),
  // where input_length equals to parameters_->sequence_length for first subgraph call, and 1 for the remaining calls.
  const TensorShape& logits_shape = logits.Get<Tensor>().Shape();
  ORT_ENFORCE(logits_shape.NumDimensions() == 3);
  auto input_length = logits_shape[1];

  // Get logits for the last token:
  //    next_token_logits = logits[:, -1, :], and the result shape is (batch_size * num_beams, vocab_size)
  // When input_length == 1, use logits directly in SoftmaxCPU below so it only need for input_length > 1.
  gsl::span<T>& next_token_logits = beam_state.next_token_logits;
  if (input_length > 1) {
    const T* current_logits = logits_data + (input_length - 1) * vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const T> source(current_logits, vocab_size);
      gsl::span<T> target = next_token_logits.subspan(i * vocab_size, vocab_size);
      gsl::copy(source, target);
      current_logits += input_length * vocab_size;
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  //DumpOrtValue("logits", logits);
  DumpTensor("next_token_logits", next_token_logits.data(), parameters_->batch_size, parameters_->num_beams, vocab_size);
#endif

  // Get scores for candidates of next token: next_token_scores = log_softmax(next_token_logits, dim=-1)
  gsl::span<T>& next_token_scores = beam_state.next_token_scores;
  Status status = SoftmaxCPU<T>(batch_beam_size,  // rows
                                vocab_size,       // elements per row
                                input_length > 1 ? next_token_logits.data() : logits_data,
                                next_token_scores.data(),
                                true,
                                thread_pool_);
  if (!status.IsOK()) {
    return status;
  }

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor("next_token_scores after softmax", next_token_scores.data(), parameters_->batch_size, parameters_->num_beams, vocab_size);
#endif

  // Apply all score processors that updates scores
  logits_processors_.Process(&(beam_state.sequences), next_token_scores);

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor("next_token_scores after logits processor", next_token_scores.data(), parameters_->batch_size, parameters_->num_beams, vocab_size);
#endif

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
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

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor("next_token_scores after adding beam_scores", next_token_scores.data(), parameters_->batch_size, parameters_->num_beams, vocab_size);
#endif

  if (parameters_->output_scores) {
    // Append next token scores to the scores output.
    gsl::copy(next_token_scores, beam_state.remaining_scores);
    beam_state.remaining_scores = beam_state.remaining_scores.subspan(next_token_scores.size());
  }

  // Apply top-k selection like the following:
  //   next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
  //   next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
  int64_t next_token_scores_dims[] = {parameters_->batch_size, parameters_->num_beams * vocab_size};
  TensorShape next_token_scores_shape(&next_token_scores_dims[0], 2);
  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue next_token_scores_value;
  Tensor::InitOrtValue(element_type, next_token_scores_shape, next_token_scores.data(), allocator->Info(), next_token_scores_value);
  const Tensor& input = next_token_scores_value.Get<Tensor>();

  constexpr int axis = 1;
  const unsigned top_k = static_cast<unsigned>(2 * parameters_->num_beams);
  constexpr bool largest = true;
  constexpr bool sorted = true;  // results returned in sorted order.

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

  // Convert indices in range [0, num_beams * vocab_size) to token ID of range [0, vocab_size) like the following:
  //   next_indices = (next_tokens / vocab_size).long()
  //   next_tokens = next_tokens % vocab_size
  gsl::span<const int64_t> next_token_indices = topk_indices->DataAsSpan<int64_t>();
  offset = 0;
  for (int i = 0; i < parameters_->batch_size; i++) {
    for (unsigned int j = 0; j < top_k; j++, offset++) {
      beam_state.next_indices[offset] = next_token_indices[offset] / vocab_size;
      beam_state.next_tokens[offset] = next_token_indices[offset] % vocab_size;
    }
  }

  gsl::span<const T> next_scores = topk_scores->DataAsSpan<T>();
  gsl::span<const int64_t> next_tokens(beam_state.next_tokens.data(), beam_state.next_tokens.size());
  gsl::span<const int64_t> next_indices(beam_state.next_indices.data(), beam_state.next_indices.size());

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor<T>("next_scores before scorer", next_scores.data(), parameters_->batch_size, top_k);
  DumpTensor<int64_t>("next_tokens before scorer", next_tokens.data(), parameters_->batch_size, top_k);
  DumpTensor<int64_t>("next_indices before scorer", next_indices.data(), parameters_->batch_size, top_k);
#endif

  beam_scorer_->Process(
      &(beam_state.sequences),
      next_scores,
      next_tokens,
      next_indices);

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int64_t>& beam_next_tokens,
    gsl::span<int64_t>& beam_indices,
    BeamSearchState<T>& beam_state) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, allocator_));

  gsl::span<T>& beam_scores = beam_scorer_->GetNextScores();
  // It is optional to clone beam_scores. Change it to use same buffer also works:
  //    beam_state.beam_scores = beam_scores
  // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
  gsl::copy(beam_scores, beam_state.beam_scores);

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  DumpTensor<T>("beam_scores after scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  DumpTensor<int64_t>("beam_next_tokens after scorer", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  DumpTensor<int64_t>("beam_indices after scorer", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  beam_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  beam_state.sequences.PrintSequences();
#endif
  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    gsl::span<int64_t>& next_positions,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices) {
  return gpt_subgraph_.UpdateFeeds(last_outputs, next_inputs, current_length, next_positions,
                                   beam_next_tokens, beam_indices, parameters_->num_beams);
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

  // Initialize resources
  AllocatorPtr temp_space_allocator;
  ORT_RETURN_IF_ERROR(context_.GetTempSpaceAllocator(&temp_space_allocator));

  BeamSearchState<T> beam_state;
  beam_state.Init(temp_space_allocator,
                  parameters_->batch_size,
                  parameters_->num_beams,
                  parameters_->vocab_size,
                  parameters_->sequence_length,
                  parameters_->max_length,
                  parameters_->output_scores);

  beam_scorer_ = std::make_unique<BeamSearchScorer<T>>(parameters_->batch_size,
                                                       parameters_->num_beams,
                                                       parameters_->max_length,
                                                       parameters_->length_penalty,
                                                       parameters_->early_stopping,
                                                       parameters_->num_return_sequences,
                                                       parameters_->pad_token_id,
                                                       parameters_->eos_token_id);
  beam_scorer_->Initialize(allocator_, parameters_->sequence_length);  // TODO: use temp_space_allocator

  CreateInitialFeeds(beam_state.next_positions, feeds);
  const OrtValue& input_ids = feeds[0];
  beam_state.sequences.Init(temp_space_allocator,
                            input_ids,
                            parameters_->BatchBeamSize(),
                            parameters_->sequence_length,
                            parameters_->max_length);

#ifdef DEBUG_BEAM_SEARCH
  DumpOrtValue("input_ids", input_ids);
  DumpOrtValue("position_ids", feeds[1]);
  DumpOrtValue("attention_mask", feeds[2]);
#endif

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
    ORT_RETURN_IF_ERROR(GenerateNextToken(logits, beam_next_tokens, beam_indices, beam_state));

    // When all batches are finished, stop earlier to avoid wasting computation.
    if (beam_scorer_->IsDone()) {
      break;
    }

    // Increase sequence length after a new token is generated.
    ++current_length;

    // Prepare inputs for next round of subgraph call.
    if (current_length < parameters_->max_length) {
      ORT_RETURN_IF_ERROR(UpdateFeeds(fetches, feeds, current_length,
                                      beam_state.next_positions,
                                      beam_next_tokens.as_span<const int64_t>(),
                                      beam_indices.as_span<const int64_t>()));
    }
    fetches.clear();

#ifdef DEBUG_BEAM_SEARCH
    if (current_length - parameters_->sequence_length == 3) {  // only dump a few steps.
      DisableTensorDump();
    }
#endif
  }

  gsl::span<const T> beam_scores(beam_state.beam_scores.data(), beam_state.beam_scores.size());
  beam_scorer_->Finalize(&(beam_state.sequences),
                         beam_scores,
                         output_sequences,
                         output_sequences_scores);

  // Output per token scores
  if (output_scores != nullptr) {
    gsl::span<T> target = output_scores->MutableDataAsSpan<T>();
    gsl::span<const T> source = gsl::span<const T>(beam_state.scores.data(), beam_state.scores.size());
    assert(target.length() == source.length());
    gsl::copy(source, target);
  }

  return status;
}

// Instantiation
template class BeamSearchImpl<float>;
template class BeamSearch<float>;

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
