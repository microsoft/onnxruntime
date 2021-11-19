// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#ifndef NDEBUG
#define DEBUG_BEAM_SEARCH 1  // TODO: remove this once this operator is ready for production.
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
      BeamSearch<T>);

REGISTER_KERNEL_TYPED(float)

// CPU does not support float16
// REGISTER_KERNEL_TYPED(MLFloat16)

GptSubgraphInfo::GptSubgraphInfo(const onnxruntime::Node& node, const GraphViewer& subgraph_in)
    : subgraph(subgraph_in) {
  num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());

  auto& subgraph_inputs = subgraph.GetInputs();
  auto& subgraph_outputs = subgraph.GetOutputs();

  // inputs: input_ids, position_ids, attention_mask, past_0, past_1, ...
  // outputs: logits, present_0, present_1, ...
  num_subgraph_inputs = static_cast<int>(subgraph_inputs.size());
  num_subgraph_outputs = static_cast<int>(subgraph_outputs.size());

  // CheckSubgraph will verify inputs and outputs later.
  subgraph_input_names.reserve(num_subgraph_inputs);
  for (int i = 0; i < num_subgraph_inputs; ++i) {
    subgraph_input_names.push_back(subgraph_inputs[i]->Name());
  }

  subgraph_output_names.reserve(num_subgraph_outputs);
  for (int i = 0; i < num_subgraph_outputs; ++i) {
    subgraph_output_names.push_back(subgraph_outputs[i]->Name());
  }
}

void Sequences::Init(const OrtValue& input_ids, int batch_beam_size, int sequence_length, int max_length) {
  // Allocate buffer (shall we use allocator instead?)
  sequences[0].assign(batch_beam_size * max_length, 0);
  sequences[1].assign(batch_beam_size * max_length, 0);

  // copying input_ids to sequences[0]
  gsl::span<const int64_t> input = input_ids.Get<Tensor>().DataAsSpan<int64_t>();
  gsl::span<int64_t> output(sequences[0]);
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<const int64_t> source = input.subspan(i * sequence_length, sequence_length);
    gsl::span<int64_t> target = output.subspan(i * max_length, sequence_length);
    gsl::copy(source, target);
  }
  current_sequences_buffer = 0;

  batch_beam_size_ = batch_beam_size;
  max_length_ = max_length;
  current_length_ = sequence_length;
}

gsl::span<const int64_t> Sequences::GetSequence(int beam_index) {
  gsl::span<int64_t> buffer(sequences[current_sequences_buffer]);
  gsl::span<int64_t> sequence = buffer.subspan(beam_index * max_length_, current_length_);
  return sequence;
}

int Sequences::GetSequenceLength() {
  return current_length_;
}

void Sequences::PrintSequences() {
#ifdef DEBUG_BEAM_SEARCH
  for (int i = 0; i < batch_beam_size_; i++) {
    gsl::span<const int64_t> sequence = GetSequence(i);
    DumpString("sequences", i, false);
    DumpTensor<int64_t>(nullptr, sequence.data(), 1, current_length_);
  }
#endif
}

void Sequences::AppendNextTokenToSequences(
    gsl::span<int64_t>& beam_indices,
    gsl::span<int64_t>& beam_next_tokens) {
  //sequences = torch.cat([sequences[beam_indices, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
  gsl::span<const int64_t> input(sequences[current_sequences_buffer]);
  gsl::span<int64_t> output(sequences[1 - current_sequences_buffer]);

  for (int i = 0; i < batch_beam_size_; i++) {
    int beam_index = static_cast<int>(beam_indices[i]);
    gsl::span<const int64_t> source = input.subspan(beam_index * max_length_, current_length_);
    gsl::span<int64_t> target = output.subspan(i * max_length_, current_length_);
    gsl::copy(source, target);
  }

  // append next token to each beam
  for (int i = 0; i < batch_beam_size_; i++) {
    output[i * max_length_ + current_length_] = beam_next_tokens[i];
  }

  ++current_length_;
  current_sequences_buffer = 1 - current_sequences_buffer;  // rotate buffer for next round
}

template <typename T>
class BeamSearchImpl {
 public:
  BeamSearchImpl(OpKernelContextInternal& context,
                 const SessionState& session_state,
                 const GptSubgraphInfo& info,
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

  Status CheckSubgraph(const std::vector<const NodeArg*>& subgraph_inputs,
                       const std::vector<const NodeArg*>& subgraph_outputs) const;

  OrtValue ExpandInputs(const OrtValue& input_ids, int num_beams) const;

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

  void ProcessNextTokenScores(gsl::span<T>& next_token_scores);

  // Reorder cache by picking the past state based on beam indices
  void PickPastState(const std::vector<OrtValue>& last_outputs,
                     std::vector<OrtValue>& next_inputs,
                     gsl::span<const int64_t>& beam_indices);

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const GptSubgraphInfo& subgraph_info_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  std::vector<int64_t> next_positions_;

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
  ORT_ENFORCE(subgraph_info_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  ORT_UNUSED_PARAMETER(attribute_name);

  const auto& node = Node();
  subgraph_info_ = std::make_unique<GptSubgraphInfo>(node, subgraph_session_state.GetGraphViewer());

  ORT_RETURN_IF(subgraph_info_->num_subgraph_outputs <= 1,
                "Invalid GPT-2 subgraph: number of outputs shall be larger than 1 (Need past state in inputs and outputs).");

  ORT_RETURN_IF(subgraph_info_->num_subgraph_inputs != subgraph_info_->num_subgraph_outputs + 2,
                "Invalid GPT-2 subgraph: number of inputs shall be number of outputs plus 2");

  std::vector<std::string> feed_names;
  feed_names.reserve(subgraph_info_->num_subgraph_inputs + subgraph_info_->num_implicit_inputs);

  // First, get the location of input_ids of current operator.
  const auto& node_inputs = node.InputDefs();
  const OrtMemoryInfo& input_ids_location = utils::FindMemoryInfoForValue(session_state, node_inputs[0]->Name());

  // position_ids, attention_mask, past_0, ... are created by this operator so the name doesn't matter.
  // as we skip them when we call FindDevicesForValues, and default them to be in the same device as input_ids
  feed_names.insert(feed_names.end(), subgraph_info_->subgraph_input_names.begin(), subgraph_info_->subgraph_input_names.end());

  for (auto& entry : node.ImplicitInputDefs()) {
    feed_names.push_back(entry->Name());
  }

  std::vector<OrtDevice> feed_locations;
  feed_locations.resize(feed_names.size());

  for (size_t i = 0, end = feed_names.size(); i < end; ++i) {
    if (i >= subgraph_info_->subgraph_input_names.size()) {  // implicit inputs
      const auto& location = utils::FindMemoryInfoForValue(session_state, feed_names[i]);
      feed_locations[i] = location.device;
    } else {
      feed_locations[i] = input_ids_location.device;
    }
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, subgraph_info_->subgraph_output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // setup the locations where we want the subgraph output to end up on
  std::vector<const OrtMemoryInfo*> fetch_locations;
  fetch_locations.reserve(subgraph_info_->num_subgraph_outputs);

  // past state need to be where we can feed them in to the next iteration, so set the fetch location to match the feed location.
  for (int i = 0; i < subgraph_info_->num_subgraph_outputs; ++i) {
    fetch_locations.push_back(&input_ids_location);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager_ = std::move(ffm);

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

  BeamSearchImpl<T> impl{*ctx_internal, *session_state, *subgraph_info_, thread_pool, stream_, parameters};

  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = impl.Execute(*feeds_fetches_manager_);

  return status;
}

template <typename T>
BeamSearchImpl<T>::BeamSearchImpl(OpKernelContextInternal& context,
                                  const SessionState& session_state,
                                  const GptSubgraphInfo& subgraph_info,
                                  concurrency::ThreadPool* thread_pool,
                                  void* stream,
                                  BeamSearchParameters& params)
    : context_(context),
      session_state_(session_state),
      subgraph_info_(subgraph_info),
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
  }

  return Status::OK();
}

template <typename T>
Status BeamSearchImpl<T>::CheckSubgraph(const std::vector<const NodeArg*>& subgraph_inputs,
                                        const std::vector<const NodeArg*>& subgraph_outputs) const {
  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids", "subgraph input 0 shall be named as input_ids, got: ",
                subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "position_ids", "subgraph input 1 shall be named as position_ids, got: ",
                subgraph_inputs[1]->Name());
  ORT_RETURN_IF(subgraph_inputs[2]->Name() != "attention_mask", "subgraph input 2 shall be named as attention_mask, got: ",
                subgraph_inputs[2]->Name());
  ORT_RETURN_IF(subgraph_inputs[3]->Name() != "past_0", "subgraph input 3 shall be named as past_0, got: ",
                subgraph_inputs[3]->Name());

  // Past state shape is like (2, batch_size, 12, past_seq_len, 64). Here 12 and 64 are constants of num_heads and hidden_size/num_heads.
  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_inputs[3]->Shape();
  ORT_RETURN_IF(past_shape->dim_size() != 5, "subgraph past state is expected to have 5 dimension, got ",
                past_shape->dim_size());

  ORT_RETURN_IF(!past_shape->dim(0).has_dim_value() || past_shape->dim(0).dim_value() != 2,
                "subgraph past state dimension 0 shall have length of 2");

  ORT_RETURN_IF(!past_shape->dim(2).has_dim_value() || past_shape->dim(2).dim_value() <= 0,
                "subgraph past state dimension 2 shall have a positive value for number of heads");

  ORT_RETURN_IF(!past_shape->dim(4).has_dim_value() || past_shape->dim(4).dim_value() <= 0,
                "subgraph past state dimension 4 shall have a positive value for hidden size per head");

  // check subgraph outputs
  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits", "subgraph output 0 shall be named as logits, got: ",
                subgraph_outputs[0]->Name());

  ORT_RETURN_IF(subgraph_outputs[1]->Name() != "present_0", "subgraph input 1 shall be named as present_0, got: ",
                subgraph_outputs[1]->Name());

  // Logits shape is like (batch_size, seq_len, 50257). Here 50257 is the vocabulary size.
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();
  ORT_RETURN_IF(logits_shape->dim_size() != 3, "subgraph logits output is expected to have 3 dimension, got ",
                logits_shape->dim_size());

  ORT_RETURN_IF(!logits_shape->dim(2).has_dim_value() || logits_shape->dim(2).dim_value() <= 0,
                "subgraph past state dimension 2 shall have a positive value for vocabulary size");

  int num_heads = static_cast<int>(past_shape->dim(2).dim_value());
  int head_size = static_cast<int>(past_shape->dim(4).dim_value());
  int vocab_size = static_cast<int>(logits_shape->dim(2).dim_value());
  int num_layers = static_cast<int>(subgraph_outputs.size()) - 1;
  parameters_->SetSubgraphParameters(num_heads, head_size, vocab_size, num_layers);

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

  auto& inputs = subgraph_info_.subgraph.GetInputs();
  auto& outputs = subgraph_info_.subgraph.GetOutputs();
  ORT_RETURN_IF_ERROR(CheckSubgraph(inputs, outputs));

  // CheckInputs shall be after CheckSubgraph due to its dependency on vocab_size
  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  return status;
}

template <typename T>
OrtValue BeamSearchImpl<T>::ExpandInputs(const OrtValue& input, int num_beams) const {
  if (num_beams == 1)
    return input;

  // Given input of shape (batch_size, sequence_length), expand the shape to be (batch_size * num_beams, sequence_length)
  const TensorShape& input_shape = input.Get<Tensor>().Shape();
  ORT_ENFORCE(input_shape.NumDimensions() == 2 && input_shape[0] == parameters_->batch_size && input_shape[1] == parameters_->sequence_length);

  const int64_t& batch_size = input_shape[0];
  const int64_t& sequence_length = input_shape[1];
  int64_t dims[] = {batch_size * num_beams, sequence_length};
  TensorShape expanded_shape(&dims[0], 2);

  auto element_type = DataTypeImpl::GetType<int64_t>();
  OrtValue expanded;
  Tensor::InitOrtValue(element_type, expanded_shape, allocator_, expanded);

  const int64_t* input_data = input.Get<Tensor>().Data<int64_t>();
  int64_t* expanded_data = expanded.GetMutable<Tensor>()->MutableData<int64_t>();
  int64_t* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      memcpy(target, input_data + i * sequence_length, sizeof(int64_t) * sequence_length);
      target += sequence_length;
    }
  }

  return expanded;
}

template <typename T>
void BeamSearchImpl<T>::CreateInitialFeeds(std::vector<OrtValue>& feeds) {
  // Subgraph inputs:
  //   input_ids: shape (B, S) wher B is batch size, and S is sequence length
  //   position_ids: shape (B, S)
  //   attention_mask: shape (B, P+S), where past_sequence_length (P) is 0
  // After expansion, their shapes will become (B, M*S), where M is num_beams.

  const OrtValue* input_ids = context_.GetInputOrtValue(0);

  const Tensor& input_ids_tensor = input_ids->Get<Tensor>();

  const TensorShape& input_ids_shape = input_ids_tensor.Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int64_t>();

  // input_ids for subgraph is int64, so we need Cast input_ids from int32 to int64.
  OrtValue subgraph_input_ids;
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator_, subgraph_input_ids);

  int64_t* subgraph_input_data = subgraph_input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  const int32_t* source = input_ids_tensor.Data<int32_t>();
  int64_t* target = subgraph_input_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++, source++, target++) {
      *target = static_cast<int64_t>(*source);
    }
  }

  OrtValue position_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator_, position_ids);

  OrtValue attention_mask;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator_, attention_mask);

  next_positions_.resize(batch_size * parameters_->num_beams);
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and cumulated sum of mask in a batch for other tokens
  int64_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int64_t>();
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  source = input_ids_tensor.Data<int32_t>();
  int64_t* mask = mask_data;
  int64_t* position = position_data;
  for (int i = 0; i < batch_size; i++) {
    int64_t abs_position = 0;
    for (int j = 0; j < sequence_length; j++, source++, mask++, position++) {
      if (*source == parameters_->pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position;
        abs_position++;
      }
    }
    for (int k = 0; k < parameters_->num_beams; k++) {
      next_positions_[i * parameters_->num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length) for input_ids, position_ids and attention_mask
  // TODO: Try expand inputs/outputs after first subgraph call instead. That may get better peroformance, but more complex to implement.
  OrtValue expanded_input_ids = ExpandInputs(subgraph_input_ids, parameters_->num_beams);
  OrtValue expanded_position_ids = ExpandInputs(position_ids, parameters_->num_beams);
  OrtValue expanded_attention_mask = ExpandInputs(attention_mask, parameters_->num_beams);

  // Initialize empty past state
  auto past_type = DataTypeImpl::GetType<T>();
  int64_t past_state_dims[] = {2, batch_size * parameters_->num_beams, parameters_->num_heads, 0, parameters_->head_size};
  TensorShape past_shape(&past_state_dims[0], 5);
  OrtValue empty_past;
  Tensor::InitOrtValue(past_type, past_shape, allocator_, empty_past);

  // The ordering is the same as used in SetupSubgraphExecutionInfo
  feeds.reserve(subgraph_info_.num_subgraph_inputs + subgraph_info_.num_implicit_inputs);
  feeds.push_back(expanded_input_ids);
  feeds.push_back(expanded_position_ids);
  feeds.push_back(expanded_attention_mask);

  // The remaing inputs are past state.
  for (int i = 3; i < subgraph_info_.num_subgraph_inputs; ++i) {
    feeds.push_back(empty_past);
  }

  // pass in implicit inputs
  for (const auto* entry : implicit_inputs_) {
    feeds.push_back(*entry);
  }
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

  // Extra processing: next_token_scores = logits_processor(input_ids, next_token_scores)
  // where input_ids is current sequences in beam_state_
  ProcessNextTokenScores(next_token_scores);

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
void BeamSearchImpl<T>::ProcessNextTokenScores(gsl::span<T>& /*next_token_scores*/) {
  return;
}

template <typename T>
void BeamSearchImpl<T>::PickPastState(const std::vector<OrtValue>& last_outputs,
                                      std::vector<OrtValue>& next_inputs,
                                      gsl::span<const int64_t>& beam_indices) {
  for (int i = 3; i < subgraph_info_.num_subgraph_inputs; ++i) {
    const OrtValue& present = last_outputs[i - 2];  // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();

    // Create a tensor with same shape.
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<T>();  // present.Type()
    Tensor::InitOrtValue(past_type, past_shape, allocator_, past);

    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    gsl::span<T> past_span = past.GetMutable<Tensor>()->MutableDataAsSpan<T>();
    gsl::span<const T> present_span = present.Get<Tensor>().DataAsSpan<T>();
    for (gsl::index j = 0; j < beam_indices.length(); j++) {
      int64_t beam_index = beam_indices[j];
      gsl::span<const T> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const T> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      gsl::span<T> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<T> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      gsl::copy(present_key, past_key);
      gsl::copy(present_value, past_value);

#ifdef DEBUG_BEAM_SEARCH
      if (i == 3)  // only dump past_0
      {
        DumpString("past_key of beam", static_cast<int>(j), true);
        DumpTensor<T>(nullptr, past_key.data(), 1, static_cast<int>(block_size_per_beam));

        DumpString("past_value of beam", static_cast<int>(j), true);
        DumpTensor<T>(nullptr, past_value.data(), 1, static_cast<int>(block_size_per_beam));
      }
#endif
    }

    next_inputs[i] = past;
  }
}

template <typename T>
Status BeamSearchImpl<T>::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices) {
  // last_outputs: logits, present_0, present_1, ...
  // next_inputs: input_ids, position_id, attention_mask, past_0, past_1

  // The following updates inputs for subgraph
  // TODO: Reuse buffer for input_ids and position_ids to reduce memory allocation.

  // Update input_ids with next tokens.
  int batch_beam_size = parameters_->batch_size * parameters_->num_beams;
  int64_t dims[] = {batch_beam_size, 1};
  TensorShape input_ids_shape(&dims[0], 2);
  auto element_type = DataTypeImpl::GetType<int64_t>();
  OrtValue input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator_, input_ids);
  int64_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = beam_next_tokens[i];
  }
  next_inputs[0] = input_ids;

  // Update position IDs
  OrtValue position_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, allocator_, position_ids);
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    position_data[i] = next_positions_[i];
    next_positions_[i]++;
  }
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const int64_t* old_mask_data = old_mask.Get<Tensor>().Data<int64_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  Tensor::InitOrtValue(element_type, mask_shape, allocator_, attention_mask);
  int64_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
  next_inputs[2] = attention_mask;

#ifdef DEBUG_BEAM_SEARCH
  DumpOrtValue("input_ids", input_ids);
  DumpOrtValue("position_ids", position_ids);
  DumpOrtValue("attention_mask", attention_mask);
#endif

  // Update past state
  if (parameters_->num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (int i = 3; i < subgraph_info_.num_subgraph_inputs; ++i) {
      next_inputs[i] = last_outputs[i - 2];
    }
  } else {
    PickPastState(last_outputs, next_inputs, beam_indices);
  }

  return Status::OK();
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

}  // namespace contrib
}  // namespace onnxruntime
