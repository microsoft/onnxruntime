// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "gsl/gsl"
#include "gpt_subgraph.h"
#include "dump_tensor.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace transformers {

GptSubgraph::GptSubgraph(
    const onnxruntime::Node& node_in,
    const std::string& attribute_name,
    const GraphViewer& subgraph_in)
    : node(node_in), attribute(attribute_name), subgraph(subgraph_in), allocator_(nullptr), is_output_float16_(false) {
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

Status GptSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                             const std::vector<const NodeArg*>& subgraph_outputs) {
  ORT_RETURN_IF(num_subgraph_outputs <= 1,
                "Invalid GPT-2 subgraph: number of outputs shall be larger than 1 (Need past state in inputs and outputs).");

  ORT_RETURN_IF(num_subgraph_inputs != num_subgraph_outputs + 2,
                "Invalid GPT-2 subgraph: number of inputs shall be number of outputs plus 2");

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

  // Save parameters related to the subgraph.
  num_heads = static_cast<int>(past_shape->dim(2).dim_value());
  head_size = static_cast<int>(past_shape->dim(4).dim_value());
  vocab_size = static_cast<int>(logits_shape->dim(2).dim_value());
  num_layers = static_cast<int>(subgraph_outputs.size()) - 1;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "subgraph input 0 (input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "subgraph input 1 (position_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "subgraph input 2 (attention_mask) shall have int32 type");

  auto output_type = subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT && output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
                "subgraph output 0 (logits) shall be float or float16 data type");

  ORT_RETURN_IF(subgraph_inputs[3]->TypeAsProto()->tensor_type().elem_type() != output_type,
                "subgraph input 3 (past_0) shall shall have same data type of logits output");
  ORT_RETURN_IF(subgraph_outputs[1]->TypeAsProto()->tensor_type().elem_type() != output_type,
                "subgraph output 1 (present_0) shall shall have same data type of logits output");

  is_output_float16_ = (output_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);

  return Status::OK();
}

Status GptSubgraph::Setup(const SessionState& session_state,
                          const SessionState& subgraph_session_state) {
  session_state_ = &session_state;
  subgraph_session_state_ = &subgraph_session_state;

  std::vector<std::string> feed_names;
  feed_names.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // Currently, input_ids is in CPU even for CUDA operator, so we have to use logits location as default.
  const OrtMemoryInfo& default_location = utils::FindMemoryInfoForValue(subgraph_session_state, "logits");

  // position_ids, attention_mask, past_0, ... are created by this operator so the name doesn't matter.
  // as we skip them when we call FindDevicesForValues, and default them to be in the same device as input_ids
  feed_names.insert(feed_names.end(), subgraph_input_names.begin(), subgraph_input_names.end());

  for (auto& entry : node.ImplicitInputDefs()) {
    feed_names.push_back(entry->Name());
  }

  std::vector<OrtDevice> feed_locations;
  feed_locations.resize(feed_names.size());

  for (size_t i = 0, end = feed_names.size(); i < end; ++i) {
    if (i >= subgraph_input_names.size()) {  // implicit inputs
      const auto& location = utils::FindMemoryInfoForValue(session_state, feed_names[i]);
      feed_locations[i] = location.device;
    } else {
      feed_locations[i] = default_location.device;
    }
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, subgraph_output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // setup the locations where we want the subgraph output to end up on
  std::vector<const OrtMemoryInfo*> fetch_locations;
  fetch_locations.reserve(num_subgraph_outputs);

  // past state need to be where we can feed them in to the next iteration, so set the fetch location to match the feed location.
  for (int i = 0; i < num_subgraph_outputs; ++i) {
    fetch_locations.push_back(&default_location);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager_ = std::move(ffm);

  // Check subgraph only need once so put in Setup function.
  auto& inputs = subgraph.GetInputs();
  auto& outputs = subgraph.GetOutputs();
  ORT_RETURN_IF_ERROR(Validate(inputs, outputs));

  return Status::OK();
}

const IExecutionProvider* GptSubgraph::GetProvider() const {
  const ExecutionProviders& providers = session_state_->GetExecutionProviders();
  const IExecutionProvider* cpu_provider = providers.Get(onnxruntime::kCpuExecutionProvider);
  const IExecutionProvider* cuda_provider = providers.Get(onnxruntime::kCudaExecutionProvider);
  const IExecutionProvider* provider = cuda_provider ? cuda_provider : cpu_provider;
  return provider;
}

Status GptSubgraph::CreateInitialFeeds(
    const Tensor& input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    OrtValue& expanded_input_ids,
    std::vector<OrtValue>& feeds,
    const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
    const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
    IAllocatorUniquePtr<char>& buffer) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  const IExecutionProvider* provider = GetProvider();

  const TensorShape& input_ids_shape = input_ids.Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];

  // Subgraph inputs:
  //   input_ids: shape (B, S) wher B is batch size, and S is sequence length
  //   position_ids: shape (B, S)
  //   attention_mask: shape (B, P+S), where past_sequence_length (P) is 0
  // After expansion, their shapes will become (B, M*S), where M is num_beams.

  // Allocate subgraph inputs to be same device as input_ids
  AllocatorPtr cpu_alloactor = session_state_->GetAllocator(input_ids.Location());

  // Store allocator, which will be used in remaining feeds
  auto default_allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  allocator_ = default_allocator;

  // Initialize empty past state
  auto past_type = IsOutputFloat16() ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  int64_t past_state_dims[] = {2, batch_size * num_beams, num_heads, 0, head_size};
  TensorShape past_shape(&past_state_dims[0], 5);
  OrtValue empty_past;
  Tensor::InitOrtValue(past_type, past_shape, default_allocator, empty_past);

  // The ordering is the same as used in Setup
  feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  OrtValue expanded_position_ids;
  OrtValue expanded_attention_mask;
  ORT_RETURN_IF_ERROR(create_inputs_func(&input_ids, num_beams, pad_token_id, sequence_lengths, cpu_alloactor, expanded_input_ids, expanded_position_ids, expanded_attention_mask));

  ORT_RETURN_IF_ERROR(add_to_feeds_func(provider, expanded_input_ids, expanded_position_ids, expanded_attention_mask, feeds, buffer));

  // The remaing inputs are past state.
  for (int i = 3; i < num_subgraph_inputs; ++i) {
    feeds.push_back(empty_past);
  }

  // pass in implicit inputs
  for (const auto* entry : implicit_inputs) {
    feeds.push_back(*entry);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
