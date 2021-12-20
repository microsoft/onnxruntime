// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "gsl/gsl"
#include "gpt_subgraph.h"
#include "dump_tensor.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef _MSC_VER
// Could reduce the chance of arithmetic overflow. TODO: fix it
#pragma warning(disable : 26451)
#endif
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace transformers {

GptSubgraph::GptSubgraph(
    const onnxruntime::Node& node_in,
    const std::string& attribute_name,
    const GraphViewer& subgraph_in)
    : node(node_in), attribute(attribute_name), subgraph(subgraph_in), allocator_(nullptr) {
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

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                "subgraph input 0 (input_ids) shall have int64 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                "subgraph input 1 (position_ids) shall have int64 type");
  // TODO: support float16
  ORT_RETURN_IF(subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
                "subgraph input 2 (attention_mask) shall have float type");
  ORT_RETURN_IF(subgraph_inputs[3]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
                "subgraph input 3 (past_0) shall have float type");
  ORT_RETURN_IF(subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
                "subgraph output 0 (logits) shall have float type");
  ORT_RETURN_IF(subgraph_outputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
                "subgraph output 1 (present_0) shall have float type");

  return Status::OK();
}

Status GptSubgraph::Setup(const SessionState& session_state,
                          const SessionState& subgraph_session_state) {
  session_state_ = &session_state;
  subgraph_session_state_ = &subgraph_session_state;

  std::vector<std::string> feed_names;
  feed_names.reserve(num_subgraph_inputs + num_implicit_inputs);

  // First, get the location of input_ids of current operator.
  const auto& node_inputs = node.InputDefs();
  const OrtMemoryInfo& input_ids_location = utils::FindMemoryInfoForValue(session_state, node_inputs[0]->Name());

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
      feed_locations[i] = input_ids_location.device;
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
    fetch_locations.push_back(&input_ids_location);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager_ = std::move(ffm);

  // Check subgraph only need once so put in Setup function.
  auto& inputs = subgraph.GetInputs();
  auto& outputs = subgraph.GetOutputs();
  ORT_RETURN_IF_ERROR(Validate(inputs, outputs));

  return Status::OK();
}

void GptSubgraph::CreateInitialFeeds(
    const Tensor& input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int num_beams,
    int pad_token_id,
    gsl::span<int64_t>& next_positions,
    std::vector<OrtValue>& feeds) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  // Subgraph inputs:
  //   input_ids: shape (B, S) wher B is batch size, and S is sequence length
  //   position_ids: shape (B, S)
  //   attention_mask: shape (B, P+S), where past_sequence_length (P) is 0
  // After expansion, their shapes will become (B, M*S), where M is num_beams.

  // Allocate subgraph inputs to be same device as input_ids
  AllocatorPtr alloactor = session_state_->GetAllocator(input_ids.Location());

  // Store allocator, which is needed in ExpandInputs.
  allocator_ = alloactor;

  const TensorShape& input_ids_shape = input_ids.Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];
  const int64_t& sequence_length = input_ids_shape[1];

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = DataTypeImpl::GetType<int64_t>();

  // input_ids for subgraph is int64, so we need Cast input_ids from int32 to int64.
  OrtValue subgraph_input_ids;
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, subgraph_input_ids);

  int64_t* subgraph_input_data = subgraph_input_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  const int32_t* source = input_ids.Data<int32_t>();
  int64_t* target = subgraph_input_data;
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
  // Set position id to be 0 for pad tokens, and cumulated sum of mask in a batch for other tokens
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();
  int64_t* position_data = position_ids.GetMutable<Tensor>()->MutableData<int64_t>();
  source = input_ids.Data<int32_t>();
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
      next_positions[i * num_beams + k] = abs_position;
    }
  }

  // Initialize empty past state
  auto past_type = DataTypeImpl::GetType<float>();
  int64_t past_state_dims[] = {2, batch_size * num_beams, num_heads, 0, head_size};
  TensorShape past_shape(&past_state_dims[0], 5);
  OrtValue empty_past;
  Tensor::InitOrtValue(past_type, past_shape, allocator_, empty_past);

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length) for input_ids, position_ids and attention_mask
  // TODO: Try expand inputs/outputs after first subgraph call instead. That may get better peroformance, but more complex to implement.
  OrtValue expanded_input_ids = ExpandInputs(subgraph_input_ids, num_beams);
  OrtValue expanded_position_ids = ExpandInputs(position_ids, num_beams);
  OrtValue expanded_attention_mask = ExpandInputs(attention_mask, num_beams);

  // The ordering is the same as used in Setup
  feeds.reserve(num_subgraph_inputs + num_implicit_inputs);
  feeds.push_back(expanded_input_ids);
  feeds.push_back(expanded_position_ids);
  feeds.push_back(expanded_attention_mask);

  // The remaing inputs are past state.
  for (int i = 3; i < num_subgraph_inputs; ++i) {
    feeds.push_back(empty_past);
  }

  // pass in implicit inputs
  for (const auto* entry : implicit_inputs) {
    feeds.push_back(*entry);
  }
}

OrtValue GptSubgraph::ExpandInputs(const OrtValue& input, int num_beams) const {
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
  Tensor::InitOrtValue(element_type, expanded_shape, allocator_, expanded);

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

// TODO: support float16
void GptSubgraph::PickPastState(const std::vector<OrtValue>& last_outputs,
                                std::vector<OrtValue>& next_inputs,
                                gsl::span<const int64_t>& beam_indices) {
  for (int i = 3; i < num_subgraph_inputs; ++i) {
    const OrtValue& present = last_outputs[i - 2];  // shape is like (2, batch_beam_size, 12, past_seq_len, 64)
    const TensorShape& past_shape = present.Get<Tensor>().Shape();

    // Create a tensor with same shape.
    OrtValue past;
    auto past_type = DataTypeImpl::GetType<float>();
    Tensor::InitOrtValue(past_type, past_shape, allocator_, past);

    auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
    auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

    gsl::span<float> past_span = past.GetMutable<Tensor>()->MutableDataAsSpan<float>();
    gsl::span<const float> present_span = present.Get<Tensor>().DataAsSpan<float>();
    for (gsl::index j = 0; j < beam_indices.length(); j++) {
      int64_t beam_index = beam_indices[j];
      gsl::span<const float> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      gsl::span<const float> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      gsl::span<float> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      gsl::span<float> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      gsl::copy(present_key, past_key);
      gsl::copy(present_value, past_value);
#ifdef DEBUG_BEAM_SEARCH
      if (i == 3)  // only dump past_0
      {
        DumpString("past_key of beam", static_cast<int>(j), true);
        DumpTensor<float>(nullptr, past_key.data(), 1, static_cast<int>(block_size_per_beam));

        DumpString("past_value of beam", static_cast<int>(j), true);
        DumpTensor<float>(nullptr, past_value.data(), 1, static_cast<int>(block_size_per_beam));
      }
#endif
    }

    next_inputs[i] = past;
  }
}

Status GptSubgraph::UpdateFeeds(
    const std::vector<OrtValue>& last_outputs,
    std::vector<OrtValue>& next_inputs,
    int current_length,
    gsl::span<int64_t>& next_positions,
    gsl::span<const int64_t> beam_next_tokens,
    gsl::span<const int64_t> beam_indices,
    int num_beams) {
  // last_outputs: logits, present_0, present_1, ...
  // next_inputs: input_ids, position_id, attention_mask, past_0, past_1

  // The following updates inputs for subgraph
  // TODO: Reuse buffer for input_ids and position_ids to reduce memory allocation.

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(beam_next_tokens.length());
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
    position_data[i] = next_positions[i];
    next_positions[i]++;
  }
  next_inputs[1] = position_ids;

  // Update attention mask
  const OrtValue& old_mask = next_inputs[2];
  const float* old_mask_data = old_mask.Get<Tensor>().Data<float>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  TensorShape mask_shape(&mask_dims[0], 2);
  OrtValue attention_mask;
  auto mask_type = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(mask_type, mask_shape, allocator_, attention_mask);
  float* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<float>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1.0f;
  }
  next_inputs[2] = attention_mask;

#ifdef DEBUG_BEAM_SEARCH
  DumpOrtValue("input_ids", input_ids);
  DumpOrtValue("position_ids", position_ids);
  DumpOrtValue("attention_mask", attention_mask);
#endif

  // Update past state
  if (num_beams == 1) {
    // feed present_* output to past_* inputs one by one
    for (int i = 3; i < num_subgraph_inputs; ++i) {
      next_inputs[i] = last_outputs[i - 2];
    }
  } else {
    PickPastState(last_outputs, next_inputs, beam_indices);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
