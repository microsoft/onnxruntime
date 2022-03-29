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

// bugbug: refactor. create subgraph class as interface
EncoderSubgraph::EncoderSubgraph(
    const onnxruntime::Node& node_in,
    const std::string& attribute_name,
    const GraphViewer& subgraph_in)
    : node(node_in), attribute(attribute_name), subgraph(subgraph_in), allocator_(nullptr), is_output_float16_(false) {
  num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());

  auto& subgraph_inputs = subgraph.GetInputs();
  auto& subgraph_outputs = subgraph.GetOutputs();

  // inputs: input_ids, attention_mask
  // outputs: encoder_outputs
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

Status EncoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                             const std::vector<const NodeArg*>& subgraph_outputs) {
  ORT_RETURN_IF(num_subgraph_outputs == 1, "bugbug");

  ORT_RETURN_IF(num_subgraph_inputs == 2, "bugbug");

  // bugbug 1,453
  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids", "subgraph input 0 shall be named as input_ids, got: ",
                subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "attention_mask", "subgraph input 1 shall be named as attention_mask, got: ",
                subgraph_inputs[1]->Name());

  // bugbug 1,453,1024
  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "encoder_outputs", "subgraph output 0 shall be named as encoder_outputs, got: ",
                subgraph_outputs[0]->Name());

  // Cannot get the following data
  //num_heads = static_cast<int>(past_shape->dim(2).dim_value());
  //head_size = static_cast<int>(past_shape->dim(4).dim_value());
  //vocab_size = static_cast<int>(logits_shape->dim(2).dim_value());
  //num_layers = static_cast<int>(subgraph_outputs.size()) - 1;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "subgraph input 0 (input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "subgraph input 1 (position_ids) shall have int32 type");

  auto output_type = subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT && output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
                "subgraph output 0 (logits) shall be float or float16 data type");

  is_output_float16_ = (output_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);

  return Status::OK();
}

Status EncoderSubgraph::Setup(const SessionState& session_state,
                              const SessionState& subgraph_session_state) {
  session_state_ = &session_state;
  subgraph_session_state_ = &subgraph_session_state;

  std::vector<std::string> feed_names;
  feed_names.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // Currently, input_ids is in CPU even for CUDA operator, so we have to use logits location as default.
  const OrtMemoryInfo& default_location = utils::FindMemoryInfoForValue(subgraph_session_state, "encoder_outputs");

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

const IExecutionProvider* EncoderSubgraph::GetProvider() const {
  const ExecutionProviders& providers = session_state_->GetExecutionProviders();
  const IExecutionProvider* cpu_provider = providers.Get(onnxruntime::kCpuExecutionProvider);
  const IExecutionProvider* cuda_provider = providers.Get(onnxruntime::kCudaExecutionProvider);
  const IExecutionProvider* provider = cuda_provider ? cuda_provider : cpu_provider;
  return provider;
}

Status EncoderSubgraph::CreateInitialFeeds(
    const Tensor& input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int pad_token_id,
    std::vector<OrtValue>& feeds,
    //const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
    //const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
    IAllocatorUniquePtr<char>& buffer) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  const IExecutionProvider* provider = GetProvider();

  const TensorShape& input_ids_shape = input_ids.Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];

  // Subgraph inputs:
  //   input_ids: shape (B, S) wher B is batch size, and S is sequence length
  //   attention_mask: shape (B, S), all 1

  // Allocate subgraph inputs to be same device as input_ids
  AllocatorPtr cpu_alloactor = session_state_->GetAllocator(input_ids.Location());

  // Store allocator, which will be used in remaining feeds
  auto default_allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  allocator_ = default_allocator;


  // The ordering is the same as used in Setup
  feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  auto element_type = DataTypeImpl::GetType<int32_t>();
  OrtValue encoder_input_ids;
  Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(input_ids)->MutableData<int32_t>(), location, encoder_input_ids);

  OrtValue attention_mask;
  Tensor::InitOrtValue(element_type, input_ids_shape, alloactor, attention_mask);

  // def _prepare_attention_mask_for_generation(
  //     self, input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int
  // ) -> torch.LongTensor:
  //     is_pad_token_in_inputs_ids = (pad_token_id is not None) and (pad_token_id in input_ids)
  //     is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
  //         (eos_token_id is not None) and (pad_token_id != eos_token_id)
  //     )
  //     if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
  //         return input_ids.ne(pad_token_id).long()
  //     return input_ids.new_ones(input_ids.shape)
  int32_t* mask_data = attention_mask.GetMutable<Tensor>()->MutableData<int32_t>();
  const int32_t* word_id = input_ids->Data<int32_t>();
  int32_t* mask = mask_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++, word_id++, mask++) {
      if (*word_id == pad_token_id) {
        *mask = 0;
      } else {
        *mask = 1;
      }
    }
  }

  feeds.push_back(encoder_input_ids);
  feeds.push_back(attention_mask);

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
