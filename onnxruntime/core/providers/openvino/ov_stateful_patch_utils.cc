// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_stateful_patch_utils.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace openvino_ep {

void LogBasicModelInfo(const std::shared_ptr<const ov::Model>& model) {
  std::cout << "Model Name: " << model->get_friendly_name() << std::endl;

  // Log detailed information about model inputs and outputs
  auto inputs = model->inputs();
  auto outputs = model->outputs();

  std::cout << "\tInputs: " << std::endl;
  for (const ov::Output<const ov::Node>& input : inputs) {
    const std::string name = input.get_any_name();
    const ov::element::Type type = input.get_element_type();
    const ov::PartialShape shape = input.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(input);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  std::cout << "\tOutputs: " << std::endl;
  for (const ov::Output<const ov::Node>& output : outputs) {
    const std::string name = output.get_any_name();
    const ov::element::Type type = output.get_element_type();
    const ov::PartialShape shape = output.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(output);

    std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
  }

  return;
}

bool ModelHasInputOutputNames(std::shared_ptr<ov::Model> model, const std::string& name_to_match) {
  for (const ov::Output<ov::Node>& input : model->inputs()) {
    auto& names = input.get_names();

    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  for (const ov::Output<ov::Node>& output : model->outputs()) {
    auto& names = output.get_names();
    for (auto& name : names) {
      if (name == name_to_match) {
        return true;
      }
    }
  }

  return false;
}

std::string GetInputOutputName(std::shared_ptr<ov::Model> ov_model,
                               const std::vector<std::string>& candidate_names) {
  for (const auto& name : candidate_names) {
    if (ModelHasInputOutputNames(ov_model, name)) {
      return name;
    }
  }
  // Return the first candidate as default if none are found
  return candidate_names.empty() ? "" : candidate_names[0];
}

void FuseCacheReorder(std::shared_ptr<ov::Model> ov_model,
                      std::vector<std::string>& not_kv_inputs,
                      const std::vector<std::string>& key_value_input_names,
                      int gather_dim,
                      const bool should_add_kvcache_reorder) {
  if (ModelHasInputOutputNames(ov_model, "beam_idx")) {
    throw std::runtime_error("Model already has fused cache");
  }

  // Define input name candidates in priority order
  const std::vector<std::string> input_name_candidates = {
      "inputs_embeds",                       // Default fallback
      "input_ids",                           // Most common
      "input_hidden_states",                 // Alternative
      "/model/embed_tokens/Gather_output_0"  // Specific model type
  };

  std::string main_input_name = GetInputOutputName(ov_model, input_name_candidates);

  auto input_batch = ov_model->input(main_input_name).get_partial_shape()[0];
  auto update_shape = ov_model->input(key_value_input_names[0]).get_partial_shape();

  auto beam_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, ov::PartialShape({std::move(input_batch)}));
  beam_idx->set_friendly_name("beam_idx");
  beam_idx->output(0).get_tensor().add_names({"beam_idx"});
  ov_model->add_parameters({beam_idx});
  not_kv_inputs.push_back(beam_idx->get_friendly_name());

  std::shared_ptr<ov::opset13::Parameter> src_idx;
  std::shared_ptr<ov::opset13::Parameter> dst_idx;

  if (should_add_kvcache_reorder) {
    src_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, ov::PartialShape({update_shape[2]}));
    src_idx->set_friendly_name("src_idx");
    src_idx->output(0).get_tensor().add_names({"src_idx"});
    ov_model->add_parameters({src_idx});
    not_kv_inputs.push_back(src_idx->get_friendly_name());

    dst_idx = std::make_shared<ov::opset13::Parameter>(ov::element::i32, update_shape);
    dst_idx->set_friendly_name("dst_idx");
    dst_idx->output(0).get_tensor().add_names({"dst_idx"});
    ov_model->add_parameters({dst_idx});
    not_kv_inputs.push_back(dst_idx->get_friendly_name());
  }

  // Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
  for (const auto& input_name : key_value_input_names) {
    auto parameter_output_port = ov_model->input(input_name);
    auto consumers = parameter_output_port.get_target_inputs();

    auto gather_op =
        std::make_shared<ov::opset13::Gather>(parameter_output_port,
                                              beam_idx,
                                              ov::opset13::Constant::create(ov::element::i64, {}, {gather_dim}));

    std::shared_ptr<ov::Node> output_node;
    if (should_add_kvcache_reorder) {
      auto updatekv_gather_op =
          std::make_shared<ov::opset13::Gather>(gather_op,
                                                src_idx,
                                                ov::opset13::Constant::create(ov::element::i64, {}, {2}));

      auto updatekv_op = std::make_shared<ov::opset12::ScatterElementsUpdate>(gather_op,
                                                                              dst_idx,
                                                                              updatekv_gather_op,
                                                                              ov::opset13::Constant::create(ov::element::i64, {}, {2}));
      output_node = updatekv_op;
    } else {
      output_node = gather_op;
    }

    // Replace the source output for all consumers of the input tensor
    for (auto& consumer : consumers) {
      consumer.replace_source_output(output_node->output(0));
    }
  }

  // Validate the modified model
  ov_model->validate_nodes_and_infer_types();
}

void MakeStateful(std::shared_ptr<ov::Model>& ov_model,
                  const std::vector<std::string>& key_value_input_names,
                  const std::vector<std::string>& key_value_output_names) {
  std::map<std::string, std::string> input_output_map;

  // Create mapping for KV-cache inputs and outputs
  for (size_t i = 0; i < key_value_input_names.size(); ++i) {
    input_output_map[key_value_input_names[i]] = key_value_output_names[i];
  }

  // Apply the transformation to make the model stateful
  ov::pass::Manager manager;
  manager.register_pass<ov::pass::MakeStateful>(input_output_map);
  manager.run_passes(ov_model);
}

// Helper function to extract KV patterns from output names dynamically
//
// Example: Given output names ["present_key_cross_0", "present_key_cross_1", "present_value_cross_0", "present_value_cross_1", "logits"]
//   key_value_output_names = ["present_key_cross_0", "present_key_cross_1", "present_value_cross_0", "present_value_cross_1"]
//   unique_patterns = {"key_cross", "value_cross"}
std::pair<std::vector<std::string>, std::unordered_set<std::string>> ExtractKVPatternsFromOutputs(const std::shared_ptr<ov::Model>& model) {
  std::vector<std::string> key_value_output_names;
  std::unordered_set<std::string> unique_patterns;

  const std::string prefix = "present_";
  const size_t prefix_len = prefix.length();
  for (const ov::Output<ov::Node>& output : model->outputs()) {
    const auto& names = output.get_names();
    for (const auto& name : names) {
      if (name.find(prefix) == 0 && name.length() > prefix_len) {
        size_t last_underscore_pos = name.rfind('_');
        // Extract pattern between "present_" and the last underscore
        if (last_underscore_pos != std::string::npos && last_underscore_pos > prefix_len) {
          std::string pattern = name.substr(prefix_len, last_underscore_pos - prefix_len);
          if (!pattern.empty()) {
            unique_patterns.insert(pattern);
            key_value_output_names.push_back(name);
          }
        }
        break;
      }
    }
  }

  if (unique_patterns.size() > 2) {
    ORT_THROW("More than two unique KV patterns found in output names.");
  }
  return std::make_pair(key_value_output_names, unique_patterns);
}

// Main function to extract KV tensors using dynamic pattern matching
//
// Example: Given input names ["input_ids", "attention_mask", "past_key_cross_0", "past_key_cross_1", "past_value_cross_0", "past_value_cross_1"]
//   kv_patterns = {"key_cross", "value_cross"}
//
//   key_value_input_names = ["past_key_cross_0", "past_key_cross_1", "past_value_cross_0", "past_value_cross_1"]
//   not_kv_inputs = ["input_ids", "attention_mask"]
std::pair<std::vector<std::string>, std::vector<std::string>> ExtractInputKVTensors(
    const std::shared_ptr<ov::Model>& model, const std::unordered_set<std::string>& kv_patterns) {
  std::vector<std::string> key_value_input_names;
  std::vector<std::string> not_kv_inputs;

  if (kv_patterns.empty()) {
    // Fallback: use original substring matching
    for (const ov::Output<ov::Node>& input : model->inputs()) {
      const auto& names = input.get_names();
      const std::string input_name = input.get_any_name();

      bool is_kv_input = false;
      for (const auto& name : names) {
        if (name.find("key_values") != std::string::npos ||
            name.find("keys") != std::string::npos ||
            name.find("values") != std::string::npos) {
          key_value_input_names.push_back(name);
          is_kv_input = true;
          break;
        }
      }

      if (!is_kv_input) {
        not_kv_inputs.push_back(input_name);
      }
    }

    return std::make_pair(key_value_input_names, not_kv_inputs);
  }

  // Inline helper function to check if name is matched with provided pattern followed by "_%d"
  auto matches_pattern = [](const std::string& name, const std::string& pattern) -> bool {
    size_t pos = name.find(pattern);
    if (pos == std::string::npos) {
      return false;
    }

    size_t after_pattern = pos + pattern.length();
    if (after_pattern >= name.length() || name[after_pattern] != '_') {
      return false;
    }

    std::string suffix = name.substr(after_pattern + 1);
    return !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
  };

  for (const ov::Output<ov::Node>& input : model->inputs()) {
    auto& names = input.get_names();
    bool found = false;

    // Check if any input name contains either key or value pattern
    for (const auto& name : names) {
      for (const auto& pattern : kv_patterns) {
        if (matches_pattern(name, pattern)) {
          key_value_input_names.push_back(name);
          found = true;
          break;
        }
      }
      if (found) break;
    }

    if (!found) {
      not_kv_inputs.push_back(input.get_any_name());
    }
  }

  return std::make_pair(key_value_input_names, not_kv_inputs);
}

// Updated PatchStatefulDecoder function
void PatchStatefulDecoder(std::shared_ptr<ov::Model> model, const bool should_add_kvcache_reorder) {
  // Use the dynamic pattern-based extraction logic
  auto [key_value_output_names, extracted_patterns] = ExtractKVPatternsFromOutputs(model);
  auto [key_value_input_names, not_kv_inputs] = ExtractInputKVTensors(model, extracted_patterns);

  if (key_value_input_names.empty() || key_value_output_names.empty()) {
    ORT_THROW("No key_value_input_names or key_value_output_names found");
  }

  if (key_value_input_names.size() != key_value_output_names.size()) {
    ORT_THROW("Found different sizes between key_value_input_names (",
              key_value_input_names.size(),
              ") and key_value_output_names (",
              key_value_output_names.size(),
              "). They couldn't be paired.");
  }

  // By default, batch is the 0 - th but chatglm uses 1 - st dimension as batch
  // TODO(ryan): Deduce from a model via ordinal reshape(? ) and topology
  // batch_dim = 1 if config.model_type == "chatglm" and not hasattr(config, "rope_ratio") else 0
  auto batch_dim = 0;

  FuseCacheReorder(model, not_kv_inputs, key_value_input_names, batch_dim, should_add_kvcache_reorder);

  MakeStateful(model, key_value_input_names, key_value_output_names);
}

// Some other utility functions copied from OpenVINO GenAI
bool HasOpWithType(const std::shared_ptr<const ov::Model>& function, const std::string& type_name) {
  for (const auto& op : function->get_ops()) {
    if (op->get_type_name() == type_name) {
      return true;
    }
  }
  return false;
}

std::tuple<std::shared_ptr<ov::Node>, int64_t> FindLLMMatmul(const std::shared_ptr<ov::Model>& model) {
  auto last_node = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
  std::shared_ptr<ov::Node> matmul = ov::as_type_ptr<ov::op::v0::MatMul>(last_node);

  // In the case of PagedAttention, all tokens are moved to the batch dimension,
  // and slicing/gathering must be performed accordingly.
  const bool pa_based_model = HasOpWithType(model, "PagedAttentionExtension");
  int64_t slice_gather_dim = pa_based_model ? 0 : 1;

  // There are several patterns for MatMul we are looking for:
  // MatMul -> Result
  // MatMul -> Add -> Result
  // MatMul -> Transpose -> Result
  // MatMul -> Divide -> Tanh -> Multiply -> Result
  // MatMul -> Convert -> Result
  if (!matmul) {
    if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
    } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
      auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr())->get_axis_vector_val();
      slice_gather_dim = order[slice_gather_dim];
    } else if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(last_node)) {
      if (auto tanh = ov::as_type_ptr<ov::op::v0::Tanh>(multiply->input_value(0).get_node_shared_ptr())) {
        if (auto divide = ov::as_type_ptr<ov::op::v1::Divide>(tanh->input_value(0).get_node_shared_ptr())) {
          matmul = ov::as_type_ptr<ov::op::v0::MatMul>(divide->input_value(0).get_node_shared_ptr());
        }
      }
    } else if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(last_node)) {
      matmul = ov::as_type_ptr<ov::op::v0::MatMul>(convert->input_value(0).get_node_shared_ptr());
    }
  }
  return std::make_tuple(matmul, slice_gather_dim);
}

void ApplySliceBeforeMatmulTransformation(std::shared_ptr<ov::Model> model) {
  std::shared_ptr<ov::Node> matmul = nullptr;
  int64_t slice_gather_dim = -1;
  std::tie(matmul, slice_gather_dim) = FindLLMMatmul(model);

  if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
    auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
    auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
    matmul->input(0).replace_source_output(slice);
  }
}

void UpdateConfig(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
  if (config.count(pair.first) == 0) {
    config.insert(pair);
  }
}

std::optional<ov::Any> PopOption(ov::AnyMap& config, const std::string& option_name) {
  if (auto it = config.find(option_name); it != config.end()) {
    std::optional<ov::Any> found = std::make_optional(it->second);
    config.erase(it);
    return found;
  }
  return std::nullopt;
}

void RenameKey(ov::AnyMap& config, const std::string& old_key, const std::string& new_key) {
  if (config.count(old_key) != 0) {
    auto opt_value = PopOption(config, old_key);
    config[new_key] = opt_value.value();
  }
}

KVAxesPosition GetKVAxesPos(std::shared_ptr<const ov::Model> model) {
  // Sequence length axis in key/values tensors. For most cases, the tensor shape is
  // [batch_size, num_kv_heads, seq_len, head_size]. Therefore, the sequence length axis
  // is usually at index 2, and the batch axis is at index 0.
  KVAxesPosition kv_pos{0u, 2u};

  // "ReadValue" node is KV cache representation in stateful model
  std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

  for (const auto& op : model->get_ops()) {
    // Check input size, as in LoRA adapters case it could be 0
    if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
      continue;
    }

    // Shape example: [-1,4,0,64]
    auto shape = op->get_input_partial_shape(0);

    for (int64_t i = 0; i < shape.rank().get_length(); i++) {
      // Find axis = 0. This would be sequence length axis.
      if (shape[i] == 0) {
        kv_pos.seq_len = i;
      } else if (shape[i].is_dynamic()) {
        // Dynamic axis is a batch
        kv_pos.batch = i;
      }
    }
    break;
  }

  return kv_pos;
}

void UpdateNPUConfig(ov::AnyMap& config, const KVAxesPosition& kv_pos, const KVDesc& kv_desc) {
  UpdateConfig(config, {"NPU_USE_NPUW", "YES"});
  UpdateConfig(config, {"NPUW_LLM", "YES"});

  UpdateConfig(config, {"NPUW_LLM_BATCH_DIM", kv_pos.batch});
  UpdateConfig(config, {"NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len});

  UpdateConfig(config, {"NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len});
  UpdateConfig(config, {"NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len});

  RenameKey(config, "++PREFILL_CONFIG", "++NPUW_LLM_PREFILL_CONFIG");
  RenameKey(config, "++GENERATE_CONFIG", "++NPUW_LLM_GENERATE_CONFIG");
  RenameKey(config, "PREFILL_CONFIG", "NPUW_LLM_PREFILL_CONFIG");
  RenameKey(config, "PREFILL_HINT", "NPUW_LLM_PREFILL_HINT");
  RenameKey(config, "GENERATE_CONFIG", "NPUW_LLM_GENERATE_CONFIG");
  RenameKey(config, "GENERATE_HINT", "NPUW_LLM_GENERATE_HINT");
}

std::optional<ov::Any> PopOptionNew(ov::AnyMap& config, const std::string& option_name) {
  if (auto it = config.find(option_name); it != config.end()) {
    std::optional<ov::Any> found = std::make_optional(it->second);
    config.erase(it);
    return found;
  }
  return std::nullopt;
}

std::optional<uint32_t> PopIntAndCast(ov::AnyMap& config, const std::string& key) {
  auto anyopt = PopOptionNew(config, key);
  if (anyopt.has_value()) {
    const auto any = anyopt.value();
    int64_t value;
    // NB: Integer value coming from python has int64_t datatype
    if (any.is<int64_t>()) {
      value = any.as<int64_t>();
    } else if (any.is<int>()) {
      value = any.as<int>();
    } else {
      OPENVINO_THROW("Failed to extract " + key + ". Type mismatch: expected types: int or int64_t");
    }
    if (value < 0) {
      OPENVINO_THROW(key + " cannot be negative!");
    }
    return std::make_optional(static_cast<uint32_t>(value));
  }
  return std::nullopt;
}

bool IsStateful(const std::shared_ptr<ov::Model>& model) {
  for (auto&& ptr : model->get_ordered_ops()) {
    if (ov::is_type<ov::op::v3::ReadValue>(ptr) ||
        ov::is_type<ov::op::v6::ReadValue>(ptr) ||
        ov::is_type<ov::op::v3::Assign>(ptr) ||
        ov::is_type<ov::op::v6::Assign>(ptr)) {
      return true;
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
