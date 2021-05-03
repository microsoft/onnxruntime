// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>

namespace onnxruntime {
namespace openvino_ep {

// Holds context applicable to the entire EP instance.
struct GlobalContext {
  InferenceEngine::Core ie_core;
  bool is_wholly_supported_graph = false;
  bool enable_vpu_fast_compile = false;
  bool use_compiled_network = false;
  bool disable_graph_partition = false;
  size_t num_of_threads;
  std::string device_type;
  std::string precision_str;
  std::string device_id;
  std::string blob_dump_path;
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::vector<std::string> deviceTags = {"0", "1", "2", "3", "4", "5", "6", "7"};
  std::string onnx_model_name;
  std::string onnx_model_path_name;
  int onnx_opset_version;
};

// Holds context specific to subgraph.
struct SubGraphContext {
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_vpu_config = false;
  bool is_constant = false;
  std::string subgraph_name;
  std::vector<int> input_indexes;
  std::unordered_map<std::string, int> input_names;
  std::unordered_map<std::string, int> output_names;
  InferenceEngine::Precision precision;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
