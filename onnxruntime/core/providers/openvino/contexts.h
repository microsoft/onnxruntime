// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

// Holds context applicable to the entire EP instance.
struct GlobalContext {
  OVCore ie_core;
  bool is_wholly_supported_graph = false;
  bool enable_opencl_throttling = false;
  bool disable_dynamic_shapes = false;
  bool ep_context_embed_mode = false;
  bool export_ep_ctx_blob = false;
  bool enable_qdq_optimizer = false;
  bool disable_cpu_fallback = false;
  bool has_external_weights = false;
  size_t num_of_threads;
  std::string device_type;
  std::string precision_str;
  std::string model_precision;
  std::string cache_dir;
  std::map<std::string, ov::AnyMap> load_config;
  std::string model_priority = "DEFAULT";
  int num_streams;
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::string onnx_model_name;
  std::string onnx_model_path_name;
  int onnx_opset_version;
  void* context = 0;
  bool use_api_2;
  std::vector<int> OpenVINO_Version = {};  // Ov Major and OV minor version from OV headers
};

// Holds context specific to subgraph.
struct SubGraphContext {
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_npu_config = false;
  bool is_constant = false;
  void* context = 0;
  std::string subgraph_name;
  std::vector<int> input_indexes;
  std::unordered_map<std::string, int> input_names;
  std::unordered_map<std::string, int> output_names;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
