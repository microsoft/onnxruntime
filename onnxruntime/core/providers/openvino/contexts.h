// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include "ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

// Holds context applicable to the entire EP instance.
struct GlobalContext {
  OVCore ie_core;
  bool is_wholly_supported_graph = false;
  bool enable_npu_fast_compile = false;
  bool enable_opencl_throttling = false;
  bool disable_dynamic_shapes = false;
  size_t num_of_threads;
  std::string device_type;
  std::string precision_str;
  std::string device_id;
  std::string cache_dir;
  int num_streams;
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::vector<std::string> deviceTags = {"0", "1", "2", "3", "4", "5", "6", "7"};
  std::string onnx_model_name;
  std::string onnx_model_path_name;
  int onnx_opset_version;
  void* context = 0;
  bool use_api_2;
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
  std::string precision;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
