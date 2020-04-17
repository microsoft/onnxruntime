// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>
// IE defines a macro 'OPTIONAL' that conflicts the remaining headers using MSVC
#if defined(_MSC_VER)
#undef OPTIONAL
#endif

namespace onnxruntime {
namespace openvino_ep {

// Holds context applicable to the entire EP instance.
struct GlobalContext {
  InferenceEngine::Core ie_core;
  bool is_wholly_supported_graph = false;
};

// Holds context specific to subgraph.
struct SubGraphContext {
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_vpu_config = false;
  std::string subgraph_name;
  std::vector<int> input_indexes;
  std::unordered_map<std::string, int> output_names;
  std::string device_id;
  InferenceEngine::Precision precision;
  std::string precision_str;
};

}  // namespace openvino_ep
}  // namespace onnxruntime