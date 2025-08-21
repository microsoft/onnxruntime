// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "flatbuffers/idl.h"
#include "core/providers/migraphx/ort_trt_int8_cal_table.fbs.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/common/path_string.h"
#include "core/framework/murmurhash3.h"

namespace fs = std::filesystem;

namespace onnxruntime {

inline bool IsGraphInput(const GraphViewer& graph, const std::string& name) {
  const auto& graph_inputs = graph.GetInputs();
  std::vector<std::string> input_names(graph_inputs.size());
  std::transform(graph_inputs.begin(), graph_inputs.end(), input_names.begin(), [](auto in) {
    return in->Name();
  });
  return (std::find(input_names.begin(), input_names.end(), name) != input_names.end());
}

inline bool IsGraphInitializer(const GraphViewer& graph, const std::string& name, [[maybe_unused]] bool check_outer_scope = true) {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  return graph.GetInitializedTensor(name, initializer);
}

inline const Node* GetInputNode(const Node& node, int arg_index) {
  int index = 0;
  for (auto nit = node.InputNodesBegin(); nit != node.InputNodesEnd(); ++nit, ++index) {
    if (index == arg_index) {
      return &(*nit);
    }
  }

  return nullptr;
}

inline std::size_t getNodeInputNum(const Node& node) {
  std::size_t node_num = 0;
  for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
    node_num++;
  }

  return node_num;
}

inline bool isInputNode(const Node* node, const std::string& name) {
  auto outputs = node->OutputDefs();
  return std::any_of(outputs.begin(), outputs.end(), [&](auto out) {
    return (out->Name() == name);
  });
}

inline bool canEvalShapeGeneral(const GraphViewer& graph, const Node* node, std::vector<NodeIndex>& input_nodes) {
  if (node == nullptr) {
    return false;
  }

  std::vector<const Node*> in_nodes;
  for (auto nit = node->InputNodesBegin(); nit != node->InputNodesEnd(); ++nit) {
    in_nodes.push_back(&(*nit));
  }

  if (node->OpType() == "Shape") {
    input_nodes.push_back(node->Index());
    return true;
  }

  auto inputs = node->InputDefs();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const std::string& input_name = inputs.at(i)->Name();
    // If it is an initializer, it can be constant folded
    if (IsGraphInitializer(graph, input_name)) {
      continue;
    }

    // Input for sure cannot be constant folded
    if (IsGraphInput(graph, input_name)) {
      return false;
    }

    // find the node corresponding to the name
    auto nit = std::find_if(in_nodes.begin(), in_nodes.end(), [&](auto n) {
      return isInputNode(n, input_name);
    });
    if (nit == in_nodes.end()) {
      return false;
    }

    auto input_node = (*nit);
    // shape node, it is OK
    if (input_node->OpType() == "Shape") {
      continue;
    }

    if (canEvalShapeGeneral(graph, input_node, input_nodes)) {
      continue;
    }

    return false;
  }

  input_nodes.push_back(node->Index());
  return true;
}

inline bool canEvalNodeArgument(const GraphViewer& graph,
                                const Node* node,
                                std::vector<std::size_t> indices,
                                std::vector<NodeIndex>& input_nodes) {
  input_nodes.clear();
  std::vector<const Node*> in_nodes;
  for (auto nit = node->InputNodesBegin(); nit != node->InputNodesEnd(); ++nit) {
    in_nodes.push_back(&(*nit));
  }

  auto inputs = node->InputDefs();
  for (auto index : indices) {
    // an initializer itself is a constant
    auto input_name = inputs.at(index)->Name();
    if (IsGraphInitializer(graph, input_name)) {
      continue;
    }

    // Input cannot be constant folded
    if (IsGraphInput(graph, input_name)) {
      return false;
    }

    // find the node corresponding to the name
    auto nit = std::find_if(in_nodes.begin(), in_nodes.end(), [&](auto n) {
      return isInputNode(n, input_name);
    });
    if (nit == in_nodes.end()) {
      return false;
    }

    if (!canEvalShapeGeneral(graph, *nit, input_nodes)) {
      return false;
    }
  }

  return true;
}

inline float ConvertSinglePrecisionIEEE754ToFloat(uint32_t input) {
  int s = (input >> 31) & 0x01;
  int e = ((input & 0x7f800000) >> 23) - 127;
  int p = -1;
  double m = 0.0;
  for (int i = 0; i < 23; ++i) {
    m += ((input >> (23 - i - 1)) & 0x01) * pow(2.0, p--);
  }
  return static_cast<float>((s ? -1 : 1) * pow(2.0, e) * (m + 1.0));
}

/*
 * Read calibration table for INT8 quantization
 * Two kind of calibration tables are supported,
 * 1. ORT generated calibration table
 * The table is pre-serialized by flatbuffers.
 * Each entry in the table is a key-value pair,
 * key: tensor name, value: maximum absolute value in floating point
 * For example,
 *   data_0 2.008338
 *   ...
 * 2. Native TensorRT generated calibration table
 * Data format is defined by TensorRT as,
 * tensor name : scale in 32-bit single precision IEEE754 format
 * For example,
 *   TRT-7103-EntropyCalibration2
 *   data_0: 4000889d
 *   ...
 *
 * Taken from the tensorRT EP to allow MIGraphX EP to reuse calibration tables for existing models
 *
 */
inline bool ReadDynamicRange(const std::filesystem::path& filename,
                             const bool is_calibration_table,
                             std::unordered_map<std::string,
                                                float>& dynamic_range_map) {
  std::ifstream infile{filename, std::ios::binary | std::ios::in};
  if (!infile.good()) {
    return false;
  }

  if (is_calibration_table) {
    // Native TensorRT generated calibration table
    std::string line;
    char delim = ':';
    if (std::getline(infile, line)) {
      std::istringstream first_line(line);
      std::string version;
      std::getline(first_line, version, delim);
      std::size_t found = version.find("TRT-");
      if (found != std::string::npos) {
        while (std::getline(infile, line)) {
          std::istringstream in_line(line);
          std::string str;
          std::getline(in_line, str, delim);
          std::string tensor_name = str;
          std::getline(in_line, str, delim);
          uint32_t scale_int = std::strtoul(str.c_str(), nullptr, 16);
          float scale_float = ConvertSinglePrecisionIEEE754ToFloat(scale_int);
          float dynamic_range = scale_float * 127.0f;
          dynamic_range_map[tensor_name] = dynamic_range;
        }
      } else {
        throw std::runtime_error("This is not a TensorRT generated calibration table " + filename.string());
      }
    }
  } else {
    // ORT generated calibration table
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data{new char[length]};
    infile.read(reinterpret_cast<char*>(data.get()), length);
    infile.close();
    auto flat_table = flatbuffers::GetRoot<CalTableFlatBuffers::TrtTable>(reinterpret_cast<char*>(data.get()));
    auto flat_dict = flat_table->dict();
    for (size_t i = 0, end = flat_dict->size(); i < end; ++i) {
      flatbuffers::uoffset_t idx = static_cast<flatbuffers::uoffset_t>(i);
      dynamic_range_map[flat_dict->Get(idx)->key()->str()] = std::stof(flat_dict->Get(idx)->value()->str());
    }
  }
  return true;
}

/*
 * Get cache by name
 *
 */
inline std::filesystem::path GetCachePath(const std::filesystem::path& root, std::string_view name) {
  return root.empty() ? std::filesystem::path{ToPathString(name)} : root / ToPathString(name);
}

inline std::string GenerateGraphId(const GraphViewer& graph_viewer) {
  HashValue model_hash;

  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  const Graph& main_graph = *cur_graph;
  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
  };

  // Use the model's file name instead of the entire path to avoid cache regeneration if a path changes
  const fs::path path{main_graph.ModelPath()};

  if (path.has_filename()) {
    const auto model_name = path.filename().string();

    LOGS_DEFAULT(INFO) << "Model name is '" << model_name << "'";
    // Ensure enough characters are hashed in case model names are too short
    const size_t model_name_length = model_name.length();
    constexpr size_t hash_string_length = 500;
    std::string repeat_model_name = model_name;
    for (size_t i = model_name_length; i > 0 && i < hash_string_length; i += model_name_length) {
      repeat_model_name += model_name;
    }
    hash_str(repeat_model_name);
  } else {
    LOGS_DEFAULT(INFO) << "Model path is empty";
  }

  // fingerprint current graph by hashing graph inputs
  for (const auto* node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    hash_str(node_arg->Name());
  }

  // hashing outputs, inputs and inputs shapes of each node
  const int number_of_ort_nodes = graph_viewer.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto& index : nodes_vector) {
    const auto& node = graph_viewer.GetNode(node_index[index]);
    for (const auto* node_arg : node->OutputDefs()) {
      if (node_arg != nullptr && node_arg->Exists()) {
        hash_str(node_arg->Name());
      }
    }
    for (const auto* node_arg : node->InputDefs()) {
      if (node_arg != nullptr && node_arg->Exists()) {
        hash_str(node_arg->Name());
        if (node_arg->Shape() == nullptr) {
          continue;
        }
        int dim_size = node_arg->Shape()->dim_size();
        for (int i = 0; i < dim_size; i++) {
          hash_str(std::to_string(node_arg->Shape()->dim(i).dim_value()));
        }
      }
    }
  }

#ifdef __linux__
  hash_str("LINUX");
#elif defined(_WIN32)
  hash_str("WINDOWS");
#endif

  model_hash = hash[0] | static_cast<uint64_t>(hash[1]) << 32;

  std::array<char, sizeof(HashValue) << 1> s{};
  auto [ptr, ec] = std::to_chars(s.data(), s.data() + s.size(), model_hash, 16);
  return std::string{s.data(), ptr};
}

inline std::string_view TrimLeft(std::string_view sv, int (*fn)(int) = std::isspace) {
  return sv.substr(0, sv.end() - std::find_if(sv.begin(), sv.end(), [fn](int ch) {
                        return fn(ch);
                      }));
}

inline std::string_view TrimRight(std::string_view sv, int (*fn)(int) = std::isspace) {
  return sv.substr(sv.end() - std::find_if(sv.rbegin(), sv.rend(), [fn](int ch) {
                                return fn(ch);
                              }).base());
}

inline std::string_view Trim(std::string_view sv, int (*fn)(int) = std::isspace) {
  return TrimRight(TrimLeft(sv, fn), fn);
}

inline int ToInteger(const std::string_view sv) {
  int result = 0;
  if (auto [_, ec] = std::from_chars(sv.data(), sv.data() + sv.length(), result); ec == std::errc()) {
    return result;
  }
  ORT_THROW("invalid input for conversion to integer");
}

}  // namespace onnxruntime
