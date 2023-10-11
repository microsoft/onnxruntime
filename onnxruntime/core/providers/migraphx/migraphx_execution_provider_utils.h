// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

bool IsGraphInput(const GraphViewer& graph, const std::string& name) {
  const auto& graph_inputs = graph.GetInputs();
  std::vector<std::string> input_names(graph_inputs.size());
  std::transform(graph_inputs.begin(), graph_inputs.end(), input_names.begin(), [](auto in) {
    return in->Name();
  });
  return (std::find(input_names.begin(), input_names.end(), name) != input_names.end());
}

bool IsGraphInitializer(const GraphViewer& graph, const std::string& name, bool check_outer_scope = true) {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  return graph.GetInitializedTensor(name, initializer);
}

const Node* GetInputNode(const Node& node, int arg_index) {
  int index = 0;
  for (auto nit = node.InputNodesBegin(); nit != node.InputNodesEnd(); ++nit, ++index) {
    if (index == arg_index) {
      return &(*nit);
    }
  }

  return nullptr;
}

std::size_t getNodeInputNum(const Node& node) {
  std::size_t node_num = 0;
  for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
    node_num++;
  }

  return node_num;
}

bool isInputNode(const Node* node, const std::string& name) {
  auto outputs = node->OutputDefs();
  return std::any_of(outputs.begin(), outputs.end(), [&](auto out) {
    return (out->Name() == name);
  });
}

bool canEvalShapeGeneral(const GraphViewer& graph, const Node* node, std::vector<NodeIndex>& input_nodes) {
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

bool canEvalNodeArgument(const GraphViewer& graph, const Node* node, std::vector<std::size_t> indices, std::vector<NodeIndex>& input_nodes) {
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
bool ReadDynamicRange(const std::string file_name, const bool is_calibration_table, std::unordered_map<std::string, float>& dynamic_range_map) {
  std::ifstream infile(file_name, std::ios::binary | std::ios::in);
  if (!infile) {
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
          unsigned long scale_int = std::strtoul(str.c_str(), nullptr, 16);
          float scale_float = ConvertSinglePrecisionIEEE754ToFloat(scale_int);
          float dynamic_range = scale_float * 127.0f;
          dynamic_range_map[tensor_name] = dynamic_range;
        }
      } else {
        throw std::runtime_error("This is not a TensorRT generated calibration table " + file_name);
      }
    }
  } else {
    // ORT generated calibration table
    infile.seekg(0, std::ios::end);
    size_t length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data{new char[length]};
    infile.read((char*)data.get(), length);
    infile.close();
    auto flat_table = flatbuffers::GetRoot<CalTableFlatBuffers::TrtTable>((const uint8_t*)data.get());
    auto flat_dict = flat_table->dict();
    for (size_t i = 0, end = flat_dict->size(); i < end; ++i) {
      flatbuffers::uoffset_t idx = static_cast<flatbuffers::uoffset_t>(i);
      dynamic_range_map[flat_dict->Get(idx)->key()->str()] = std::stof(flat_dict->Get(idx)->value()->str());
    }
  }
  return true;
}

}  // namespace onnxruntime
