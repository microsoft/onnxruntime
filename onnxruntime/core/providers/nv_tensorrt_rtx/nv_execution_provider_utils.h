// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <filesystem>
#include "flatbuffers/idl.h"
#include "nv_includes.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/common/path_string.h"
#include "core/framework/murmurhash3.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace fs = std::filesystem;

namespace onnxruntime {

/*
 * Get number of profile setting.
 *
 * profile_min_shapes/profile_max_shapes/profile_opt_shapes may contain multiple profile settings.
 * Note: TRT EP currently only supports one profile setting.
 *
 * {
 *   tensor_a: [[dim_0_value_0, dim_1_value_1, dim_2_value_2]],
 *   tensor_b: [[dim_0_value_3, dim_1_value_4, dim_2_value_5]]
 * }
 *
 */
static int GetNumProfiles(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes) {
  int num_profile = 0;
  for (auto it = profile_shapes.begin(); it != profile_shapes.end(); it++) {
    num_profile = static_cast<int>(it->second.size());
    if (num_profile > 0) {
      break;
    }
  }
  return num_profile;
}

/*
 * Get cache by name
 *
 */
static std::string GetCachePath(const std::string& root, const std::string& name) {
  if (root.empty()) {
    return name;
  } else {
    fs::path path = root;
    path.append(name);
    return path.string();
  }
}

/*
 * Get compute capability
 *
 */
static std::string GetComputeCapability(const cudaDeviceProp& prop) {
  const std::string compute_capability = std::to_string(prop.major * 10 + prop.minor);
  return compute_capability;
}

/**
 * <summary>
 * Helper class to generate engine id via model name/model content/env metadata
 * </summary>
 * <remarks>
 * The TensorRT Execution Provider is used in multiple sessions and the underlying infrastructure caches
 * compiled kernels, so the name must be unique and deterministic across models and sessions.
 * </remarks>
 */
static HashValue TRTGenerateId(const GraphViewer& graph_viewer, std::string trt_version, std::string cuda_version) {
  HashValue model_hash = 0;

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

  // Use the model's file name instead of the entire path to avoid cache regeneration if path changes
  if (main_graph.ModelPath().has_filename()) {
    std::string model_name = PathToUTF8String(main_graph.ModelPath().filename());

    LOGS_DEFAULT(INFO) << "[NvTensorRTRTX EP] Model name is " << model_name;
    // Ensure enough characters are hashed in case model names are too short
    const size_t model_name_length = model_name.size();
    constexpr size_t hash_string_length = 500;
    std::string repeat_model_name = model_name;
    for (size_t i = model_name_length; i > 0 && i < hash_string_length; i += model_name_length) {
      repeat_model_name += model_name;
    }
    hash_str(repeat_model_name);
  } else {
    LOGS_DEFAULT(INFO) << "[NvTensorRTRTX EP] Model path is empty";
  }

  // fingerprint current graph by hashing graph inputs
  for (const auto* node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    hash_str(node_arg->Name());
  }

  // hashing output of each node
  const int number_of_ort_nodes = graph_viewer.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto& index : nodes_vector) {
    const auto& node = graph_viewer.GetNode(node_index[index]);
    for (const auto* node_arg : node->OutputDefs()) {
      if (node_arg->Exists()) {
        hash_str(node_arg->Name());
      }
    }
  }

#ifdef __linux__
  hash_str("LINUX");
#elif defined(_WIN32)
  hash_str("WINDOWS");
#endif

#ifdef ORT_VERSION
  hash_str(ORT_VERSION);
#endif

#ifdef CUDA_VERSION
  hash_str(cuda_version);
#endif

#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR)
  hash_str(trt_version);
#endif

  model_hash = hash[0] | (uint64_t(hash[1]) << 32);

  // return the current unique id
  return model_hash;
}

static bool ValidateProfileShapes(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                                  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                                  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes) {
  if (profile_min_shapes.empty() && profile_max_shapes.empty() && profile_opt_shapes.empty()) {
    return true;
  }

  if ((profile_min_shapes.size() != profile_max_shapes.size()) &&
      (profile_min_shapes.size() != profile_opt_shapes.size()) &&
      (profile_max_shapes.size() != profile_opt_shapes.size())) {
    return false;
  }

  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>::iterator it;
  for (it = profile_min_shapes.begin(); it != profile_min_shapes.end(); it++) {
    auto input_name = it->first;
    auto num_profile = it->second.size();

    // input_name must also be in max/opt profile
    if ((profile_max_shapes.find(input_name) == profile_max_shapes.end()) ||
        (profile_opt_shapes.find(input_name) == profile_opt_shapes.end())) {
      return false;
    }

    // number of profiles should be the same
    if ((num_profile != profile_max_shapes[input_name].size()) ||
        (num_profile != profile_opt_shapes[input_name].size())) {
      return false;
    }
  }

  return true;
}

/*
 * Make input-name and shape as a pair.
 * This helper function is being used by ParseProfileShapes().
 *
 * For example:
 * The input string is "input_id:32x1",
 * after the string is being parsed, the pair object is returned as below.
 * pair("input_id", [32, 1])
 *
 * Return true if string can be successfully parsed or false if string has wrong format.
 */
static bool MakeInputNameShapePair(std::string pair_string, std::pair<std::string, std::vector<int64_t>>& pair) {
  if (pair_string.empty()) {
    return true;
  }

  LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] " << pair_string;

  std::stringstream input_string_stream(pair_string);
  char first_delim = ':';
  char second_delim = 'x';
  std::string input_name;
  std::string shape;
  std::getline(input_string_stream, input_name, first_delim);
  std::getline(input_string_stream, shape, first_delim);

  std::vector<int64_t> shapes;
  std::stringstream shape_string_stream(shape);
  std::string value;
  while (std::getline(shape_string_stream, value, second_delim)) {
    shapes.push_back(std::stoi(value));
  }

  // wrong input string
  if (input_name.empty() || shapes.empty()) {
    return false;
  }

  pair.first = input_name;
  pair.second = shapes;

  return true;
}

/*
 * Parse explicit profile min/max/opt shapes from Nv EP provider options.
 *
 * For example:
 * The provider option is --trt_profile_min_shapes="input_id:32x1,attention_mask:32x1,input_id:32x41,attention_mask:32x41",
 * after string is being parsed, the profile shapes has two profiles and is being represented as below.
 * {"input_id": [[32, 1], [32, 41]], "attention_mask": [[32, 1], [32, 41]]}
 *
 * Return true if string can be successfully parsed or false if string has wrong format.
 */
static bool ParseProfileShapes(std::string profile_shapes_string, std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes) {
  if (profile_shapes_string.empty()) {
    return true;
  }

  std::stringstream input_string_stream(profile_shapes_string);
  char delim = ',';
  std::string input_name_with_shape;  // input_name:shape, ex: "input_id:32x1"
  while (std::getline(input_string_stream, input_name_with_shape, delim)) {
    std::pair<std::string, std::vector<int64_t>> pair;
    if (!MakeInputNameShapePair(input_name_with_shape, pair)) {
      return false;
    }

    std::string input_name = pair.first;
    if (profile_shapes.find(input_name) == profile_shapes.end()) {
      std::vector<std::vector<int64_t>> profile_shape_vector;
      profile_shapes[input_name] = profile_shape_vector;
    }
    profile_shapes[input_name].push_back(pair.second);

    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] " << input_name;
    std::string shape_string = "";
    for (auto v : pair.second) {
      shape_string += std::to_string(v);
      shape_string += ", ";
    }
    LOGS_DEFAULT(VERBOSE) << "[NvTensorRTRTX EP] " << shape_string;
  }

  return true;
}

/*
 * Checks if there is a an element with value `-1` in nvinfer1::Dims
 */
static bool checkTrtDimIsDynamic(nvinfer1::Dims dims) {
  for (int j = 0, end = dims.nbDims; j < end; ++j) {
    if (dims.d[j] == -1) {
      return true;
    }
  }
  return false;
}

/*
 * Checks if an nvinfer1::ITensor signales a dynamic shape,
 * either due to dynamic shapes or due to it being a shape tensor
 */
static bool checkTrtTensorIsDynamic(nvinfer1::ITensor* tensor) {
  if (tensor->isShapeTensor()) {
    return true;
  } else {
    // Execution tensor
    return checkTrtDimIsDynamic(tensor->getDimensions());
  }
}
}  // namespace onnxruntime
