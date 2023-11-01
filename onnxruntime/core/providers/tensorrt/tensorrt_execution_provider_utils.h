// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <unordered_map>
#include <string>
#include <iostream>
#include <filesystem>
#include <experimental/filesystem>
#include "flatbuffers/idl.h"
#include "ort_trt_int8_cal_table.fbs.h"
#include <NvInferVersion.h>
#include "core/providers/cuda/cuda_pch.h"
#include "core/common/path_string.h"
#include "core/framework/murmurhash3.h"

static const std::string EP_CONTEXT_OP_TYPE = "EPContext";
static const std::string EP_CONTEXT_ATTR_EMBED_MODE = "embed_mode";
static const std::string EP_CONTEXT_ATTR_CACHE_CTX = "ep_cache_context";

namespace fs = std::experimental::filesystem;

namespace onnxruntime {

float ConvertSinglePrecisionIEEE754ToFloat(unsigned long input) {
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
 */
bool ReadDynamicRange(const std::string file_name, const bool is_trt_calibration_table, std::unordered_map<std::string, float>& dynamic_range_map) {
  std::ifstream infile(file_name, std::ios::binary | std::ios::in);
  if (!infile) {
    return false;
  }

  if (is_trt_calibration_table) {
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

int GetNumProfiles(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes) {
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>::iterator it;
  for (it = profile_shapes.begin(); it != profile_shapes.end(); it++) {
    return static_cast<int>(it->second.size());
  }
  return 0;
}

/*
 * Seralize engine profile
 * The profile contains min/max shape ranges of dynamic shape dimensions of each input tensor
 * For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
 * has one dynamic shape dimension: dim_1. The data in profile will be,
 * key: tensor_a, value: dim_0 min_shape max_shape dim_2 min_shape max_shape
 * key: tensor_b, value: dim_1 min_shape max_shape
 *
 * [Deprecated] Use SerializeProfileV2
 */
void SerializeProfile(const std::string& file_name, std::unordered_map<std::string, std::unordered_map<size_t, std::pair<int64_t, int64_t>>>& shape_ranges) {
  // Serialize profile
  flexbuffers::Builder builder;
  auto profile_start = builder.StartMap();
  for (auto outer_it = shape_ranges.begin(); outer_it != shape_ranges.end(); ++outer_it) {
    builder.TypedVector(outer_it->first.c_str(), [&] {
      for (auto inner_it = outer_it->second.begin(); inner_it != outer_it->second.end(); ++inner_it) {
        builder.Int(inner_it->first);
        builder.Int(inner_it->second.first);
        builder.Int(inner_it->second.second);
      }
    });
  }
  builder.EndMap(profile_start);
  builder.Finish();

  // Save flexbuffer
  std::ofstream file(file_name, std::ios::binary | std::ios::out);
  auto buf = builder.GetBuffer();
  size_t size = builder.GetSize();
  file.write(reinterpret_cast<const char*>(&buf[0]), size);
  file.close();
}

// Deserialize engine profile
// [Deprecated] Use DeserializeProfileV2
std::unordered_map<std::string, std::unordered_map<size_t, std::pair<int64_t, int64_t>>> DeserializeProfile(std::ifstream& infile) {
  // Load flexbuffer
  infile.seekg(0, std::ios::end);
  size_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data{new char[length]};
  infile.read((char*)data.get(), length);
  infile.close();

  // Deserialize profile
  std::unordered_map<std::string, std::unordered_map<size_t, std::pair<int64_t, int64_t>>> shape_ranges;
  auto tensors_range_entries = flexbuffers::GetRoot((const uint8_t*)data.get(), length).AsMap();
  auto keys = tensors_range_entries.Keys();
  auto values = tensors_range_entries.Values();
  for (size_t i = 0, end = keys.size(); i < end; ++i) {
    auto dim_range_vectors = values[i].AsTypedVector();
    std::unordered_map<size_t, std::pair<int64_t, int64_t>> inner_map;
    for (size_t j = 0, end = dim_range_vectors.size() / 3; j < end; ++j) {
      size_t idx = 3 * j;
      inner_map[dim_range_vectors[idx].AsInt64()] = std::make_pair(dim_range_vectors[idx + 1].AsInt64(), dim_range_vectors[idx + 2].AsInt64());
    }
    shape_ranges[keys[i].AsString().c_str()] = inner_map;
  }
  return shape_ranges;
}

/*
 * Seralize engine profile. (This function starts from ORT 1.15)
 *
 *
 * (1) Single profile case:
 * Assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2,
 * and tensor_b has one dynamic shape dimension: dim_1.
 *
 * The data before serialization will be:
 * {
 *   tensor_a: {
 *     dim_0: [[min_shape_0, max_shape_0, opt_shape_0]],
 *     dim_2: [[min_shape_2, max_shape_2, opt_shape_2]]
 *   },
 *   tensor_b: {
 *     dim_1: [[min_shape_1, max_shape_1, opt_shape_1]]
 *   }
 * }
 *
 * The data after serialization will be:
 * {
 *   tensor_a: [dim_0, min_shape_0, max_shape_0, opt_shape_0, dim_2, min_shape_2, max_shape_2, opt_shape_2]
 *   tensor_b: [dim_1, min_shape_1, max_shape_1, opt_shape_1]
 * }
 *
 *
 * (2) Multiple profiles case:
 * For example, if the data before serialization is:
 * {
 *   tensor_a: {
 *     dim_0: [[min_shape_0, max_shape_0, opt_shape_0], [min_shape_1, max_shape_1, opt_shape_1]]
 *   },
 *   tensor_b: {
 *     dim_1: [[min_shape_2, max_shape_2, opt_shape_2], [min_shape_3, max_shape_3, opt_shape_3]]
 *   }
 * }
 *
 * The data after serialization will be:
 * {
 *   tensor_a: [dim_0, min_shape_0, max_shape_0, opt_shape_0, dim_0, min_shape_1, max_shape_1, opt_shape_1]
 *              |                                          |  |                                          |
 *              ---------------- profile 0 -----------------  ---------------- profile 1 -----------------
 *
 *   tensor_b: [dim_1, min_shape_2, max_shape_2, opt_shape_2, dim_1, min_shape_3, max_shape_3, opt_shape_3]
 *              |                                          |  |                                          |
 *              ---------------- profile 0 -----------------  ---------------- profile 1 -----------------
 * }
 *
 */
void SerializeProfileV2(const std::string& file_name, std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>& shape_ranges) {
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] In SerializeProfileV2()";
  // Serialize profile
  flexbuffers::Builder builder;
  auto tensor_map_start = builder.StartMap();
  for (auto tensor_it = shape_ranges.begin(); tensor_it != shape_ranges.end(); tensor_it++) {  // iterate tensors
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] input tensor is '" << tensor_it->first.c_str() << "'";
    builder.TypedVector(tensor_it->first.c_str(), [&] {
      for (auto dim_it = tensor_it->second.begin(); dim_it != tensor_it->second.end(); dim_it++) {
        size_t num_profiles = dim_it->second.size();
        for (size_t i = 0; i < num_profiles; i++) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] profile #" << i << ", dim is " << dim_it->first;
          builder.Int(dim_it->first);
          builder.Int(dim_it->second[i][0]);
          builder.Int(dim_it->second[i][1]);
          builder.Int(dim_it->second[i][2]);
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << dim_it->first << ", " << dim_it->second[i][0] << ", " << dim_it->second[i][1] << ", " << dim_it->second[i][2];
        }
      }
    });
  }
  builder.EndMap(tensor_map_start);
  builder.Finish();

  // Save flexbuffer
  std::ofstream file(file_name, std::ios::binary | std::ios::out);
  auto buf = builder.GetBuffer();
  size_t size = builder.GetSize();
  file.write(reinterpret_cast<const char*>(&buf[0]), size);
  file.close();
}

/*
 * Deserialize engine profile. (This function starts from ORT 1.15)
 *
 *
 * (1) Single profile case:
 * Assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2,
 * and tensor_b has one dynamic shape dimension: dim_1.
 *
 * The data in profile file will be:
 * {
 *   tensor_a: [dim_0, min_shape_0, max_shape_0, opt_shape_0, dim_2, min_shape_2, max_shape_2, opt_shape_2]
 *   tensor_b: [dim_1, min_shape_1, max_shape_1, opt_shape_1]
 * }
 *
 * The data after deserialization will be:
 * {
 *   tensor_a: {
 *     dim_0: [[min_shape_0, max_shape_0, opt_shape_0]],
 *     dim_2: [[min_shape_2, max_shape_2, opt_shape_2]]
 *   },
 *   tensor_b: {
 *     dim_1: [[min_shape_1, max_shape_1, opt_shape_1]]
 *   }
 * }
 *
 *
 * (2) Multiple profiles case:
 * For example, if the data in profile file is:
 * {
 *   tensor_a: [dim_0, min_shape_0, max_shape_0, opt_shape_0, dim_0, min_shape_1, max_shape_1, opt_shape_1]
 *              |                                          |  |                                          |
 *              ---------------- profile 0 -----------------  ---------------- profile 1 -----------------
 *
 *   tensor_b: [dim_1, min_shape_2, max_shape_2, opt_shape_2, dim_1, min_shape_3, max_shape_3, opt_shape_3]
 *              |                                          |  |                                          |
 *              ---------------- profile 0 -----------------  ---------------- profile 1 -----------------
 * }
 *
 * The data after deserialization will be:
 * {
 *   tensor_a: {
 *     dim_0: [[min_shape_0, max_shape_0, opt_shape_0], [min_shape_1, max_shape_1, opt_shape_1]]
 *   },
 *   tensor_b: {
 *     dim_1: [[min_shape_2, max_shape_2, opt_shape_2], [min_shape_3, max_shape_3, opt_shape_3]]
 *   }
 * }
 */
std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> DeserializeProfileV2(std::ifstream& infile) {
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] In DeserializeProfileV2()";
  // Load flexbuffer
  infile.seekg(0, std::ios::end);
  size_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data{new char[length]};
  infile.read((char*)data.get(), length);
  infile.close();

  // Deserialize profile
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> shape_ranges;
  auto tensors_range_entries = flexbuffers::GetRoot((const uint8_t*)data.get(), length).AsMap();
  auto keys = tensors_range_entries.Keys();
  auto values = tensors_range_entries.Values();
  for (size_t i = 0, end = keys.size(); i < end; ++i) {  // iterate tensors
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] input tensor is '" << keys[i].AsString().c_str() << "'";
    auto dim_range_vector = values[i].AsTypedVector();
    std::unordered_map<size_t, std::vector<std::vector<int64_t>>> inner_map;
    std::vector<std::vector<int64_t>> profile_vector;

    for (size_t k = 0; k < (dim_range_vector.size() / 4); k++) {  // iterate dim, min, max, opt for all profiles
      std::vector<int64_t> shape_vector;
      auto idx = 4 * k;
      auto dim = dim_range_vector[idx].AsInt64();
      shape_vector.push_back(dim_range_vector[idx + 1].AsInt64());  // min shape
      shape_vector.push_back(dim_range_vector[idx + 2].AsInt64());  // max shape
      shape_vector.push_back(dim_range_vector[idx + 3].AsInt64());  // opt shape

      if (inner_map.find(dim) == inner_map.end()) {
        inner_map[dim] = profile_vector;
      }
      inner_map[dim].push_back(shape_vector);
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << dim << ", " << shape_vector[0] << ", " << shape_vector[1] << ", " << shape_vector[2];
    }
    shape_ranges[keys[i].AsString().c_str()] = inner_map;
  }
  return shape_ranges;
}

/*
 * Compare profile shapes from profile file (.profile) with explicit profile min/max/opt shapes.
 * Return false meaning no need to rebuild engine if everything is same.
 * Otherwise return true and engine needs to be rebuilt.
 */
bool CompareProfiles(const std::string& file_name,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes) {
  std::ifstream profile_file(file_name, std::ios::binary | std::ios::in);
  if (!profile_file) {
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << file_name << " doesn't exist.";
    return true;
  }

  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> shape_ranges;
  shape_ranges = DeserializeProfileV2(profile_file);

  /* The format of the two data structures are below, for example:
   *
   * shape_ranges:
   * {
   *   tensor_a: {
   *     dim_0: [[min_shape, max_shape, opt_shape]],
   *     dim_2: [[min_shape, max_shape, opt_shape]]
   *   },
   *   tensor_b: {
   *     dim_1: [[min_shape, max_shape, opt_shape]]
   *   }
   * }
   *
   * profile_min_shapes:
   * {
   *   tensor_a: [[dim_0_value_0, dim_1_value_1, dim_2_value_2]],
   *   tensor_b: [[dim_0_value_3, dim_1_value_4, dim_2_value_5]]
   * }
   *
   */

  // Check number of dynamic shape inputs
  if (profile_min_shapes.size() != shape_ranges.size()) {
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Numbers of dynamic shape inputs are not the same.";
    return true;
  }

  // Iterate through shape_ranges map
  for (auto tensor_it = shape_ranges.begin(); tensor_it != shape_ranges.end(); tensor_it++) {  // iterate tensors
    auto tensor_name = tensor_it->first;
    if (profile_min_shapes.find(tensor_name) == profile_min_shapes.end()) {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Tensor name '" << tensor_name << "' doesn't exist in trt_profile_min_shapes.";
      return true;
    }

    for (auto dim_it = tensor_it->second.begin(); dim_it != tensor_it->second.end(); dim_it++) {  // iterate dimensions
      auto dim = dim_it->first;
      auto num_profiles = GetNumProfiles(profile_min_shapes);

      if (dim_it->second.size() != static_cast<size_t>(num_profiles)) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Numbers of profiles are not the same.";
        return true;
      }

      for (size_t i = 0; i < dim_it->second.size(); i++) {  // iterate (multiple) profile(s)
        auto shape_values = dim_it->second[i];
        if (dim > (profile_min_shapes[tensor_name][i].size() - 1)) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] dimension " << dim << " of '" << tensor_name << "' in " << file_name << " exceeds the total dimension of trt_profile_min_shapes.";
          return true;
        }

        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_min_shapes[tensor_name][i][dim];
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[0] << " in " << file_name;
        if (profile_min_shapes[tensor_name][i][dim] != shape_values[0]) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }

        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_max_shapes[tensor_name][i][dim];
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[1] << " in " << file_name;
        if (profile_max_shapes[tensor_name][i][dim] != shape_values[1]) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }

        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_opt_shapes[tensor_name][i][dim];
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[2] << " in " << file_name;
        if (profile_opt_shapes[tensor_name][i][dim] != shape_values[2]) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }
      }
    }
  }
  return false;
}

/*
 * Get cache by name
 *
 */
std::string GetCachePath(const std::string& root, const std::string& name) {
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
std::string GetComputeCapacity(const cudaDeviceProp& prop) {
  const std::string compute_capability = std::to_string(prop.major * 10 + prop.minor);
  return compute_capability;
}

/*
 * Get Timing by compute capability
 *
 */
std::string GetTimingCachePath(const std::string& root, cudaDeviceProp prop) {
  // append compute capability of the GPU as this invalidates the cache and TRT will throw when loading the cache
  const std::string timing_cache_name = "TensorrtExecutionProvider_cache_sm" +
                                        GetComputeCapacity(prop) + ".timing";
  return GetCachePath(root, timing_cache_name);
}

/*
 * Get cache by type
 *
 * \param root root path of the cache
 * \param file_extension It could be ".engine", ".profile" or ".timing"
 */
std::vector<fs::path> GetCachesByType(const std::string& root, std::string file_extension) {
  std::vector<fs::path> cache_files;
  for (const auto& entry : fs::directory_iterator(root)) {
    if (fs::path(file_extension) == fs::path(entry).extension()) {
      cache_files.push_back(fs::path(entry));
    }
  }
  return cache_files;
}

bool IsCacheExistedByType(const std::string& root, std::string file_extension) {
  auto cache_files = GetCachesByType(root, file_extension);
  if (cache_files.size() == 0) {
    return false;
  }
  return true;
}

void RemoveCachesByType(const std::string& root, std::string file_extension) {
  auto cache_files = GetCachesByType(root, file_extension);
  for (const auto& entry : cache_files) {
    fs::remove(entry);
  }
}

// Helper class to generate engine id via model name/model content/env metadata
HashValue TRTGenerateId(const GraphViewer& graph_viewer) {
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
  const auto& model_path_components = main_graph.ModelPath().GetComponents();

  if (!model_path_components.empty()) {
    std::string model_name = PathToUTF8String(model_path_components.back());

    LOGS_DEFAULT(INFO) << "[TensorRT EP] Model name is " << model_name;
    // Ensure enough characters are hashed in case model names are too short
    const size_t model_name_length = model_name.size();
    constexpr size_t hash_string_length = 500;
    std::string repeat_model_name = model_name;
    for (size_t i = model_name_length; i > 0 && i < hash_string_length; i += model_name_length) {
      repeat_model_name += model_name;
    }
    hash_str(repeat_model_name);
  } else {
    LOGS_DEFAULT(INFO) << "[TensorRT EP] Model path is empty";
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
  hash_str(std::to_string(CUDA_VERSION));
#endif

#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR)
  std::string TRT_VERSION = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR);
  hash_str(TRT_VERSION);
#endif

  model_hash = hash[0] | (uint64_t(hash[1]) << 32);

  // return the current unique id
  return model_hash;
}

bool ValidateProfileShapes(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
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
bool MakeInputNameShapePair(std::string pair_string, std::pair<std::string, std::vector<int64_t>>& pair) {
  if (pair_string.empty()) {
    return true;
  }

  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << pair_string;

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
 * Parse explicit profile min/max/opt shapes from TensorRT EP provider options.
 *
 * For example:
 * The provider option is --trt_profile_min_shapes="input_id:32x1,attention_mask:32x1,input_id:32x41,attention_mask:32x41",
 * after string is being parsed, the profile shapes has two profiles and is being represented as below.
 * {"input_id": [[32, 1], [32, 41]], "attention_mask": [[32, 1], [32, 41]]}
 *
 * Return true if string can be successfully parsed or false if string has wrong format.
 */
bool ParseProfileShapes(std::string profile_shapes_string, std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes) {
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

    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << input_name;
    std::string shape_string = "";
    for (auto v : pair.second) {
      shape_string += std::to_string(v);
      shape_string += ", ";
    }
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << shape_string;
  }

  return true;
}


bool CheckPrecompiledEngine(const GraphViewer& graph) {
  if (graph.NumberOfNodes() == 1 && graph.GetNode(0)->OpType() == EP_CONTEXT_OP_TYPE) {
    return true;
  }
  return false;
}

const onnxruntime::Path& GetModelPath(const GraphViewer& graph_viewer) {
  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  const Graph& main_graph = *cur_graph;
  return main_graph.ModelPath();
}

std::filesystem::path LocateEnginePath(const onnxruntime::Path& model_path, std::string ep_cache_context) {
  std::filesystem::path path(model_path.ToPathString());
  std::filesystem::path parent_path = path.parent_path();
  std::filesystem::path engine_path = parent_path.append(ep_cache_context);
  return engine_path;
}

/*
 * The sanity check for EP context contrib op.
 */
bool IsValidEPContextNode(const GraphViewer& graph) {
  assert(graph.NumberOfNodes() == 1);
  assert(graph.GetNode(0)->OpType() == EP_CONTEXT_OP_TYPE);
  auto node = graph.GetNode(0);
  auto& attrs = node->GetAttributes();
  // "embed_mode" attr and "ep_cache_context" attr should be present
  if (attrs.count(EP_CONTEXT_ATTR_EMBED_MODE) > 0 && attrs.count(EP_CONTEXT_ATTR_CACHE_CTX) > 0) {
    // ep_cache_context: payload of the execution provider context if embed_mode=1, or path to the context file if embed_mode=0
    if (attrs.at(EP_CONTEXT_ATTR_EMBED_MODE).i() == 0) {
      std::filesystem::path engine_path_default = LocateEnginePath(GetModelPath(graph), attrs.at(EP_CONTEXT_ATTR_CACHE_CTX).s());
      std::filesystem::path engine_path(attrs.at(EP_CONTEXT_ATTR_CACHE_CTX).s()); 
      if (!std::filesystem::exists(engine_path_default) && !std::filesystem::exists(engine_path)) {
        LOGS_DEFAULT(ERROR) << "Can't find " << engine_path_default.string() << " or " << engine_path.string() << " TensorRT engine";
        return false;
      }
    }
  }
  return true;
}
}  // namespace onnxruntime
