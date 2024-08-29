#pragma once
#include <string>
#include <filesystem>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <gsl/gsl>
#include "flatbuffers/idl.h"
#include "ort_trt_int8_cal_table.fbs.h"
#include "murmurhash3.h"

namespace fs = std::filesystem;

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
  int num_profile = 0;
  for (auto it = profile_shapes.begin(); it != profile_shapes.end(); it++) {
    num_profile = static_cast<int>(it->second.size());
    if (num_profile > 0) {
      break;
    }
  }
  return num_profile;
}

void SerializeProfileV2(const std::string& file_name, std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>& shape_ranges) {
  //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] In SerializeProfileV2()";
  // Serialize profile
  flexbuffers::Builder builder;
  auto tensor_map_start = builder.StartMap();
  for (auto tensor_it = shape_ranges.begin(); tensor_it != shape_ranges.end(); tensor_it++) {  // iterate tensors
    //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] input tensor is '" << tensor_it->first.c_str() << "'";
    builder.TypedVector(tensor_it->first.c_str(), [&] {
      for (auto dim_it = tensor_it->second.begin(); dim_it != tensor_it->second.end(); dim_it++) {
        size_t num_profiles = dim_it->second.size();
        for (size_t i = 0; i < num_profiles; i++) {
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] profile #" << i << ", dim is " << dim_it->first;
          builder.Int(dim_it->first);
          builder.Int(dim_it->second[i][0]);
          builder.Int(dim_it->second[i][1]);
          builder.Int(dim_it->second[i][2]);
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << dim_it->first << ", " << dim_it->second[i][0] << ", " << dim_it->second[i][1] << ", " << dim_it->second[i][2];
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

std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> DeserializeProfileV2(std::ifstream& infile) {
  //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] In DeserializeProfileV2()";
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
    //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] input tensor is '" << keys[i].AsString().c_str() << "'";
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
      //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << dim << ", " << shape_vector[0] << ", " << shape_vector[1] << ", " << shape_vector[2];
    }
    shape_ranges[keys[i].AsString().c_str()] = inner_map;
  }
  return shape_ranges;
}

bool CompareProfiles(const std::string& file_name,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                     std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes) {
  std::ifstream profile_file(file_name, std::ios::binary | std::ios::in);
  if (!profile_file) {
    //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << file_name << " doesn't exist.";
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
    //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Numbers of dynamic shape inputs are not the same.";
    return true;
  }

  // Iterate through shape_ranges map
  for (auto tensor_it = shape_ranges.begin(); tensor_it != shape_ranges.end(); tensor_it++) {  // iterate tensors
    auto tensor_name = tensor_it->first;
    if (profile_min_shapes.find(tensor_name) == profile_min_shapes.end()) {
      //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Tensor name '" << tensor_name << "' doesn't exist in trt_profile_min_shapes.";
      return true;
    }

    for (auto dim_it = tensor_it->second.begin(); dim_it != tensor_it->second.end(); dim_it++) {  // iterate dimensions
      auto dim = dim_it->first;
      auto num_profiles = GetNumProfiles(profile_min_shapes);

      if (dim_it->second.size() != static_cast<size_t>(num_profiles)) {
        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Numbers of profiles are not the same.";
        return true;
      }

      for (size_t i = 0; i < dim_it->second.size(); i++) {  // iterate (multiple) profile(s)
        auto shape_values = dim_it->second[i];
        if (dim > (profile_min_shapes[tensor_name][i].size() - 1)) {
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] dimension " << dim << " of '" << tensor_name << "' in " << file_name << " exceeds the total dimension of trt_profile_min_shapes.";
          return true;
        }

        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_min_shapes[tensor_name][i][dim];
        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[0] << " in " << file_name;
        if (profile_min_shapes[tensor_name][i][dim] != shape_values[0]) {
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] min shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }

        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_max_shapes[tensor_name][i][dim];
        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[1] << " in " << file_name;
        if (profile_max_shapes[tensor_name][i][dim] != shape_values[1]) {
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] max shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }

        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape value of dimension " << dim << " of '" << tensor_name << "' is " << profile_opt_shapes[tensor_name][i][dim];
        //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape value of dimension " << dim << " of '" << tensor_name << "' is " << shape_values[2] << " in " << file_name;
        if (profile_opt_shapes[tensor_name][i][dim] != shape_values[2]) {
          //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] opt shape values of dimension " << dim << " of '" << tensor_name << "' are not the same";
          return true;
        }
      }
    }
  }
  return false;
}

std::string GetCachePath(const std::string& root, const std::string& name) {
  if (root.empty()) {
    return name;
  } else {
    fs::path path = root;
    path.append(name);
    return path.string();
  }
}

std::string GetComputeCapacity(const cudaDeviceProp& prop) {
  const std::string compute_capability = std::to_string(prop.major * 10 + prop.minor);
  return compute_capability;
}

std::string GetTimingCachePath(const std::string& root, std::string& compute_cap) {
  // append compute capability of the GPU as this invalidates the cache and TRT will throw when loading the cache
  const std::string timing_cache_name = "TensorrtExecutionProvider_cache_sm" +
                                        compute_cap + ".timing";
  return GetCachePath(root, timing_cache_name);
}

HashValue TRTGenerateId(const OrtGraphViewer* graph_viewer) {
  HashValue model_hash = 0;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtGraph* cur_graph = nullptr;
  api->OrtGraph_GetOrtGraph(graph_viewer, &cur_graph);
  bool is_subgraph = false;
  api->OrtGraph_IsSubgraph(graph_viewer, &is_subgraph);
  while (is_subgraph) {
    const OrtGraph* parent_graph = nullptr;
    api->OrtGraph_GetParentGraph(cur_graph, &parent_graph);
    cur_graph = parent_graph;
    api->OrtGraph_IsSubgraph(graph_viewer, &is_subgraph);
  }

  const OrtGraph* main_graph = cur_graph;
  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
  };

  const std::filesystem::path* model_path = nullptr;
  api->OrtGraph_GetModelPath(graph_viewer, (const void**)&model_path);

  // Use the model's file name instead of the entire path to avoid cache regeneration if path changes
  if (model_path->has_filename()) {
    std::string model_name = model_path->filename();

    // LOGS_DEFAULT(INFO) << "[TensorRT EP] Model name is " << model_name;
    // Ensure enough characters are hashed in case model names are too short
    const size_t model_name_length = model_name.size();
    constexpr size_t hash_string_length = 500;
    std::string repeat_model_name = model_name;
    for (size_t i = model_name_length; i > 0 && i < hash_string_length; i += model_name_length) {
      repeat_model_name += model_name;
    }
    hash_str(repeat_model_name);
  } else {
    // LOGS_DEFAULT(INFO) << "[TensorRT EP] Model path is empty";
  }

  // fingerprint current graph by hashing graph inputs
  // const std::vector<const char*>& input_names = nullptr;
  const char** input_names = nullptr;
  size_t input_count = 0;
  api->OrtGraph_GetInputsIncludingInitializers(graph_viewer, &input_count, &input_names);
  for (size_t i = 0; i < input_count; ++i) {
    hash_str(input_names[i]);
  }

  // hashing output of each node
  const int number_of_ort_nodes = api->OrtGraph_NumberOfNodes(graph_viewer);
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  size_t nodes_count = 0;
  const size_t* nodes_index = nullptr;
  api->OrtGraph_GetNodesIndexInTopologicalOrder(graph_viewer, 0, &nodes_count, &nodes_index);
  for (const auto& index : nodes_vector) {
    const OrtNode* node = nullptr;
    api->OrtGraph_GetOrtNode(graph_viewer, nodes_index[index], &node);
    size_t output_size = 0;
    api->OrtNode_GetOutputSize(node, &output_size);
    for (size_t i = 0; i < output_size; ++i) {
      const char* output_name = nullptr;
      api->OrtNode_GetIthOutputName(node, i, &output_name);
      if (output_name != nullptr) {
        hash_str(output_name);
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

std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::string join(const std::vector<std::string>& vec, const std::string& delimiter) {
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result += vec[i];
    if (i < vec.size() - 1) {
      result += delimiter;
    }
  }
  return result;
}

std::string GetCacheSuffix(const std::string& fused_node_name, const std::string& trt_node_name_with_precision) {
  std::vector<std::string> split_fused_node_name = split(fused_node_name, '_');
  if (split_fused_node_name.size() >= 3) {
    // Get index of model hash from fused_node_name
    std::string model_hash = split_fused_node_name[split_fused_node_name.size() - 3];
    size_t index = fused_node_name.find(model_hash);
    // Parse suffix from trt_node_name_with_precision, as it has additional precision info
    std::vector<std::string> suffix_group = split(trt_node_name_with_precision.substr(index), '_');
    if (suffix_group.size() > 2) {
      suffix_group.erase(suffix_group.begin() + 2);
    }
    return join(suffix_group, "_");
  }
  return "";
}
}
