// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <unordered_map>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include "flatbuffers/idl.h"
#include "ort_trt_int8_cal_table.fbs.h"
#include "murmurhash3.h"
#include <NvInferVersion.h>
#include "core/providers/cuda/cuda_pch.h"

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

/*
* Seralize engine profile
* The profile contains min/max shape ranges of dynamic shape dimensions of each input tensor
* For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
* has one dynamic shape dimension: dim_1. The data in profile will be,
* key: tensor_a, value: dim_0 min_shape max_shape dim_2 min_shape max_shape
* key: tensor_b, value: dim_1 min_shape max_shape
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
 * Get cache by type
 *
 * \param root root path of the cache  
 * \param file_extension It could be ".engine", ".profile" or ".timing"
*/
std::vector<fs::path> GetCachesByType(const std::string& root, std::string file_extension) {
  std::vector<fs::path> cache_files;
  for (const auto & entry : fs::directory_iterator(root)) {
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
  for (const auto & entry : cache_files) {
    fs::remove(entry);
  }
}

// Helper class to generate engine id via model name/model content/env metadata
class TRTModelMetadefIdGenerator {
 public:
  int TRTGenerateId(const GraphViewer& graph_viewer, HashValue& model_hash) {
    model_hash = 0;

    // find the top level graph
    const Graph* cur_graph = &graph_viewer.GetGraph();
    while (cur_graph->IsSubgraph()) {
      cur_graph = cur_graph->ParentGraph();
    }

    uint32_t instance_hash[4] = {0, 0, 0, 0};

    const Graph& main_graph = *cur_graph;

    // hash the bytes in the Graph instance. we can't just use the address as a new Graph instance may use
    // the same memory (unit tests prove this can occur). the raw bytes of the Graph instance should be a unique
    // fingerprint for the instance that can use used as the key to the hash of the graph name/inputs&outputs/metadata.
    MurmurHash3::x86_128(&main_graph, gsl::narrow_cast<int32_t>(sizeof(Graph)), instance_hash[0], &instance_hash);
    HashValue graph_instance_hash = instance_hash[0] | (uint64_t(instance_hash[1]) << 32);

    // if we've already hashed this main graph instance use the cached value
    auto entry = trt_main_graph_hash_.find(graph_instance_hash);
    if (entry != trt_main_graph_hash_.cend()) {
      model_hash = entry->second;
    } else {
      uint32_t hash[4] = {0, 0, 0, 0};

      // Use graph name instead of path to avoid cache regeneration if path changes
      const auto& model_name_str = main_graph.Name();
      if (!model_name_str.empty()) {
        MurmurHash3::x86_128(model_name_str.data(), gsl::narrow_cast<int32_t>(model_name_str.size()), hash[0], &hash);
      }

      auto hash_str = [&hash](const std::string& str) {
        MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
      };

      // fingerprint the main graph by hashing graph inputs
      for (const auto* node_arg : main_graph.GetInputsIncludingInitializers()) {
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
      trt_main_graph_hash_[graph_instance_hash] = model_hash;
    }

    // return the current unique id, and increment to update
    return trt_model_metadef_id_[model_hash]++;
  }

 private:
  std::unordered_map<HashValue, HashValue> trt_main_graph_hash_;  // map graph instance hash to model contents hash
  std::unordered_map<HashValue, int> trt_model_metadef_id_;       // current unique id for model
};

std::unique_ptr<TRTModelMetadefIdGenerator> trt_metadef_id_generator_ = std::make_unique<TRTModelMetadefIdGenerator>();

// Calll TRTGenerateMetaDefId to generate hash id for TRT engine cache
int TRTGenerateMetaDefId(const GraphViewer& graph_viewer, HashValue& model_hash) {
  // if the EP is shared across multiple sessions there's a very small potential for concurrency issues.
  // use a lock when generating an id to be paranoid
  static OrtMutex mutex;
  std::lock_guard<OrtMutex> lock(mutex);
  return trt_metadef_id_generator_->TRTGenerateId(graph_viewer, model_hash);
}
}
