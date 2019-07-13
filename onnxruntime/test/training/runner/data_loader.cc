// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/util/protobuf_parsing_utils.h"
#include "test/training/runner/data_loader.h"
#include <fstream>

using namespace std;

namespace onnxruntime {
namespace training {

using FileInputStream = google::protobuf::io::FileInputStream;
using CodedInputStream = google::protobuf::io::CodedInputStream;

vector<PATH_STRING_TYPE> GetAllDataFiles(const PATH_STRING_TYPE& dir_path) {
  vector<PATH_STRING_TYPE> data_files;
  LoopDir(dir_path,
          [&data_files, &dir_path](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
            PATH_STRING_TYPE filename_str = filename;
            if (filename_str[0] == '.' ||
                f_type != OrtFileType::TYPE_REG ||
                !HasExtensionOf(filename_str, ORT_TSTR("pb"))) {
              return true;
            }
            data_files.push_back(ConcatPathComponent<PATH_CHAR_TYPE>(dir_path, filename_str));
            return true;
          });
  return data_files;
}

DataLoader::DataLoader(const MapStringToString& input_name_map,
                       const PATH_STRING_TYPE& dir_path,
                       size_t max_num_files_preload,
                       size_t shard_index,
                       size_t total_shard)
    : input_name_map_(input_name_map),
      active_data_set_index_(0),
      max_num_files_preload_(max_num_files_preload) {
  input_tensor_names_.reserve(input_name_map.size());

  size_t index = 0;
  for (const auto& pair : input_name_map) {
    input_tensor_names_.push_back(pair.second);
    input_to_feature_index_map_[pair.first] = index++;
  }

  data_files_ = GetAllDataFiles(dir_path);
  vector<PATH_STRING_TYPE> partial_training_files;
  // If only need to load partial data for data-parallelism training
  if (total_shard > 1) {
    if (shard_index < total_shard) {
      ORT_THROW("shard_index must be 0~", total_shard - 1);
    }

    int count = 0;
    for (const auto& file : data_files_) {
      if ((count++ % total_shard) == shard_index) {
        partial_training_files.push_back(file);
      }
    }
    data_files_ = partial_training_files;
  }
  data_sets_.resize(data_files_.size());
}

Status DataLoader::Load() {
  for (size_t i = 0; i < max_num_files_preload_; ++i) {
    ORT_RETURN_IF_ERROR(LoadFile(data_files_[i], data_sets_[i]));
  }
  return Status::OK();
}

std::shared_ptr<DataSet> DataLoader::NextShard() {
  size_t last_active_index = active_data_set_index_;
  active_data_set_index_ = (active_data_set_index_ + 1) % NumShards();

  //TODO: Release and Load Next File in another thread
  data_sets_[last_active_index].reset();

  size_t index_to_load = (active_data_set_index_ + max_num_files_preload_ - 1) % NumShards();
  Status s = LoadFile(data_files_[index_to_load], data_sets_[index_to_load]);
  if (!s.IsOK()) {
    ORT_THROW("Error Loading file ", data_files_[index_to_load].c_str());
  }

  return MutableDataSet();
}

Status DataLoader::LoadFile(const PATH_STRING_TYPE& file_path, std::shared_ptr<DataSet>& data_set) {
  int tensor_fd;
  ORT_RETURN_IF_ERROR(Env::Default().FileOpenRd(file_path, tensor_fd));
  FileInputStream f(tensor_fd);
  CodedInputStream coded_in(&f);
  f.SetCloseOnDelete(true);

  if (data_set == nullptr) {
    data_set = std::make_shared<DataSet>(input_tensor_names_);
  }

  uint32_t sample_size;
  while (coded_in.ReadRaw(&sample_size, SIZEOF_UINT32)) {
    Status s = LoadOneSample(coded_in, sample_size, data_set);
    if (!s.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parse file '", ToMBString(file_path), "' failed");
    }
  }
  return Status::OK();
}

Status DataLoader::LoadOneSample(CodedInputStream& coded_in,
                                 uint32_t sample_size,
                                 std::shared_ptr<DataSet>& data_set) {
  uint32_t read = 0;
  std::vector<ONNX_NAMESPACE::TensorProto> features(NumInputs());

  while (read < sample_size) {
    uint32_t feature_size;
    coded_in.ReadRaw(&feature_size, SIZEOF_UINT32);
    std::string s;
    coded_in.ReadString(&s, feature_size);

    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromString(s)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to parse one TensoProto");
    }

    const std::string& input_name = tensor.name();
    auto it = input_to_feature_index_map_.find(input_name);
    if (it != input_to_feature_index_map_.end()) {
      size_t idx = it->second;
      features[idx] = tensor;
    }

    read += SIZEOF_UINT32 + feature_size;
  }

  ORT_RETURN_IF_ERROR(data_set->AddData(features));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
