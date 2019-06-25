// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include "core/util/protobuf_parsing_utils.h"
#include "onnx/onnx-ml.pb.h"
#include "test/training/runner/data_loader.h"
#include <fstream>

using namespace std;

namespace onnxruntime {
namespace training {

using FileInputStream = google::protobuf::io::FileInputStream;
using CodedInputStream = google::protobuf::io::CodedInputStream;

//load tensors from a list of pb files.
static Status LoadTensors(const vector<PATH_STRING_TYPE>& pb_files,
                          vector<ONNX_NAMESPACE::TensorProto>& input_pbs) {
  for (size_t i = 0; i != pb_files.size(); ++i) {
    int tensor_fd;
    auto st = Env::Default().FileOpenRd(pb_files.at(i), tensor_fd);
    ORT_RETURN_IF_ERROR(st);
    FileInputStream f(tensor_fd);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromZeroCopyStream(&f)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parse file '", ToMBString(pb_files.at(i)), "' failed");
    }
    input_pbs.emplace_back(tensor);
  }
  return Status::OK();
}

void GetDataFiles(const PATH_STRING_TYPE& dir_path, unordered_map<PATH_STRING_TYPE, vector<PATH_STRING_TYPE>>& sample_inputs_map) {
  LoopDir(dir_path,
          [&sample_inputs_map, &dir_path](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
            PATH_STRING_TYPE filename_str = filename;
            if (filename_str[0] == '.' ||
                f_type != OrtFileType::TYPE_REG ||
                !HasExtensionOf(filename_str, ORT_TSTR("pb"))) {
              return true;
            }
            // Filename is like "<sample_name>_input_<count>.pb"
            // e.g. xxxxx_input_0.pb
            const PATH_STRING_TYPE delimiter = ORT_TSTR("_input_");
            PATH_STRING_TYPE::size_type pos;
            if ((pos = filename_str.find(delimiter)) != PATH_STRING_TYPE::npos) {
              PATH_STRING_TYPE sample_name = filename_str.substr(0, pos);
              PATH_STRING_TYPE::size_type count_start_pos = pos + delimiter.size();
              PATH_STRING_TYPE::size_type count_end_pos;
              if ((count_end_pos = filename_str.find('.', count_start_pos)) != PATH_STRING_TYPE::npos) {
                auto count_str = filename_str.substr(count_start_pos, count_end_pos - count_start_pos);
                int count = stoi(count_str);
                auto& file_list = sample_inputs_map[sample_name];
                if (static_cast<int>(file_list.size()) <= count) {
                  file_list.resize(count + 1);
                }
                sample_inputs_map[sample_name][count] = ConcatPathComponent<PATH_CHAR_TYPE>(dir_path, filename_str);
              }
            }
            return true;
          });
}

Status DataLoader::AddData(const vector<ONNX_NAMESPACE::TensorProto>& inputs) {
  DataSet::SampleType sample = make_unique<vector<MLValue>>();

  for (const auto& tensor_proto : inputs) {
    size_t cpu_tensor_length;
    ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor_proto, &cpu_tensor_length));
    MLValue mlvalue;
    OrtAllocatorInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
    std::unique_ptr<Tensor> p_tensor;
    OrtCallback deleter;
    ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(
        Env::Default(), nullptr, tensor_proto, MemBuffer(data.get(), cpu_tensor_length, info), mlvalue, deleter));

    sample->push_back(mlvalue);
    buffer_for_mlvalues_.emplace_back(std::move(data));
    if (deleter.f != nullptr) {
      deleter_for_mlvalues_.emplace_back(deleter);
    }
  }

  // TODO: Initialize data_set_ in constructor
  // Currently, we do it here because we need to know the names of tensors
  if (data_set_ == nullptr) {
    std::vector<std::string> tensor_names;
    for (const auto& tensor_proto : inputs) {
      tensor_names.push_back(tensor_proto.name());
    }
    data_set_ = std::make_unique<DataSet>(tensor_names);
  }

  return data_set_->AddData(std::move(sample));
}

Status DataLoader::Load(const PATH_STRING_TYPE& dir_path, size_t shard_index, size_t total_shard) {
  unordered_map<PATH_STRING_TYPE, vector<PATH_STRING_TYPE>> sample_inputs_map;
  GetDataFiles(dir_path, sample_inputs_map);

  // If only need to load partial data for data-parallelism training
  if (total_shard > 1) {
    ORT_RETURN_IF_NOT(shard_index < total_shard, "shard_index must be 0~", total_shard - 1);

    unordered_map<PATH_STRING_TYPE, vector<PATH_STRING_TYPE>> partial_inputs_map;
    int count = 0;
    for (const auto& kv : sample_inputs_map) {
      if ((count++ % total_shard) == shard_index) {
        partial_inputs_map[kv.first] = kv.second;
      }
    }
    swap(partial_inputs_map, sample_inputs_map);
  }

  unordered_map<PATH_STRING_TYPE, vector<ONNX_NAMESPACE::TensorProto>> sample_tensor_map;
  for (const auto& kv : sample_inputs_map) {
    ORT_RETURN_IF_ERROR(LoadTensors(kv.second, sample_tensor_map[kv.first]));
  }
  ORT_RETURN_IF_NOT(!sample_tensor_map.empty());

  // Set input names
  vector<string> tensor_names;
  for (const auto& tensor : sample_tensor_map.begin()->second) {
    ORT_RETURN_IF_NOT(find(tensor_names.begin(), tensor_names.end(), tensor.name()) == tensor_names.end(),
                      "Load data set error: input has duplicated names");
    tensor_names.push_back(tensor.name());
  }
  data_set_ = make_unique<DataSet>(tensor_names);

  // Set input MLValues
  for (const auto& kv : sample_tensor_map) {
    ORT_RETURN_IF_ERROR(AddData(kv.second));
  }
  return Status::OK();
}

DataLoader ::~DataLoader() {
  for (OrtCallback& deleter : deleter_for_mlvalues_) {
    deleter.f(deleter.param);
  }
}

vector<PATH_STRING_TYPE> GetAllTrainingFiles(const PATH_STRING_TYPE& dir_path) {
  vector<PATH_STRING_TYPE> training_files;
  LoopDir(dir_path,
          [&training_files, &dir_path](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
            PATH_STRING_TYPE filename_str = filename;
            if (filename_str[0] == '.' ||
                f_type != OrtFileType::TYPE_REG ||
                !HasExtensionOf(filename_str, ORT_TSTR("pb"))) {
              return true;
            }
            training_files.push_back(ConcatPathComponent<PATH_CHAR_TYPE>(dir_path, filename_str));
            return true;
          });
  return training_files;
}

Status BertDataLoader::Load(const PATH_STRING_TYPE& dir_path, size_t shard_index, size_t total_shard) {
  vector<PATH_STRING_TYPE> training_files = GetAllTrainingFiles(dir_path);
  vector<PATH_STRING_TYPE> partial_training_files;
  // If only need to load partial data for data-parallelism training
  if (total_shard > 1) {
    ORT_RETURN_IF_NOT(shard_index < total_shard, "shard_index must be 0~", total_shard - 1);
    int count = 0;
    for (const auto& file : training_files) {
      if ((count++ % total_shard) == shard_index) {
        partial_training_files.push_back(file);
      }
    }
    training_files = partial_training_files;
  }
  for (auto file_path : training_files) {
    ORT_RETURN_IF_ERROR(LoadFile(file_path));
  }
  return Status::OK();
}

Status BertDataLoader::LoadFile(const PATH_STRING_TYPE& file_path) {
  int tensor_fd;
  ORT_RETURN_IF_ERROR(Env::Default().FileOpenRd(file_path, tensor_fd));
  FileInputStream f(tensor_fd);
  CodedInputStream coded_in(&f);
  f.SetCloseOnDelete(true);

  uint32_t sample_size;
  while (coded_in.ReadRaw(&sample_size, SIZEOF_UINT32)) {
    Status s = LoadOneSample(coded_in, sample_size);
    if (!s.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parse file '", ToMBString(file_path), "' failed");
    }
  }
  return Status::OK();
}

Status BertDataLoader::LoadOneSample(CodedInputStream& coded_in, uint32_t sample_size) {
  uint32_t read = 0;
  std::vector<ONNX_NAMESPACE::TensorProto> features;
  while (read < sample_size) {
    uint32_t feature_size;
    coded_in.ReadRaw(&feature_size, SIZEOF_UINT32);
    std::string s;
    coded_in.ReadString(&s, feature_size);

    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromString(s)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to parse one TensoProto");
    }

    features.emplace_back(tensor);
    read += SIZEOF_UINT32 + feature_size;
  }

  ORT_RETURN_IF_ERROR(AddData(features));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
