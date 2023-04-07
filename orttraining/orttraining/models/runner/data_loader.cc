// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/util/protobuf_parsing_utils.h"
#include "orttraining/models/runner/data_loader.h"
#include <fstream>

namespace onnxruntime {
namespace training {

using FileInputStream = google::protobuf::io::FileInputStream;
using CodedInputStream = google::protobuf::io::CodedInputStream;
using ThreadPool = onnxruntime::concurrency::ThreadPool;

static std::vector<PathString> GetAllDataFiles(const PathString& dir_path) {
  std::vector<PathString> data_files;
  LoopDir(dir_path,
          [&data_files, &dir_path](const PathChar* filename, OrtFileType f_type) -> bool {
            PathString filename_str = filename;
            if (filename_str[0] == '.' ||
                f_type != OrtFileType::TYPE_REG ||
                !HasExtensionOf(filename_str, ORT_TSTR("pb"))) {
              return true;
            }
            data_files.push_back(ConcatPathComponent<PathChar>(dir_path, filename_str));
            return true;
          });

  // Sort to ensure the view on training files are identical on all the workers
  std::sort(data_files.begin(), data_files.end());

  return data_files;
}

DataLoader::DataLoader(const MapStringToString& input_name_map,
                       const PathString& dir_path,
                       size_t max_num_files_preload,
                       size_t world_rank,
                       size_t world_size)
    : input_name_map_(input_name_map),
      max_num_files_preload_(max_num_files_preload) {
  ORT_ENFORCE(max_num_files_preload > 0);

  input_tensor_names_.reserve(input_name_map.size());

  size_t index = 0;
  for (const auto& pair : input_name_map) {
    input_tensor_names_.push_back(pair.second);
    input_to_feature_index_map_[pair.first] = index++;
  }

  data_files_ = GetAllDataFiles(dir_path);
  // If only need to load partial data for data-parallelism training
  if (world_size > 1) {
    if (world_rank >= world_size) {
      ORT_THROW("world_rank must be 0~", world_size - 1);
    }

    std::vector<PathString> partial_training_files;
    int count = 0;
    for (const auto& file : data_files_) {
      if ((count++ % world_size) == world_rank) {
        partial_training_files.push_back(file);
      }
    }
    data_files_ = std::move(partial_training_files);
  }

  data_loader_thread_pool_ = std::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), ORT_TSTR("DataLoaderPool"), thread_pool_size_, true);
}

Status DataLoader::InitializeDataSetIndex(size_t initial_data_set_index) {
  if (initial_data_set_index == active_file_index_) return Status::OK();

  ORT_RETURN_IF(is_preloaded_, "is_preloaded_ was true");
  ORT_RETURN_IF_NOT(initial_data_set_index < NumShards(), "initial_data_set_index >= NumShards()");

  active_file_index_ = initial_data_set_index;

  return InitialPreLoadAsync();
}

std::shared_ptr<DataSet> DataLoader::MoveToNextDataSet() {
  EnsurePreloadedOrThrow();

  const size_t old_active_file_index = active_file_index_;
  active_file_index_ = (active_file_index_ + 1) % NumShards();

  if (max_num_files_preload_ < NumShards()) {
    const size_t index_to_remove = old_active_file_index;
    const size_t index_to_load = (active_file_index_ + max_num_files_preload_ - 1) % NumShards();
    LoadAndRemoveInternalAsync(index_to_load, true, index_to_remove);
  }

  return CurrentDataSet();
}

Status DataLoader::InitialPreLoadAsync() {
  ORT_RETURN_IF(is_preloaded_, "is_preloaded_ was true");

  for (size_t i = 0; i < std::min(max_num_files_preload_, NumShards()); ++i) {
    const auto data_set_index = (active_file_index_ + i) % NumShards();
    LoadAndRemoveInternalAsync(data_set_index, false, 0);
  }

  is_preloaded_ = true;

  return Status::OK();
}

void DataLoader::EnsurePreloadedOrThrow() {
  if (is_preloaded_) return;
  const auto status = InitialPreLoadAsync();
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
}

void DataLoader::LoadAndRemoveInternalAsync(size_t index_to_load, bool need_remove, size_t index_to_remove) {
  ThreadPool::Schedule(data_loader_thread_pool_.get(), [this, index_to_load, need_remove, index_to_remove]() {
    std::shared_ptr<DataSet> data_set = std::make_shared<DataSet>(input_tensor_names_);
    if (index_to_load >= NumShards()) {
      LOGS_DEFAULT(WARNING)
          << "Value of index_to_load (" << index_to_load << ") is not in valid range ([0, " << NumShards() << "))";
      return;
    }
    Status s = LoadFile(data_files_[index_to_load], data_set);
    if (!s.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to load file at index " << index_to_load << ": " << s.ErrorMessage();
      buffer_.Set(index_to_load, nullptr);
    } else {
      buffer_.Set(index_to_load, data_set);
    }

    // Put data removal in forked thread since it is observed calling Remove in main thread will
    // block the main thread execution (possibly because the removal triggering some heap re-org).
    if (need_remove) {
      buffer_.Remove(index_to_remove);
    }
  });
}

Status DataLoader::LoadFile(const PathString& file_path, std::shared_ptr<DataSet>& data_set) {
  int tensor_fd;
  ORT_RETURN_IF_ERROR(Env::Default().FileOpenRd(file_path, tensor_fd));
  FileInputStream f(tensor_fd);
  CodedInputStream coded_in(&f);
  f.SetCloseOnDelete(true);

  uint32_t sample_size;
  while (coded_in.ReadLittleEndian32(&sample_size)) {
    Status s = LoadOneSample(coded_in, sample_size, data_set);
    if (!s.IsOK()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL, "Failed to parse file '", ToUTF8String(file_path), "': ", s.ErrorMessage());
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
    coded_in.ReadLittleEndian32(&feature_size);
    std::string s;
    coded_in.ReadString(&s, feature_size);

    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromString(s)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to parse TensorProto");
    }

    const std::string& input_name = tensor.name();
    auto it = input_to_feature_index_map_.find(input_name);
    if (it != input_to_feature_index_map_.end()) {
      size_t idx = it->second;
      features[idx] = tensor;
    }

    read += sizeof(uint32_t) + feature_size;
  }

  ORT_RETURN_IF_ERROR(data_set->AddData(features));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
