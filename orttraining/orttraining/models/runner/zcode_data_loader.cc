// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/util/protobuf_parsing_utils.h"
#include "orttraining/models/runner/zcode_data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace onnxruntime {
namespace training {

using ThreadPool = onnxruntime::concurrency::ThreadPool;

ZcodeDataLoader::ZcodeDataLoader(const MapStringToString& input_name_map,
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

  data_loader_thread_pool_ = onnxruntime::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), ORT_TSTR("DataLoaderPool"), thread_pool_size_, true);
}

std::vector<PathString> ZcodeDataLoader::GetAllDataFiles(const PathString& dir_path) {
  std::vector<PathString> data_files;
  
  PathString filename_str = "training_data.csv";
            
  data_files.push_back(ConcatPathComponent<PathChar>(dir_path, filename_str));

  return data_files;
}

Status ZcodeDataLoader::InitializeDataSetIndex(size_t initial_data_set_index) {
  if (initial_data_set_index == active_file_index_) return Status::OK();

  ORT_RETURN_IF_NOT(!is_preloaded_);
  ORT_RETURN_IF_NOT(initial_data_set_index < NumShards());

  active_file_index_ = initial_data_set_index;

  return InitialPreLoadAsync();
}

std::shared_ptr<DataSet> ZcodeDataLoader::MoveToNextDataSet() {
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

Status ZcodeDataLoader::InitialPreLoadAsync() {
  ORT_RETURN_IF_NOT(!is_preloaded_);

  for (size_t i = 0; i < std::min(max_num_files_preload_, NumShards()); ++i) {
    const auto data_set_index = (active_file_index_ + i) % NumShards();
    LoadAndRemoveInternalAsync(data_set_index, false, 0);
  }

  is_preloaded_ = true;

  return Status::OK();
}

void ZcodeDataLoader::EnsurePreloadedOrThrow() {
  if (is_preloaded_) return;
  const auto status = InitialPreLoadAsync();
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
}

void ZcodeDataLoader::LoadAndRemoveInternalAsync(size_t index_to_load, bool need_remove, size_t index_to_remove) {
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

Status ZcodeDataLoader::LoadFile(const PathString& file_path, std::shared_ptr<DataSet>& data_set) {
  std::ifstream fin;
  std::string line;

  fin.open(file_path);
  // skip the first line (header)
  std::getline(fin, line);
  while(std::getline(fin, line))
  {
      Status s = LoadOneSample(line, data_set);
      if (!s.IsOK()) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, FAIL, "Failed to parse file '", ToMBString(file_path), "': ", s.ErrorMessage());
      }
  }
  return Status::OK();
}

Status ZcodeDataLoader::LoadOneSample(std::string line,
                                 std::shared_ptr<DataSet>& data_set) {
  
  VectorString feature_strings;

  std::stringstream linestream(line);
  while(linestream.good()) {
      std::string feature;
      std::getline(linestream, feature, ',');
      feature_strings.push_back(feature);
  }

  DataSet::SampleType sample = make_unique<std::vector<OrtValue>>();
  for (size_t i = 0; i < feature_strings.size(); i++) {

    std::string& feature_str = feature_strings[i];

    std::stringstream feature_stream(feature_str);
    std::vector<int64_t> feature;
    int64_t num;
    while ( feature_stream >> num )
    {
      feature.push_back( num );
    }
    
    int64_t feature_size = static_cast<int64_t>(feature.size());

    TensorShape shape(std::vector<int64_t>({feature_size}));
    MLDataType element_type = DataTypeImpl::GetType<int64_t>();

    auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, TrainingUtil::GetCpuAllocator());
    void* buffer = p_tensor->MutableDataRaw();
    memcpy(buffer, feature.data(), feature.size() * sizeof(int64_t));

    OrtValue ort_value(p_tensor.release(),
                        DataTypeImpl::GetType<Tensor>(),
                        DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());  
    
    sample->push_back(ort_value);
  }
  
  ORT_RETURN_IF_ERROR(data_set->AddData(move(sample)));

  return Status::OK();
}

// Status ZcodeDataLoader::LoadOneSample(std::string line,
//                                  std::shared_ptr<DataSet>& data_set) {
  
//   std::vector<std::vector<OrtValue>> features(NumInputs());
  
//   VectorString feature_strings;

//   std::stringstream linestream(line);
//   while(linestream.good()) {
//       std::string feature;
//       std::getline(linestream, feature, ',');
//       feature_strings.push_back(feature);
//   }
  
//   for (size_t i = 0; i < feature_strings.size(); i++)
//   {
//     std::string& feature_str = feature_strings[i];

//     std::stringstream feature_stream(feature_str);
//     std::vector<uint64_t> feature;
//     uint64_t num;
//     while ( feature_stream >> num )
//     {
//       feature.push_back( num );
//     }
    
//     int64_t feature_size = static_cast<int64_t>(feature.size());

//     TensorShape shape(std::vector<int64_t>({1, feature_size}));
//     MLDataType element_type = DataTypeImpl::GetType<uint64_t>();

//     auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, TrainingUtil::GetCpuAllocator());
//     void* buffer = p_tensor->MutableDataRaw();
//     memcpy(buffer, feature.data(), feature.size() * sizeof(uint64_t));
//     std::vector<OrtValue> ort_value;
//     ort_value.emplace_back(p_tensor.release(),
//                            DataTypeImpl::GetType<Tensor>(),
//                            DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());  
    
//     const std::string& input_name = input_tensor_names_[i];
//     auto it = input_to_feature_index_map_.find(input_name);
//     if (it != input_to_feature_index_map_.end()) {
//       size_t idx = it->second;
//       features[idx] = ort_value;
//     }
//   }
  
//   ORT_RETURN_IF_ERROR(data_set->AddData(features));

//   return Status::OK();
// }

}  // namespace training
}  // namespace onnxruntime
