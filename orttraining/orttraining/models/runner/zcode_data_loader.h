// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include "orttraining/models/runner/data_loader.h"
#include "orttraining/models/runner/training_util.h"

namespace onnxruntime {
namespace training {


/*
Training file is organized in the following format:
  input_ids,attention_mask,label
  [Sample input id sequence],[attention_mask],[label id]
  next sample ...

*/
class ZcodeDataLoader : public IDataLoader {
 public:
  ZcodeDataLoader(const MapStringToString& input_name_map,
                    const PathString& dir_path,
                    size_t max_num_files_preload = 2,
                    size_t world_rank = 0,
                    size_t world_size = 1);

  Status InitializeDataSetIndex(size_t initial_data_set_index) override;

  std::shared_ptr<DataSet> CurrentDataSet() override {
    EnsurePreloadedOrThrow();
    return buffer_.Get(active_file_index_);
  }

  size_t CurrentDataSetIndex() const override { return active_file_index_; }

  size_t NumShards() const override { return data_files_.size(); }

  std::shared_ptr<DataSet> MoveToNextDataSet() override;

  const VectorString& DataSetTensorNames() const override {
    return input_tensor_names_;
  }

 private:
  std::vector<PathString> GetAllDataFiles(const PathString& dir_path);

  common::Status InitialPreLoadAsync();

  void EnsurePreloadedOrThrow();

  size_t NumInputs() const { return input_tensor_names_.size(); }

  common::Status LoadFile(const PathString& file_path, std::shared_ptr<DataSet>& data_set);

  common::Status LoadOneSample(std::string line,
                               std::shared_ptr<DataSet>& data_set);

  void LoadAndRemoveInternalAsync(size_t index_to_load, bool need_remove, size_t index_to_remove);

  // TensorName in File -> Input Name for Graph
  MapStringToString input_name_map_;

  // Input Name for Graph
  VectorString input_tensor_names_;

  // TensorName in File -> Index in input_tensor_names_
  std::map<std::string, size_t> input_to_feature_index_map_;

  const size_t max_num_files_preload_;

  std::vector<PathString> data_files_;

  size_t active_file_index_ = 0;

  DataSetBuffer buffer_;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> data_loader_thread_pool_;

  // number of thread pool threads
  // NOTE: Currently, thread pool requests should be serialized for correctness
  //     (i.e., this should be 1). Otherwise, race conditions can occur - e.g.,
  //     a data set gets removed and then waited for indefinitely.
  const int32_t thread_pool_size_ = 1;

  // indicates whether initial preloading has occurred
  bool is_preloaded_ = false;
};

// Loader that only load one single DataSet.
class SingleZcodeDataLoader : public IDataLoader {
 public:
  SingleZcodeDataLoader(std::shared_ptr<DataSet> single_data_set, VectorString input_tensor_names)
      : data_set_(single_data_set), input_tensor_names_(input_tensor_names) {}

  Status InitializeDataSetIndex(size_t initial_data_set_index) override {
    ORT_RETURN_IF_NOT(initial_data_set_index == 0);
    return Status::OK();
  }

  std::shared_ptr<DataSet> CurrentDataSet() override {
    return data_set_;
  }

  size_t CurrentDataSetIndex() const override { return 0; }

  size_t NumShards() const override { return 1; }

  std::shared_ptr<DataSet> MoveToNextDataSet() override {
    return data_set_;
  }

  const VectorString& DataSetTensorNames() const override {
    return input_tensor_names_;
  }

 private:
  const std::shared_ptr<DataSet> data_set_;
  const VectorString input_tensor_names_;
};

}  // namespace training
}  // namespace onnxruntime
