// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/framework/path_lib.h"
#include "test/training/runner/training_util.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace training {

typedef std::basic_string<PATH_CHAR_TYPE> PATH_STRING_TYPE;

/*
This buffer is thread safe, to make sure data set
could be loaded asynchronously.
*/
class DataSetBuffer {
 public:
  DataSetBuffer() {
  }

  // For our data prefetch scenario, when we are getting one data set
  // which's not ready, we should wait until it's ready.
  std::shared_ptr<DataSet> Get(size_t index) {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return data_sets_.count(index) != 0; });
    return data_sets_[index];
  }

  void Set(size_t index, std::shared_ptr<DataSet> data_set) {
    std::unique_lock<std::mutex> lk(mutex_);
    data_sets_[index] = data_set;
    lk.unlock();
    cv_.notify_one();
  }

  bool Remove(size_t index) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (data_sets_.count(index) == 0) {
      return false;
    }

    data_sets_[index].reset();
    return true;
  }

 private:
  std::unordered_map<size_t, std::shared_ptr<DataSet>> data_sets_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class IDataLoader {
 public:
  IDataLoader(){};

  virtual ~IDataLoader() = default;

  virtual std::shared_ptr<DataSet> CurrentDataSet() = 0;

  virtual size_t NumShards() const = 0;

  virtual std::shared_ptr<DataSet> MoveToNextDataSet() = 0;

  // Input Name for Graph
  virtual const VectorString& DataSetTensorNames() const = 0;
};

/*
Training file is organized in the following format:
  
  [Sample ByteSize] [Feature0 ByteSize] [Feature0 TensorProto] ... [FeatureN ByteSize] [FeatureN TensorProto]
  next sample ...

All the bytesize fields are stored as 4 bytes uint32_t
*/
class DataLoader : public IDataLoader {
 public:
  DataLoader(const MapStringToString& input_name_map,
             const PATH_STRING_TYPE& dir_path,
             size_t max_num_files_preload = 2,
             size_t shard_index = 0,
             size_t total_shard = 1);

  common::Status InitialPreLoadAsync();

  std::shared_ptr<DataSet> CurrentDataSet() override {
    return buffer_.Get(active_file_index_);
  }

  size_t NumShards() const override { return data_files_.size(); }

  std::shared_ptr<DataSet> MoveToNextDataSet() override;

  const VectorString& DataSetTensorNames() const override {
    return input_tensor_names_;
  }

 private:
  size_t NumInputs() const { return input_tensor_names_.size(); }

  common::Status PreloadAsync(size_t index_to_load);

  common::Status LoadFile(const PATH_STRING_TYPE& file_path, std::shared_ptr<DataSet>& data_set);

  common::Status LoadOneSample(google::protobuf::io::CodedInputStream& coded_in,
                               uint32_t sample_size,
                               std::shared_ptr<DataSet>& data_set);

  // TensorName in File -> Input Name for Graph
  MapStringToString input_name_map_;

  // Input Name for Graph
  VectorString input_tensor_names_;

  // TensorName in File -> Index in input_tensor_names_
  std::map<std::string, size_t> input_to_feature_index_map_;

  const size_t max_num_files_preload_;

  const static size_t SIZEOF_UINT32 = 4;

  std::vector<PATH_STRING_TYPE> data_files_;

  size_t active_file_index_ = 0;

  DataSetBuffer buffer_;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> data_loader_thread_pool_;

  const int32_t thread_pool_size_ = 2;
};

// Loader that only load one single DataSet.
class SingleDataLoader : public IDataLoader {
 public:
  SingleDataLoader(std::shared_ptr<DataSet> single_data_set, VectorString input_tensor_names)
      : data_set_(single_data_set), input_tensor_names_(input_tensor_names) {}

  std::shared_ptr<DataSet> CurrentDataSet() override {
    return data_set_;
  }

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
