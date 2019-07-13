// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include <string>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/framework/path_lib.h"
#include "test/training/runner/training_util.h"

namespace onnxruntime {
namespace training {

typedef std::basic_string<PATH_CHAR_TYPE> PATH_STRING_TYPE;

/*
Training file is organized in the following format:
  
  [Sample ByteSize] [Feature0 ByteSize] [Feature0 TensorProto] ... [FeatureN ByteSize] [FeatureN TensorProto]
  next sample ...

All the bytesize fields are stored as 4 bytes uint32_t
*/
class DataLoader {
 public:
  DataLoader(const MapStringToString& input_name_map,
             const PATH_STRING_TYPE& dir_path,
             size_t max_num_files_preload = 2,
             size_t shard_index = 0,
             size_t total_shard = 1);

  common::Status Load();

  std::shared_ptr<DataSet> MutableDataSet() {
    return data_sets_[active_data_set_index_];
  }

  std::shared_ptr<DataSet> NextShard();

  size_t NumShards() const { return data_files_.size(); }

 private:
  common::Status LoadFile(const PATH_STRING_TYPE& file_path, std::shared_ptr<DataSet>& data_set);

  common::Status LoadOneSample(google::protobuf::io::CodedInputStream& coded_in,
                               uint32_t sample_size,
                               std::shared_ptr<DataSet>& data_set);

  size_t NumInputs() const { return input_tensor_names_.size(); }

  // TensorName in File -> Input Name for Graph
  MapStringToString input_name_map_;

  // Input Name for Graph
  VectorString input_tensor_names_;

  // TensorName in File -> Index in input_tensor_names_
  std::map<std::string, size_t> input_to_feature_index_map_;

  size_t active_data_set_index_;
  const size_t max_num_files_preload_;
  const static size_t SIZEOF_UINT32 = 4;

  std::vector<PATH_STRING_TYPE> data_files_;
  std::vector<std::shared_ptr<DataSet>> data_sets_;
};

}  // namespace training
}  // namespace onnxruntime
