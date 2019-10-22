// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/framework/callback.h"
#include "test/training/runner/training_util.h"
#include "test/training/runner/data_loader.h"

using namespace onnxruntime;
using namespace onnxruntime::training;

namespace onnxruntime {
namespace test {

TEST(TrainingDataLoaderTest, SingleDataLoader_RandomDataSet) {
  const int batch_size = 2;
  const int num_of_batch = 1;
  std::vector<std::string> tensor_names = {"input1",
                                           "input2",
                                           "input3"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, 3},
                                            {batch_size, 3},
                                            {batch_size}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_INT64};

  const size_t num_of_perf_samples = num_of_batch * batch_size;
  auto random_data = std::make_shared<RandomDataSet>(num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);
  SingleDataLoader data_loader(random_data, tensor_names);
  ASSERT_EQ(1, data_loader.NumShards());
  ASSERT_EQ(3, data_loader.CurrentDataSet()->NumInputs());
  ASSERT_EQ(num_of_batch, data_loader.CurrentDataSet()->TotalBatch(batch_size));
}

TEST(TrainingDataLoaderTest, DataLoader_OneSingleFile) {
  const size_t max_num_files_preload = 3;
  const MapStringToString input_name_map = {
      {"input_ids", "input1"},
      {"segment_ids", "input2"},
      {"input_mask", "input3"},
      {"masked_lm_positions", "masked_lm_positions"},
      {"masked_lm_ids", "masked_lm_ids"},
      {"masked_lm_weights", "masked_lm_weights"},
      {"next_sentence_label", "next_sentence_labels"}};
  std::string data_dir = "testdata/dataloader/";
  PATH_STRING_TYPE train_data_dir;
  train_data_dir.assign(data_dir.begin(), data_dir.end());
  DataLoader data_loader(input_name_map,
                         train_data_dir,
                         max_num_files_preload);
  data_loader.InitialPreLoadAsync();

  ASSERT_NE(nullptr, data_loader.CurrentDataSet());
  auto next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
  next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
  next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
}

TEST(TrainingDataLoaderTest, DataLoader_OneSingleFileFailParsing) {
  const size_t max_num_files_preload = 3;
  const MapStringToString input_name_map = {
      {"input_ids_invalid_by_intention", "input1"},
      {"segment_ids", "input2"},
      {"input_mask", "input3"},
      {"masked_lm_positions", "masked_lm_positions"},
      {"masked_lm_ids", "masked_lm_ids"},
      {"masked_lm_weights", "masked_lm_weights"},
      {"next_sentence_label", "next_sentence_labels"}};
  std::string data_dir = "testdata/dataloader/";
  PATH_STRING_TYPE train_data_dir;
  train_data_dir.assign(data_dir.begin(), data_dir.end());
  DataLoader data_loader(input_name_map,
                         train_data_dir,
                         max_num_files_preload);
  data_loader.InitialPreLoadAsync();
  ASSERT_EQ(nullptr, data_loader.CurrentDataSet());
  auto next_data = data_loader.MoveToNextDataSet();
  ASSERT_EQ(nullptr, next_data);
}

}  // namespace test
}  // namespace onnxruntime
