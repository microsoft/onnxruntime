// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/callback.h"
#include "core/platform/env.h"
#include "core/util/protobuf_parsing_utils.h"
#include "orttraining/models/runner/training_util.h"
#include "core/platform/path_lib.h"  // TODO fix include ordering dependency
#include "orttraining/models/runner/data_loader.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

using onnxruntime::test::TemporaryDirectory;

namespace onnxruntime {
namespace training {
namespace test {

namespace {
Status WriteInputDataFile(
    const PathString& path,
    const uint32_t num_samples,
    const std::vector<std::string>& sample_tensor_names,
    const uint32_t tensor_data_value) {
  const uint32_t num_features = static_cast<uint32_t>(sample_tensor_names.size());
  ORT_RETURN_IF_NOT(num_samples > 0 && num_features > 0, "num_samples > 0 && num_features > 0 was false");

  // feature tensors have dimension of {1} and data value of tensor_data_value
  std::vector<std::vector<ONNX_NAMESPACE::TensorProto>> samples;
  for (uint32_t i = 0; i < num_samples; ++i) {
    std::vector<ONNX_NAMESPACE::TensorProto> features;
    for (uint32_t j = 0; j < num_features; ++j) {
      ONNX_NAMESPACE::TensorProto t{};
      t.set_name(sample_tensor_names[j]);
      t.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
      t.add_dims(1);
      t.add_uint64_data(tensor_data_value);
      features.emplace_back(t);
    }
    samples.emplace_back(features);
  }

  const uint32_t feature_size = static_cast<uint32_t>(samples[0][0].ByteSizeLong());  // they're all the same size
  const uint32_t sample_size = (feature_size + sizeof(uint32_t)) * num_features;

  int fd;
  ORT_RETURN_IF_ERROR(Env::Default().FileOpenWr(path, fd));
  google::protobuf::io::FileOutputStream file_stream{fd};
  file_stream.SetCloseOnDelete(true);
  google::protobuf::io::CodedOutputStream coded_stream{&file_stream};

  for (const auto& sample : samples) {
    coded_stream.WriteLittleEndian32(sample_size);
    for (const auto& feature : sample) {
      coded_stream.WriteLittleEndian32(feature_size);
      feature.SerializeToCodedStream(&coded_stream);
    }
  }

  return Status::OK();
}

Status CreateInputDataFiles(
    const PathString& directory_path,
    const size_t num_input_files,
    const std::vector<std::string>& sample_tensor_names) {
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(directory_path));
  for (size_t i = 0; i < num_input_files; ++i) {
    const PathString input_file_path = ConcatPathComponent(
        directory_path, ToPathString(MakeString("input.", i, ".pb")));
    ORT_RETURN_IF_ERROR(WriteInputDataFile(
        input_file_path, 3, sample_tensor_names, static_cast<uint32_t>(i)));
  }
  return Status::OK();
}

// check for expected values loaded from WriteInputDataFile()-produced file
void CheckDataSetValue(DataSet* data_set, uint32_t expected_value) {
  ASSERT_NE(data_set, nullptr);
  const auto& ort_values = data_set->GetKthBatch(1, 0);
  for (const auto& ort_value : ort_values) {
    const auto& tensor = ort_value.Get<Tensor>();
    ASSERT_EQ(tensor.Shape().Size(), 1);
    ASSERT_EQ(*tensor.Data<uint32_t>(), expected_value);
  }
}
}  // namespace

TEST(TrainingDataLoaderTest, SingleDataLoader_RandomDataSet) {
  constexpr int batch_size = 2;
  constexpr int num_of_batch = 1;
  std::vector<std::string> tensor_names = {"input1",
                                           "input2",
                                           "input3"};
  std::vector<TensorShape> tensor_shapes = {{batch_size, 3},
                                            {batch_size, 3},
                                            {batch_size}};
  std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_FLOAT,
                                                          onnx::TensorProto_DataType_INT64};

  constexpr size_t num_of_perf_samples = num_of_batch * batch_size;
  auto random_data = std::make_shared<RandomDataSet>(num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);
  SingleDataLoader data_loader(random_data, tensor_names);
  ASSERT_EQ(1, data_loader.NumShards());
  ASSERT_EQ(3, data_loader.CurrentDataSet()->NumInputs());
  ASSERT_EQ(num_of_batch, data_loader.CurrentDataSet()->TotalBatch(batch_size));
}

TEST(TrainingDataLoaderTest, DataLoader_OneSingleFile) {
  constexpr size_t max_num_files_preload = 3;
  const MapStringToString input_name_map = {{"a", "a"}, {"b", "b"}, {"c", "c"}};
  TemporaryDirectory tmp_dir{ORT_TSTR("training_data_loader_test_dir")};
  const PathString& train_data_dir = ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("single_file"));
  ASSERT_STATUS_OK(CreateInputDataFiles(train_data_dir, 1, {"a", "b", "c"}));
  DataLoader data_loader(input_name_map,
                         train_data_dir,
                         max_num_files_preload);
  ASSERT_NE(nullptr, data_loader.CurrentDataSet());
  auto next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
  next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
  next_data = data_loader.MoveToNextDataSet();
  ASSERT_NE(nullptr, next_data);
}

TEST(TrainingDataLoaderTest, DataLoader_OneSingleFileFailParsing) {
  constexpr size_t max_num_files_preload = 3;
  const MapStringToString input_name_map = {{"a_invalid", "a"}, {"b", "b"}, {"c", "c"}};
  TemporaryDirectory tmp_dir{ORT_TSTR("training_data_loader_test_dir")};
  const PathString& train_data_dir = ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("single_file"));
  ASSERT_STATUS_OK(CreateInputDataFiles(train_data_dir, 1, {"a", "b", "c"}));
  DataLoader data_loader(input_name_map,
                         train_data_dir,
                         max_num_files_preload);
  ASSERT_EQ(nullptr, data_loader.CurrentDataSet());
  auto next_data = data_loader.MoveToNextDataSet();
  ASSERT_EQ(nullptr, next_data);
}

namespace {
void TestDataLoaderWithMultipleFiles(
    const size_t num_input_files, const size_t max_num_files_preload,
    const size_t* const start_data_set_index = nullptr) {
  const MapStringToString input_name_map = {{"a", "a"}, {"b", "b"}, {"c", "c"}};
  TemporaryDirectory tmp_dir{ORT_TSTR("training_data_loader_test_dir")};
  const PathString& train_data_dir = ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("multiple_files"));

  ASSERT_STATUS_OK(CreateInputDataFiles(
      train_data_dir, num_input_files, {"a", "b", "c"}));

  DataLoader data_loader(input_name_map,
                         train_data_dir,
                         max_num_files_preload);

  ASSERT_EQ(data_loader.NumShards(), num_input_files);

  if (start_data_set_index) {
    ASSERT_STATUS_OK(data_loader.InitializeDataSetIndex(*start_data_set_index));
  }

  const size_t initial_data_set_index = start_data_set_index ? *start_data_set_index : 0;

  auto data_set_ptr = data_loader.CurrentDataSet();
  CheckDataSetValue(
      data_set_ptr.get(), static_cast<uint32_t>(initial_data_set_index));

  for (int i = 0; i < 100; ++i) {
    data_set_ptr = data_loader.MoveToNextDataSet();
    CheckDataSetValue(
        data_set_ptr.get(),
        static_cast<uint32_t>((initial_data_set_index + i + 1) % num_input_files));
  }
}
}  // namespace

TEST(TrainingDataLoaderTest, DataLoader_MultipleFiles_InitializeDataSetIndex) {
  constexpr size_t start_index = 2;
  TestDataLoaderWithMultipleFiles(3, 2, &start_index);
}

TEST(TrainingDataLoaderTest, DataLoader_MultipleFiles_PreloadOneFile) {
  TestDataLoaderWithMultipleFiles(3, 1);
}

TEST(TrainingDataLoaderTest, DataLoader_MultipleFiles_PreloadCountEqualsFileCount) {
  TestDataLoaderWithMultipleFiles(3, 3);
}

TEST(TrainingDataLoaderTest, DataLoader_MultipleFiles_PreloadCountExceedsFileCount) {
  TestDataLoaderWithMultipleFiles(3, 4);
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
