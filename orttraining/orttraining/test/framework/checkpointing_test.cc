// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpointing.h"

#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/path_lib.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

using onnxruntime::test::TemporaryDirectory;

namespace onnxruntime {
namespace training {
namespace test {

namespace {
const OrtMemoryInfo cpu_alloc_info(onnxruntime::CPU, OrtDeviceAllocator);

class OrtValueTensorData {
 public:
  OrtValueTensorData(TensorShape shape, std::vector<float> data) {
    ORT_ENFORCE(shape.Size() == static_cast<int64_t>(data.size()));
    shape_ = std::move(shape);
    data_ = std::move(data);
  }

  OrtValue GetOrtValue() {
    return OrtValue(
        new Tensor(DataTypeImpl::GetType<float>(), shape_, data_.data(), cpu_alloc_info),
        DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

 private:
  TensorShape shape_;
  std::vector<float> data_;
};

void CompareOrtValuesToTensorProtoValues(
    const PathString& model_path,
    const NameMLValMap& name_to_ort_value,
    const std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto>& name_to_tensor_proto) {
  ASSERT_EQ(name_to_ort_value.size(), name_to_tensor_proto.size());

  NameMLValMap name_to_ort_value_from_tensor_proto{};
  std::vector<std::vector<char>> tensor_buffers{};

  for (const auto& name_and_tensor_proto : name_to_tensor_proto) {
    const auto& name = name_and_tensor_proto.first;
    const auto& tensor_proto = name_and_tensor_proto.second;
    TensorShape shape{tensor_proto.dims().data(), static_cast<size_t>(tensor_proto.dims().size())};
    ASSERT_EQ(tensor_proto.data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    std::vector<char> tensor_buffer(shape.Size() * sizeof(float));
    MemBuffer m(tensor_buffer.data(), tensor_buffer.size(), cpu_alloc_info);
    OrtValue ort_value;
    ASSERT_STATUS_OK(utils::TensorProtoToMLValue(
        Env::Default(), model_path.c_str(), tensor_proto, m, ort_value));

    name_to_ort_value_from_tensor_proto.emplace(name, ort_value);
    tensor_buffers.emplace_back(std::move(tensor_buffer));
  }

  for (const auto& name_and_ort_value : name_to_ort_value) {
    const auto& name = name_and_ort_value.first;
    const auto& ort_value = name_and_ort_value.second;
    const auto& ort_value_from_tensor_proto = name_to_ort_value_from_tensor_proto.at(name);

    ASSERT_TRUE(ort_value.IsTensor() && ort_value_from_tensor_proto.IsTensor());
    const Tensor& a = ort_value.Get<Tensor>();
    const Tensor& b = ort_value_from_tensor_proto.Get<Tensor>();
    ASSERT_TRUE(
        a.DataType() == b.DataType() &&
        a.SizeInBytes() == b.SizeInBytes() &&
        std::memcmp(a.DataRaw(), b.DataRaw(), a.SizeInBytes()) == 0);
  }
}
}  // namespace

TEST(CheckpointingTest, SaveAndLoad) {
  std::unordered_map<std::string, OrtValueTensorData> name_to_ort_value_data{
      {"first", {{3}, {1.0f, 2.0f, 3.0f}}},
      {"second", {{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}},
  };

  NameMLValMap name_to_ort_value{};
  for (auto& name_and_ort_value_data : name_to_ort_value_data) {
    name_to_ort_value.emplace(
        name_and_ort_value_data.first, name_and_ort_value_data.second.GetOrtValue());
  }

  std::unordered_map<std::string, std::string> properties{
      {"one", "1"},
      {"two", "2"},
      {"three", "3"},
  };

  TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};

  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_checkpoint"))};
  // this path doesn't need to exist, we just consider its parent directory
  PathString model_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_model.onnx"))};

  DataTransferManager data_transfer{};
  data_transfer.RegisterDataTransfer(std::make_unique<CPUDataTransfer>());

  ASSERT_STATUS_OK(SaveModelCheckpoint(
      checkpoint_path, data_transfer, name_to_ort_value, properties));

  std::vector<ONNX_NAMESPACE::TensorProto> loaded_tensor_protos{};
  std::unordered_map<std::string, std::string> loaded_properties{};

  ASSERT_STATUS_OK(LoadModelCheckpoint(
      checkpoint_path, model_path, loaded_tensor_protos, loaded_properties));

  ASSERT_EQ(loaded_properties, properties);

  std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto> name_to_loaded_tensor_proto{};
  std::transform(
      loaded_tensor_protos.begin(), loaded_tensor_protos.end(),
      std::inserter(name_to_loaded_tensor_proto, name_to_loaded_tensor_proto.end()),
      [](const ONNX_NAMESPACE::TensorProto& tensor_proto) {
        return std::make_pair(tensor_proto.name(), tensor_proto);
      });

  CompareOrtValuesToTensorProtoValues(
      model_path, name_to_ort_value, name_to_loaded_tensor_proto);
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
