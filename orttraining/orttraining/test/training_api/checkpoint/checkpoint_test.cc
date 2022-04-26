// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpointing.h"

#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/path_lib.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/utilities.h"
#include "orttraining/training_api/interfaces.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "test/test_environment.h"
#include "test/util/include/test/test_environment.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
using onnxruntime::test::TemporaryDirectory;

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {
#define MODEL_FOLDER ORT_TSTR("testdata/")

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

TEST(CheckPointApiTest, Save_CPU) {
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/e2e.onnx";
  std::shared_ptr<Model> p_model;

  const auto& default_logger = onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, default_logger));
  Graph& graph = p_model->MainGraph();

  std::vector<std::string> trainable_weight_names{
      "bert.encoder.layer.2.output.LayerNorm.weight",
      "bert.encoder.layer.2.output.LayerNorm.bias",
      "add1_initializerr",
      "cls.predictions.transform.LayerNorm.weight",
      "cls.predictions.transform.LayerNorm.bias",
      "bert.embeddings.word_embeddings.weight_transposed",
      "cls.predictions.bias",
  };

  std::vector<const ONNX_NAMESPACE::TensorProto*> trainable_weight_values;
  trainable_weight_values.reserve(trainable_weight_names.size());
  for (size_t i = 0; i < trainable_weight_names.size(); ++i) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    ORT_ENFORCE(graph.GetInitializedTensor(trainable_weight_names[i], tensor_proto), "Failed to find weight values");
    trainable_weight_values.emplace_back(tensor_proto);
  }

  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }

  TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ORT_ENFORCE(api_test::CheckpointUtils::SaveORTCheckpoint(trainable_weight_values, checkpoint_path).IsOK());

  // Check the ckpt files in the directory.
}
//   std::unordered_map<std::string, OrtValueTensorData>
//       name_to_ort_value_data{
//           {"first", {{3}, {1.0f, 2.0f, 3.0f}}},
//           {"second", {{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}},
//       };

//   NameMLValMap name_to_ort_value{};
//   for (auto& name_and_ort_value_data : name_to_ort_value_data) {
//     name_to_ort_value.emplace(
//         name_and_ort_value_data.first, name_and_ort_value_data.second.GetOrtValue());
//   }

//   std::unordered_map<std::string, std::string> properties{
//       {"one", "1"},
//       {"two", "2"},
//       {"three", "3"},
//   };

//   TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};

//   struct ModuleCheckpointStates {
//    public:
//     std::unordered_map<std::string, std::shared_ptr<api_test::Parameter>> named_parameters;
//     DataTransferManager* train_session_data_transfer_mgr_;
//   };

//   api_test::CheckpointStates state_dicts_to_save;
//   ORT_ENFORCE(module.GetStateDict(state_dicts_to_save.module_checkpoint_states).IsOK());
//   ORT_ENFORCE(optimizer.GetStateDict(state_dicts_to_save.optimizer_checkpoint_states).IsOK());
//   std::string ckpt_file = params.output_dir + "/ckpt_" + params.model_name + std::to_string(batch_idx);
//   ORT_ENFORCE(CheckpointUtils::SaveORTCheckpoint(state_dicts_to_save, ckpt_file).IsOK());

//   PathString checkpoint_path{
//       ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_checkpoint"))};
//   // this path doesn't need to exist, we just consider its parent directory
//   PathString model_path{
//       ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_model.onnx"))};

//   DataTransferManager data_transfer{};
//   ASSERT_STATUS_OK(data_transfer.RegisterDataTransfer(std::make_unique<CPUDataTransfer>()));

//   ASSERT_STATUS_OK(SaveModelCheckpoint(
//       checkpoint_path, data_transfer, name_to_ort_value, properties));

//   std::vector<ONNX_NAMESPACE::TensorProto> loaded_tensor_protos{};
//   std::unordered_map<std::string, std::string> loaded_properties{};

//   ASSERT_STATUS_OK(LoadModelCheckpoint(
//       checkpoint_path, model_path, loaded_tensor_protos, loaded_properties));

//   ASSERT_EQ(loaded_properties, properties);

//   std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto> name_to_loaded_tensor_proto{};
//   std::transform(
//       loaded_tensor_protos.begin(), loaded_tensor_protos.end(),
//       std::inserter(name_to_loaded_tensor_proto, name_to_loaded_tensor_proto.end()),
//       [](const ONNX_NAMESPACE::TensorProto& tensor_proto) {
//         return std::make_pair(tensor_proto.name(), tensor_proto);
//       });

//   CompareOrtValuesToTensorProtoValues(
//       model_path, name_to_ort_value, name_to_loaded_tensor_proto);
// }

// TEST(CheckPointApiTest, SaveAndLoad) {
//   std::unordered_map<std::string, OrtValueTensorData> name_to_ort_value_data{
//       {"first", {{3}, {1.0f, 2.0f, 3.0f}}},
//       {"second", {{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}},
//   };

//   NameMLValMap name_to_ort_value{};
//   for (auto& name_and_ort_value_data : name_to_ort_value_data) {
//     name_to_ort_value.emplace(
//         name_and_ort_value_data.first, name_and_ort_value_data.second.GetOrtValue());
//   }

//   std::unordered_map<std::string, std::string> properties{
//       {"one", "1"},
//       {"two", "2"},
//       {"three", "3"},
//   };

//   TemporaryDirectory tmp_dir{ORT_TSTR("checkpointing_test_dir")};

//   PathString checkpoint_path{
//       ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_checkpoint"))};
//   // this path doesn't need to exist, we just consider its parent directory
//   PathString model_path{
//       ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("test_model.onnx"))};

//   DataTransferManager data_transfer{};
//   ASSERT_STATUS_OK(data_transfer.RegisterDataTransfer(std::make_unique<CPUDataTransfer>()));

//   ASSERT_STATUS_OK(SaveModelCheckpoint(
//       checkpoint_path, data_transfer, name_to_ort_value, properties));

//   std::vector<ONNX_NAMESPACE::TensorProto> loaded_tensor_protos{};
//   std::unordered_map<std::string, std::string> loaded_properties{};

//   ASSERT_STATUS_OK(LoadModelCheckpoint(
//       checkpoint_path, model_path, loaded_tensor_protos, loaded_properties));

//   ASSERT_EQ(loaded_properties, properties);

//   std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto> name_to_loaded_tensor_proto{};
//   std::transform(
//       loaded_tensor_protos.begin(), loaded_tensor_protos.end(),
//       std::inserter(name_to_loaded_tensor_proto, name_to_loaded_tensor_proto.end()),
//       [](const ONNX_NAMESPACE::TensorProto& tensor_proto) {
//         return std::make_pair(tensor_proto.name(), tensor_proto);
//       });

//   CompareOrtValuesToTensorProtoValues(
//       model_path, name_to_ort_value, name_to_loaded_tensor_proto);
// }

}  // namespace
}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime