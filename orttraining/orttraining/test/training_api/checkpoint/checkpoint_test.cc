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

TEST(CheckPointApiTest, SaveOnnxModelAsCheckpoint_ThenLoad_CPU) {
  // Model path and trainable parameter name definitions.
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/e2e.onnx";
  std::vector<std::string> expected_trainable_param_names{
      "bert.encoder.layer.2.output.LayerNorm.weight",
      "bert.encoder.layer.2.output.LayerNorm.bias",
      "add1_initializerr",
      "cls.predictions.transform.LayerNorm.weight",
      "cls.predictions.transform.LayerNorm.bias",
      "bert.embeddings.word_embeddings.weight_transposed",
      "cls.predictions.bias",
  };

  // Extract a weight value baseline to compare.
  // expected_param_name_to_ort_value is used to compare with the values after restoring from checkpoint.
  auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
  std::shared_ptr<Model> p_model;
  ORT_ENFORCE(Model::Load(model_uri, p_model, nullptr, *logger_ptr).IsOK());
  Graph& graph = p_model->MainGraph();

  std::vector<const ONNX_NAMESPACE::TensorProto*> trainable_param_values;
  trainable_param_values.reserve(expected_trainable_param_names.size());
  for (size_t i = 0; i < expected_trainable_param_names.size(); ++i) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    ORT_ENFORCE(graph.GetInitializedTensor(expected_trainable_param_names[i], tensor_proto),
                "Failed to find weight values: ", expected_trainable_param_names[i]);
    trainable_param_values.emplace_back(tensor_proto);
  }

  std::unordered_map<std::string, OrtValue> expected_param_name_to_ort_value;
  ORT_ENFORCE(api_test::CreateOrtValuesFromTensorProtos(trainable_param_values, expected_param_name_to_ort_value).IsOK());

  // Remove the tempoprary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(api_test::CheckpointUtils::SaveORTCheckpoint(model_uri, expected_trainable_param_names, checkpoint_path));

  // Check the ckpt files in the directory.
  std::set<PathString> expected_file_names{"param_tensors.pbseq"};  //
  std::set<PathString> valid_file_names;
  LoopDir(checkpoint_path,
          [&valid_file_names, &checkpoint_path](const PathChar* filename, OrtFileType file_type) -> bool {
            PathString filename_str = filename;
            bool is_valid_ckpt_file_exts =
                HasExtensionOf(filename_str, ORT_TSTR("pbseq")) || HasExtensionOf(filename_str, ORT_TSTR("bin"));
            if (filename_str[0] == '.' || file_type == OrtFileType::TYPE_DIR || !is_valid_ckpt_file_exts) {
              return true;
            }
            valid_file_names.emplace(filename_str);
            return true;
          });

  ASSERT_EQ(expected_file_names, valid_file_names);

  // Call Load APIs
  api_test::CheckpointStates checkpoint_states;
  ASSERT_STATUS_OK(api_test::CheckpointUtils::LoadORTCheckpoint(checkpoint_path, checkpoint_states));
  api_test::ModuleCheckpointStates module_states = checkpoint_states.module_checkpoint_states;
  const auto& param_states = module_states.named_parameters;
  std::unordered_map<std::string, OrtValue> restored_param_name_to_ort_values;
  std::vector<std::string> restored_trainable_param_names;
  for (auto it = param_states.begin(); it != param_states.end(); ++it) {
    restored_param_name_to_ort_values.insert({it->first, it->second->Data()});
    if (it->second->RequiresGrad()) {
      restored_trainable_param_names.emplace_back(it->first);
    }
  }

  // Check loaded parameter's values are same with original ones.
  ASSERT_EQ(expected_param_name_to_ort_value.size(), restored_param_name_to_ort_values.size());

  std::sort(expected_trainable_param_names.begin(), expected_trainable_param_names.end());
  std::sort(restored_trainable_param_names.begin(), restored_trainable_param_names.end());
  ASSERT_EQ(expected_trainable_param_names, restored_trainable_param_names);

  for (const auto& name_and_ort_value : restored_param_name_to_ort_values) {
    const auto& name = name_and_ort_value.first;
    const auto& restored_ort_value = name_and_ort_value.second;
    const auto& expected_ort_value = expected_param_name_to_ort_value.at(name);

    ASSERT_TRUE(restored_ort_value.IsTensor() && expected_ort_value.IsTensor());
    const Tensor& restored_tensor = restored_ort_value.Get<Tensor>();
    const Tensor& expected_tensor = expected_ort_value.Get<Tensor>();
    ASSERT_EQ(expected_tensor.DataType(), restored_tensor.DataType());
    ASSERT_EQ(expected_tensor.SizeInBytes(), restored_tensor.SizeInBytes());
    ASSERT_EQ(expected_tensor.DataType(), restored_tensor.DataType());

    ASSERT_TRUE(std::memcmp(expected_tensor.DataRaw(), restored_tensor.DataRaw(), expected_tensor.SizeInBytes()) == 0);
  }
}

const OrtMemoryInfo cpu_alloc_info(onnxruntime::CPU, OrtDeviceAllocator);
class OrtValueTensorData {
 public:
  OrtValueTensorData(TensorShape shape, std::vector<float> data) {
    ORT_ENFORCE(shape.Size() == static_cast<int64_t>(data.size()));
    shape_ = std::move(shape);
    data_ = std::move(data);
  }

  OrtValue GetOrtValue() {
    return OrtValue(new Tensor(DataTypeImpl::GetType<float>(), shape_, data_.data(), cpu_alloc_info),
                    DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  }

 private:
  TensorShape shape_;
  std::vector<float> data_;
};

TEST(CheckPointApiTest, SaveOptimizerStateAsCheckpoint_ThenLoad_CPU) {
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/e2e.onnx";
  std::unordered_map<std::string, OrtValueTensorData> name_to_ort_value_data{
      {"param1", {{3}, {1.0f, 2.0f, 3.0f}}},
      {"param2", {{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}},
      {"param3", {{3}, {1.0f, 2.0f, 3.0f}}},
      {"param4", {{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}},
  };

  std::vector<std::string> trainable_param_names{"param1", "param4"};
  NameMLValMap name_to_ort_value{};
  for (auto& name_and_ort_value_data : name_to_ort_value_data) {
    name_to_ort_value.emplace(
        name_and_ort_value_data.first, name_and_ort_value_data.second.GetOrtValue());
  }

  // Optimizer creation and trainable parameter name definitions.
  std::unordered_map<std::string, std::shared_ptr<api_test::Parameter>> named_parameters;
  for (auto it = name_to_ort_value.begin(); it != name_to_ort_value.end(); ++it) {
    auto param = std::make_shared<api_test::Parameter>(it->first, it->second);
    bool is_trainable =
        std::find(trainable_param_names.begin(), trainable_param_names.end(), param->Name()) != trainable_param_names.end();
    ASSERT_STATUS_OK(param->SetRequiresGrad(is_trainable));
    named_parameters.insert({it->first, param});
  }
  auto optimizer = api_test::Optimizer(model_uri, named_parameters);

  api_test::CheckpointStates state_dicts_to_save;
  ORT_ENFORCE(optimizer.GetStateDict(state_dicts_to_save.optimizer_checkpoint_states).IsOK());

  // Remove the tempoprary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(api_test::CheckpointUtils::SaveORTCheckpoint(state_dicts_to_save, checkpoint_path));

  // Check the ckpt files in the directory.
  std::set<PathString> expected_file_names{
      "optim_group0_momentum0_tensors.pbseq",
      "optim_group0_momentum1_tensors.pbseq",
      "optim_group0_properties.pbseq",
  };

  std::set<PathString> valid_file_names;
  LoopDir(checkpoint_path,
          [&valid_file_names, &checkpoint_path](const PathChar* filename, OrtFileType file_type) -> bool {
            PathString filename_str = filename;
            bool is_valid_ckpt_file_exts =
                HasExtensionOf(filename_str, ORT_TSTR("pbseq")) || HasExtensionOf(filename_str, ORT_TSTR("bin"));
            if (filename_str[0] == '.' || file_type == OrtFileType::TYPE_DIR || !is_valid_ckpt_file_exts) {
              return true;
            }
            valid_file_names.emplace(filename_str);
            return true;
          });

  ASSERT_EQ(expected_file_names, valid_file_names);

  // Call Load APIs
  api_test::CheckpointStates checkpoint_states;
  ASSERT_STATUS_OK(api_test::CheckpointUtils::LoadORTCheckpoint(checkpoint_path, checkpoint_states));
  api_test::OptimizerCheckpointStates optimizer_states = checkpoint_states.optimizer_checkpoint_states;
  std::unordered_map<std::string, std::shared_ptr<api_test::GroupOptimizerState>>&
      group_optimizer_states = optimizer_states.group_named_optimizer_states;

  ASSERT_EQ(group_optimizer_states.size(), 1);
  ASSERT_EQ(group_optimizer_states.begin()->first, "group0");

  std::unordered_map<std::string, api_test::ParameterOptimizerState>&
      param_named_optimizer_states = group_optimizer_states["group0"]->param_named_optimizer_states_;

  ASSERT_EQ(param_named_optimizer_states.size(), 2);
  auto it = param_named_optimizer_states.begin();
  ASSERT_EQ(it->first, "param1");
  std::advance(it, 1);
  ASSERT_EQ(it->first, "param4");

  for (auto it = param_named_optimizer_states.begin(); it != param_named_optimizer_states.end(); ++it) {
    for (auto& state_pair : it->second.states_) {
      ASSERT_TRUE(state_pair.first == "momentum0" || state_pair.first == "momentum1");
      const OrtValue& restored_ort_value = *(state_pair.second);
      const OrtValue& expected_ort_value = name_to_ort_value[it->first];
      ASSERT_TRUE(restored_ort_value.IsTensor() && expected_ort_value.IsTensor());
      const Tensor& restored_tensor = restored_ort_value.Get<Tensor>();
      const Tensor& expected_tensor = expected_ort_value.Get<Tensor>();

      ASSERT_EQ(expected_tensor.DataType(), restored_tensor.DataType());
      ASSERT_EQ(expected_tensor.SizeInBytes(), restored_tensor.SizeInBytes());
      ASSERT_EQ(expected_tensor.DataType(), restored_tensor.DataType());
    }
  }
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
