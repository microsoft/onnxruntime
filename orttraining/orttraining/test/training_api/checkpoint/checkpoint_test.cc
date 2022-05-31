// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/framework_common.h"
#include "core/framework/data_transfer.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/platform/path_lib.h"

#include "orttraining/core/framework/checkpoint_common.h"
#include "orttraining/training_api/include/interfaces.h"

#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test/test_environment.h"

using onnxruntime::test::TemporaryDirectory;
using namespace onnxruntime::training::api;

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

#define MODEL_FOLDER ORT_TSTR("testdata/")

/**
 * Load ONNX model from file path, save into ORT checkpoint files,
 * Then load it into ORT, compare with the initial parameter values.
 */
TEST(CheckpointApiTest, SaveOnnxModelAsCheckpoint_ThenLoad_CPU) {
  /// Phase 1 - Test Preparison
  /// Prepare the data and dest folder for saving checkpoint.
  /// Also cooked the data for test result comparision.

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
  // expected_trainable_param_name_to_ort_value is used to compare with the values after restoring from checkpoint.
  auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
  std::shared_ptr<Model> p_model;
  ORT_ENFORCE(Model::Load(model_uri, p_model, nullptr, *logger_ptr).IsOK());
  Graph& graph = p_model->MainGraph();

  std::vector<ONNX_NAMESPACE::TensorProto> trainable_param_values;
  trainable_param_values.reserve(expected_trainable_param_names.size());
  std::vector<ONNX_NAMESPACE::TensorProto> non_trainable_param_values;
  const auto& initializer_tensors = graph.GetAllInitializedTensors();
  for (const std::pair<std::string, const ONNX_NAMESPACE::TensorProto*>& pair : initializer_tensors) {
    if (std::find(expected_trainable_param_names.begin(), expected_trainable_param_names.end(), pair.first) != expected_trainable_param_names.end()) {
      trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*pair.second));
    } else {
      non_trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*pair.second));
    }
  }

  std::unordered_map<std::string, OrtValue> expected_trainable_param_name_to_ort_value;
  ORT_ENFORCE(CreateOrtValuesFromTensorProtos(trainable_param_values, expected_trainable_param_name_to_ort_value).IsOK());

  // Remove the tempoprary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  /// Phase 2 - Run save checkpoint APIs.
  /// And check the result checkpoint files.

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(trainable_param_values, non_trainable_param_values, checkpoint_path));

  // Check the ckpt files in the directory.
  std::set<PathString> expected_file_names{"paramfrozen_tensors.pbseq", "paramtrain_tensors.pbseq"};
  std::set<PathString> valid_file_names;
  LoopDir(checkpoint_path,
          [&valid_file_names, &checkpoint_path](const PathChar* filename, OrtFileType file_type) -> bool {
            PathString filename_str = filename;
            bool is_valid_ckpt_file_exts = HasExtensionOf(filename_str, ORT_TSTR("pbseq"));
            if (filename_str[0] == '.' || file_type == OrtFileType::TYPE_DIR || !is_valid_ckpt_file_exts) {
              return true;
            }
            valid_file_names.emplace(filename_str);
            return true;
          });

  ASSERT_EQ(expected_file_names, valid_file_names);

  /// Phase 3 - Run load checkpoint APIs.
  /// And check the result comparible with initial parameter values.

  // Call Load APIs
  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_path, checkpoint_state_to_load));
  ModuleCheckpointState module_state = checkpoint_state_to_load.module_checkpoint_state;
  const auto& param_states = module_state.named_parameters;
  std::unordered_map<std::string, OrtValue> restored_param_name_to_ort_values;
  std::vector<std::string> restored_trainable_param_names;
  for (auto it = param_states.begin(); it != param_states.end(); ++it) {
    restored_param_name_to_ort_values.insert({it->first, it->second->Data()});
    if (it->second->RequiresGrad()) {
      restored_trainable_param_names.emplace_back(it->first);
    }
  }

  // Check loaded parameter's values are same with original ones.
  ASSERT_EQ(expected_trainable_param_name_to_ort_value.size(), restored_trainable_param_names.size());
  ASSERT_EQ(expected_trainable_param_name_to_ort_value.size(), 7);
  ASSERT_EQ(restored_param_name_to_ort_values.size(), 9);

  std::sort(expected_trainable_param_names.begin(), expected_trainable_param_names.end());
  std::sort(restored_trainable_param_names.begin(), restored_trainable_param_names.end());
  ASSERT_EQ(expected_trainable_param_names, restored_trainable_param_names);

  for (const auto& name : restored_trainable_param_names) {
    const auto& restored_ort_value = restored_param_name_to_ort_values[name];
    const auto& expected_ort_value = expected_trainable_param_name_to_ort_value.at(name);

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

/**
 * Create Optimizer with sets of parameters,
 * Save Optimizer states into ORT checkpoint files,
 * Then load it into ORT, compare with the initial optimizer states values.
 */
TEST(CheckpointApiTest, SaveOptimizerStateAsCheckpoint_ThenLoad_CPU) {
  /// Phase 1 - Test Preparison
  /// Prepare the data and dest folder for saving checkpoint.
  /// Also cooked the data for test result comparision.

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
  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters;
  for (auto it = name_to_ort_value.begin(); it != name_to_ort_value.end(); ++it) {
    bool is_trainable =
        std::find(trainable_param_names.begin(), trainable_param_names.end(), it->first) != trainable_param_names.end();
    auto param = std::make_shared<Parameter>(it->first, it->second, is_trainable);
    named_parameters.insert({it->first, param});
  }
  auto optimizer = Optimizer(model_uri, named_parameters);

  /// Phase 2 - Run Optimizer.GetStateDict and call save checkpoint APIs.
  /// And check the result checkpoint files.

  CheckpointState checkpoint_state;
  ORT_ENFORCE(optimizer.GetStateDict(checkpoint_state.optimizer_checkpoint_state).IsOK());

  // Remove the tempoprary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(checkpoint_state, checkpoint_path));

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

  /// Phase 3 - Run load checkpoint APIs.
  /// And check the result comparible with initial optimizer state values.

  // Call Load APIs
  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_path, checkpoint_state_to_load));
  OptimizerCheckpointState optimizer_state = checkpoint_state_to_load.optimizer_checkpoint_state;
  std::unordered_map<std::string, std::shared_ptr<GroupOptimizerState>>&
      group_optimizer_states = optimizer_state.group_named_optimizer_states;

  ASSERT_EQ(group_optimizer_states.size(), 1);
  ASSERT_EQ(group_optimizer_states.begin()->first, "group0");

  std::unordered_map<std::string, ParameterOptimizerState>&
      param_named_optimizer_states = group_optimizer_states["group0"]->param_named_optimizer_states;

  ASSERT_EQ(param_named_optimizer_states.size(), 2);
  auto it = param_named_optimizer_states.begin();
  ASSERT_EQ(it->first, "param1");
  std::advance(it, 1);
  ASSERT_EQ(it->first, "param4");

  for (auto it = param_named_optimizer_states.begin(); it != param_named_optimizer_states.end(); ++it) {
    for (auto& state_pair : it->second.momentum_named_states) {
      ASSERT_TRUE(state_pair.first == "momentum0" || state_pair.first == "momentum1");
      const OrtValue& restored_ort_value = state_pair.second;
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

/**
 * Create PropertyBag with sets of properties,
 * Save properties into ORT checkpoint files,
 * Then load it into ORT, compare with the initial properties' values.
 */
TEST(CheckpointApiTest, SaveCustomPropertyAsCheckpoint_ThenLoad_CPU) {
  /// Phase 1 - Test Preparison
  /// Prepare the data and dest folder for saving checkpoint.

  CheckpointState checkpoint_state;
  PropertyBag& property_bag = checkpoint_state.property_bag;

  float f_data = 0.5f;
  std::string f_property_name("float_number");
  property_bag.AddProperty<float>(f_property_name, f_data);

  int64_t i_data = 400;
  std::string i_property_name("dataset_epoch_index");
  property_bag.AddProperty<int64_t>(i_property_name, i_data);

  std::string s_data("/data/path/train.bin");
  std::string s_property_name("train_data_path");
  property_bag.AddProperty<std::string>(s_property_name, s_data);

  // Remove the tempoprary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  if (Env::Default().FolderExists(ckpt_test_root_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(ckpt_test_root_dir).IsOK());
  }
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  /// Phase 2 - Call save checkpoint APIs.
  /// And check the result checkpoint files.

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(checkpoint_state, checkpoint_path));

  // Check the ckpt files in the directory.
  std::set<PathString> expected_file_names{
      "custom_properties.pbseq",
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
  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_path, checkpoint_state_to_load));
  PropertyBag& restored_property_bag = checkpoint_state_to_load.property_bag;
  ASSERT_EQ(restored_property_bag.Size(), 3);
  float restored_f_data = restored_property_bag.GetProperty<float>(f_property_name);
  ASSERT_FLOAT_EQ(f_data, restored_f_data);
  int64_t restored_i_data = restored_property_bag.GetProperty<int64_t>(i_property_name);
  ASSERT_EQ(i_data, restored_i_data);
  std::string restored_s_data = restored_property_bag.GetProperty<std::string>(s_property_name);
  ASSERT_EQ(s_data, restored_s_data);
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
