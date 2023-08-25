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
#include "orttraining/training_api/module.h"
#include "orttraining/training_api/optimizer.h"
#include "orttraining/training_api/checkpoint_property.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/lr_scheduler.h"

#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test_environment.h"
#include "orttraining/test/training_api/common/synthetic_data_loader.h"
#include "orttraining/test/training_api/core/data_utils.h"
#include "default_providers.h"

using onnxruntime::test::TemporaryDirectory;
using namespace onnxruntime::training::api;

namespace onnxruntime::training::test {

#define MODEL_FOLDER ORT_TSTR("testdata/")

/**
 * Load ONNX model from file path, save into ORT checkpoint files,
 * Then load it into ORT, compare with the initial parameter values.
 */
TEST(CheckpointApiTest, SaveOnnxModelAsCheckpoint_ThenLoad_CPU) {
  /// Phase 1 - Test Preparation
  /// Prepare the data and dest folder for saving checkpoint.
  /// Also cooked the data for test result comparison.

  // Model path and trainable parameter name definitions.
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/gathernd/e2e.onnx";
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
  for (const auto& [initializer_name, tensor_proto] : initializer_tensors) {
    if (std::find(expected_trainable_param_names.begin(), expected_trainable_param_names.end(), initializer_name) !=
        expected_trainable_param_names.end()) {
      trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*tensor_proto));
    } else {
      non_trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*tensor_proto));
    }
  }

  std::unordered_map<std::string, OrtValue> expected_trainable_param_name_to_ort_value;
  ORT_ENFORCE(CreateOrtValuesFromTensorProtos(trainable_param_values, expected_trainable_param_name_to_ort_value)
                  .IsOK());

  // Remove the temporary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  /// Phase 2 - Run save checkpoint APIs.
  /// And check the result checkpoint files.

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(trainable_param_values, non_trainable_param_values, checkpoint_path));

  /// Phase 3 - Run load checkpoint APIs.
  /// And check the result comparable with initial parameter values.

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

    ASSERT_EQ(std::memcmp(expected_tensor.DataRaw(), restored_tensor.DataRaw(), expected_tensor.SizeInBytes()), 0);
  }
}

/**
 * Load ONNX model from file path, save into ORT checkpoint files,
 * Then load it into a bytes buffer and then load the buffer to a checkpoint, compare with the initial parameter values.
 */
TEST(CheckpointApiTest, SaveOnnxModelAsCheckpointThenLoadFromBufferCPU) {
  /// Phase 1 - Test Preparation
  /// Prepare the data and dest folder for saving checkpoint.
  /// Also cooked the data for test result comparison.

  // Model path and trainable parameter name definitions.
  auto model_uri = MODEL_FOLDER "transform/computation_reduction/gathernd/e2e.onnx";
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
  for (const auto& [initializer_name, tensor_proto] : initializer_tensors) {
    if (std::find(expected_trainable_param_names.begin(), expected_trainable_param_names.end(), initializer_name) !=
        expected_trainable_param_names.end()) {
      trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*tensor_proto));
    } else {
      non_trainable_param_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*tensor_proto));
    }
  }

  std::unordered_map<std::string, OrtValue> expected_trainable_param_name_to_ort_value;
  ORT_ENFORCE(CreateOrtValuesFromTensorProtos(trainable_param_values, expected_trainable_param_name_to_ort_value)
                  .IsOK());

  // Remove the temporary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  /// Phase 2 - Run save checkpoint APIs.
  /// And check the result checkpoint files.

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(trainable_param_values, non_trainable_param_values, checkpoint_path));

  /// Phase 3 - Run load checkpoint APIs.
  /// And check the result comparable with initial parameter values.

  // Call Load APIs
  size_t num_bytes = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(checkpoint_path.c_str(), num_bytes));
  std::vector<uint8_t> checkpoint_bytes(num_bytes);

  std::ifstream bytes_stream(checkpoint_path, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(checkpoint_bytes.data()), num_bytes);

  ASSERT_TRUE(bytes_stream);

  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpointFromBuffer(checkpoint_bytes, checkpoint_state_to_load));
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

    ASSERT_EQ(std::memcmp(expected_tensor.DataRaw(), restored_tensor.DataRaw(), expected_tensor.SizeInBytes()), 0);
  }
}

/**
 * Load ONNX model with parameters set to 0 from file path, Load Checkpoint weights into the Model,
 * Then compare the new weights to 0 to make sure they were changed after loading checkpoint to model.
 */
TEST(CheckpointApiTest, LoadCheckpointToModel) {
  // Phase 1: Load a Model with weights set to zero.
  auto model_uri = MODEL_FOLDER "training_api/zero_model.onnx";
  ONNX_NAMESPACE::ModelProto p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model));
  // Phase 2: Load the checkpoint weights into the Model.
  // Call Load APIs
  auto checkpoint_uri = MODEL_FOLDER "training_api/checkpoint.ckpt";
  ASSERT_STATUS_OK(LoadCheckpointToModel(checkpoint_uri, p_model));

  // Phase 3: Make sure the Model's weights are not equal to zero after loading the new ones.
  // Load imported initializers into the Model
  for (auto& init : p_model.graph().initializer()) {
    // Convert the tensor bytes to a float vector to compare.
    ASSERT_TRUE(init.has_raw_data());
    size_t len = init.raw_data().size() / sizeof(float);
    InlinedVector<float> float_values(len);
    std::copy(init.raw_data().data(), init.raw_data().data() + init.raw_data().size(), reinterpret_cast<char*>(&float_values.front()));

    // Make sure the weights are no longer a zero.
    for (size_t i = 0; i < len; i++) {
      ASSERT_NE(float_values[i], 0.0f);
    }
  }
}

/**
 * Create Module passing in checkpoint state,
 * Create Optimizer passing in checkpoint state.
 * Save Optimizer states into ORT checkpoint files,
 * Then load it into ORT, compare with the initial optimizer states values.
 */
TEST(CheckpointApiTest, SaveOptimizerStateAsCheckpoint_ThenLoad) {
  /// Phase 1 - Test Preparation
  /// Prepare the data and dest folder for saving checkpoint.
  /// Also cooked the data for test result comparison.
  auto model_uri = "testdata/training_api/training_model.onnx";
  auto optim_uri = "testdata/training_api/adamw.onnx";

  // Generate randomized weight values using synthetic data generator.
  constexpr int64_t fc2_weight_dim_in = 10, fc2_weight_dim_out = 500,
                    fc1_weight_dim_in = 500, fc1_weight_dim_out = 784;
  const std::vector<int64_t> fc1_weight_shape{fc1_weight_dim_in, fc1_weight_dim_out};
  const std::vector<int64_t> fc1_bias_shape{fc1_weight_dim_in};
  const std::vector<int64_t> fc2_weight_shape{fc2_weight_dim_in, fc2_weight_dim_out};
  const std::vector<int64_t> fc2_bias_shape{fc2_weight_dim_in};

  onnxruntime::training::test::training_api::SyntheticDataLoader data_loader;
  auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
  sample.AddFloatInput(fc1_weight_shape);
  sample.AddFloatInput(fc1_bias_shape);
  sample.AddFloatInput(fc2_weight_shape);
  sample.AddFloatInput(fc2_bias_shape);
  data_loader.AddSyntheticSampleBatch(std::move(sample));

  std::vector<Ort::Value> all_weights_values;
  data_loader.GetNextSampleBatch(all_weights_values);
  ASSERT_EQ(all_weights_values.size(), 4);
  NameMLValMap name_to_ort_value{
      {"fc1.weight", *all_weights_values[0]},
      {"fc1.bias", *all_weights_values[1]},
      {"fc2.weight", *all_weights_values[2]},
      {"fc2.bias", *all_weights_values[3]},
  };

  // Module/Optimizer creation and trainable parameter name definitions.
  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters;
  for (auto it = name_to_ort_value.begin(); it != name_to_ort_value.end(); ++it) {
    auto param = std::make_shared<Parameter>(it->first, it->second, true /*is_trainable*/);
    named_parameters.insert({it->first, param});
  }

  auto state = CheckpointState();
  state.module_checkpoint_state.named_parameters = named_parameters;

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));
  std::vector<std::shared_ptr<IExecutionProvider>> providers;
#if defined(USE_CUDA)
  providers.push_back(onnxruntime::test::DefaultCudaExecutionProvider());
#endif
  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                           std::nullopt,
                                           std::optional<std::string>(onnxruntime::ToUTF8String(optim_uri)));
  auto model = std::make_unique<Module>(model_identifier, &state, session_option,
                                        *env, providers);
  auto optimizer = std::make_unique<Optimizer>(model_identifier, &state, session_option,
                                               *env, providers);

  // Remove the temporary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(state, checkpoint_path, true));

  /// Phase 2 - Run load checkpoint APIs.
  /// Validate the result matches with initial optimizer state values.

  // Call Load APIs
  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_path, checkpoint_state_to_load));
  OptimizerCheckpointState optimizer_state = checkpoint_state_to_load.optimizer_checkpoint_state;
  InlinedHashMap<std::string, std::shared_ptr<GroupOptimizerState>>&
      group_optimizer_states = optimizer_state.group_named_optimizer_states;

  ASSERT_EQ(group_optimizer_states.size(), 1);
  ASSERT_EQ(group_optimizer_states.begin()->first, "group0");

  InlinedHashMap<std::string, ParameterOptimizerState>&
      param_named_optimizer_states = group_optimizer_states["group0"]->param_named_optimizer_states;

  ASSERT_EQ(param_named_optimizer_states.size(), named_parameters.size());

  for (auto it = param_named_optimizer_states.begin(); it != param_named_optimizer_states.end(); ++it) {
    ASSERT_TRUE(named_parameters.find(it->first) != named_parameters.end());
    for (auto& [momentum_name, restored_ort_value] : it->second) {
      ASSERT_TRUE(momentum_name == "momentum0" || momentum_name == "momentum1");
      const OrtValue& param_ort_value = name_to_ort_value[it->first];
      ASSERT_TRUE(restored_ort_value.IsTensor() && param_ort_value.IsTensor());
      const Tensor& restored_tensor = restored_ort_value.Get<Tensor>();
      const Tensor& param_tensor = param_ort_value.Get<Tensor>();

      ASSERT_EQ(param_tensor.DataType(), restored_tensor.DataType());
      ASSERT_EQ(param_tensor.SizeInBytes(), restored_tensor.SizeInBytes());

      std::vector<float> state_vect;
      CpuOrtValueToVec(restored_ort_value, state_vect);
      for (size_t i = 0; i < state_vect.size(); i++) {
        ASSERT_EQ(state_vect[i], 0.0f);
      }
    }
  }
}

/**
 * Create PropertyBag with sets of properties,
 * Save properties into ORT checkpoint files,
 * Then load it into ORT, compare with the initial properties' values.
 */
TEST(CheckpointApiTest, SaveCustomPropertyAsCheckpoint_ThenLoad_CPU) {
  /// Phase 1 - Test Preparation
  /// Prepare the data and dest folder for saving checkpoint.

  CheckpointState checkpoint_state;
  PropertyBag& property_bag = checkpoint_state.property_bag;

  float f_data = 0.5f;
  std::string f_property_name("float_number");
  property_bag.AddProperty(f_property_name, f_data);

  int64_t i_data = 400;
  std::string i_property_name("dataset_epoch_index");
  property_bag.AddProperty(i_property_name, i_data);

  std::string s_data("/data/path/train.bin");
  std::string s_property_name("train_data_path");
  property_bag.AddProperty(s_property_name, s_data);

  // Remove the temporary directory if it already exists.
  auto ckpt_test_root_dir = ORT_TSTR("checkpointing_api_test_dir");
  TemporaryDirectory tmp_dir{ckpt_test_root_dir};

  /// Phase 2 - Call save checkpoint APIs.
  /// And check the result checkpoint files.

  // Call Save APIs.
  PathString checkpoint_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("e2e_ckpt_save_cpu"))};
  ASSERT_STATUS_OK(SaveCheckpoint(checkpoint_state, checkpoint_path, false));

  // Call Load APIs
  CheckpointState checkpoint_state_to_load;
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_path, checkpoint_state_to_load));
  PropertyBag& restored_property_bag = checkpoint_state_to_load.property_bag;
  ASSERT_EQ(restored_property_bag.size(), 3);
  float restored_f_data = restored_property_bag.GetProperty<float>(f_property_name);
  ASSERT_FLOAT_EQ(f_data, restored_f_data);
  int64_t restored_i_data = restored_property_bag.GetProperty<int64_t>(i_property_name);
  ASSERT_EQ(i_data, restored_i_data);
  std::string restored_s_data = restored_property_bag.GetProperty<std::string>(s_property_name);
  ASSERT_EQ(s_data, restored_s_data);
}
}  // namespace onnxruntime::training::test
