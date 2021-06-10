// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/test/session/training_session_test_utils.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "test/util/include/default_providers.h"

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

namespace onnxruntime {
namespace test {
namespace training_session_test_utils {

void GenerateOptimizerConfig(const std::string optimizer_name,
                             const bool use_mixed_precision_moments,
                             TrainingSession::TrainingConfiguration& config) {
  TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
  opt.name = optimizer_name;
  opt.learning_rate_input_name = "Learning_Rate";
  opt.weight_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, float>(); };
  opt.weight_int_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, int64_t>(); };
  opt.use_mixed_precision_moments = use_mixed_precision_moments;
  opt.do_all_reduce_in_mixed_precision_type = true;
  opt.use_nccl = true;
  config.optimizer_config = opt;
}

template <class T>
void GenerateOptimizerInitialState(const std::string& optimizer_op_name, const T init_moment_value, TrainingSession::OptimizerState& optimizer_state) {
  TrainingSession::OptimizerState result;
  std::vector<int64_t> uc_value = {4};
  MLValue mlValue;
  NameMLValMap shared_states;
  for (auto& weight_name : WEIGHT_NAMES) {
    NameMLValMap optim_state;

    std::vector<int64_t> param_dims = WEIGHT_TO_SHAPE_MAP.at(weight_name);
    int64_t num_ele = std::accumulate(param_dims.begin(), param_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

    for (auto& param_prefix : MOMENTS_PREFIXES) {
      std::vector<T> param_value(num_ele, init_moment_value);
      TrainingUtil::CreateCpuMLValue<T>(param_dims, param_value, &mlValue);
      optim_state.insert(std::make_pair(param_prefix, std::move(mlValue)));
    }
    if (optimizer_op_name == k_adam_optimizer_op_name) {
      CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, uc_value, &mlValue);
      optim_state.insert(std::make_pair(ADAM_UC_PREFIX, std::move(mlValue)));
    }
    result.insert(std::make_pair(weight_name, std::move(optim_state)));
  }
  if (optimizer_op_name == k_lamb_optimizer_op_name) {
    // add "Step" for lamb
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, uc_value, &mlValue);
    shared_states.insert(std::make_pair(LAMB_STEP_TENSOR_NAME, std::move(mlValue)));
    result.insert(std::make_pair(onnxruntime::training::SHARED_OPTIMIZER_STATES_KEY, std::move(shared_states)));
  }
  optimizer_state = std::move(result);
}

template void GenerateOptimizerInitialState<float>(const std::string& optimizer_op_name, const float init_moment_value, TrainingSession::OptimizerState& optimizer_state);
template void GenerateOptimizerInitialState<MLFloat16>(const std::string& optimizer_op_name, const MLFloat16 init_moment_value, TrainingSession::OptimizerState& optimizer_state);

void SeparateStateTensors(const NameMLValMap& training_state, NameMLValMap& model_state, TrainingSession::OptimizerState& optimizer_state) {
  NameMLValMap result;
  std::transform(
      WEIGHT_NAMES.begin(), WEIGHT_NAMES.end(), std::inserter(result, result.end()),
      [training_state](const std::string& weight_name) {
        return std::make_pair(
            weight_name, training_state.at(weight_name));
      });

  model_state = std::move(result);
  for (auto& weight_name : WEIGHT_NAMES) {
    NameMLValMap optim_state;
    std::vector<std::string> per_weight_states_prefixes(MOMENTS_PREFIXES);
    per_weight_states_prefixes.push_back(ADAM_UC_PREFIX);
    for (auto& param_prefix : per_weight_states_prefixes) {
      std::string param_name = param_prefix + "_" + weight_name;
      const auto& param_state_it = training_state.find(param_name);
      if (param_state_it != training_state.end()) {
        optim_state.insert(std::make_pair(param_prefix, param_state_it->second));
      }
    }
    optimizer_state.insert(std::make_pair(weight_name, optim_state));
  }
  NameMLValMap shared_optim_state;
  const auto& param_state_it = training_state.find(LAMB_STEP_TENSOR_NAME);
  if (param_state_it != training_state.end()) {
    shared_optim_state.insert(std::make_pair(LAMB_STEP_TENSOR_NAME, param_state_it->second));
    optimizer_state.insert(std::make_pair(onnxruntime::training::SHARED_OPTIMIZER_STATES_KEY, shared_optim_state));
  }
}

void VerifyState(const DataTransferManager& data_transfer_mgr, const NameMLValMap& expected_state, const NameMLValMap& actual_state) {
  for (auto& a_state_it : actual_state) {
    std::string key = a_state_it.first;
    const auto& e_state_it = expected_state.find(key);
    ORT_ENFORCE(e_state_it != expected_state.end());
    auto& expected_tensor = e_state_it->second.Get<Tensor>();
#if defined(USE_CUDA) || defined(USE_ROCM)
    auto& actual_gpu_tensor = a_state_it.second.Get<Tensor>();

    // Copying tensor to CPU when cuda is enabled.
    auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
    Tensor actual_tensor{actual_gpu_tensor.DataType(), actual_gpu_tensor.Shape(), cpu_allocator};
    ORT_ENFORCE(data_transfer_mgr.CopyTensor(actual_gpu_tensor, actual_tensor).IsOK());
#else
    ORT_UNUSED_PARAMETER(data_transfer_mgr);
    auto& actual_tensor = a_state_it.second.Get<Tensor>();
#endif
    if (expected_tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      // compare "Update_Count" or "Step"
      ASSERT_EQ(actual_tensor.GetElementType(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      ASSERT_EQ(expected_tensor.Shape(), actual_tensor.Shape());
      std::vector<int64_t> dims = {1};
      ASSERT_EQ(expected_tensor.Shape().GetDims(), dims);
      auto size = expected_tensor.Shape().Size();
      const std::vector<int64_t> expected(expected_tensor.template Data<int64_t>(), expected_tensor.template Data<int64_t>() + size);
      const std::vector<int64_t> actual(actual_tensor.template Data<int64_t>(), actual_tensor.template Data<int64_t>() + size);
      // the step value will be incremented by 1 after a train step
      ASSERT_EQ(expected[0] + 1, actual[0]);
    } else if (expected_tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
               expected_tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      // adding a tolerance as after a train step, the moment tensor value will be updated
      horizontal_parallel_test_utils::VerifyOutputs(expected_tensor, actual_tensor, true, 1e-8f, 1e-7f, 0.32f);
    }
  }
}

void VerifyOptimizerState(const DataTransferManager& data_transfer_manager, const TrainingSession::OptimizerState& expected_state, const TrainingSession::OptimizerState& actual_state) {
  for (const auto& a_state_it : actual_state) {
    std::string key = a_state_it.first;
    const auto& e_state_it = expected_state.find(key);
    ORT_ENFORCE(e_state_it != expected_state.end());
    VerifyState(data_transfer_manager, e_state_it->second, a_state_it.second);
  }
}

std::unordered_set<std::string> GetModelOutputNames(const InferenceSession& session) {
  const auto outputs_result = session.GetModelOutputs();
  ORT_ENFORCE(outputs_result.first.IsOK(), "Failed to get model outputs: ", outputs_result.first.ErrorMessage());
  std::unordered_set<std::string> output_names{};
  for (const auto* output : *outputs_result.second) {
    output_names.insert(output->Name());
  }
  return output_names;
}

TrainingSession::TrainingConfiguration MakeBasicTrainingConfig() {
  TrainingSession::TrainingConfiguration config{};
  config.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  config.loss_function_config = TrainingSession::TrainingConfiguration::LossFunctionConfiguration{};
  config.loss_function_config.value().loss_function_info =
      LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  return config;
}

std::unique_ptr<TrainingSession> BuildAndRunTrainingSessionWithChecks(
    const SessionOptions& so, const PathString& forward_model_file,
    const TrainingSession::TrainingConfiguration& config) {
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  std::unique_ptr<TrainingSession> training_session = std::make_unique<TrainingSession>(so, *env);

  std::cout << "Loading source model file = " << ToMBString(forward_model_file) << "\n";

  ORT_THROW_IF_ERROR(training_session->Load(forward_model_file));

  TrainingSession::TrainingConfigurationResult config_result{};
  ORT_THROW_IF_ERROR(training_session->ConfigureForTraining(config, config_result));

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  ORT_THROW_IF_ERROR(res.first);
  ORT_ENFORCE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << '\n';

#ifdef USE_CUDA
  ORT_THROW_IF_ERROR(training_session->RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#elif USE_ROCM
  ROCMExecutionProviderInfo xp_info;
  ORT_THROW_IF_ERROR(training_session->RegisterExecutionProvider(std::make_unique<ROCMExecutionProvider>(xp_info)));
#endif
  ORT_THROW_IF_ERROR(training_session->Initialize());

  std::vector<MLValue> gradient_fetches;
  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;
  run_options.training_mode = true;

  // Create dummy feeds
  std::vector<int64_t> image_dims = {1, 784};
  std::vector<int64_t> label_dims = {1, 10};
  std::vector<float> image_value(784, 1);
  std::vector<float> label_value(10, 1);

  MLValue imageMLValue;
  TrainingUtil::CreateCpuMLValue(image_dims, image_value, &imageMLValue);
  MLValue labelMLValue;
  TrainingUtil::CreateCpuMLValue(label_dims, label_value, &labelMLValue);

  auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X", "labels"}, {imageMLValue, labelMLValue});

  if (config.optimizer_config.has_value()) {
    auto optim_config = config.optimizer_config.value();
    auto lr_feed_name = optim_config.learning_rate_input_name;

    float lr = 0.001f;
    MLValue lrMLValue;
    TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{lr}, &lrMLValue);
    fw_feeds.first.push_back(lr_feed_name);
    fw_feeds.second.push_back(lrMLValue);
  }

  if (config_result.mixed_precision_config_result.has_value()) {
    const std::string& loss_scale_input_name =
        config_result.mixed_precision_config_result.value().loss_scale_input_name;
    float loss_scale = 2048.0f;
    MLValue loss_scaleMLValue;
    TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{loss_scale}, &loss_scaleMLValue);
    fw_feeds.first.push_back(loss_scale_input_name);
    fw_feeds.second.push_back(loss_scaleMLValue);
  }

  auto output_names_include_gradients = GetModelOutputNames(*training_session);
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  auto start_time = std::chrono::high_resolution_clock::now();

  ORT_THROW_IF_ERROR(training_session->Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed = TimeDiffMicroSeconds(start_time, end_time);
  std::cout << "Training session run completed in " << elapsed << " microseconds.\n";

  return training_session;
}

}  // namespace training_session_test_utils
}  // namespace test
}  // namespace onnxruntime
