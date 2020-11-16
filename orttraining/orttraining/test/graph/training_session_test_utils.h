// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/test/optimizer/horizontal_parallel_test_utils.h"

#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

#ifdef USE_CUDA
#include "bert_toy_fetches.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

namespace onnxruntime {
namespace test {

namespace {
constexpr auto ORIGINAL_MODEL_PATH = ORT_TSTR("testdata/test_training_model.onnx");
constexpr auto BACKWARD_MODEL_PATH = ORT_TSTR("testdata/temp_backward_model.onnx");
constexpr auto CONCAT_MODEL_PATH = ORT_TSTR("testdata/transform/concat_trainable.onnx");
constexpr const char* const k_adam_optimizer_op_name = "AdamOptimizer";
constexpr const char* const k_lamb_optimizer_op_name = "LambOptimizer";
const std::vector<std::string> WEIGHT_NAMES = {"W1", "W2", "W3", "B1", "B2", "B3"};
const std::unordered_map<std::string, std::vector<int64_t>> WEIGHT_TO_SHAPE_MAP = {
    {"B3", {10}},
    {"W1", {784, 128}},
    {"W2", {128, 32}},
    {"B2", {32}},
    {"W3", {32, 10}},
    {"B1", {128}}};
const std::vector<std::string> MOMENT_PREFIX = {"Moment_1", "Moment_2"};
const std::vector<std::string> MOMENT_UC_PREFIX = {"Moment_1", "Moment_2", "Update_Count"};
constexpr char STEP_TENSOR_NAME[] = "Step";
constexpr char UC_TENSOR_NAME[] = "Update_Count";

void GenerateOptimizerConfig(const std::string optimizer_name,
                             const bool use_mixed_precision,
                             TrainingSession::TrainingConfiguration& config) {
  TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
  opt.name = optimizer_name;
  opt.learning_rate_input_name = "Learning_Rate";
  opt.weight_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, float>(); };
  opt.weight_int_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, int64_t>(); };
  opt.use_mixed_precision_moments = use_mixed_precision;
  opt.do_all_reduce_in_mixed_precision_type = use_mixed_precision;
  opt.use_nccl = true;
  config.optimizer_config = opt;
}

void GenerateOpimizerInitialState(const std::string& optimizer_op_name, TrainingSession::OptimizerState& optimizer_state) {
  TrainingSession::OptimizerState result;
  std::vector<int64_t> uc_value = {4};
  MLValue mlValue;
  NameMLValMap shared_states;
  for (auto& weight_name : WEIGHT_NAMES) {
    NameMLValMap optim_state;

    std::vector<int64_t> param_dims = WEIGHT_TO_SHAPE_MAP.at(weight_name);
    int64_t num_ele = std::accumulate(param_dims.begin(), param_dims.end(), 1, std::multiplies<int64_t>());
    for (auto& param_prefix : MOMENT_PREFIX) {
      std::vector<float> param_value(num_ele, 2.5f);

      TrainingUtil::CreateCpuMLValue(param_dims, param_value, &mlValue);
      optim_state.insert(std::make_pair(param_prefix, std::move(mlValue)));
    }
    if (optimizer_op_name == k_adam_optimizer_op_name) {
      CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, uc_value, &mlValue);
      optim_state.insert(std::make_pair(UC_TENSOR_NAME, std::move(mlValue)));
    }
    result.insert(std::make_pair(weight_name, std::move(optim_state)));
  }
  if (optimizer_op_name == k_lamb_optimizer_op_name) {
    // add "Step" for lamb
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, uc_value, &mlValue);
    shared_states.insert(std::make_pair(STEP_TENSOR_NAME, std::move(mlValue)));
    result.insert(std::make_pair(onnxruntime::training::SHARED_STATES_KEY, std::move(shared_states)));
  }
  optimizer_state = std::move(result);
}

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
    for (auto& param_prefix : MOMENT_UC_PREFIX) {
      std::string param_name = param_prefix + "_" + weight_name;
      const auto& param_state_it = training_state.find(param_name);
      if (param_state_it != training_state.end()) {
        optim_state.insert(std::make_pair(param_prefix, param_state_it->second));
      }
    }
    optimizer_state.insert(std::make_pair(weight_name, optim_state));
  }
  NameMLValMap shared_optim_state;
  const auto& param_state_it = training_state.find(STEP_TENSOR_NAME);
  if (param_state_it != training_state.end()) {
    shared_optim_state.insert(std::make_pair(STEP_TENSOR_NAME, param_state_it->second));
    optimizer_state.insert(std::make_pair(onnxruntime::training::SHARED_STATES_KEY, shared_optim_state));
  }
}

void VerifyState(const DataTransferManager& data_transfer_mgr, const NameMLValMap& expected_state, const NameMLValMap& actual_state) {
  for (auto& a_state_it : actual_state) {
    std::string key = a_state_it.first;
    const auto& e_state_it = expected_state.find(key);
    ORT_ENFORCE(e_state_it != expected_state.end());
    auto& expected_tensor = e_state_it->second.Get<Tensor>();
#ifdef USE_CUDA
    auto& actual_gpu_tensor = a_state_it.second.Get<Tensor>();

    // Copying tensor to CPU when cuda is enabled.
    auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
    Tensor actual_tensor{actual_gpu_tensor.DataType(), actual_gpu_tensor.Shape(), cpu_allocator};
    ORT_ENFORCE(data_transfer_mgr.CopyTensor(actual_gpu_tensor, actual_tensor).IsOK());
#else
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
    } else {  // adding a tolerance as after a train step, the moment tensor value will be updated
      horizontal_parallel_test_utils::VerifyOutputs(expected_tensor, actual_tensor, true, 1e-8, 1e-7, 0.32f);
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
}  // namespace

static TrainingSession::TrainingConfiguration MakeBasicTrainingConfig() {
  TrainingSession::TrainingConfiguration config{};
  config.model_with_training_graph_path = BACKWARD_MODEL_PATH;
  config.loss_function_config = TrainingSession::TrainingConfiguration::LossFunctionConfiguration{};
  config.loss_function_config.value().loss_function_info =
      LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  return config;
}

static Status BuildBackPropGraph(
    const PathString& forward_model_file,
    const TrainingSession::TrainingConfiguration& config,
    PathString& backward_model_file) {
  std::unique_ptr<Environment> env;
  ORT_RETURN_IF_ERROR(Environment::Create(nullptr, env));

  SessionOptions so{};
  TrainingSession training_session{so, *env};

  std::cout << "Loading source model file = " << ToMBString(forward_model_file) << "\n";

  ORT_RETURN_IF_ERROR(training_session.Load(forward_model_file));

  TrainingSession::TrainingConfigurationResult config_result{};
  ORT_RETURN_IF_ERROR(training_session.ConfigureForTraining(config, config_result));

  backward_model_file = config.model_with_training_graph_path.value();

  return Status::OK();
}

/**
 * Run a training session for this model for 1 epoch, using batch size of 1 and synthetic input data.
 * @param so - SessionOptions for this run.
 * @param forward_model_file - Model file to be run.
 * @param config - Training session config
 * @return TrainingSession for this run.
 */
static std::unique_ptr<TrainingSession> BuildAndRunTrainingSessionWithChecks(
    const SessionOptions& so, const PathString& forward_model_file,
    const TrainingSession::TrainingConfiguration& config) {
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  std::unique_ptr<TrainingSession> training_session = onnxruntime::make_unique<TrainingSession>(so, *env);

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
  CUDAExecutionProviderInfo xp_info;
  ORT_THROW_IF_ERROR(training_session->RegisterExecutionProvider(onnxruntime::make_unique<CUDAExecutionProvider>(xp_info)));
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

    float lr = 0.001;
    MLValue lrMLValue;
    TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{lr}, &lrMLValue);
    fw_feeds.first.push_back(lr_feed_name);
    fw_feeds.second.push_back(lrMLValue);
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

/**
 * Run a training session for this model for 1 epoch, using batch size of 1 and synthetic input data.
 * @param so - SessionOptions for this run.
 * @param backprop_model_file - Model file to be run. This should already contain loss function and backward prop nodes.
 * @return TrainingSession for this run.
 */
static std::unique_ptr<TrainingSession> RunTrainingSessionWithChecks(
    const SessionOptions& so, const PathString& backprop_model_file) {
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  std::unique_ptr<TrainingSession> training_session = onnxruntime::make_unique<TrainingSession>(so, *env);

  ORT_THROW_IF_ERROR(training_session->Load(backprop_model_file));

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  ORT_THROW_IF_ERROR(res.first);
  ORT_ENFORCE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << '\n';

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

  auto output_names_include_gradients = GetModelOutputNames(*training_session);
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  auto start_time = std::chrono::high_resolution_clock::now();

  ORT_THROW_IF_ERROR(training_session->Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed = TimeDiffMicroSeconds(start_time, end_time);
  std::cout << "Training session run completed in " << elapsed << " microseconds.\n";

  return training_session;
}

#ifdef USE_CUDA
static void RunBertTrainingWithChecks(
    const SessionOptions& so,
    const PathString& backprop_model_file) {
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));

  std::unique_ptr<TrainingSession> training_session = onnxruntime::make_unique<TrainingSession>(so, *env);

  ASSERT_STATUS_OK(training_session->Load(backprop_model_file));

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  ASSERT_STATUS_OK(res.first);
  ASSERT_TRUE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << '\n';

  CUDAExecutionProviderInfo xp_info;
  ASSERT_STATUS_OK(training_session->RegisterExecutionProvider(onnxruntime::make_unique<CUDAExecutionProvider>(xp_info)));

  ASSERT_STATUS_OK(training_session->Initialize());

  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;
  run_options.training_mode = true;

  // Creating feeds
  int batch_size = 13;
  int max_seq_len_in_batch = 7;
  std::vector<std::string> feed_names = {
      "input_ids",
      "token_type_ids",
      "input_mask",
      "masked_lm_ids",
      "next_sentence_labels",
      "masked_lm_positions",
  };
  std::vector<TensorShape> tensor_shapes = {
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch},
      {batch_size},
      {batch_size, max_seq_len_in_batch},
      {batch_size, max_seq_len_in_batch}};

  std::vector<std::vector<int64_t>> tensor_values = {
      /*input_ids*/
      {49, 97, 53, 5, 33, 65, 62,
       51, 38, 61, 45, 74, 27, 64,
       17, 36, 17, 96, 12, 79, 32,
       68, 90, 77, 18, 39, 12, 93,
       9, 87, 42, 60, 71, 12, 45,
       55, 40, 78, 81, 26, 70, 61,
       56, 66, 33, 7, 70, 1, 11,
       92, 51, 90, 85, 80, 0, 78,
       63, 42, 31, 93, 41, 90, 8,
       24, 72, 28, 30, 18, 69, 57,
       11, 10, 40, 65, 62, 13, 38,
       70, 37, 90, 15, 70, 42, 69,
       26, 77, 70, 75, 36, 56, 11},
      /*token_type_ids*/
      {12, 13, 1, 8, 15, 12, 9,
       15, 11, 6, 4, 9, 4, 3,
       8, 4, 9, 3, 2, 10, 15,
       3, 11, 13, 10, 6, 15, 14,
       8, 1, 0, 2, 12, 0, 15,
       10, 7, 10, 2, 6, 7, 7,
       4, 14, 2, 2, 10, 15, 3,
       9, 9, 3, 10, 6, 9, 14,
       2, 12, 10, 7, 9, 5, 6,
       5, 1, 8, 15, 2, 2, 4,
       4, 1, 2, 12, 8, 7, 6,
       13, 8, 14, 15, 11, 2, 10,
       3, 15, 10, 6, 7, 0, 8},
      /*input_mask*/
      {1, 1, 0, 1, 1, 1, 1,
       1, 1, 0, 0, 1, 0, 0,
       1, 0, 1, 0, 0, 1, 1,
       0, 1, 1, 1, 0, 1, 1,
       1, 0, 0, 0, 1, 0, 1,
       1, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 1, 0,
       1, 1, 0, 1, 0, 1, 1,
       0, 1, 1, 0, 1, 0, 0,
       0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 0, 1,
       0, 1, 1, 0, 0, 0, 1},
      /*masked_lm_ids*/
      {1, 1, 0, 1, 2, 1, 1,
       1, 1, 1, 2, 0, 2, 0,
       1, 0, 0, 2, 1, 2, 2,
       2, 0, 1, 0, 2, 0, 2,
       1, 1, 2, 0, 1, 1, 1,
       2, 2, 0, 2, 1, 1, 2,
       1, 0, 2, 0, 0, 2, 1,
       2, 2, 2, 0, 2, 1, 1,
       0, 2, 1, 2, 0, 0, 2,
       0, 0, 0, 2, 1, 0, 0,
       1, 2, 1, 0, 1, 2, 1,
       2, 0, 2, 1, 2, 0, 2,
       2, 2, 1, 1, 0, 2, 1},
      /*next_sentence_labels*/
      {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0},
      /*masked_lm_positions*/
      {0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6,
       0, 1, 2, 3, 4, 5, 6}};

  std::vector<OrtValue> feeds(feed_names.size());
  for (size_t i = 0; i < 6; ++i) {
    TrainingUtil::CreateCpuMLValue(tensor_shapes[i].GetDims(), tensor_values[i], &feeds[i]);
  }

  auto output_names_include_gradients = GetModelOutputNames(*training_session);
  std::vector<std::string> fetch_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(training_session->Run(run_options, feed_names, feeds, fetch_names, &fetches));

  for (size_t i = 0; i < fetch_names.size(); ++i) {
    if (!fetches[i].IsAllocated() || !!fetches[i].IsTensor())
      continue;

    const Tensor& tensor = fetches[i].Get<Tensor>();
    if (DataTypeImpl::GetType<float>() != tensor.DataType()) {
      continue;
    }

    const std::string& name = fetch_names[i];
    if (BERT_TOY_FETCHES.find(name) == BERT_TOY_FETCHES.end()) {
      continue;
    }

    auto gradient_ref = BERT_TOY_FETCHES.at(name);
    EXPECT_TRUE(static_cast<size_t>(tensor.Shape().Size()) == gradient_ref.size());

    float max_diff = 0;
    float max_percent_diff = 0;
    const float* data = tensor.template Data<float>();
    for (size_t idx = 0; idx < gradient_ref.size(); ++idx) {
      float diff = std::fabs(static_cast<float>(gradient_ref[idx]) - data[idx]);
      max_diff = std::fmax(max_diff, diff);
      max_percent_diff = std::fmax(max_percent_diff, diff / data[idx]);
    }
    EXPECT_TRUE(max_diff < 1e-5) << name << " is incorrect: max_diff is " << max_diff;
    if (max_diff > 1e-10) {
      EXPECT_TRUE(max_percent_diff < 0.01f) << name << " is incorrect: max_percent_diff is "
                                            << max_percent_diff;
    }
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
