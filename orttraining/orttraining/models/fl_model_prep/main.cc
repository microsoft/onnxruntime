// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/bfc_arena.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "core/graph/model.h"

#include <condition_variable>
#include <mutex>
#include <tuple>

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;


static std::vector<FreeDimensionOverride> training_overrides = {};
static SessionOptions TRAINING_SESSION_OPTION = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
    ExecutionOrder::PRIORITY_BASED,    //execution_order
    false,                             //enable_profiling
    ORT_TSTR(""),                      //optimized_model_filepath
    true,                              //enable_mem_pattern
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level1,          //graph_optimization_level
    {},                                //intra_op_param
    {},                                //inter_op_param
    training_overrides,                //free_dimension_overrides
    true,                              //use_per_session_threads
    true,                              //thread_pool_allow_spinning
    false,                             //use_deterministic_compute
    {},                                //session_configurations
    {},                                // initializers_to_share_map
};

static SessionOptions INFERENCE_SESSION_OPTION = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
    ExecutionOrder::PRIORITY_BASED,    //execution_order
    false,                             //enable_profiling
    ORT_TSTR(""),                      //optimized_model_filepath
    true,                              //enable_mem_pattern
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level2,          //graph_optimization_level
    {},                                //intra_op_param
    {},                                //inter_op_param
    training_overrides,                //free_dimension_overrides
    true,                              //use_per_session_threads
    true,                              //thread_pool_allow_spinning
    false,                             //use_deterministic_compute
    {},                                //session_configurations
    {},                                // initializers_to_share_map
};


struct ModelPrepParameters {
  std::string starting_onnx_model_path;
  std::string training_onnx_model_path;
  std::string modified_training_onnx_model_path;
  std::string training_optimized_ort_model_path;

  std::string predicted_value_name;
  std::string expected_value_name;

  std::string loss_output_name;
  std::string loss_function;

  std::string optimizer_type;
  std::string learning_rate_input_name;
};

Status ConfigModelPrepParameters(int argc, char* argv[], ModelPrepParameters& params) {
  cxxopts::Options options("POC Training", "Derive an ORT format training graph that can be exeuted in an InferenceSession, from an ONNX format inference graph.");
  // clang-format off
  options
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("optimizer_type", "optimizer to be used for training.", cxxopts::value<std::string>())
      ("loss_function", "loss function to be used for training.", cxxopts::value<std::string>())
      ("loss_output_name", "the name of the graph output to be used for the loss function.", cxxopts::value<std::string>())
      ("learning_rate_input_name", "the name of the input to be used for the learning rate.", cxxopts::value<std::string>())
      ("expected_value_name", "the name of the input which will take the expected value.", cxxopts::value<std::string>())
      ("predicted_value_name", "the name of the graph output that holds the predicted value.", cxxopts::value<std::string>());

  std::string model_base_path;
  try {
    auto flags = options.parse(argc, argv);
    model_base_path = flags["model_name"].as<std::string>();
    params.loss_output_name = flags["loss_output_name"].as<std::string>();
    params.predicted_value_name = flags["predicted_value_name"].as<std::string>();
    params.expected_value_name = flags["expected_value_name"].as<std::string>();
    params.optimizer_type = flags["optimizer_type"].as<std::string>();
    params.loss_function = flags["loss_function"].as<std::string>();
    params.learning_rate_input_name = flags["learning_rate_input_name"].as<std::string>();

    params.starting_onnx_model_path = model_base_path + ".onnx";
    params.training_onnx_model_path = model_base_path + "_bw.onnx";
    params.modified_training_onnx_model_path = model_base_path + "_bw_modified.onnx";
    params.training_optimized_ort_model_path = model_base_path + "_bw_modified_optimized.ort";
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return Status(ONNXRUNTIME, FAIL, msg);
  }

  return Status::OK();
}

onnxruntime::common::Status CreateTrainingGraph(const ModelPrepParameters& params, const Environment& env) {
  onnxruntime::training::TrainingSession::TrainingConfiguration trainer_config;
  TrainingSession::TrainingConfiguration::LossFunctionConfiguration lf{};
  lf.loss_function_info = LossFunctionInfo(OpDef(params.loss_function, kMSDomain, 1),
                                           params.loss_output_name,
                                           {params.predicted_value_name, params.expected_value_name});
  trainer_config.loss_function_config = lf;

  TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
  opt.name = params.optimizer_type;
  opt.learning_rate_input_name = params.learning_rate_input_name;
  opt.weight_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, float>(); };
  opt.weight_int_attributes_generator = [](const std::string&) { return std::unordered_map<std::string, int64_t>(); };
  opt.use_mixed_precision_moments = false;
  opt.do_all_reduce_in_mixed_precision_type = false;
  opt.use_nccl = false;
  opt.enable_grad_norm_clip = true;
  trainer_config.optimizer_config = opt;

  TrainingSession::TrainingConfigurationResult config_result{};
  auto session = std::make_unique<TrainingSession>(TRAINING_SESSION_OPTION, env);
  auto status = session->Load(params.starting_onnx_model_path);
  if (!status.IsOK()) {
    std::cerr << "Failed to load model in a training session: " << status.ErrorMessage() << "\n";
    return status;
  }

  status = session->ConfigureForTraining(trainer_config, config_result);
  if (!status.IsOK()) {
    std::cerr << "Failed to configure training session: " << status.ErrorMessage() << "\n";
    return status;
  }

  status = session->Save(ToPathString(params.training_onnx_model_path), TrainingSession::SaveOption::NO_RELOAD);
  if (!status.IsOK()) {
    std::cerr << "Failed to save training graph: " << status.ErrorMessage() << "\n";
    return status;
  }

  std::cout << "Training graph written to " << params.training_onnx_model_path << "\n";

  return status;
}

onnxruntime::common::Status PrepareTrainingGraphForInferenceSession(const ModelPrepParameters& params) {
  std::vector<std::string> initializers_to_remove = {};
  std::vector<std::string> optimizer_state = {};
  std::map<std::string, std::vector<float>> trainable_weights = {};
  std::vector<const NodeArg*> new_inputs = {};
  std::vector<const NodeArg*> new_outputs = {};
  std::shared_ptr<onnxruntime::Model> new_model;

  auto status = onnxruntime::Model::Load(ToPathString(params.training_onnx_model_path), new_model, nullptr, logging::LoggingManager::DefaultLogger());
  if (!status.IsOK()) {
    std::cerr << "Failed to load training graph: " << status.ErrorMessage() << "\n";
    return status;
  }

  auto graph_outputs = new_model->MainGraph().GetOutputs();
  auto graph_inputs = new_model->MainGraph().GetInputsIncludingInitializers();
  for (
      auto current_node = new_model->MainGraph().Nodes().begin();
      current_node != new_model->MainGraph().Nodes().end();
      current_node++) {
    if (current_node->OpType() == params.optimizer_type) {
      auto optimizer_input_defs = current_node->InputDefs();
      for (auto optimizer_input = optimizer_input_defs.begin(); optimizer_input != optimizer_input_defs.end(); optimizer_input++) {
        std::string optimizer_input_name = (*optimizer_input)->Name();
        if (new_model->MainGraph().IsInitializedTensor(optimizer_input_name)) {
          initializers_to_remove.push_back(optimizer_input_name);
          // if the initiaizer is also an input then consider it a trainable weight, otherwise it's optimizer state
          bool is_trainable_weight = false;

          for (auto input = graph_inputs.begin(); input != graph_inputs.end(); input++) {
            if ((*input)->Name() == optimizer_input_name) {
              is_trainable_weight = true;
            }
          }
          if (is_trainable_weight) {
            std::vector<float> weight = {};
            const TensorProto* proto;
            new_model->MainGraph().GetInitializedTensor(optimizer_input_name, proto);
            trainable_weights.insert({optimizer_input_name, weight});

          } else {
            optimizer_state.push_back(optimizer_input_name);
            new_inputs.push_back(*optimizer_input);
          }
        }
      }
      auto optimizer_output_defs = current_node->OutputDefs();
      for (auto optimizer_output = optimizer_output_defs.begin(); optimizer_output != optimizer_output_defs.end(); optimizer_output++) {
        if (!(*optimizer_output)->Name().empty()) {
          new_outputs.push_back(*optimizer_output);
        }
      }
    }
  }
  for (auto new_input = new_inputs.begin(); new_input != new_inputs.end(); new_input++) {
    graph_inputs.push_back(*new_input);
  }
  new_model->MainGraph().SetInputs(graph_inputs);
  for (auto new_output = new_outputs.begin(); new_output != new_outputs.end(); new_output++) {
    graph_outputs.push_back(*new_output);
  }
  new_model->MainGraph().SetOutputs(graph_outputs);
  for (auto initializer = initializers_to_remove.begin(); initializer < initializers_to_remove.end(); initializer++) {
    new_model->MainGraph().RemoveInitializedTensor(*initializer);
  }

  status = onnxruntime::Model::Save(*new_model, params.modified_training_onnx_model_path);
  if (!status.IsOK()) {
    std::cerr << "Failed to save training graph: " << status.ErrorMessage() << "\n";
    return status;
  }

  std::cout << "Modified training graph written to " << params.modified_training_onnx_model_path << "\n";

  return status;
}

onnxruntime::common::Status SaveOptimizedOrtModel(const std::string& source_model_path, const std::string& destination_model_path, const Environment& env) {
  INFERENCE_SESSION_OPTION.optimized_model_filepath = ToPathString(destination_model_path);
  auto inference_session = std::make_unique<InferenceSession>(INFERENCE_SESSION_OPTION, env, source_model_path);
  auto status = inference_session->Load();
  if (!status.IsOK()) {
    std::cerr << "Failed to load model in an inference session: " << status.ErrorMessage() << "\n";
    return status;
  }
  
  status = inference_session->Initialize();
  if (!status.IsOK()) {
    std::cerr << "Failed to initialize inference session: " << status.ErrorMessage() << "\n";
    return status;
  }

  std::cout << "Optimized training graph written to " << destination_model_path << "\n";

  return status;
}


int main(int argc, char* args[]) {
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kERROR,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env).IsOK());

  ModelPrepParameters params = {};
  auto status = ConfigModelPrepParameters(argc, args, params);
  if (!status.IsOK()) {
    return 1;
  }

  status = CreateTrainingGraph(params, *env);
  if (!status.IsOK()) {
    return 1;
  }

  status = PrepareTrainingGraphForInferenceSession(params);
  if (!status.IsOK()) {
    return 1;
  }

  status = SaveOptimizedOrtModel(params.modified_training_onnx_model_path, params.training_optimized_ort_model_path, *env);
  if (!status.IsOK()) {
    return 1;
  }
}
