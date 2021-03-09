// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "core/graph/model.h"

#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/utils.h"


#include <algorithm>
#include <condition_variable>
#include<fstream>
#include<iostream>
#include <mutex>
#include <tuple>

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
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
    false,                              //enable_cpu_mem_arena
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
  std::string required_ops_path;
  std::string weight_file_path;

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
    params.required_ops_path = model_base_path + "_required_ops.txt";
    params.weight_file_path = model_base_path + "_weights.dat";
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << std::endl
         << options.help() << std::endl;
    return Status(ONNXRUNTIME, FAIL, msg);
  }

  return Status::OK();
}

onnxruntime::common::Status CreateTrainingGraph(const ModelPrepParameters& params, const Environment& env) {

  std::cout << std::endl;
  std::cout << "Creating training graph based on inference graph" << std::endl;

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
    std::cerr << "Failed to load model in a training session: " << status.ErrorMessage() << std::endl;
    return status;
  }

  status = session->ConfigureForTraining(trainer_config, config_result);
  if (!status.IsOK()) {
    std::cerr << "Failed to configure training session: " << status.ErrorMessage() << std::endl;
    return status;
  }

  status = session->Save(ToPathString(params.training_onnx_model_path), TrainingSession::SaveOption::NO_RELOAD);
  if (!status.IsOK()) {
    std::cerr << "Failed to save training graph: " << status.ErrorMessage() << std::endl;
    return status;
  }

  std::cout << "Training graph written to " << params.training_onnx_model_path << std::endl;

  return status;
}

void WriteWeightsToFile(std::map<std::string, std::vector<float>> weights, const std::string& output_path) {
  ofstream f(output_path, ios::out | ios::binary);
  // construct metadata
  std::string key_list;
  std::string length_list;
  for (auto weight_iter = weights.begin(); weight_iter != weights.end(); weight_iter++) {
    if (weight_iter != weights.begin()) {
      key_list += ",";
      length_list += ",";
    }
    key_list += "\"" + weight_iter->first + "\"";
    length_list += std::to_string(weight_iter->second.size());
  }
  std::string metadata = "{";
  metadata += "\"version\":1.0,";
  metadata += "\"key_list\":[" + key_list + "],";
  metadata += "\"length_list\":[" + length_list + "]}";
//   bool needs_swap = endian::native == endian::little;
  uint32_t metadata_length = (uint32_t)(metadata.size() * sizeof(metadata[0]));
//   if (needs_swap) {
//     metadata_length = ntohl(metadata_length);
//   }
  f.write((char *) &metadata_length, sizeof(uint32_t)); 
  f.write(metadata.data(), metadata.size());
  for (const auto& weight : weights) {
      for (const auto& value : weight.second) {
          uint32_t u_value = 0;
          memcpy(&u_value, &value, sizeof(float));
        //   if (needs_swap) {
        //     u_value = ntohl(u_value);
        //   }
          f.write((char*) &value, sizeof(float));
      }
  }
  /*for (auto weight_iter = weights.begin(); weight_iter != weights.end(); weight_iter++) {
      for (auto value_iter = weight_iter->second.begin(); value_iter != weight_iter->second.end(); value_iter++) {
      }
  }*/
  f.close();
}

onnxruntime::common::Status ExtractTrainableWeights(const ModelPrepParameters& params) {

  std::cout << "Extracting trainable weights" << std::endl;

  std::map<std::string, std::vector<float>> weights_to_export = {};
  std::shared_ptr<onnxruntime::Model> model;
  auto status = onnxruntime::Model::Load(ToPathString(params.starting_onnx_model_path), model, nullptr, logging::LoggingManager::DefaultLogger());
  if (!status.IsOK()) {
    std::cerr << "Failed to load model: " << status.ErrorMessage() << std::endl;
    return status;
  }
  auto input_list = model->MainGraph().GetAllInitializedTensors();
  for (auto it = input_list.begin(); it != input_list.end(); it++) {
      
      size_t raw_data_size = it->second->raw_data().length();
      size_t expect_element_count = raw_data_size / sizeof(float); 
      std::vector<float> weight(expect_element_count, 0.0);
      auto unpack_status = onnxruntime::utils::UnpackTensor(
          *(it->second),
          it->second->raw_data().data(),
          raw_data_size,
          weight.data(),
          expect_element_count);
      if (unpack_status.IsOK())
      {
          weights_to_export.insert({it->first, weight});
      }
  }
  WriteWeightsToFile(weights_to_export, params.weight_file_path);

  std::cout << "Trainable weights written to " << params.weight_file_path << std::endl;

  return Status::OK();
}

void WriteRequiredOpsToFile(const std::map<std::string, std::set<std::string>>& required_ops, const std::string& output_path) {
   ofstream f(output_path, ios::out);
   if(!f) {
       std::cerr << "Failed to open file: " << output_path << std::endl;
       return;
   }
   for (auto namespace_iter = required_ops.begin(); namespace_iter != required_ops.end(); namespace_iter++) {
       f << namespace_iter->first;
       bool first_op = true;
       for (auto op_iter = namespace_iter->second.begin(); op_iter != namespace_iter->second.end(); op_iter++) {
           if (first_op) {
               f << ";";
               first_op = false;
           }
           else {
               f << ",";
           }
           f << (*op_iter);
       }
       f << std::endl;
   }
   f.close();
}

onnxruntime::common::Status PrepareTrainingGraphForInferenceSession(const ModelPrepParameters& params) {

  std::cout << std::endl;
  std::cout << "Modifying training graph to run in inference session" << std::endl;

  std::vector<std::string> initializers_to_remove = {};
  std::map<std::string, std::set<std::string>> required_ops = {};
  std::vector<const NodeArg*> new_inputs = {};
  std::vector<const NodeArg*> new_outputs = {};
  std::shared_ptr<onnxruntime::Model> new_model;

  auto status = onnxruntime::Model::Load(ToPathString(params.training_onnx_model_path), new_model, nullptr, logging::LoggingManager::DefaultLogger());
  if (!status.IsOK()) {
    std::cerr << "Failed to load training graph: " << status.ErrorMessage() << std::endl;
    return status;
  }

  auto graph_outputs = new_model->MainGraph().GetOutputs();
  auto graph_inputs = new_model->MainGraph().GetInputsIncludingInitializers();
  for (
      auto current_node = new_model->MainGraph().Nodes().begin();
      current_node != new_model->MainGraph().Nodes().end();
      current_node++) {
    int version = current_node->SinceVersion();
    std::string domain = current_node->Domain().empty() ? "ai.onnx" : current_node->Domain();
    std::string opset_key = domain + ";" + std::to_string(version);
    if (required_ops.find(opset_key) == required_ops.end()) {
        required_ops.insert({opset_key, {}});
    }
    required_ops[opset_key].insert(current_node->OpType());

    if (current_node->OpType() == params.optimizer_type) {
      auto optimizer_input_defs = current_node->InputDefs();
      for (auto optimizer_input = optimizer_input_defs.begin(); optimizer_input != optimizer_input_defs.end(); optimizer_input++) {
        std::string optimizer_input_name = (*optimizer_input)->Name();
        if (new_model->MainGraph().IsInitializedTensor(optimizer_input_name)) {
          initializers_to_remove.push_back(optimizer_input_name);
          if (params.optimizer_type == "AdamOptimizer" && optimizer_input_name.rfind("Update_Count_") == 0)
          {
              auto& batch_number_input = new_model->MainGraph().GetOrCreateNodeArg("batch_number", (*optimizer_input)->TypeAsProto());
              current_node->ReplaceDefs({{*optimizer_input,&batch_number_input}});
          }
          else {
            bool input_exists = std::any_of(graph_inputs.begin(), graph_inputs.end(), [optimizer_input_name](const NodeArg* input){return input->Name() == optimizer_input_name;});
            if (!input_exists) {
              new_inputs.push_back(*optimizer_input);
            }
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

  // Save required ops to a file
  WriteRequiredOpsToFile(required_ops, params.required_ops_path);

  if (params.optimizer_type == "AdamOptimizer") {
      auto& batch_number_input = new_model->MainGraph().GetOrCreateNodeArg("batch_number", nullptr);
      graph_inputs.push_back(&batch_number_input);
  }

  // Add new inputs
  for (auto new_input = new_inputs.begin(); new_input != new_inputs.end(); new_input++) {
    graph_inputs.push_back(*new_input);
  }
  new_model->MainGraph().SetInputs(graph_inputs);

  // Ad new outputs
  for (auto new_output = new_outputs.begin(); new_output != new_outputs.end(); new_output++) {
    graph_outputs.push_back(*new_output);
  }
  new_model->MainGraph().SetOutputs(graph_outputs);

  // Remove initializers
  for (auto initializer = initializers_to_remove.begin(); initializer < initializers_to_remove.end(); initializer++) {
    new_model->MainGraph().RemoveInitializedTensor(*initializer);
  }

  status = onnxruntime::Model::Save(*new_model, params.modified_training_onnx_model_path);
  if (!status.IsOK()) {
    std::cerr << "Failed to save training graph: " << status.ErrorMessage() << std::endl;
    return status;
  }

  std::cout << "Modified training graph written to " << params.modified_training_onnx_model_path << std::endl;

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

  status = ExtractTrainableWeights(params);
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
}