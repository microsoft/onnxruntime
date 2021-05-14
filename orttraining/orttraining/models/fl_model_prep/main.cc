// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h "
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

#include "onnx/defs/attr_proto_util.h"

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


struct BinaryCrossEntropy : public ILossFunction {
  GraphAugmenter::GraphDefs operator()(const Graph& graph, const LossFunctionInfo& loss_func_info) override {
    const std::string& loss_name = loss_func_info.loss_name;
    const VectorString& args = loss_func_info.loss_builder_args;
    ORT_ENFORCE(args.size() == 2, "Expected 2 arguments.");
    const std::string& prediction_name = args[0];
    const std::string& label_name = args[1];

    GraphAugmenter::GraphDefs graph_defs;

    graph_defs.AddGraphInputs({label_name});
    graph_defs.AddGraphOutputs({loss_name});

    std::vector<NodeDef> new_nodes;

    const NodeArg* prediction_arg = graph.GetNodeArg(prediction_name);
    ORT_ENFORCE(prediction_arg != nullptr,
                "Prediction arg ", prediction_name, " is not found in the graph.");
    TypeProto* label_type_proto = graph_defs.CopyTypeProto(prediction_arg);

    // p(x) * log(q(x))
    {
      new_nodes.emplace_back(NodeDef("Log",  // Op
                                     {
                                         ArgDef(prediction_name)},
                                     {
                                         ArgDef("BCE_Log_Pred"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Log_Pred"  // name
                                     ));
      new_nodes.emplace_back(NodeDef("Mul",  // Op
                                     {
                                         ArgDef(label_name, label_type_proto),
                                         ArgDef("BCE_Log_Pred")},
                                     {
                                         ArgDef("BCE_Mul_Label_Log_Pred"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Mul_Label_Log_Pred"  // name
                                     ));
    }

    // (1-p(x)) * log(1-q(x))
    {
      onnx::TensorProto tensor_proto;
      tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      tensor_proto.add_float_data(1.f);
      tensor_proto.set_name("one");
      graph_defs.AddInitializers({tensor_proto});

      new_nodes.emplace_back(NodeDef("Sub",  // Op
                                     {
                                         ArgDef("one"),
                                         ArgDef(prediction_name)},
                                     {
                                         ArgDef("BCE_InversePred")  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_InversePred"  // name
                                     ));
      new_nodes.emplace_back(NodeDef("Sub",  // Op
                                     {
                                         ArgDef("one"),
                                         ArgDef(label_name, label_type_proto)},
                                     {
                                         ArgDef("BCE_InverseLabel"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_InverseLabel"  // name
                                     ));
      new_nodes.emplace_back(NodeDef("Log",  // Op
                                     {
                                         ArgDef("BCE_InversePred")},
                                     {
                                         ArgDef("BCE_Log_InversePred"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Log_InversePred"  // name
                                     ));
      new_nodes.emplace_back(NodeDef("Mul",  // Op
                                     {
                                         ArgDef("BCE_InverseLabel"),
                                         ArgDef("BCE_Log_InversePred")},
                                     {
                                         ArgDef("BCE_Mul_InverseLabel_Log_InversePred"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Mul_InverseLabel_Log_InversePred"  // name
                                     ));
    }

    // add
    {
      new_nodes.emplace_back(NodeDef("Add",  // Op
                                     {
                                         ArgDef("BCE_Mul_Label_Log_Pred"),
                                         ArgDef("BCE_Mul_InverseLabel_Log_InversePred")},
                                     {
                                         ArgDef("BCE_Sum"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Sum"  // name
                                     ));
      new_nodes.emplace_back(NodeDef("Neg",  // Op
                                     {
                                         ArgDef("BCE_Sum")},
                                     {
                                         ArgDef("BCE_Sum_Neg"),  // Outputs
                                     },
                                     NodeAttributes(),
                                     "BCE_Sum_Neg"  // name
                                     ));
    }

    // ReduceMean
    {
      new_nodes.emplace_back(NodeDef("ReduceMean",  // Op
                                     {
                                         ArgDef("BCE_Sum_Neg"),  // Inputs
                                     },
                                     {
                                         ArgDef(loss_name),  // Outputs
                                     },
                                     NodeAttributes(),
                                     //{ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(0))},
                                     "BCE_reduce_mean"  // name
                                     ));
    }

    graph_defs.AddNodeDefs(new_nodes);

    return graph_defs;
  };
};

class TracingCPUAllocator : public IAllocator {
 public:
  //explicit TracingCPUAllocator (const OrtMemoryInfo& memory_info);

  TracingCPUAllocator();

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  size_t OutstandingAllocation();
  size_t PeakAllocation();

 private:
  std::map<size_t, size_t> allocation_history;
  size_t outstanding_allocation;
  size_t peak_allocation;
};

TracingCPUAllocator::TracingCPUAllocator()
    : IAllocator(OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)), allocation_history(), outstanding_allocation(0), peak_allocation(0) {}

void* TracingCPUAllocator::Alloc(size_t size) {
  void* p = utils::DefaultAlloc(size);
  allocation_history.insert({reinterpret_cast<size_t>(p), size});
  outstanding_allocation += size;
  peak_allocation = outstanding_allocation > peak_allocation ? outstanding_allocation : peak_allocation;
  return p;
}

void TracingCPUAllocator::Free(void* p) {
  size_t allocated_size = allocation_history.erase(reinterpret_cast<size_t>(p));
  outstanding_allocation -= allocated_size;
  utils::DefaultFree(p);
}

size_t TracingCPUAllocator::OutstandingAllocation() {
  return outstanding_allocation;
}

size_t TracingCPUAllocator::PeakAllocation() {
  return peak_allocation;
}

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
  bool needs_swap = endian::native == endian::little;
  uint32_t metadata_length = (uint32_t)(metadata.size() * sizeof(metadata[0]));
  if (needs_swap) {
    metadata_length = ntohl(metadata_length);
  }
  f.write((char *) &metadata_length, sizeof(uint32_t)); 
  f.write(metadata.data(), metadata.size());
  for (const auto& weight : weights) {
      for (const auto& value : weight.second) {
          uint32_t u_value = 0;
          memcpy(&u_value, &value, sizeof(float));
          if (needs_swap) {
            u_value = ntohl(u_value);
          }
          f.write((char*) &value, sizeof(float));
      }
  }
  /*for (auto weight_iter = weights.begin(); weight_iter != weights.end(); weight_iter++) {
      for (auto value_iter = weight_iter->second.begin(); value_iter != weight_iter->second.end(); value_iter++) {
      }
  }*/
  f.close();
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

template <typename T>
static void CreateOrtValue(const std::vector<int64_t>& dims,
                             const std::vector<T>& value,
                             OrtValue* p_mlvalue,
                             AllocatorPtr alloc) {
  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, alloc);

  if (value.size() > 0) {
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

onnxruntime::common::Status SaveOptimizedOrtModel(const std::string& source_model_path, const std::string& destination_model_path, const Environment& env, std::shared_ptr<TracingCPUAllocator>& alloc) {
  
  std::cout << std::endl;
  std::cout << "Preparing optimized ORT model" << std::endl;

  INFERENCE_SESSION_OPTION.optimized_model_filepath = ToPathString(destination_model_path);
  INFERENCE_SESSION_OPTION.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  auto inference_session = std::make_unique<InferenceSession>(INFERENCE_SESSION_OPTION, env, source_model_path);
  auto status = inference_session->Load();
  if (!status.IsOK()) {
    std::cerr << "Failed to load model in an inference session: " << status.ErrorMessage() << std::endl;
    return status;
  }
  
  status = inference_session->Initialize();
  if (!status.IsOK()) {
    std::cerr << "Failed to initialize inference session: " << status.ErrorMessage() << std::endl;
    return status;
  }

  std::cout << "Optimized training graph written to " << destination_model_path << std::endl;

  std::cout << std::endl;
  std::cout << "Executing training batch" << std::endl;
  
  onnxruntime::NameMLValMap inputs = {};
  std::vector<std::string> output_names = {}; 
  std::vector<OrtValue> output_values = {};
  auto model_inputs = inference_session->GetModelInputs();
  for (auto input_it = model_inputs.second->begin(); input_it != model_inputs.second->end(); input_it++) {
      auto shape = (*input_it)->Shape();
      onnx::DataType data_type = (*input_it)->Type();
      OrtValue v = {};
      std::vector<int64_t> dims = {};
      int64_t total_size = 1;
      for (int i = 0; i < shape->dim_size(); i++) {
          int64_t d = 1;
          if (shape->dim()[i].has_dim_value()) {
              d = shape->dim()[i].dim_value();
          }
          else if (shape->dim()[i].has_dim_param()) {
              auto param_name = shape->dim()[i].dim_param();
              if (param_name == "batch_size"){
                  d = 6;
              } else if (param_name == "query_length") {
                  d = 13;
              } else if (param_name == "doc_length") {
                  d = 13;
              }
          }
          //TODO: handle variable axes
          dims.push_back(d);
          total_size *= d;
      }
      if (data_type->compare("tensor(float)") == 0) { 
        auto value = std::vector<float>(total_size, 1.0);
        CreateOrtValue(dims, value, &v, alloc);
      } else if (data_type->compare("tensor(int64)") == 0) { 
        auto value = std::vector<int64_t>(total_size, 1);
        CreateOrtValue(dims, value, &v, alloc);
      }
      else {
          std::cerr << "Unsupported data type: " << (*data_type) << std::endl;
      }
      inputs.insert({(*input_it)->Name(), v});
  }
  auto model_outputs = inference_session->GetModelOutputs();
  std::transform(
      model_outputs.second->begin(),
      model_outputs.second->end(),
      std::back_inserter(output_names),
      [](const NodeArg* node){return node->Name();});
  std::cout << "Peak tensor memory allocated for input: " << alloc->PeakAllocation() << " bytes" << std::endl;
  
  status = inference_session->Run(inputs, output_names, &output_values);
  if (!status.IsOK()) {
    std::cerr << "Failed to run inference session: " << status.ErrorMessage() << std::endl;
    return status;
  }
  std::cout << "Peak tensor memory allocated for all training: " << alloc->PeakAllocation() << " bytes" << std::endl;

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

  LossFunctionRegistry::GetInstance().Register<BinaryCrossEntropy>("BinaryCrossEntropy");

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
  auto cpu_allocator = std::make_shared<TracingCPUAllocator>();
  ORT_ENFORCE(env->RegisterAllocator(cpu_allocator).IsOK());
  status = SaveOptimizedOrtModel(params.modified_training_onnx_model_path, params.training_optimized_ort_model_path, *env, cpu_allocator);
  if (!status.IsOK()) {
    return 1;
  }
}
