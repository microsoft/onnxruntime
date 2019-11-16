// u
// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>
#include <fstream>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/logging/logging.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include <transform/transformations/convert_fp32_to_fp16.hpp>
#include "intel_graph.h"
//#include "intel_custom_op.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cpp/ie_cnn_network.h>
#include <ngraph/function.hpp>
#include <ie_ir_reader.hpp>
#include <ngraph/serializer.hpp>

namespace onnxruntime {
namespace intel_ep {

//std::shared_ptr<InferenceEngine::CNNNetwork> cnetwork;

#define NGRAPH_EP_LRU_CACHE_DEFAULT_SIZE 500
const std::string IntelGraph::log_tag = "[Intel-EP] ";

InferenceEngine::CNNNetwork IntelGraph::cnetwork;

IntelGraph::IntelGraph(const onnxruntime::Node* fused_node) {
  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  std::string precision_str = "FP32";

#ifdef INTEL_CONFIG_CPU_FP32
  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  precision_str = "FP32";
#endif
#ifdef INTEL_CONFIG_GPU_FP32
  device_id_ = "GPU";
  precision_ = InferenceEngine::Precision::FP32;
  precision_str = "FP32";
#endif
#ifdef INTEL_CONFIG_GPU_FP16
  device_id_ = "GPU";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif
#ifdef INTEL_CONFIG_MYRIAD
  device_id_ = "MYRIAD";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif
#ifdef INTEL_CONFIG_VAD_M
  device_id_ = "HDDL";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif

  // Infer Request class represents OpenVINO's logical hardware instance. These logical
  // instances are bound to physical hardware instances at runtime depending
  // on the physical hardware availability. If multiple Infer Requests are mapped to
  // the same physical hardware instance, then the inference operations requests from
  // the Infer Requests are serialized before they are scheduled on the physical hardware.
  // If the different Infer Requests are scheduled on different hardware instances, inference
  // operations associated with the Infer Requests may be scheduled in parallel.
  // Infer Requests hold resources representing the entire network on their target hardware. So,
  // having more Infer Requests than needed would waste system resources.
  // In VAD-M (HDDL) accelerator, there are 8 parallel execution units. So, creating 8 instances
  // of Infer Requests only if the VAD-M accelerator is being used.
  // sets number of maximum parallel inferences
  num_inf_reqs_ = (device_id_ == "HDDL") ? 8 : 1;

  fused_node_ = fused_node;

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  std::map<std::string, int> inputdef_index_map;
  auto input_defs = fused_node_->InputDefs();
  int i = 0;
  for (auto idef : input_defs) {
    inputdef_index_map.insert({idef->Name(), i});
    i++;
  }

  auto inputs = fused_node_->GetFunctionBody()->Body().GetInputs();
  for (auto input : inputs) {
    auto it = inputdef_index_map.find(input->Name());
    if (it == inputdef_index_map.end()) {
      throw "Input not found in the input defs list";
    }

    int index = it->second;
    input_indexes_.push_back(index);
  }
}

std::vector<std::string> IntelGraph::GetEnvLdLibraryPath() const {
  std::string plugin_path = std::getenv("LD_LIBRARY_PATH");
  std::vector<std::string> paths;
  std::string token;
  std::istringstream tokenStream(plugin_path);
  char delimiter = ':';

  while (std::getline(tokenStream, token, delimiter)) {
    paths.push_back(token);
  }
  return paths;
}

InferenceEngine::Precision IntelGraph::ConvertPrecisionONNXToIntel(
    ONNX_NAMESPACE::DataType onnx_type) {
  if (*onnx_type == "float" || *onnx_type == "tensor(float)") {
    return InferenceEngine::Precision::FP32;
  } else if (*onnx_type == "float16" || *onnx_type == "tensor(float16)") {
    return InferenceEngine::Precision::FP16;
  } else if (*onnx_type == "int32" || *onnx_type == "tensor(int32)") {
    return InferenceEngine::Precision::I32;
  } else if (*onnx_type == "int16" || *onnx_type == "tensor(int16)") {
    return InferenceEngine::Precision::I16;
  } else if (*onnx_type == "int8" || *onnx_type == "tensor(int8)") {
    return InferenceEngine::Precision::I8;
  } else if (*onnx_type == "uint16" || *onnx_type == "tensor(uint16)") {
    return InferenceEngine::Precision::U16;
  } else if (*onnx_type == "uint8" || *onnx_type == "tensor(uint8)") {
    return InferenceEngine::Precision::U8;
  } else {
    throw "Unsupported Data type";
  }
}

void IntelGraph::GetExecutableHandle(
    std::shared_ptr<InferenceEngine::CNNNetwork> network) {
  LOGS_DEFAULT(INFO) << log_tag << "Loaded plugins";
  // Configure input & output
  // Prepare input blobs
  if (network)
    std::cout << "Network is not NULL" << std::endl;
  auto inputInfo = network->getInputsInfo();
  LOGS_DEFAULT(INFO) << log_tag << "Loaded plugins";
  auto onnx_input_defs = fused_node_->InputDefs();
  LOGS_DEFAULT(INFO) << log_tag << "Loaded plugins";

  int input_idx = 0;
  for (auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter, ++input_idx) {
    // Get the onnx index for the corresponding input (ignoring initializers)
    auto tracked_input_idx = input_indexes_[input_idx];
    auto precision = ConvertPrecisionONNXToIntel(onnx_input_defs[tracked_input_idx]->Type());
    iter->second->setPrecision(precision);

    // Choose the appropriate OpenVINO layout for input tensor
    // based on dims size
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for input data map for: " + iter->first;
    }
  }

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  auto onnx_output_defs = fused_node_->OutputDefs();

  int output_idx = 0;
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter, ++output_idx) {
    auto precision = ConvertPrecisionONNXToIntel(onnx_output_defs[output_idx]->Type());
    iter->second->setPrecision(precision);

    // Choose the appropriate OpenVINO layout for output tensor
    // based on dims size
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for output data map for: " + iter->first;
    }
  }
}

size_t IntelGraph::DeduceBatchSize(Ort::CustomOpApi ort, const OrtValue* input_tensor,
                                   InferenceEngine::SizeVector graph_dims) {
  size_t batch_size = 1;

  // All the inputs and outputs are batched the same way.
  // So it is sufficient to use any one of these tensors to deduce the batch size.
  const auto& input_shape = ort.GetTensorShape(ort.GetTensorTypeAndShape(input_tensor));

  if ((input_shape.size() == graph_dims.size() && input_shape[0] > 1 && graph_dims[0] == 1) || (input_shape.size() == graph_dims.size() + 1)) {
    batch_size = input_shape[0];
  }

  LOGS_DEFAULT(INFO) << log_tag << "Deduced batch size: " << batch_size;

  return batch_size;
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void IntelGraph::StartAsyncInference(Ort::CustomOpApi ort, const OrtValue* input_tensors[],
                                     size_t batch_slice_idx,
                                     size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];
  auto graph_input_info = intel_network_->getInputsInfo();

  size_t i = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {
    // Get OpenVINO's input buffer
    auto graph_input_blob = infer_request->GetBlob(input_info_iter->first);
    auto graph_input_buffer =
        graph_input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t input_data_size = graph_input_blob->byteSize();
    const char* tensor_data = ort.GetTensorData<char>(input_tensors[i]);
    const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;

    // Copy input data into OpenVINO's input buffer
    std::memcpy(graph_input_buffer, batch_memory_offset, input_data_size);
  }

  // Start Async inference
  infer_request->StartAsync();
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void IntelGraph::CompleteAsyncInference(Ort::CustomOpApi ort, OrtValue* output_tensors[],
                                        size_t batch_slice_idx,
                                        size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];

  // Wait for Async inference completion
  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  auto graph_output_info = intel_network_->getOutputsInfo();

  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    // Get OpenVINO's output blob
    auto graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    auto graph_output_buffer =
        graph_output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t output_data_size = graph_output_blob->byteSize();
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensors[i]);
    char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;

    // Copy output results back to ONNX-RT's output buffers
    std::memcpy(batch_memory_offset, graph_output_buffer, output_data_size);
  }
}

void IntelGraph::GetInputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, const OrtValue* input_tensors[]) {
  size_t input_count = intel_network_->getInputsInfo().size();

  for (size_t i = 0; i < input_count; i++) {
    input_tensors[i] = ort.KernelContext_GetInput(context, input_indexes_[i]);
  }
}

void IntelGraph::GetOutputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, OrtValue* output_tensors[], size_t batch_size) {
  auto graph_output_info = intel_network_->getOutputsInfo();

  // All infer_requests process identical tensor slices from the batch.
  // So using info from first infer_request to allocate all output tensors.
  auto infer_request = infer_requests_[0];

  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    auto graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();

    if (batch_size > 1) {
      // Add the batch size as dim 0.
      graph_output_dims.insert(graph_output_dims.begin(), batch_size);
    }

    size_t num_dims = graph_output_dims.size();
    int64_t output_shape[num_dims];
    for (size_t j = 0; j < num_dims; j++) {
      output_shape[j] = static_cast<int64_t>(graph_output_dims[j]);
    }

    output_tensors[i] = ort.KernelContext_GetOutput(context, i, output_shape, num_dims);
  }
}

void IntelGraph::CreateNGraphFunc(const ONNX_NAMESPACE::ModelProto& model_proto) {
  model_proto_ = model_proto;
  std::cout << "In CreateNgraphFunc" << std::endl;
  std::istringstream model_stream{model_proto_.SerializeAsString()};
  std::shared_ptr<ngraph::Function> ng_function;
  try {
    ng_function = ngraph::onnx_import::import_onnx_model(model_stream);
    LOGS_DEFAULT(INFO) << "ONNX Import Done";
  } catch (const std::exception& exp) {
    LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] "
                        << " - " << name_ << " - "
                        << "Exception while importing model to nGraph: " << std::string(exp.what());
    throw;
  } catch (...) {
    LOGS_DEFAULT(FATAL) << "[NGRAPHCustomOp] "
                        << " - " << name_ << " - "
                        << "Unknown exception while importing model to nGraph";
    throw;
  }

  //Serializing nGraph function
  std::string json_string = serialize(ng_function,4);
  std::ofstream out("serialize_function_before_PM.json");
  out << json_string;
    
  //Pass Manager for V1 transformations
  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<ngraph::pass::Opset1Upgrade>();
  pass_manager.run_passes(ng_function);

  if (precision_ == InferenceEngine::Precision::FP16)
  {
     std::cout << "FP16" << std::endl;
     //FP16 transformations
     ngraph::pass::ConvertFP32ToFP16().run_on_function(ng_function);
     ng_function->validate_nodes_and_infer_types();
  }

  //Serializing nGraph function
  std::string json_string_pm = serialize(ng_function,4);
  std::ofstream out_pm("serialize_function_after_PM.json");
  out_pm << json_string_pm;

  //IE wrapper for nGraph function
  InferenceEngine::CNNNetwork network(ng_function);

  //Serialize CNNNetwork
  //network.serialize("IR.xml", "IR.bin");
  
  intel_network_ = std::make_shared<InferenceEngine::CNNNetwork>(network);
}

void IntelGraph::Infer(const ONNX_NAMESPACE::ModelProto& model_proto, Ort::CustomOpApi ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  std::lock_guard<std::mutex> lock(compute_lock_);

  CreateNGraphFunc(model_proto);
  GetExecutableHandle(intel_network_);

  InferenceEngine::Core ie;
  //Loading model to the plugin
  InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(*intel_network_, device_id_);

  LOGS_DEFAULT(INFO) << log_tag << "Network loaded into accelerator plug-in succesfully";

  //Create infer request
  for (size_t i = 0; i < num_inf_reqs_; i++) {
    auto infRequest = exeNetwork.CreateInferRequestPtr();

    infer_requests_.push_back(infRequest);
  }
  LOGS_DEFAULT(INFO) << log_tag << "Infer requests created: " << num_inf_reqs_;

  // Get Input and Output tensors
  size_t input_count = intel_network_->getInputsInfo().size();
  size_t output_count = intel_network_->getOutputsInfo().size();
  const OrtValue* input_tensors[input_count];
  OrtValue* output_tensors[output_count];

  GetInputTensors(ort, context, input_tensors);

  // Calculate the batch_size from the input tensor shape.
  auto batch_size = DeduceBatchSize(ort, input_tensors[0],
                                    intel_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  GetOutputTensors(ort, context, output_tensors, batch_size);

  // Distribute the batched inputs among available Infer Requests
  // for parallel inference.

  // Run parallel inferences as sets of num_inf_reqs_
  for (size_t set = 0; set < full_parallel_runs; set++) {
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      StartAsyncInference(ort, input_tensors, batch_slice_idx, inf_req_idx);
    }
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      CompleteAsyncInference(ort, output_tensors, batch_slice_idx, inf_req_idx);
    }
  }

  // Run parallel inferences for remaining batch slices
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    StartAsyncInference(ort, input_tensors, batch_slice_idx, inf_req_idx);
  }
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    CompleteAsyncInference(ort, output_tensors, batch_slice_idx, inf_req_idx);
  }

  std::cout << "Inference successful" << std::endl;
  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace intel_ep
}  // namespace onnxruntime
