// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include <inference_engine.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"

#include "../backend_utils.h"
#include "basic_backend.h"

namespace onnxruntime {
namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {

  ie_cnn_network_ = CreateCNNNetwork(model_proto, subgraph_context_.device_id, subgraph_context_.precision);
  SetIODefs(model_proto, ie_cnn_network_);
  InferenceEngine::ExecutableNetwork exe_network;

  // Loading model to the plugin
  std::map<std::string, std::string> config;
  if(subgraph_context_.device_id == "MYRIAD" && subgraph_context_.set_vpu_config){
    config["VPU_DETECT_NETWORK_BATCH"] = CONFIG_VALUE(NO);
  }
  try {
    exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, subgraph_context_.device_id, config);
  } catch (InferenceEngine::details::InferenceEngineException e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
  }
  LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";

  // Create infer request
  try {
    infer_request_ = exe_network.CreateInferRequestPtr();
  } catch (InferenceEngine::details::InferenceEngineException e) {
    ORT_THROW(log_tag + " Exception while creating InferRequest object: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + "Exception while creating InferRequest object");
  }
  LOGS_DEFAULT(INFO) << log_tag << "Infer request created";
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void BasicBackend::StartAsyncInference(Ort::CustomOpApi& ort,
                                       std::vector<const OrtValue*> input_tensors,
                                       InferenceEngine::InferRequest::Ptr infer_request,
                                       std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  auto graph_input_info = ie_cnn_network->getInputsInfo();

  size_t i = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {
    // Get OpenVINO's input buffer
    InferenceEngine::Blob::Ptr graph_input_blob;
    try {
      graph_input_blob = infer_request->GetBlob(input_info_iter->first);
    } catch (InferenceEngine::details::InferenceEngineException e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_info_iter->first + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_info_iter->first);
    }

    auto graph_input_buffer = graph_input_blob->buffer()
                                  .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    size_t input_data_size = graph_input_blob->byteSize();
    const char* tensor_data = ort.GetTensorData<char>(input_tensors[i]);

    // Copy input data into OpenVINO's input buffer
    std::memcpy(graph_input_buffer, tensor_data, input_data_size);
  }

  // Start Async inference
  try {
    infer_request->StartAsync();
  } catch (InferenceEngine::details::InferenceEngineException e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Couldn't start Inference");
  }
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void BasicBackend::CompleteAsyncInference(Ort::CustomOpApi& ort,
                                          std::vector<OrtValue*> output_tensors,
                                          InferenceEngine::InferRequest::Ptr infer_request,
                                          std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  // Wait for Async inference completion
  try {
    infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  } catch (InferenceEngine::details::InferenceEngineException e) {
    ORT_THROW(log_tag + " Exception with completing Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception with completing Inference");
  }
  auto graph_output_info = ie_cnn_network->getOutputsInfo();

  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    // Get OpenVINO's output blob
    InferenceEngine::Blob::Ptr graph_output_blob;
    try {
      graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    } catch (InferenceEngine::details::InferenceEngineException e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_info_iter->first + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_info_iter->first);
    }
    auto graph_output_buffer = graph_output_blob->buffer()
                                   .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    size_t output_data_size = graph_output_blob->byteSize();
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensors[i]);

    // Copy output results back to ONNX-RT's output buffers
    std::memcpy(tensor_data, graph_output_buffer, output_data_size);
  }
}

void BasicBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  std::lock_guard<std::mutex> lock(compute_lock_);

  size_t batch_size = 1;
  // Get Input and Output tensors
  auto input_tensors = GetInputTensors(ort, context, ie_cnn_network_, subgraph_context_.input_indexes);
  auto output_tensors = GetOutputTensors(ort, context, batch_size, infer_request_, ie_cnn_network_, subgraph_context_.output_names);

  StartAsyncInference(ort, input_tensors, infer_request_, ie_cnn_network_);
  CompleteAsyncInference(ort, output_tensors, infer_request_, ie_cnn_network_);

  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace openvino_ep
}  // namespace onnxruntime