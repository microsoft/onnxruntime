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
#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "basic_backend.h"

namespace onnxruntime {
namespace openvino_ep {

using namespace backend_utils;


struct static_cast_int64
{
  template <typename T1> // T1 models type statically convertible to T
  int64_t operator()(const T1& x) const { return static_cast<int64_t>(x); }
};

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {

  ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
  SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_);
  InferenceEngine::ExecutableNetwork exe_network;

#if defined(OPENVINO_2020_4)
  if(const_outputs_map_.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;
#endif

  // Loading model to the plugin
  if(subgraph_context_.is_constant)
    return;
  std::map<std::string, std::string> config;
  if(global_context_.device_type == "MYRIAD"){

    if(subgraph_context_.set_vpu_config) {
      config["VPU_DETECT_NETWORK_BATCH"] = CONFIG_VALUE(NO);
    }

    if(global_context_.enable_vpu_fast_compile) {
      config["VPU_HW_INJECT_STAGES"] = CONFIG_VALUE(NO);
      config["VPU_COPY_OPTIMIZATION"] = CONFIG_VALUE(NO);
    }
  }
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  try {
    exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, hw_target, config);
  } catch (InferenceEngine::details::InferenceEngineException e) {
    ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + ": " +  e.what());
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
                                       OrtKernelContext* context,
                                       InferenceEngine::InferRequest::Ptr infer_request,
                                       std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  auto graph_input_info = ie_cnn_network->getInputsInfo();

  size_t i = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {
    // Get OpenVINO's input buffer
    InferenceEngine::Blob::Ptr graph_input_blob;
    std::string input_name = input_info_iter->first;
    try {
      graph_input_blob = infer_request->GetBlob(input_name);

    } catch (InferenceEngine::details::InferenceEngineException e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
    }
    auto precision = input_info_iter->second->getPrecision();
    auto graph_input_buffer = graph_input_blob->buffer()
                                  .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    size_t input_data_size = graph_input_blob->byteSize();

    #if (defined OPENVINO_2020_2) || (defined OPENVINO_2020_3)
    const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context_.input_indexes[i]);
    #else
    const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context_.input_names.at(input_name));
    #endif

    auto tensor_shape = ort.GetTensorTypeAndShape(tensor);
    auto elem_type = ort.GetTensorElementType(tensor_shape);

    if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
        (precision == InferenceEngine::Precision::I32)) {

      const int64_t* tensor_data_64 = ort.GetTensorData<int64_t>(tensor);
      auto data_len = (input_data_size * 2) / sizeof(int64_t)  ;

      std::copy(tensor_data_64, tensor_data_64+data_len, (uint32_t*)graph_input_buffer);
    } else {

      // Copy input data into OpenVINO's input buffer
      const char* tensor_data = ort.GetTensorData<char>(tensor);
      std::memcpy(graph_input_buffer, tensor_data, input_data_size);

    }

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

    auto tensor_shape = ort.GetTensorTypeAndShape(output_tensors[i]);
    auto elem_type = ort.GetTensorElementType(tensor_shape);
    auto precision = output_info_iter->second->getPrecision();

   if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
       (precision == InferenceEngine::Precision::I32)) {

      int64_t* tensor_data = ort.GetTensorMutableData<int64_t>(output_tensors[i]);

      auto data_len = output_data_size/sizeof(int32_t);
      std::transform((int32_t*)graph_output_buffer,((int32_t*)graph_output_buffer) + data_len, tensor_data, static_cast_int64());

    } else {
      char* tensor_data = ort.GetTensorMutableData<char>(output_tensors[i]);
      std::memcpy(tensor_data, graph_output_buffer, output_data_size);

    }
  }
#if defined(OPENVINO_2020_4)
  if(!const_outputs_map_.empty()){
    size_t j = i;
    for(auto item : const_outputs_map_){

      auto node = item.second;
      FillOutputsWithConstantData(ort,node,output_tensors[j]);
      j++;
    }
  }
#endif
}

void BasicBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  std::lock_guard<std::mutex> lock(compute_lock_);

  size_t batch_size = 1;
  auto output_tensors = GetOutputTensors(ort, context, batch_size, infer_request_, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_);
  if(subgraph_context_.is_constant){
#if defined(OPENVINO_2020_4)
    size_t i = 0;
    for(auto item : const_outputs_map_){
      auto node = item.second;
      FillOutputsWithConstantData(ort,node, output_tensors[i]);
      i++;
    }
#endif
  }
  else{
    StartAsyncInference(ort, context, infer_request_, ie_cnn_network_);
    CompleteAsyncInference(ort, output_tensors, infer_request_, ie_cnn_network_);
  }
  // Get Output tensors
  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace openvino_ep
}  // namespace onnxruntime
