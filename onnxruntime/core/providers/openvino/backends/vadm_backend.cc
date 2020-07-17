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

#include "../contexts.h"
#include "../backend_utils.h"
#include "vadm_backend.h"
#include <vpu/hddl_plugin_config.hpp>

namespace onnxruntime {
namespace openvino_ep {

using namespace backend_utils;

VADMBackend::VADMBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                         GlobalContext& global_context,
                         const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {
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
  num_inf_reqs_ = 8;

  ie_cnn_network_ = CreateCNNNetwork(model_proto, subgraph_context_.device_id, subgraph_context_.precision);

  SetIODefs(model_proto, ie_cnn_network_);
  std::map<std::string, std::string> config;

  int i = 0;
  // Loading model to the plugin
  //If graph is fully supported and batching is enabled, load the network onto all VPU's and infer
  std::vector<InferenceEngine::ExecutableNetwork> exe_networks;
  if(global_context_.is_wholly_supported_graph && subgraph_context_.enable_batching){
    for(int j = 0; j < 8; j++){
      InferenceEngine::ExecutableNetwork exe_network;
      config[VPU_HDDL_CONFIG_KEY(DEVICE_TAG)] = global_context_.deviceTags[j];
      try {
        exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, "HDDL", config);
      } catch (InferenceEngine::details::InferenceEngineException e) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + e.what());
      } catch (...) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
      }
      exe_networks.push_back(exe_network);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    for(size_t j = 0; j < num_inf_reqs_; j++) {
      InferenceEngine::InferRequest::Ptr infRequest;
      try {
        infRequest = exe_networks[j].CreateInferRequestPtr();
      } catch(InferenceEngine::details::InferenceEngineException e) {
        ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
      } catch (...) {
        ORT_THROW(log_tag + "Exception while creating InferRequest object.");
      }
      infer_requests_.push_back(infRequest);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Infer Requests created: " << num_inf_reqs_ << std::endl;
  }
  //If the graph is not fully supported, need to schedule each subgraph on different VPU
  //If batching is disabled just schedule on the first VPU
  else {
    i = GetFirstAvailableDevice(global_context);
    LOGS_DEFAULT(INFO) << log_tag << "Device Tag is: " << i;
    config[VPU_HDDL_CONFIG_KEY(DEVICE_TAG)] = global_context_.deviceTags[i];
    InferenceEngine::ExecutableNetwork exe_network;
    try {
      exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, "HDDL", config);
    } catch (InferenceEngine::details::InferenceEngineException e) {
      ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    InferenceEngine::InferRequest::Ptr infRequest;
    try{
      infRequest = exe_network.CreateInferRequestPtr();
    } catch(InferenceEngine::details::InferenceEngineException e) {
      ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
    } catch (...) {
      ORT_THROW(log_tag + "Exception while creating InferRequest object.");
    }
    infer_requests_.push_back(infRequest);
    LOGS_DEFAULT(INFO) << log_tag << "Infer Requests created: 1" << std::endl;
  }
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void VADMBackend::StartAsyncInference(Ort::CustomOpApi& ort, std::vector<const OrtValue*> input_tensors,
                                      size_t batch_slice_idx, size_t infer_req_idx,
                                      std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests,
                                      std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  auto infer_request = infer_requests[infer_req_idx];
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

    auto graph_input_buffer =
        graph_input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t input_data_size = graph_input_blob->byteSize();
    const char* tensor_data = ort.GetTensorData<char>(input_tensors[i]);
    const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;

    // Copy input data into OpenVINO's input buffer
    std::memcpy(graph_input_buffer, batch_memory_offset, input_data_size);
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
void VADMBackend::CompleteAsyncInference(Ort::CustomOpApi& ort, std::vector<OrtValue*> output_tensors,
                                         size_t batch_slice_idx,
                                         size_t infer_req_idx, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests,
                                         std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  auto infer_request = infer_requests[infer_req_idx];

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
    auto graph_output_buffer =
        graph_output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t output_data_size = graph_output_blob->byteSize();
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensors[i]);
    char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;

    // Copy output results back to ONNX-RT's output buffers
    std::memcpy(batch_memory_offset, graph_output_buffer, output_data_size);
  }
}
size_t DeduceBatchSize(Ort::CustomOpApi ort, const OrtValue* input_tensor,
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

void VADMBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time
  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  std::lock_guard<std::mutex> lock(compute_lock_);

  // Get Input and Output tensors
  auto input_tensors = GetInputTensors(ort, context, ie_cnn_network_, subgraph_context_.input_indexes);

  size_t batch_size = 1;

  if (subgraph_context_.enable_batching) {
    // Calculate the batch_size from the input tensor shape.
    batch_size = DeduceBatchSize(ort, input_tensors[0],
                                 ie_cnn_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());
  }

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  // All infer_requests process identical tensor slices from the batch.
  // So using info from first infer_request to allocate all output tensors.
  auto output_tensors = GetOutputTensors(ort, context, batch_size, infer_requests_[0], ie_cnn_network_, subgraph_context_.output_names);

  // Distribute the batched inputs among available Infer Requests
  // for parallel inference.

  // Run parallel inferences as sets of num_inf_reqs_
  for (size_t set = 0; set < full_parallel_runs; set++) {
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      StartAsyncInference(ort, input_tensors, batch_slice_idx, inf_req_idx, infer_requests_, ie_cnn_network_);
    }
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      CompleteAsyncInference(ort, output_tensors, batch_slice_idx, inf_req_idx, infer_requests_, ie_cnn_network_);
    }
  }

  // Run parallel inferences for remaining batch slices
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    StartAsyncInference(ort, input_tensors, batch_slice_idx, inf_req_idx, infer_requests_, ie_cnn_network_);
  }
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    CompleteAsyncInference(ort, output_tensors, batch_slice_idx, inf_req_idx, infer_requests_, ie_cnn_network_);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace openvino_ep
}  // namespace onnxruntime