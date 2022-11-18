// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <mutex>
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/shared_library/provider_api.h"
#include "../contexts.h"
#include "../backend_utils.h"
#include "vadm_backend.h"
#include <vpu/hddl_config.hpp>

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

  #if defined(OV_API_20)
    ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
  #else
    ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
    SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
  #endif
  OVConfig config;
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    config["PERF_COUNT"] = CONFIG_VALUE(YES);
  }
#endif

  if (const_outputs_map_.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;

  int i = 0;
  if (subgraph_context_.is_constant)
    return;
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  // Loading model to the plugin
  //If graph is fully supported and batching is enabled, load the network onto all VPU's and infer
  std::vector<OVExeNetwork> exe_networks;
  if (global_context_.is_wholly_supported_graph && subgraph_context_.enable_batching) {
    for (int j = 0; j < 8; j++) {
      OVExeNetwork exe_network;
      config[InferenceEngine::HDDL_DEVICE_TAG] = global_context_.deviceTags[j];
      exe_network = global_context_.ie_core.LoadNetwork(ie_cnn_network_, hw_target, config, subgraph_context_.subgraph_name);
      exe_networks.push_back(exe_network);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    for (size_t j = 0; j < num_inf_reqs_; j++) {
      OVInferRequestPtr infRequest;
      infRequest = std::make_shared<OVInferRequest>(exe_networks[j].CreateInferRequest());
      infer_requests_.push_back(infRequest);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Infer Requests created: " << num_inf_reqs_ << std::endl;
  }
  //If the graph is not fully supported, need to schedule each subgraph on different VPU
  //If batching is disabled just schedule on the first VPU
  else {
    i = GetFirstAvailableDevice(global_context);
    LOGS_DEFAULT(INFO) << log_tag << "Device Tag is: " << i;
    config[InferenceEngine::HDDL_DEVICE_TAG] = global_context_.deviceTags[i];
    OVExeNetwork exe_network;
    exe_network = global_context_.ie_core.LoadNetwork(ie_cnn_network_, hw_target, config, subgraph_context_.subgraph_name);
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    OVInferRequestPtr infRequest;
    infRequest = std::make_shared<OVInferRequest>(exe_network.CreateInferRequest());
    infer_requests_.push_back(infRequest);
    LOGS_DEFAULT(INFO) << log_tag << "Infer Requests created: 1" << std::endl;
  }
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void VADMBackend::StartAsyncInference(Ort::KernelContext& context,
                                      size_t batch_slice_idx, size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];
  
  #if defined (OV_API_20)
  auto graph_input_info = ie_cnn_network_->inputs();
    int input_idx = 0;
    for (auto input_info_iter = graph_input_info.begin();
      input_info_iter != graph_input_info.end(); ++input_info_iter) {
      auto input_names = input_info_iter->get_names();
      std::string onnx_input_name;
      std::string input_name;
      // use names retrieved from original ONNX model to assign the right onnx input name for the graph
      for (auto it = subgraph_context_.input_names.begin(); it != subgraph_context_.input_names.end(); ++it) {
        if (it->second == input_idx) {
          onnx_input_name = it->first;
          break;
        }
      }
      // using the input name retrieved from ONNX original to match with the input names returned by OV tensors 
      if (input_names.find(onnx_input_name) != input_names.end()) {
          input_name = onnx_input_name;
      } else {
        ORT_THROW(log_tag + "Input names mismatch between OpenVINO and ONNX. " + onnx_input_name + " doesn't exist in the list of OpenVINO input tensor names");
      }
      OVTensorPtr graph_input_blob; 
      graph_input_blob = infer_request->GetTensor(input_name);
      FillInputBlob(graph_input_blob, batch_slice_idx, input_name, context, subgraph_context_);
      input_idx++;
    }
  #else 
    auto graph_input_info = ie_cnn_network_->getInputsInfo();
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter) {
    // Get OpenVINO's input buffer
    std::string input_name = input_info_iter->first;
    auto precision = input_info_iter->second->getPrecision();
    auto graph_input_blob = infer_request->GetTensor(input_name);
    FillInputBlob(graph_input_blob, batch_slice_idx, input_name, context, precision, subgraph_context_);
  }
  #endif 

  // Start Async inference
  infer_request->StartAsync();
  
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void VADMBackend::CompleteAsyncInference(Ort::KernelContext& context,
                                         size_t batch_slice_idx, size_t infer_req_idx,
                                         size_t batch_size) {
  auto infer_request = infer_requests_[infer_req_idx];

  // Wait for Async inference completion
  infer_request->WaitRequest();
  
  #if defined (OV_API_20)
  auto graph_output_info = ie_cnn_network_->outputs();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    OVTensorPtr graph_output_blob;
    auto output_names = output_info_iter->get_names();
    std::string onnx_output_name;
    std::string output_name;
    bool output_name_found = false;
    // using the output name retrieved from ONNX original to match with the output names returned by OV tensors
    for (auto it = subgraph_context_.output_names.begin(); it != subgraph_context_.output_names.end(); ++it) {
      onnx_output_name = it->first;
      if (output_names.find(onnx_output_name) != output_names.end()) {
        //Assigning the output_name
        output_name = it->first;
        output_name_found = true;
        break;
      }
    }
    if(!output_name_found) {
      ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX. [ONNX Output: ] " + onnx_output_name + " doesn't exist in the list of OpenVINO output tensor names");
    }
    graph_output_blob = infer_request->GetTensor(output_name);
    auto output_tensor = GetOutputTensor(context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    FillOutputBlob(graph_output_blob, output_tensor, batch_slice_idx);
  }
  #else
  auto graph_output_info = ie_cnn_network_->getOutputsInfo();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    // Get OpenVINO's output blob
    OVTensorPtr graph_output_blob;
    auto output_name = output_info_iter->first;
    graph_output_blob = infer_request->GetTensor(output_name);
    auto output_tensor = GetOutputTensor(context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto precision = output_info_iter->second->getPrecision();
    FillOutputBlob(graph_output_blob, output_tensor, precision, batch_slice_idx);
  }
  #endif 
  if (!const_outputs_map_.empty()) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(node, output_tensor);
    }
  }
}
size_t DeduceBatchSize(const Ort::ConstValue& input_tensor,
                       InferenceEngine::SizeVector graph_dims) {
  size_t batch_size = 1;

  // All the inputs and outputs are batched the same way.
  // So it is sufficient to use any one of these tensors to deduce the batch size.
  const auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  if ((input_shape.size() == graph_dims.size() && input_shape[0] > 1 && graph_dims[0] == 1) || (input_shape.size() == graph_dims.size() + 1)) {
    batch_size = input_shape[0];
  }

  LOGS_DEFAULT(INFO) << log_tag << "Deduced batch size: " << batch_size;

  return batch_size;
}

void VADMBackend::Infer(OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time
  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  Ort::KernelContext ctx(context);
  
  std::lock_guard<std::mutex> lock(compute_lock_);

  size_t batch_size = 1;

  if (subgraph_context_.enable_batching) {
    // Calculate the batch_size from the input tensor shape.
    auto tensor = ctx.GetInput(subgraph_context_.input_indexes[0]);
    #if defined (OV_API_20)
    batch_size = DeduceBatchSize(tensor, ie_cnn_network_->get_result()->get_shape());
    #else
    batch_size = DeduceBatchSize(tensor, ie_cnn_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());
    #endif                             
  }

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  if (subgraph_context_.is_constant) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ctx, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(node, output_tensor);
    }
  } else {
    // Distribute the batched inputs among available Infer Requests
    // for parallel inference.

    // Run parallel inferences as sets of num_inf_reqs_
    for (size_t set = 0; set < full_parallel_runs; set++) {
      for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
        size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
        StartAsyncInference(ctx, batch_slice_idx, inf_req_idx);
      }
      for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
        size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
        CompleteAsyncInference(ctx, batch_slice_idx, inf_req_idx, batch_size);
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
          printPerformanceCounts(infer_requests_[inf_req_idx], std::cout, hw_target);
        }
#endif
      }
    }

    // Run parallel inferences for remaining batch slices
    for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
      size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
      StartAsyncInference(ctx, batch_slice_idx, inf_req_idx);
    }
    for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
      size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
      CompleteAsyncInference(ctx, batch_slice_idx, inf_req_idx, batch_size);
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
        printPerformanceCounts(infer_requests_[inf_req_idx], std::cout, hw_target);
      }
#endif
    }
  }
  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace openvino_ep
}  // namespace onnxruntime
