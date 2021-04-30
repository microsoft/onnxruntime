// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <mutex>

#include <inference_engine.hpp>

#ifdef OPENVINO_2021_4
using Exception = InferenceEngine::Exception;
using WaitMode = InferenceEngine::InferRequest::WaitMode;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
using WaitMode = InferenceEngine::IInferRequest::WaitMode;
#endif

#include "core/providers/shared_library/provider_api.h"

#include "../contexts.h"
#include "../backend_utils.h"
#include "vadm_backend.h"
#if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || \
    defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
#include <vpu/hddl_config.hpp>
#else
#include <vpu/hddl_plugin_config.hpp>
#endif

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

  ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);

  SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
  std::map<std::string, std::string> config;
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    config["PERF_COUNT"] = CONFIG_VALUE(YES);
  }
#endif

#if defined(OPENVINO_2020_4)
  if (const_outputs_map_.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;
#endif

  int i = 0;
  if (subgraph_context_.is_constant)
    return;
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  // Loading model to the plugin
  //If graph is fully supported and batching is enabled, load the network onto all VPU's and infer
  std::vector<InferenceEngine::ExecutableNetwork> exe_networks;
  if (global_context_.is_wholly_supported_graph && subgraph_context_.enable_batching) {
    for (int j = 0; j < 8; j++) {
      InferenceEngine::ExecutableNetwork exe_network;
#if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || \
    defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
      config[InferenceEngine::HDDL_DEVICE_TAG] = global_context_.deviceTags[j];
#else
      config[VPU_HDDL_CONFIG_KEY(DEVICE_TAG)] = global_context_.deviceTags[j];
#endif
      try {
        exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, hw_target, config);
      } catch (const Exception& e) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + e.what());
      } catch (...) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
      }
      exe_networks.push_back(exe_network);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    for (size_t j = 0; j < num_inf_reqs_; j++) {
      InferenceEngine::InferRequest::Ptr infRequest;
      try {
        infRequest = std::make_shared<InferenceEngine::InferRequest>(exe_networks[j].CreateInferRequest());
      } catch (const Exception& e) {
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
#if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || \
    defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
    config[InferenceEngine::HDDL_DEVICE_TAG] = global_context_.deviceTags[i];
#else
    config[VPU_HDDL_CONFIG_KEY(DEVICE_TAG)] = global_context_.deviceTags[i];
#endif
    InferenceEngine::ExecutableNetwork exe_network;
    try {
      exe_network = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, hw_target, config);
    } catch (const Exception& e) {
      ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    InferenceEngine::InferRequest::Ptr infRequest;
    try {
      infRequest = std::make_shared<InferenceEngine::InferRequest>(exe_network.CreateInferRequest());
    } catch (const Exception& e) {
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
void VADMBackend::StartAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context,
                                      size_t batch_slice_idx, size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];
  auto graph_input_info = ie_cnn_network_->getInputsInfo();

  size_t index = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++index) {
    // Get OpenVINO's input buffer
    InferenceEngine::Blob::Ptr graph_input_blob;
    std::string input_name = input_info_iter->first;
    try {
      graph_input_blob = infer_request->GetBlob(input_name);
    } catch (const Exception& e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
    }
    auto precision = input_info_iter->second->getPrecision();
    FillInputBlob(graph_input_blob, index, batch_slice_idx, input_name, ort, context, precision, subgraph_context_);
  }

  // Start Async inference
  try {
    infer_request->StartAsync();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Couldn't start Inference");
  }
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void VADMBackend::CompleteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context,
                                         size_t batch_slice_idx, size_t infer_req_idx,
                                         size_t batch_size) {
  auto infer_request = infer_requests_[infer_req_idx];

  // Wait for Async inference completion
  try {
    infer_request->Wait(WaitMode::RESULT_READY);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception with completing Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception with completing Inference");
  }
  auto graph_output_info = ie_cnn_network_->getOutputsInfo();

  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    // Get OpenVINO's output blob
    InferenceEngine::Blob::Ptr graph_output_blob;
    auto output_name = output_info_iter->first;
    try {
      graph_output_blob = infer_request->GetBlob(output_name);
    } catch (const Exception& e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_name);
    }

    auto output_tensor = GetOutputTensor(ort, context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto precision = output_info_iter->second->getPrecision();

    FillOutputBlob(graph_output_blob, output_tensor, ort, precision, batch_slice_idx);
  }
#if defined(OPENVINO_2020_4)
  if (!const_outputs_map_.empty()) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(ort, node, output_tensor);
    }
  }
#endif
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

  size_t batch_size = 1;

  if (subgraph_context_.enable_batching) {
    // Calculate the batch_size from the input tensor shape.
    const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context_.input_indexes[0]);

    batch_size = DeduceBatchSize(ort, tensor,
                                 ie_cnn_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());
  }

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  if (subgraph_context_.is_constant) {
#if defined(OPENVINO_2020_4) || defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || \
    defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(ort, node, output_tensor);
    }
#endif
  } else {
    // Distribute the batched inputs among available Infer Requests
    // for parallel inference.

    // Run parallel inferences as sets of num_inf_reqs_
    for (size_t set = 0; set < full_parallel_runs; set++) {
      for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
        size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
        StartAsyncInference(ort, context, batch_slice_idx, inf_req_idx);
      }
      for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
        size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
        CompleteAsyncInference(ort, context, batch_slice_idx, inf_req_idx, batch_size);
#ifndef NDEBUG
        if (openvino_ep::backend_utils::IsDebugEnabled()) {
          std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
          printPerformanceCounts(*infer_requests_[inf_req_idx], std::cout, hw_target);
        }
#endif
      }
    }

    // Run parallel inferences for remaining batch slices
    for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
      size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
      StartAsyncInference(ort, context, batch_slice_idx, inf_req_idx);
    }
    for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
      size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
      CompleteAsyncInference(ort, context, batch_slice_idx, inf_req_idx, batch_size);
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
        printPerformanceCounts(*infer_requests_[inf_req_idx], std::cout, hw_target);
      }
#endif
    }
  }
  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace openvino_ep
}  // namespace onnxruntime
