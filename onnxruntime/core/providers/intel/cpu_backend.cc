// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include <inference_engine.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

// FIXME: These should not be needed after v1 ops
// are fully integrated into onnx importer
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/opset1_upgrade.hpp>
#include <ngraph/pass/convert_fp32_to_fp16.hpp>

// FIXME: Remove before production
#include <ngraph/serializer.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"

#include "ov_backend.h"
#include "cpu_backend.h"

namespace onnxruntime {
namespace intel_ep {

#define NGRAPH_EP_LRU_CACHE_DEFAULT_SIZE 500
// const std::string CPUBackend::log_tag = "[Intel-EP] ";

CPUBackend::CPUBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes, std::string device_id, InferenceEngine::Precision precision)
    : OVBackend(precision, input_indexes){

  (void) device_id;

  InferenceEngine::Core ie;
  ie_cnn_network_ = CreateCNNNetwork(model_proto);

  SetIODefs(model_proto, ie_cnn_network_);

  // Loading model to the plugin
  InferenceEngine::ExecutableNetwork exe_network_ = ie.LoadNetwork(*ie_cnn_network_, "CPU");
  LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";

  // Create infer request
  infer_request_ = exe_network_.CreateInferRequestPtr();

  LOGS_DEFAULT(INFO) << log_tag << "Infer request created";
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void CPUBackend::StartAsyncInference(Ort::CustomOpApi& ort, const OrtValue* input_tensors[],
                                     InferenceEngine::InferRequest::Ptr infer_request,
                                     std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  auto graph_input_info = ie_cnn_network->getInputsInfo();

  size_t i = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {
    // Get OpenVINO's input buffer
    auto graph_input_blob = infer_request->GetBlob(input_info_iter->first);
    auto graph_input_buffer =
        graph_input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t input_data_size = graph_input_blob->byteSize();
    const char* tensor_data = ort.GetTensorData<char>(input_tensors[i]);

    // Copy input data into OpenVINO's input buffer
    std::memcpy(graph_input_buffer, tensor_data, input_data_size);
  }

  // Start Async inference
  infer_request->StartAsync();
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void CPUBackend::CompleteAsyncInference(Ort::CustomOpApi& ort, OrtValue* output_tensors[],
                                        InferenceEngine::InferRequest::Ptr infer_request,
                                        std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network) {
  // Wait for Async inference completion
  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  auto graph_output_info = ie_cnn_network->getOutputsInfo();

  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    // Get OpenVINO's output blob
    auto graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    auto graph_output_buffer =
        graph_output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t output_data_size = graph_output_blob->byteSize();
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensors[i]);

    // Copy output results back to ONNX-RT's output buffers
    std::memcpy(tensor_data, graph_output_buffer, output_data_size);
  }
}

void CPUBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";
  std::lock_guard<std::mutex> lock(compute_lock_);

  // Get Input and Output tensors
  size_t input_count = ie_cnn_network_->getInputsInfo().size();
  size_t output_count = ie_cnn_network_->getOutputsInfo().size();
  const OrtValue* input_tensors[input_count];
  OrtValue* output_tensors[output_count];

  GetInputTensors(ort, context, input_tensors, ie_cnn_network_);

  size_t batch_size = 1;
  GetOutputTensors(ort, context, output_tensors, batch_size, infer_request_, ie_cnn_network_);


  StartAsyncInference(ort, input_tensors, infer_request_, ie_cnn_network_);
  CompleteAsyncInference(ort, output_tensors, infer_request_, ie_cnn_network_);

  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
}

}  // namespace intel_ep
}  // namespace onnxruntime