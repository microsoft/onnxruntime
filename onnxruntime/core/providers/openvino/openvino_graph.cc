// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <Python.h>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "openvino_graph.h"

namespace onnxruntime{
namespace openvino_ep {

OpenVINOGraph::OpenVINOGraph(const onnxruntime::Node* fused_node, std::string /*device_info*/) {
  // TODO: stop passing the unused device_info

  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  std::string precision_str = "FP32";

#ifdef OPENVINO_CONFIG_CPU_FP32
  device_id_ = "CPU";
  precision_ = InferenceEngine::Precision::FP32;
  precision_str = "FP32";
#endif
#ifdef OPENVINO_CONFIG_GPU_FP32
  device_id_ = "GPU";
  precision_ = InferenceEngine::Precision::FP32;
  precision_str = "FP32";
#endif
#ifdef OPENVINO_CONFIG_GPU_FP16
  device_id_ = "GPU";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif
#ifdef OPENVINO_CONFIG_MYRIAD
  device_id_ = "MYRIAD";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif
#ifdef OPENVINO_CONFIG_VAD_R
  device_id_ = "HDDL";
  precision_ = InferenceEngine::Precision::FP16;
  precision_str = "FP16";
#endif

  std::cout << "[OpenVINO-EP] device:" << device_id_ << std::endl;
  std::cout << "[OpenVINO-EP] precision:" << precision_str << std::endl;


  // Infer Request class represents OpenVINO's logical hardware instance. These logical
  // instances are bound to physical hardware instances at runtime depending
  // on the physical hardware availability. If multiple Infer Requests are mapped to
  // the same physical hardware instance, then the inference operations requests from
  // the Infer Requests are serialized before they are scheduled on the physical hardware.
  // If the different Infer Requests are scheduled on different hardware instances, inference
  // operations associated with the Infer Requests may be scheduled in parallel.
  // Infer Requests hold resources representing the entire network on their target hardware. So,
  // having more Infer Requests than needed would waste system resources.
  // In VAD-R (HDDL) accelerator, there are 8 parallel execution units. So, creating 8 instances
  // of Infer Requests only if the VAD-R accelerator is being used.
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

  // Create hardware agnostic OpenVINO network representation
  openvino_network_ = BuildOpenVINONetworkWithMO();

  // Create hardware specific OpenVINO network representation
  infer_requests_ = GetExecutableHandle(openvino_network_, device_id_, precision_);
}

std::vector<std::string> OpenVINOGraph::GetEnvLdLibraryPath() const {
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

void OpenVINOGraph::ConvertONNXModelToOpenVINOIR(const std::string& onnx_model,
    std::string& openvino_xml, std::string& openvino_bin, bool precision_fp32) {

  Py_Initialize();
  if (!Py_IsInitialized()) {
    throw "Python environment initialization failure";
  }

  PyObject* pModule = NULL;
  pModule = PyImport_ImportModule("openvino_mo");
  if (pModule == NULL) {
    throw "Python module import failure";
  }

  PyObject* pFunc = NULL;
  if (precision_fp32) {
    pFunc = PyObject_GetAttrString(pModule, "convert_fp32");
  } else {
    pFunc = PyObject_GetAttrString(pModule, "convert_fp16");
  }
  if (pFunc == NULL || !PyCallable_Check(pFunc)) {
    throw "Python module function check failure";
  }

  // Prepare ModelProto Input to Python
  PyObject* pFileName = PyByteArray_FromStringAndSize(
      onnx_model.c_str(), onnx_model.size());
  PyObject* pArgs = PyTuple_New(1);
  PyTuple_SetItem(pArgs, 0, pFileName);

  PyObject* pOutputTuple = NULL;

  // Call the Python function
  pOutputTuple = PyObject_CallObject(pFunc, pArgs);

  if (pOutputTuple == NULL || !PyTuple_CheckExact(pOutputTuple)) {
    throw "Python function call failure";
  }

  // Retrieve the weights byte array
  PyObject* pArg1 = PyTuple_GetItem(pOutputTuple, 0);
  PyObject* pWeights = PyByteArray_FromObject(pArg1);
  const char* weights_bytes = PyByteArray_AsString(pWeights);
  unsigned long weights_size = PyByteArray_Size(pWeights);
  std::string weights_string(weights_bytes, weights_size);
  openvino_bin = weights_string;

  // Retrieve the xml string
  PyObject* pArg2 = PyTuple_GetItem(pOutputTuple, 1);
  PyObject* pXML = PyObject_Repr(pArg2);
  openvino_xml = PyUnicode_AsUTF8(pXML);

  // Calling Py_Finalize here prevents multiple invocations
  // of the interpreter from the same process. Relying on
  // OS process clean up routines for python shutdown.

}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::BuildOpenVINONetworkWithMO() {

  const auto& attributes = fused_node_->GetAttributes();
  std::string xml_string = attributes.at("xml_str").s();
  std::string weights_string = attributes.at("weights_str").s();
  InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr(
      new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8,
                                          InferenceEngine::Layout::C, {weights_string.size()}));
  weightsPtr->allocate();

  std::memcpy(weightsPtr->buffer(), static_cast<const void*>(weights_string.c_str()), weights_string.size());

  InferenceEngine::CNNNetReader networkReader;
  networkReader.ReadNetwork(static_cast<const char*>(xml_string.c_str()), xml_string.size());
  networkReader.SetWeights(weightsPtr);

  return std::make_shared<InferenceEngine::CNNNetwork>(networkReader.getNetwork());
}

std::vector<InferenceEngine::InferRequest::Ptr> OpenVINOGraph::GetExecutableHandle(
    std::shared_ptr<InferenceEngine::CNNNetwork> network,
    const std::string& device, InferenceEngine::Precision precision) {
  // TODO: make this better

  precision = InferenceEngine::Precision::FP32;

  // Load Plugin for inference engine
  std::cout << "[OpenVINO-EP]Loading plugin" << std::endl;

  std::vector<std::string> plugin_path = GetEnvLdLibraryPath();
  plugin_path.push_back("");
  InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher(
                                                plugin_path)
                                                .getPluginByDevice(device);

  // Configure input & output
  // Prepare input blobs
  std::cout << "[OpenVINO-EP]Preparing input blobs" << std::endl;
  size_t first_dim = 1;

  auto inputInfo = network->getInputsInfo();
  for (auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter) {
    iter->second->setPrecision(precision);
    auto dims = iter->second->getTensorDesc().getDims();
    if (dims.size() == 2 || dims.size() == 4 || dims.size() == 5) {
      if (first_dim == 1)
        first_dim = iter->second->getTensorDesc().getDims()[0];
    }

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

  network->setBatchSize(first_dim);

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
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

  // Loading model to the plugin
  std::cout << "[OpenVINO-EP]Loading model to the plugin" << std::endl;
  InferenceEngine::ExecutableNetwork exeNetwork = plugin.LoadNetwork(*network,
                                                                     {});

  // Create infer request
  //std::cout << "[OpenVINO-EP]Creating Infer requests : " << num_inf_reqs_ << std::endl;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests;
  for (size_t i = 0; i < num_inf_reqs_; i++) {
    infer_requests.push_back(exeNetwork.CreateInferRequestPtr());
  }
  return infer_requests;
}

size_t OpenVINOGraph::DeduceBatchSize(Ort::CustomOpApi ort, const OrtValue* input_tensor,
                                      InferenceEngine::SizeVector graph_dims) {
  size_t batch_size = 1;

  // All the inputs and outputs are batched the same way.
  // So it is sufficient to use any one of these tensors to deduce the batch size.
  const auto& input_shape = ort.GetTensorShape(ort.GetTensorTypeAndShape(input_tensor));

  if ((input_shape.size() == graph_dims.size() && input_shape[0] > 1 && graph_dims[0] == 1) || (input_shape.size() == graph_dims.size() + 1)) {
    batch_size = input_shape[0];
  }

  return batch_size;
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void OpenVINOGraph::StartAsyncInference(Ort::CustomOpApi ort, const OrtValue* input_tensors[],
                                        size_t batch_slice_idx,
                                        size_t infer_req_idx) {

  auto infer_request = infer_requests_[infer_req_idx];
  auto graph_input_info = openvino_network_->getInputsInfo();

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
void OpenVINOGraph::CompleteAsyncInference(Ort::CustomOpApi ort, OrtValue* output_tensors[],
                                           size_t batch_slice_idx,
                                           size_t infer_req_idx) {

  auto infer_request = infer_requests_[infer_req_idx];

  // Wait for Async inference completion
  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  auto graph_output_info = openvino_network_->getOutputsInfo();

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

void OpenVINOGraph::GetInputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, const OrtValue* input_tensors[]) {
  size_t input_count = openvino_network_->getInputsInfo().size();

  for (size_t i = 0; i < input_count; i++) {
    input_tensors[i] = ort.KernelContext_GetInput(context, input_indexes_[i]);
  }
}

void OpenVINOGraph::GetOutputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, OrtValue* output_tensors[], size_t batch_size) {
  auto graph_output_info = openvino_network_->getOutputsInfo();

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

void OpenVINOGraph::Infer(Ort::CustomOpApi ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time
  std::lock_guard<std::mutex> lock(compute_lock_);

  std::cout << "[OpenVINO-EP] Inference Started\n";

  // Get Input and Output tensors
  size_t input_count = openvino_network_->getInputsInfo().size();
  size_t output_count = openvino_network_->getOutputsInfo().size();
  const OrtValue* input_tensors[input_count];
  OrtValue* output_tensors[output_count];

  GetInputTensors(ort, context, input_tensors);

  // Calculate the batch_size from the input tensor shape.
  auto batch_size = DeduceBatchSize(ort, input_tensors[0],
                                    openvino_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());
  if (batch_size != 1) {
    std::cout << "[OpenVINO-EP] Batch Size: " << batch_size << std::endl;
  }

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  GetOutputTensors(ort, context, output_tensors, batch_size);

  // Distribute the batched inputs among available Infer Requests
  // for parallel inference.

  // Run parallel inferences as sets of num_inf_reqs_
  for (size_t set = 0; set < full_parallel_runs; set++) {
    //std::cout << "[OpenVINO-EP] Running " << num_inf_reqs_
    //          << " parallel inferences\n";
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
  //std::cout << "[OpenVINO-EP] Running " << remainder_parallel_runs
  //          << " parallel inferences\n";
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    StartAsyncInference(ort, input_tensors, batch_slice_idx, inf_req_idx);
  }
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    CompleteAsyncInference(ort, output_tensors, batch_slice_idx, inf_req_idx);
  }
  std::cout << "[OpenVINO-EP] Inference Completed\n";
}

}  // namespace openvino_ep
} // namespace onnxruntime
