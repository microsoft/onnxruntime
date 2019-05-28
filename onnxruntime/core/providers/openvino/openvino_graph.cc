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

#include "openvino_graph.h"

namespace openvino_ep {

OpenVINOGraph::OpenVINOGraph(onnxruntime::Node* fused_node, std::string /*device_info*/) {
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

  // sets number of maximum parallel inferences
  num_inf_reqs_ = 8;

  fused_node_ = fused_node;

  cnn_network_ = BuildCNNNetworkWithMO();

  infer_requests_ = GetExecutableHandle(cnn_network_, device_id_, precision_);
}

std::vector<std::string> OpenVINOGraph::GetEnvLdLibraryPath() {
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

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::BuildCNNNetworkWithMO() {
  const auto& attributes = fused_node_->GetAttributes();
  std::string modelProtoStr = attributes.at("model_proto_str").s();

  Py_Initialize();
  if (!Py_IsInitialized()) {
    std::cout << "Python Interpreter initialization failed \n";
    throw "Python Interpreter initialization failed";
  }

  // Load the MO python module
  PyObject* pModule = PyImport_ImportModule("openvino_mo");
  if (pModule == NULL) {
    std::cout << "Python module not found " << std::endl;
    Py_Finalize();
    throw "Python module not found";
  }

  // Load the relevant function
  PyObject* pFunc = NULL;
  if (precision_ == InferenceEngine::Precision::FP32) {
    pFunc = PyObject_GetAttrString(pModule, "convert_fp32");
  } else if (precision_ == InferenceEngine::Precision::FP16) {
    pFunc = PyObject_GetAttrString(pModule, "convert_fp16");
  }

  if ((pFunc == NULL) || (PyCallable_Check(pFunc) == 0)) {
    std::cout << "Python Function not found" << std::endl;
    Py_DECREF(pModule);
    Py_Finalize();
    throw "Python Function not found";
  }

  // Prepare ModelProto Input to Python
  PyObject* pFileName = PyByteArray_FromStringAndSize(modelProtoStr.c_str(), modelProtoStr.size());
  PyObject* pArgs = PyTuple_New(1);
  PyTuple_SetItem(pArgs, 0, pFileName);

  // Call the Python function
  PyObject* pOutputTuple = PyObject_CallObject(pFunc, pArgs);

  if (pOutputTuple == NULL) {
    std::cout << "Model Optimizer call Failed\n";
    throw "Model Optimizer Failed";
  }

  // Retrieve the weights byte array
  PyObject* pArg1 = PyTuple_GetItem(pOutputTuple, 0);
  PyObject* pWeights = PyByteArray_FromObject(pArg1);
  const char* weights_bytes = PyByteArray_AsString(pWeights);
  unsigned long weights_size = PyByteArray_Size(pWeights);

  // Retrieve the xml string
  PyObject* pArg2 = PyTuple_GetItem(pOutputTuple, 1);
  PyObject* pXML = PyObject_Repr(pArg2);
  std::string xml_string = PyUnicode_AsUTF8(pXML);

  InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr(
      new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8,
                                          InferenceEngine::Layout::C, {weights_size}));
  weightsPtr->allocate();

  std::memcpy(weightsPtr->buffer(), (void*)weights_bytes, weights_size);

  InferenceEngine::CNNNetReader networkReader;
  networkReader.ReadNetwork((const char*)xml_string.c_str(), xml_string.size());
  networkReader.SetWeights(weightsPtr);

  // TODO: Cleanup Python interpreter resources
  //    Py_DECREF(pXML);
  //    Py_DECREF(pArg2);
  //    Py_DECREF(pWeights);
  //    Py_DECREF(pArg1);
  //    Py_DECREF(pOutputTuple);
  //    Py_DECREF(pArgs);
  //    Py_DECREF(pFunc);
  //    Py_DECREF(pModule);
  //Py_FinalizeEx();

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
  std::cout << "[OpenVINO-EP]Creating Infer requests : " << num_inf_reqs_ << std::endl;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests;
  for (size_t i = 0; i < num_inf_reqs_; i++) {
    infer_requests.push_back(exeNetwork.CreateInferRequestPtr());
  }
  return infer_requests;
}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::GetCNNNetwork() {
  return cnn_network_;
}

size_t OpenVINOGraph::DeduceBatchSize(onnxruntime::ONNXRunTimeTensor* input_tensor,
                                      InferenceEngine::SizeVector graph_dims) {
  size_t batch_size = 1;

  std::cout << "[OpenVINO-EP] Input dims: ";
  for (size_t i = 0; i < input_tensor->ndim; i++) {
    std::cout << input_tensor->shape[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "[OpenVINO-EP] Graph dims: ";
  for (auto dim : graph_dims) {
    std::cout << dim << ", ";
  }
  std::cout << std::endl;

  if ((input_tensor->ndim == graph_dims.size() && input_tensor->shape[0] > 1 && graph_dims[0] == 1) || (input_tensor->ndim == graph_dims.size() + 1)) {
    batch_size = input_tensor->shape[0];
  }

  return batch_size;
}

void OpenVINOGraph::AllocateOutputTensors(
    size_t batch_size,
    onnxruntime::ONNXRunTimeTensor* output_tensors,
    onnxruntime::AllocateFunc& output_allocator_func,
    onnxruntime::AllocatorHandle& output_allocator_handle) {
  auto graph_output_info = cnn_network_->getOutputsInfo();

  // All infer_requests process identical tensor slices from the batch.
  // So using info from first infer_request to allocate all output tensors.
  auto infer_request = infer_requests_[0];

  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    auto graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    auto output_blob_size = graph_output_blob->byteSize();
    auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();

    if (batch_size > 1) {
      // Add the batch size as dim 0.
      graph_output_dims.insert(graph_output_dims.begin(), batch_size);
    }

    auto num_dims = graph_output_dims.size();

    // TODO: Memory Leak!!!!
    // Allocate using the AllocateFunc instead???
    // fix before shipping.
    output_tensors[i].shape = new int64_t[num_dims];
    for (size_t j = 0; j < num_dims; j++) {
      output_tensors[i].shape[j] = (int64_t)graph_output_dims[j];
    }

    output_tensors[i].ndim = num_dims;
    output_tensors[i].dtype = onnxruntime::DType::TFloat32;
    output_tensors[i].data = (*output_allocator_func)(output_allocator_handle,
                                                      64, output_blob_size * batch_size);
  }
}

void OpenVINOGraph::StartAsyncInference(onnxruntime::ONNXRunTimeTensor* input_tensors,
                                        size_t batch_slice_idx,
                                        size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];

  auto graph_input_info = cnn_network_->getInputsInfo();
  size_t i = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {
    // Get OpenVINO's input buffer
    auto graph_input_blob = infer_request->GetBlob(input_info_iter->first);
    auto graph_input_buffer =
        graph_input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t input_data_size = graph_input_blob->byteSize();

    char* batch_memory_offset = (char*)(input_tensors[i].data) + input_data_size * batch_slice_idx;

    // Copy input data into OpenVINO's input buffer
    std::memcpy(graph_input_buffer, batch_memory_offset, input_data_size);
  }

  // Start Async inference
  infer_request->StartAsync();
}

void OpenVINOGraph::CompleteAsyncInference(onnxruntime::ONNXRunTimeTensor* output_tensors,
                                           size_t batch_slice_idx,
                                           size_t infer_req_idx) {
  auto infer_request = infer_requests_[infer_req_idx];

  // Wait for Async inference completion
  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

  auto graph_output_info = cnn_network_->getOutputsInfo();
  size_t i = 0;
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter, ++i) {
    auto graph_output_blob = infer_request->GetBlob(output_info_iter->first);
    auto graph_output_buffer =
        graph_output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t output_data_size = graph_output_blob->byteSize();

    char* batch_memory_offset = (char*)(output_tensors[i].data) + output_data_size * batch_slice_idx;

    std::memcpy(batch_memory_offset, graph_output_buffer, output_data_size);
  }
}

void OpenVINOGraph::Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
                          size_t num_inputs,
                          onnxruntime::ONNXRunTimeTensor* output_tensors,
                          size_t num_outputs,
                          onnxruntime::AllocateFunc& output_allocator_func,
                          onnxruntime::AllocatorHandle& output_allocator_handle) {
  // Preliminary thread safety mechanism
  // TODO: reduce lock scope to just infer_request objects
  std::lock_guard<std::mutex> lock(compute_lock_);

  std::cout << "[OpenVINO-EP] Inference Started\n";
  // Check I/O sizes
  auto graph_input_info = cnn_network_->getInputsInfo();
  std::cout << "[OpenVINO-EP] num_inputs: " << num_inputs << std::endl;
  if (num_inputs != graph_input_info.size()) {
    throw "OpenVINO Inference: Inputs count mismatch!";
  }

  auto graph_output_info = cnn_network_->getOutputsInfo();
  std::cout << "[OpenVINO-EP] num_outputs: " << num_outputs << std::endl;
  if (num_outputs != graph_output_info.size()) {
    throw "OpenVINO Inference: Outputs count mismatch!";
  }

  auto batch_size = DeduceBatchSize(input_tensors,
                                    cnn_network_->getInputsInfo().begin()->second->getTensorDesc().getDims());
  std::cout << "[OpenVINO-EP] Batch Size: " << batch_size << std::endl;

  size_t full_parallel_runs = batch_size / num_inf_reqs_;
  size_t remainder_parallel_runs = batch_size % num_inf_reqs_;

  AllocateOutputTensors(batch_size, output_tensors, output_allocator_func,
                        output_allocator_handle);

  // Run parallel inferences as sets of num_inf_reqs_
  for (size_t set = 0; set < full_parallel_runs; set++) {
    std::cout << "[OpenVINO-EP] Running " << num_inf_reqs_
              << " parallel inferences\n";
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      StartAsyncInference(input_tensors, batch_slice_idx, inf_req_idx);
    }
    for (size_t inf_req_idx = 0; inf_req_idx < num_inf_reqs_; inf_req_idx++) {
      size_t batch_slice_idx = set * num_inf_reqs_ + inf_req_idx;
      CompleteAsyncInference(output_tensors, batch_slice_idx, inf_req_idx);
    }
  }

  // Run parallel inferences for remaining batch slices
  std::cout << "[OpenVINO-EP] Running " << remainder_parallel_runs
            << " parallel inferences\n";
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    StartAsyncInference(input_tensors, batch_slice_idx, inf_req_idx);
  }
  for (size_t inf_req_idx = 0; inf_req_idx < remainder_parallel_runs; inf_req_idx++) {
    size_t batch_slice_idx = full_parallel_runs * num_inf_reqs_ + inf_req_idx;
    CompleteAsyncInference(output_tensors, batch_slice_idx, inf_req_idx);
  }
}

}  // namespace openvino_ep
