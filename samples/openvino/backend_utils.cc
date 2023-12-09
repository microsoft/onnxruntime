// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include "ov_interface.h"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "backend_utils.h"
#include "onnx/onnx_pb.h"

#if defined(OV_API_20)
using Exception = ov::Exception;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
using WaitMode = InferenceEngine::IInferRequest::WaitMode;
#endif

namespace onnxruntime {
namespace openvino_ep {
namespace backend_utils {

#ifndef NDEBUG
bool IsDebugEnabled() {
//  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_DEBUG");
//  if (!env_name.empty()) {
//    return true;
//  }
  return false;
}
#endif

bool IsCILogEnabled() {
//  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG");
//  if (!env_name.empty()) {
//    return true;
//  }
  return false;
}

struct static_cast_int64 {
  template <typename T1>  // T1 models type statically convertible to T
  int64_t operator()(const T1& x) const { return static_cast<int64_t>(x); }
};

std::shared_ptr<OVNetwork>
CreateOVModel(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context,
              const SubGraphContext& subgraph_context,
              std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map) {
  if (IsCILogEnabled()) {
    std::cout << "CreateNgraphFunc" << std::endl;
  }
  const std::string model = model_proto.SerializeAsString();
  try {
    auto cnn_network = global_context.ie_core.ReadModel(model);
    if ((subgraph_context.precision == "FP16") &&
        (global_context.device_type.find("VPUX") == std::string::npos)) {
      // FP16 transformations
      ov::pass::ConvertFP32ToFP16 pass_obj;
      pass_obj.run_on_model(cnn_network);
      cnn_network->validate_nodes_and_infer_types();

      auto proc = ov::preprocess::PrePostProcessor(cnn_network);
      for (size_t i = 0; i < cnn_network->inputs().size(); i++) {
        if (cnn_network->inputs()[i].get_element_type() == ov::element::f16) {
          proc.input(i).tensor().set_element_type(ov::element::f32);
          proc.input(i).preprocess().convert_element_type(ov::element::f16);
        }
      }

      for (size_t i = 0; i < cnn_network->outputs().size(); i++) {
        if (cnn_network->outputs()[i].get_element_type() == ov::element::f16) {
          proc.output(i).postprocess().convert_element_type(ov::element::f32);
        }
      }
      cnn_network = proc.build();
    }

    // Check for Constant Folding
    if (!global_context.is_wholly_supported_graph) {
      ov::pass::ConstantFolding pass_const_obj;
      pass_const_obj.run_on_model(cnn_network);
      auto& results = const_cast<ov::ResultVector&>(cnn_network.get()->get_results());
      size_t index = results.size() - 1;

      for (auto it = results.rbegin(); it != results.rend(); ++it) {
        if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
          const_outputs_map[(*it)->get_friendly_name()] = const_node;
          results.erase(results.begin() + index);
        }
        --index;
      }
    }
#ifndef NDEBUG
#if defined(OPENVINO_2022_3) || (OPENVINO_2023_0) || (OPENVINO_2023_1)
    if (IsDebugEnabled()) {
      std::string name = cnn_network->get_friendly_name();
      ov::pass::Serialize serializer(name + ".xml", name + ".bin");
      serializer.run_on_model(cnn_network);
    }
#endif
#endif
    return cnn_network;
  } catch (std::string const& msg) {
    throw msg;
  }
}

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context, size_t batch_size,
                OVInferRequestPtr infer_request,
                std::string output_name,
                std::unordered_map<std::string, int> output_names) {
  auto graph_output_blob = infer_request->GetTensor(output_name);

  auto graph_output_dims = graph_output_blob->get_shape();

  if (batch_size > 1) {
    // Add the batch size as dim 0.
    graph_output_dims.insert(graph_output_dims.begin(), batch_size);
  }
  size_t num_dims = graph_output_dims.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(graph_output_dims[j]);
  }
  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    throw std::string(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  return context.GetOutput(index, output_shape.get(), num_dims);
}

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context,
                std::string output_name,
                std::unordered_map<std::string, int> output_names,
                std::shared_ptr<ov::Node> node) {
  // Find position of '/' in the output_name
  int pos = output_name.find("/");
  // Copy the substring from start to pos
  output_name = output_name.substr(0, pos);

  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    throw std::string(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  auto shape = node->get_shape();

  size_t num_dims = shape.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(shape[j]);
  }
  return context.GetOutput(index, output_shape.get(), num_dims);
}

int GetFirstAvailableDevice(GlobalContext& global_context) {
  int i = 0;
  // Get the first available VAD-M device and set the device to busy
  while (i < 8) {
    bool device = global_context.deviceAvailableList[i];
    if (device) {
      global_context.deviceAvailableList[i] = false;
      break;
    }
    i++;
  }
  // If all of the devices are busy, assign the first device and
  // make all remaining devices free
  if (i == 8) {
    i = 0;
    global_context.deviceAvailableList[i] = false;
    for (int j = 1; j < 8; j++) {
      global_context.deviceAvailableList[j] = true;
    }
  }
  return i;
}

void FillOutputsWithConstantData(std::shared_ptr<ov::Node> node, Ort::UnownedValue& out_tensor) {
  switch (node->get_element_type()) {
    case ov::element::Type_t::f32: {
      FillOutputHelper<float>(out_tensor, node);
      break;
    }
    case ov::element::Type_t::boolean: {
      FillOutputHelper<char>(out_tensor, node);
      break;
    }
    case ov::element::Type_t::i32: {
      FillOutputHelper<int32_t>(out_tensor, node);
      break;
    }
    case ov::element::Type_t::i64: {
      FillOutputHelper<int64_t>(out_tensor, node);
      break;
    }
    case ov::element::Type_t::f16: {
      FillOutputHelper<float>(out_tensor, node);
      break;
    }
    default:
      throw std::string(log_tag + "Unsupported output data type");
  }
}

#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif

template <typename T>
void FillOutputHelper(Ort::UnownedValue& out_tensor, std::shared_ptr<ov::Node> node) {
  auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
  auto res = const_node->cast_vector<T>();
  T* tensor_data = out_tensor.GetTensorMutableData<T>();
  std::copy(res.begin(), res.end(), tensor_data);
}

#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif

void FillInputBlob(OVTensorPtr inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::KernelContext& context,
                   const SubGraphContext& subgraph_context) {
  size_t input_data_size = inputBlob->get_byte_size();
  auto input_data = inputBlob->data();
  auto tensor = context.GetInput(subgraph_context.input_names.at(input_name));
  auto mem_info = tensor.GetTensorMemoryInfo();
  if (mem_info.GetAllocatorName() == "OpenVINO_GPU") {  // TODO: or include allocator.h
    throw std::string(log_tag + "IO Buffering is not enabled, Please enable Input on CPU");
  }
  // Copy input data into OpenVINO's input buffer
  const char* tensor_data = tensor.GetTensorData<char>();
  const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;
  std::memcpy(input_data, batch_memory_offset, input_data_size);
}

void FillOutputBlob(OVTensorPtr outputBlob, Ort::UnownedValue& output_tensor,
                    size_t batch_slice_idx) {
  auto output_data = outputBlob->data();
  size_t output_data_size = outputBlob->get_byte_size();
  char* tensor_data = output_tensor.GetTensorMutableData<char>();
  char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;
  std::memcpy(batch_memory_offset, output_data, output_data_size);
}

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName) {
  long long totalTime = 0;
  // Print performance counts
  stream << std::endl
         << "performance counts:" << std::endl
         << std::endl;

  for (const auto& it : performanceMap) {
    std::string toPrint(it.node_name);
    const int maxLayerName = 30;

    if (it.node_name.length() >= maxLayerName) {
      toPrint = it.node_name.substr(0, maxLayerName - 4);
      toPrint += "...";
    }
    stream << std::setw(maxLayerName) << std::left << toPrint;
    switch (it.status) {
      case OVProfilingInfo::Status::EXECUTED:
        stream << std::setw(15) << std::left << "EXECUTED";
        break;
      case OVProfilingInfo::Status::NOT_RUN:
        stream << std::setw(15) << std::left << "NOT_RUN";
        break;
      case OVProfilingInfo::Status::OPTIMIZED_OUT:
        stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
        break;
    }
    stream << std::setw(30) << std::left << "layerType: " + std::string(it.node_type) + " ";
    stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.real_time.count());
    stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.cpu_time.count());
    stream << " execType: " << it.exec_type << std::endl;
    if (it.real_time.count() > 0) {
      totalTime += it.real_time.count();
    }
  }
  stream << std::setw(20) << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Full device name: " << deviceName << std::endl;
  std::cout << std::endl;
}

void printPerformanceCounts(OVInferRequestPtr request, std::ostream& stream, std::string deviceName) {
  auto performanceMap = request->GetNewObj().get_profiling_info();
  printPerformanceCounts(performanceMap, stream, deviceName);
}

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
