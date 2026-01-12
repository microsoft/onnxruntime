// Copyright (C) Intel Corporation
// Licensed under the MIT License
#include <algorithm>
#include <sstream>
#include <fstream>
#include <utility>

#include <filesystem>
#include <stdexcept>

#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/ov_interface.h"

using Exception = ov::Exception;

namespace onnxruntime {
namespace openvino_ep {

namespace backend_utils {

bool IsDebugEnabled() {
  static std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_DEBUG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

bool IsCILogEnabled() {
  static std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

std::shared_ptr<const OVNetwork>
CreateOVModel(std::string&& model,
              const SessionContext& session_context,
              std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map) {
  if (IsCILogEnabled()) {
    std::cout << "CreateNgraphFunc" << std::endl;
  }
  try {
    auto ov_model = OVCore::Get()->ReadModel(std::move(model), session_context.onnx_model_path_name.string());

    if (!session_context.reshape.empty()) {
      LOGS_DEFAULT(INFO) << log_tag << "Reshaping the ov tensor to specified shape";
      ov_model->reshape(session_context.reshape);
    }

    if (!session_context.layout.empty()) {
      LOGS_DEFAULT(INFO) << log_tag << "Setting the ov tensor layout to specified layout";
      ov_model = Set_Layout(ov_model, session_context.layout);
    }
    // Check for Constant Folding
    if ((session_context.device_type != "NPU") && !session_context.is_wholly_supported_graph) {
      ov::pass::ConstantFolding pass_const_obj;
      pass_const_obj.run_on_model(ov_model);
      auto& results = const_cast<ov::ResultVector&>(ov_model.get()->get_results());
      size_t index = results.size() - 1;

      for (auto it = results.rbegin(); it != results.rend(); ++it) {
        if (auto const_node =
                std::dynamic_pointer_cast<ov::op::v0::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
          const_outputs_map[(*it)->get_friendly_name()] = const_node;
          results.erase(results.begin() + index);
        }
        --index;
      }
    }
#ifndef NDEBUG
    if (IsDebugEnabled()) {
      std::string name = ov_model->get_friendly_name();
      ov::pass::Serialize serializer(name + ".xml", name + ".bin");
      serializer.run_on_model(ov_model);
    }
#endif
    return ov_model;
  } catch (std::string const& msg) {
    ORT_THROW(msg);
  }
}

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context,
                std::string output_name,
                const SubGraphContext::string_index_map_t& output_names,
                std::shared_ptr<ov::Node> node) {
  // Find position of '/' in the output_name
  auto pos = output_name.find("/");
  // Copy the substring from start to pos
  output_name = output_name.substr(0, pos);

  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  auto output_shape = ParameterShape::ToOrtShape(node->get_shape());

  return context.GetOutput(index, output_shape);
}

std::shared_ptr<OVNetwork> Set_Layout(std::shared_ptr<OVNetwork> ov_model, const layout_t& layout) {
  ov::preprocess::PrePostProcessor preproc(ov_model);

  const auto& inputs = ov_model->inputs();
  const auto& outputs = ov_model->outputs();

  auto find_tensor_index = [](const std::vector<ov::Output<ov::Node>>& tensors, const std::string& name) -> std::optional<size_t> {
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      if (tensor.get_any_name() == name || tensor.get_tensor().get_names().count(name) > 0) {
        return i;
      }
    }
    return std::nullopt;
  };

  for (const auto& [tensor_name, layout_value] : layout) {
    bool tensor_found = false;

    if (auto input_idx = find_tensor_index(inputs, tensor_name)) {
      preproc.input(*input_idx).tensor().set_layout(layout_value);
      tensor_found = true;
    } else if (auto output_idx = find_tensor_index(outputs, tensor_name)) {
      preproc.output(*output_idx).tensor().set_layout(layout_value);
      tensor_found = true;
    }

    if (!tensor_found) {
      LOGS_DEFAULT(WARNING) << "Tensor '" << tensor_name << "' not found in model inputs or outputs";
    }
  }

  return preproc.build();
}

int GetFirstAvailableDevice(SessionContext& session_context) {
  int i = 0;
  // Get the first available VAD-M device and set the device to busy
  while (i < 8) {
    bool device = session_context.deviceAvailableList[i];
    if (device) {
      session_context.deviceAvailableList[i] = false;
      break;
    }
    i++;
  }
  // If all of the devices are busy, assign the first device and
  // make all remaining devices free
  if (i == 8) {
    i = 0;
    session_context.deviceAvailableList[i] = false;
    for (int j = 1; j < 8; j++) {
      session_context.deviceAvailableList[j] = true;
    }
  }
  return i;
}

void FillOutputsWithConstantData(std::shared_ptr<ov::Node> node, Ort::UnownedValue& out_tensor) {
  switch (node->get_element_type()) {
    case ov::element::Type_t::f32: {
      FillOutputHelper<float>(out_tensor, std::move(node));
      break;
    }
    case ov::element::Type_t::boolean: {
      FillOutputHelper<char>(out_tensor, std::move(node));
      break;
    }
    case ov::element::Type_t::i32: {
      FillOutputHelper<int32_t>(out_tensor, std::move(node));
      break;
    }
    case ov::element::Type_t::i64: {
      FillOutputHelper<int64_t>(out_tensor, std::move(node));
      break;
    }
    case ov::element::Type_t::f16: {
      FillOutputHelper<float>(out_tensor, std::move(node));
      break;
    }
    default:
      ORT_THROW(log_tag + "Unsupported output data type");
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
  if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
    ORT_THROW(log_tag + "IO Buffering is not enabled, Please enable Input on CPU");
  }
  // Copy input data into OpenVINO's input buffer
  const char* tensor_data = tensor.GetTensorData<char>();
  const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;
  std::memcpy(input_data, batch_memory_offset, input_data_size);
}

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName) {
  int64_t totalTime = 0;
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
  auto performanceMap = request->GetInfReq().get_profiling_info();
  printPerformanceCounts(performanceMap, stream, std::move(deviceName));
}

bool IsModelStreamXML(std::istream& model_stream) {
  std::streampos originalPos = model_stream.tellg();

  // first, get the total size of model_stream in bytes
  model_stream.seekg(0, std::ios::end);
  auto end_pos = model_stream.tellg();
  //  Restore the stream position
  model_stream.seekg(originalPos);
  auto total_size = end_pos - originalPos;

  // Choose 32 bytes to hold content of:
  // '<?xml version-"1.0"?> <net '
  const std::streamsize header_check_len = 32;
  ORT_ENFORCE(total_size > header_check_len);

  // read 32 bytes into header
  std::string header(header_check_len, '\0');
  model_stream.read(&header[0], header_check_len);
  // Clear any read errors
  model_stream.clear();
  // Restore the stream position
  model_stream.seekg(originalPos);

  // return true if the header starts with '<?xml' and also includes '<net '
  return ((header.rfind("<?xml", 0) == 0) && (header.find("<net ") != std::string::npos));
}

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
