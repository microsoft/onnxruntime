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

SharedContext::SharedWeights::WeightsFile::WeightsFile(std::filesystem::path filename) : file_(filename, std::ios::in | std::ios::binary) {
  try {
    file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    weights_size_ = file_.seekg(0, std::ios::end).tellg();
  } catch (std::ifstream::failure& e) {
    ORT_THROW("Error: Failed to open weight file at ", filename.string(), " ", e.what());
  }
}

void SharedContext::SharedWeights::WeightsFile::load_weights(size_t file_offset, void* data, size_t size) {
  ORT_ENFORCE(file_offset < weights_size_ && size <= weights_size_ && (file_offset <= weights_size_ - size), "Error: File offset is out of bounds.");
  file_.seekg(file_offset);
  file_.read(reinterpret_cast<char*>(data), size);
}

std::ostream& operator<<(std::ostream& stream, const SharedContext::SharedWeights::Metadata::Map& metadata) {
  try {
    stream << metadata.size();

    // Write each key-value pair
    // Put elements in separate lines to facilitate reading
    for (const auto& [key, value] : metadata) {
      stream << std::endl
             << key.name;
      stream << std::endl
             << value.location;
      stream << std::endl
             << value.data_offset;
      stream << std::endl
             << value.size;
      stream << std::endl
             << value.dimensions.size();
      for (const auto& dim : value.dimensions) {
        stream << std::endl
               << dim;
      }
      stream << std::endl
             << value.element_type;
    }
  } catch (const Exception& e) {
    ORT_THROW("Error: Failed to write map data.", e.what());
  } catch (...) {
    ORT_THROW("Error: Failed to write map data.");
  }

  ORT_ENFORCE(stream.good(), "Error: Failed to write map data.");
  return stream;
}

std::istream& operator>>(std::istream& stream, SharedContext::SharedWeights::Metadata::Map& metadata) {
  size_t map_size{0};
  try {
    stream >> map_size;

    while (!stream.eof()) {
      SharedContext::SharedWeights::Metadata::Key key;
      SharedContext::SharedWeights::Metadata::Value value;
      stream >> key.name;
      stream >> value.location;
      stream >> value.data_offset;
      stream >> value.size;
      size_t num_dimensions;
      stream >> num_dimensions;

      if (stream.fail()) {
        ORT_THROW("Error: Failed to read num_dimensions from stream.");
      }

      constexpr size_t MAX_SAFE_DIMENSIONS = 1024;

      size_t safe_num_dimensions = num_dimensions;

      if (num_dimensions == 0 || safe_num_dimensions > MAX_SAFE_DIMENSIONS) {
        ORT_THROW("Invalid number of dimensions provided.");
      }
      try {
        value.dimensions.resize(safe_num_dimensions);
      } catch (const std::bad_alloc&) {
        ORT_THROW("Error: Memory allocation failed while resizing dimensions.");
      }

      for (auto& dim : value.dimensions) {
        stream >> dim;
      }
      stream >> value.element_type;
      metadata.emplace(key, value);
    }
  } catch (const Exception& e) {
    ORT_THROW("Error: Failed to read map data.", e.what());
  } catch (...) {
    ORT_THROW("Error: Failed to read map data.");
  }

  ORT_ENFORCE(metadata.size() == map_size, "Error: Inconsistent map data.");

  return stream;
}

namespace backend_utils {

bool IsDebugEnabled() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_DEBUG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

bool IsCILogEnabled() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

std::shared_ptr<const OVNetwork>
CreateOVModel(const std::string model,
              const SessionContext& session_context,
              std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map) {
  if (IsCILogEnabled()) {
    std::cout << "CreateNgraphFunc" << std::endl;
  }
  try {
    auto ov_model = OVCore::Get()->ReadModel(model, session_context.onnx_model_path_name.string());

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
GetOutputTensor(Ort::KernelContext& context, size_t batch_size,
                OVInferRequestPtr infer_request,
                std::string output_name,
                const SubGraphContext::string_index_map_t& output_names) {
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
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  return context.GetOutput(index, output_shape.get(), num_dims);
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
  auto shape = node->get_shape();

  size_t num_dims = shape.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(shape[j]);
  }
  return context.GetOutput(index, output_shape.get(), num_dims);
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
  auto performanceMap = request->GetNewObj().get_profiling_info();
  printPerformanceCounts(performanceMap, stream, std::move(deviceName));
}

ov::element::Type GetOpenVINOElementType(ONNX_NAMESPACE::TensorProto_DataType dt) {
  static std::unordered_map<ONNX_NAMESPACE::TensorProto_DataType, ov::element::Type> map{
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ov::element::f32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, ov::element::u8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, ov::element::i8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, ov::element::u16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, ov::element::i16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, ov::element::i32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, ov::element::i64},
      {ONNX_NAMESPACE::TensorProto_DataType_STRING, ov::element::string},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, ov::element::boolean},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, ov::element::f16},
      {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ov::element::f64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, ov::element::u32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, ov::element::u64},
      //{ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64, ov::element::undefined},
      //{ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128, ov::element::undefined},
      {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, ov::element::bf16},
      //{ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, ov::element::undefined},
      //{ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ, ov::element::undefined},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, ov::element::f8e5m2},
      //{ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ, ov::element::undefined},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT4, ov::element::u4},
      {ONNX_NAMESPACE::TensorProto_DataType_INT4, ov::element::i4},
  };

  if (auto result = map.find(dt); result != map.end()) {
    return result->second;
  } else {
    throw std::runtime_error("Unsupported ONNX data type: " + std::to_string(dt));
  }
}

// Function to handle tensor creation from external data
void CreateOVTensors(const std::string& device_name,
                     SharedContext::SharedWeights::Metadata::Map& metadata_map,
                     SharedContext::SharedWeights::WeightsFile& weights) {
  for (auto& [key, value] : metadata_map) {
    if (value.tensor) continue;

    // Get element data type
    auto onnx_element_type = (ONNX_NAMESPACE::TensorProto_DataType)value.element_type;

    ov::element::Type ov_elementType = GetOpenVINOElementType(onnx_element_type);  // Map to OpenVINO data type

    // Create OpenVINO Tensor
    if (device_name == "NPU") {
      // Use remote tensors
      auto npu_context = OVCore::Get()->core.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
      auto&& remote_tensor = npu_context.create_l0_host_tensor(ov_elementType, value.dimensions, ov::intel_npu::TensorType::INPUT);

      // Copy data to remote tensor
      weights.load_weights(value.data_offset, remote_tensor.get(), value.size);
      value.tensor = std::make_shared<ov::Tensor>(remote_tensor);
    } else {
      // Use vanilla tensors
      value.tensor = std::make_shared<ov::Tensor>(ov_elementType, value.dimensions);
      weights.load_weights(value.data_offset, value.tensor->data(), value.size);
    }
    ORT_ENFORCE(value.tensor->get_byte_size() == value.size, "Unexpected tensor size mismatch");
  }
}

void DestroyOVTensors(SharedContext::SharedWeights::Metadata::Map& metadata_map) {
  for (auto& [key, value] : metadata_map) {
    if (value.tensor) {
      value.tensor.reset();
    }
  }
  metadata_map.clear();
}

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
