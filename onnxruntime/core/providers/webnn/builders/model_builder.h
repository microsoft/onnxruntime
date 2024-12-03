// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include <core/graph/graph_viewer.h>

#include "model.h"
#include "core/framework/execution_provider.h"
#include "core/providers/webnn/builders/helper.h"

#include <sstream>
#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
               const emscripten::val& context, const DataLayout preferred_layout,
               const WebnnDeviceType wnn_device_type, const emscripten::val& wnn_limits);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;

  // Accessors for members.
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  InitializedTensorSet GetInitializerTensors();

  const emscripten::val& GetBuilder() const { return wnn_builder_; }
  const emscripten::val& GetContext() const { return wnn_context_; }
  const emscripten::val& GetOperand(const std::string& name) const { return wnn_operands_.at(name); }
  const emscripten::val& GetOpSupportLimits() const { return wnn_limits_; }

  void AddOperand(const std::string& name, const emscripten::val& operand);

  template <typename T>
  const emscripten::val& CreateOrGetConstant(const int32_t& data_type, T value,
                                             const std::vector<uint32_t>& shape = {});

  // Use the buffers to persist WebNN allocated data like transposed weight.
  // It ensures the validity during inference session.
  std::vector<std::unique_ptr<uint8_t[]>> mem_persist_buffers_;
  // Add a constant operand (allocate persist buffer and move the ownership to mem_persist_buffers_).
  Status AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
      const size_t size, const std::vector<uint32_t> shape, const int32_t data_type);

  DataLayout GetPreferredLayout() const { return preferred_layout_; }

  WebnnDeviceType GetWebnnDeviceType() const { return wnn_device_type_; }

  // The initializer will be processed separately, skip it as an initializer.
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to WebNN model, since WebNN does not like input unused.
  void AddInputToSkip(const std::string& input_name);

  std::string GetUniqueName(const std::string& base_name);

 private:
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;

  emscripten::val wnn_context_ = emscripten::val::undefined();
  emscripten::val wnn_builder_ = emscripten::val::undefined();
  DataLayout preferred_layout_;
  WebnnDeviceType wnn_device_type_;
  emscripten::val wnn_limits_ = emscripten::val::undefined();
  InlinedHashMap<std::string, emscripten::val> wnn_operands_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<uint8_t>> unpacked_tensors_;

  InlinedHashMap<std::string, OnnxTensorInfo> input_output_info_;

  InlinedHashSet<std::string> skipped_initializers_;
  InlinedHashSet<std::string> skipped_inputs_;

  uint32_t name_token_{0};
  InlinedHashSet<std::string> unique_names_;

  // Convert the onnx model to WebNN operands
  Status Initialize() ORT_MUST_USE_RESULT;

  void PreprocessInitializers();

  // Copy and process all the initializers to WebNN constants.
  Status RegisterInitializers() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;
  Status RegisterModelInputs() ORT_MUST_USE_RESULT;
  Status RegisterModelOutputs() ORT_MUST_USE_RESULT;
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) ORT_MUST_USE_RESULT;

  static const IOpBuilder* GetOpBuilder(const Node& node);
};

// Create or retrieve one of the following:
// - A WebNN constant MLOperand filled with the specified value, data type, and shape.
// - A WebNN scalar constant MLOperand with the specified value and data type.
// For scalar constant, it is workaround for builer.constant(type, value) method since
// it has not been implemented now.
// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-type-value
//
// This function enforces a mapping between the data_type and the value types:
// - TensorProto_DataType_INT4    <-> int8_t
// - TensorProto_DataType_UINT4   <-> int8_t
// - TensorProto_DataType_BOOL    <-> bool
// - TensorProto_DataType_UINT8   <-> uint8_t
// - TensorProto_DataType_INT8    <-> int8_t
// - TensorProto_DataType_FLOAT16 <-> float
// - TensorProto_DataType_FLOAT   <-> float
// - TensorProto_DataType_INT32   <-> int32_t
// - TensorProto_DataType_INT64   <-> int64_t
// - TensorProto_DataType_UINT32  <-> uint32_t
// - TensorProto_DataType_UINT64  <-> uint64_t
template <typename T>
const emscripten::val& ModelBuilder::CreateOrGetConstant(const int32_t& data_type, T value,
                                                         const std::vector<uint32_t>& shape) {
  std::string name = "webnn_constant_" + std::to_string(data_type) + "_" + std::to_string(value);
  emscripten::val dims = emscripten::val::array();
  if (!shape.empty()) {
    dims = emscripten::val::array(shape);
    std::ostringstream name_stream;
    name_stream << name;
    for (const auto& dim : shape) {
      name_stream << "_" << dim;
    }
    name = name_stream.str();
  }

  // If the operand does not exist, create it.
  if (wnn_operands_.find(name) == wnn_operands_.end()) {
    emscripten::val desc = emscripten::val::object();
    desc.set("shape", dims);
    desc.set("dimensions", dims);
    emscripten::val buffer = emscripten::val::undefined();
    if (!SetWebnnDataType(desc, data_type)) {
      ORT_THROW("Unsupported data type: " + std::to_string(data_type));
    }
    auto num_elements = Product(shape);
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
        // For WebNN int4 and uint4 tensors are stored in Uint8Array,
        // so we need to adjust the number of elements.
        num_elements = (num_elements + 1) / 2;
        buffer = emscripten::val::global("Uint8Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(PackInt8ToUint8AsNibble(value, data_type)));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        buffer = emscripten::val::global("Uint8Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        buffer = emscripten::val::global("Int8Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        buffer = emscripten::val::global("Uint16Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(PackFloat32ToUint16AsFloat16(value)));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        buffer = emscripten::val::global("Float32Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        buffer = emscripten::val::global("Int32Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        buffer = emscripten::val::global("Uint32Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        buffer = emscripten::val::global("BigInt64Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val::global("BigInt")(value));
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        buffer = emscripten::val::global("BigUint64Array").new_(num_elements);
        if (value) {
          buffer.call<void>("fill", emscripten::val::global("BigInt")(value));
        }
        break;
      default:
        break;
    }

    const emscripten::val constant = wnn_builder_.call<emscripten::val>("constant", desc, buffer);
    wnn_operands_.insert(std::make_pair(name, constant));
  }

  return wnn_operands_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime
