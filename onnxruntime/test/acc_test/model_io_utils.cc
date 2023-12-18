// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model_io_utils.h"
#include <iostream>

bool GetTensorElemDataSize(ONNXTensorElementDataType data_type, size_t& size) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      size = sizeof(float);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      size = sizeof(uint8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      size = sizeof(int8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      size = sizeof(uint16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      size = sizeof(int16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      size = sizeof(int32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      size = sizeof(int64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      size = sizeof(bool);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      size = sizeof(double);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      size = sizeof(uint32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      size = sizeof(uint64_t);
      break;
    default:
      std::cerr << "[ERROR]: Unsupported tensor element data type: " << data_type << std::endl;
      return false;
  }

  return true;
}

AccMetrics ComputeAccuracyMetric(Ort::ConstValue ort_output, Span<const char> raw_expected_output,
                                 const IOInfo& output_info) {
  AccMetrics metrics = {};
  switch (output_info.data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      Span<const float> expected_output = ReinterpretBytesAsSpan<const float>(raw_expected_output);
      Span<const float> actual_output(ort_output.GetTensorData<float>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      Span<const uint8_t> expected_output = ReinterpretBytesAsSpan<const uint8_t>(raw_expected_output);
      Span<const uint8_t> actual_output(ort_output.GetTensorData<uint8_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      Span<const int8_t> expected_output = ReinterpretBytesAsSpan<const int8_t>(raw_expected_output);
      Span<const int8_t> actual_output(ort_output.GetTensorData<int8_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      Span<const uint16_t> expected_output = ReinterpretBytesAsSpan<const uint16_t>(raw_expected_output);
      Span<const uint16_t> actual_output(ort_output.GetTensorData<uint16_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      Span<const int16_t> expected_output = ReinterpretBytesAsSpan<const int16_t>(raw_expected_output);
      Span<const int16_t> actual_output(ort_output.GetTensorData<int16_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      Span<const int32_t> expected_output = ReinterpretBytesAsSpan<const int32_t>(raw_expected_output);
      Span<const int32_t> actual_output(ort_output.GetTensorData<int32_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      Span<const int64_t> expected_output = ReinterpretBytesAsSpan<const int64_t>(raw_expected_output);
      Span<const int64_t> actual_output(ort_output.GetTensorData<int64_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      Span<const bool> expected_output = ReinterpretBytesAsSpan<const bool>(raw_expected_output);
      Span<const bool> actual_output(ort_output.GetTensorData<bool>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      Span<const double> expected_output = ReinterpretBytesAsSpan<const double>(raw_expected_output);
      Span<const double> actual_output(ort_output.GetTensorData<double>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      Span<const uint32_t> expected_output = ReinterpretBytesAsSpan<const uint32_t>(raw_expected_output);
      Span<const uint32_t> actual_output(ort_output.GetTensorData<uint32_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
      Span<const uint64_t> expected_output = ReinterpretBytesAsSpan<const uint64_t>(raw_expected_output);
      Span<const uint64_t> actual_output(ort_output.GetTensorData<uint64_t>(), expected_output.size());
      GetAccuracy(expected_output, actual_output, metrics);
      break;
    }
    default:
      // Note: shouldn't get here because we've already validated expected output data types when loading model.
      std::cerr << "[ERROR]: Unsupported tensor element data type: " << output_info.data_type << std::endl;
      std::abort();
  }

  return metrics;
}

bool ModelIOInfo::Init(ModelIOInfo& model_info, Ort::ConstSession session) {
  Ort::AllocatorWithDefaultOptions allocator;

  // Get model input info (name, shape, type)
  {
    size_t num_inputs = session.GetInputCount();
    model_info.inputs.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
      Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        std::cerr << "[ERROR]: Only support models with tensor inputs" << std::endl;
        return false;
      }

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      IOInfo input_info;
      if (!IOInfo::Init(input_info, session.GetInputNameAllocated(i, allocator).get(),
                        tensor_info.GetElementType(), tensor_info.GetShape())) {
        std::cerr << "[ERROR]: Unsupported tensor element type (" << tensor_info.GetElementType()
                  << ") for input at index " << i << std::endl;
        return false;
      }

      model_info.inputs.push_back(std::move(input_info));
    }
  }

  // Get model output info (name, shape, type)
  {
    size_t num_outputs = session.GetOutputCount();
    model_info.outputs.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; i++) {
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        std::cerr << "[ERROR]: Only support models with tensor outputs" << std::endl;
        return false;
      }

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      IOInfo output_info;
      if (!IOInfo::Init(output_info, session.GetOutputNameAllocated(i, allocator).get(),
                        tensor_info.GetElementType(), tensor_info.GetShape())) {
        std::cerr << "[ERROR]: Unsupported tensor element type (" << tensor_info.GetElementType()
                  << ") for output at index " << i << std::endl;
        return false;
      }

      model_info.outputs.push_back(std::move(output_info));
    }
  }

  return true;
}

size_t ModelIOInfo::GetTotalInputSize() const {
  size_t total_size = 0;

  for (const auto& input_info : inputs) {
    total_size += input_info.total_data_size;
  }

  return total_size;
}

size_t ModelIOInfo::GetTotalOutputSize() const {
  size_t total_size = 0;

  for (const auto& output_info : outputs) {
    total_size += output_info.total_data_size;
  }

  return total_size;
}
