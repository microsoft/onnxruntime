// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>

#include "dummy_gemm.h"

#ifndef ORT_ENFORCE
#define ORT_ENFORCE(cond, ...) \
  if (!(cond)) ORT_CXX_API_THROW("Initialization failed.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
#endif

namespace Cpu {

void* CustomGemmOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char* CustomGemmOp::GetName() const { return op_name_; }

const char* CustomGemmOp::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

size_t CustomGemmOp::GetInputTypeCount() const { return 6; }

ONNXTensorElementDataType CustomGemmOp::GetInputType(size_t index) const {
  switch (index) {
    case 0:  // A
    case 1:  // B
      return ab_type_;
    case 2:  // C
      return c_type_;
    case 3:  // scale A
    case 4:  // scale B
    case 5:  // scale Y
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORT_CXX_API_THROW("Input index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

OrtCustomOpInputOutputCharacteristic CustomGemmOp::GetInputCharacteristic(size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    case 2:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    case 3:
    case 4:
    case 5:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    default:
      ORT_CXX_API_THROW("Input index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

size_t CustomGemmOp::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomGemmOp::GetOutputType(size_t index) const {
  // D, scale D
  switch (index) {
    case 0:
      return d_type_;
    default:
      ORT_CXX_API_THROW("Output index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

OrtCustomOpInputOutputCharacteristic CustomGemmOp::GetOutputCharacteristic(size_t index) const {
  switch (index) {
    case 0:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    default:
      ORT_CXX_API_THROW("Output index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

CustomGemmKernel::CustomGemmKernel(const OrtApi&, const OrtKernelInfo*) {}

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue& input, std::vector<int64_t>& shape, bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();
  ORT_ENFORCE(shape.size() == 2);
  if (swap) {
    std::swap(shape[0], shape[1]);
  }
  return t.GetElementType();
}

void CustomGemmKernel::Compute(OrtKernelContext* context) {
  // The function does nothing related to Gemm operator. It creates an output with the same dimensions as
  // the model used in the unit tests and fills it with the first integer.
  Ort::KernelContext ctx(context);

  auto n_inputs = ctx.GetInputCount();
  ORT_ENFORCE(n_inputs >= 2);
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);

  std::vector<int64_t> shape_A, shape_B;
  GetTypeAndShape(input_A, shape_A);
  GetTypeAndShape(input_B, shape_B);
  ORT_ENFORCE(shape_A.size() == 2);
  ORT_ENFORCE(shape_B.size() == 2);
  std::vector<int64_t> dimensions{shape_A[0], shape_B[1]};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  float* out = Y.GetTensorMutableData<float>();
  size_t end = static_cast<size_t>(dimensions[0] * dimensions[1]);
  for (size_t i = static_cast<size_t>(0); i < end; ++i) {
    out[i] = static_cast<float>(i);
  }
}

}  // namespace Cpu
