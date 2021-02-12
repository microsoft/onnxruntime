// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*);
#endif

void MyCustomKernel::Compute(OrtKernelContext* context) {
  // The kernel compute's logic handles the following cases:
  // 1) First input: float, Second input: float, Output: float
  // 2) First input: float, second input: double, Output: float
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);

  const float* X = ort_.GetTensorData<float>(input_X);

  const float* Y_float = nullptr;
  const double* Y_double = nullptr;
  OrtTensorTypeAndShapeInfo* input_Y_info = ort_.GetTensorTypeAndShape(input_Y);
  ONNXTensorElementDataType input_Y_type = ort_.GetTensorElementType(input_Y_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_Y_info);

  if (input_Y_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Y_float = ort_.GetTensorData<float>(input_Y);
  } else if (input_Y_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    Y_double = ort_.GetTensorData<double>(input_Y);
  }

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

// Do computation
#ifdef USE_CUDA
  if (Y_float) {
    cuda_add<float, float, float>(size, out, X, Y_float);
  } else if (Y_double) {
    cuda_add<float, double, float>(size, out, X, Y_double);
  }
  cudaStreamSynchronize(nullptr);
#else
  if (Y_float) {
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y_float[i];
    }
  } else if (Y_double) {
    for (int64_t i = 0; i < size; i++) {
      out[i] = static_cast<float>(X[i] + Y_double[i]);
    }
  }
#endif
}
