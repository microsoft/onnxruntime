// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*);
#endif

void MyCustomKernel::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const float* X = ort_.GetTensorData<float>(input_X);
  const float* Y = ort_.GetTensorData<float>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
#ifdef USE_CUDA
  cuda_add(size, out, X, Y);
  cudaStreamSynchronize(nullptr);
#else
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
#endif
}

void MyCustomKernelMultipleDynamicInputs::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  // Even though this kernel backs an operator where-in both inputs can be any type and need not be homogeneous
  // as a proof-of-concept, support the case where-in the first input is of float type and the second input
  // is of double type. Users need to extend this logic to handle any arbitrary type should the need arise.
  const float* X = ort_.GetTensorData<float>(input_X);
  const double* Y = ort_.GetTensorData<double>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
#ifdef USE_CUDA
  cuda_add(size, out, X, Y);
  cudaStreamSynchronize(nullptr);
#else
  for (int64_t i = 0; i < size; i++) {
    out[i] = static_cast<float>(X[i] + Y[i]);
  }
#endif
}

void MyCustomKernelWithOptionalInput::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X1 = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_X2 = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* input_X3 = ort_.KernelContext_GetInput(context, 2);

  const float* X1 = ort_.GetTensorData<float>(input_X1);
  // The second input may or may not be present
  const float* X2 = (input_X2 != nullptr) ? ort_.GetTensorData<float>(input_X2) : nullptr;
  const float* X3 = ort_.GetTensorData<float>(input_X3);

  // Setup output
  int64_t output_dim_value = 1;
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, &output_dim_value, 1);
  float* out = ort_.GetTensorMutableData<float>(output);

  // Only CPU EP is supported in this kernel
  for (int64_t i = 0; i < output_dim_value; i++) {
    out[i] = X1[i] + (X2 != nullptr ? X2[i] : 0) + X3[i];
  }
}
