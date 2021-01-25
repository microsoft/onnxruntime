// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
void cuda_add(int64_t, float*, const float*, const float*);
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
#else
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
#endif
}
