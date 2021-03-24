// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_op_utils.h"
#include "core/common/common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

template <typename T>
void cuda_slice(const T*, int64_t, int64_t, T*, cudaStream_t compute_stream);
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
  // Launch on stream 0 or user provided stream
  cuda_add(size, out, X, Y, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
  // If everything is setup correctly, custom op implementations need not have such synchronization logic.
  // To make sure custom ops and ORT CUDA kernels are implicitly synchronized, create your session with a compute stream
  // passed in via SessionOptions and use the same compute stream ti launch the custom op (as shown in this example)
  // cudaStreamSynchronize(nullptr);
#else
  ORT_UNUSED_PARAMETER(compute_stream_);
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
  // Launch on stream 0 or user provided stream
  cuda_add(size, out, X, Y, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
  // If everything is setup correctly, custom op implementations need not have such synchronization logic.
  // To make sure custom ops and ORT CUDA kernels are implicitly synchronized, create your session with a compute stream
  // passed in via SessionOptions and use the same compute stream ti launch the custom op (as shown in this example)
  // cudaStreamSynchronize(nullptr);
#else
  ORT_UNUSED_PARAMETER(compute_stream_);
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

void MyCustomKernelWithAttributes::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // This kernel only supports CPU EP
  if (string_arr_ == "add") {  // Test that the string attribute parsing went correctly
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] +
               float_attr_ + static_cast<float>(int_attr_) +
               floats_attr_[0] + floats_attr_[1] +
               static_cast<float>(ints_attr_[0]) + static_cast<float>(ints_attr_[1]);
    }
  } else {  // if the string attribute parsing had not gone correctly - it will trigger this path and fail the test due to result mis-match
    for (int64_t i = 0; i < size; i++) {
      out[i] = 0.f;
    }
  }
}

template <typename T>
static void custom_slice(const T* X, int64_t from, int64_t to, T* Y, void* compute_stream) {
#ifdef USE_CUDA
  // Launch on stream 0 or user provided stream
  cuda_slice(X, from, to, Y, compute_stream == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream));
  // If everything is setup correctly, custom op implementations need not have such synchronization logic.
  // To make sure custom ops and ORT CUDA kernels are implicitly synchronized, create your session with a compute stream
  // passed in via SessionOptions and use the same compute stream ti launch the custom op (as shown in this example)
  // cudaStreamSynchronize(nullptr);
#else
  ORT_UNUSED_PARAMETER(compute_stream);
  for (auto i = from; i < to; i++) {
    Y[i - from] = X[i];
  }
#endif
}

void SliceCustomOpKernel::Compute(OrtKernelContext* context) {
  // Setup inputs and outputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_from = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* input_to = ort_.KernelContext_GetInput(context, 2);
  OrtTensorTypeAndShapeInfo* input_X_info = ort_.GetTensorTypeAndShape(input_X);
  ONNXTensorElementDataType input_X_type = ort_.GetTensorElementType(input_X_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_X_info);
#if USE_CUDA
  int64_t slice_from = 0;
  int64_t slice_to = 0;
  cudaMemcpy(&slice_from, ort_.GetTensorData<int64_t>(input_from), sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&slice_to, ort_.GetTensorData<int64_t>(input_to), sizeof(int64_t), cudaMemcpyDeviceToHost);
#else
  int64_t slice_from = *ort_.GetTensorData<int64_t>(input_from);
  int64_t slice_to = *ort_.GetTensorData<int64_t>(input_to);
#endif
  std::vector<int64_t> output_dims = {slice_to - slice_from};
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
  // do slice
  switch (input_X_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:

      custom_slice(ort_.GetTensorData<float>(input_X), slice_from, slice_to,
                   ort_.GetTensorMutableData<float>(output), compute_stream_);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      custom_slice(ort_.GetTensorData<double>(input_X), slice_from, slice_to,
                   ort_.GetTensorMutableData<double>(output), compute_stream_);
      break;
    default:
      ORT_THROW("Unsupported input type");
  }
}
