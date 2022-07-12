// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/core/ivalue.h>
#include "core/framework/ortdevice.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {
namespace lazytensor {
// Scalar type translation from ONNX to Pytorch.
c10::ScalarType CreateC10ScalarType(const onnxruntime::PrimitiveDataTypeBase* elem_type);
// Scalar type translation from Pytorch to ORT.
onnxruntime::MLDataType CreateOrtScalarType(at::ScalarType dtype);
// Device translation from Pytorch to ORT.
OrtDevice CreateOrtDevice(const c10::Device device);
// Device translation from ORT to Pytorch.
c10::Device CreateC10Device(const OrtDevice& device);
// Create a tensor from a Pytorch tensor. No memory copy.
// Conceptually, the returned tensor is a view of the input tensor.
OrtValue CreateOrtTensorValue(const at::Tensor& tensor);
// Similarly, create a Pytorch tensor from an OrtValue without
// memory copy.
// The created at::Tensor and onnxruntime::Tensor have
// the same lifetime.
c10::IValue CreateC10IvalueTensor(OrtValue value);
// Map Pytorch scalar to tensor with empty shape in ORT.
OrtValue CreateOrtScalarValue(const at::Scalar& scalar);
// Wrap ORT scalar as c10::IValue (a scalar).
c10::IValue CreateC10IvalueScalar(OrtValue value);
}  // namespace lazytensor
}  // namespace onnxruntime
