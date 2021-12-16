// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ort_util.h"
#include <core/framework/ort_value.h>
#include <core/eager/ort_kernel_invoker.h>

namespace torch_ort {
namespace eager {

template <template<class> class V>
OrtValue reshape_copy(
  onnxruntime::ORTInvoker& invoker,
  const OrtValue& input,
  V<int64_t> shape) {
  // TODO: actual reshape on buffer
  const onnxruntime::Tensor& input_tensor = input.Get<onnxruntime::Tensor>();
  auto new_shape = at::infer_size(shape, input_tensor.Shape().Size());
  OrtValue shape_tensor;
  //todo: avoid the copy on this small shape vector;
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, {(int64_t)new_shape.size(),}, &shape_tensor);
  auto* ort_shape_tensor = shape_tensor.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>(invoker, new_shape, *ort_shape_tensor);
  std::vector<OrtValue> result(1);
  ORT_THROW_IF_ERROR(invoker.Invoke("Reshape", {input, shape_tensor}, result, nullptr));
  return result[0];
}

OrtValue add(onnxruntime::ORTInvoker& invoker,
             const OrtValue& A,
             const OrtValue& B);

void copy(onnxruntime::ORTInvoker& invoker, 
          const OrtValue& src, OrtValue& dst);

} // namespace eager
} // namespace torch_ort