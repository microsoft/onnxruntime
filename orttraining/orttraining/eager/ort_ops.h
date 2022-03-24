// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ort_util.h"
#include <core/framework/ort_value.h>
#include <core/eager/ort_kernel_invoker.h>

namespace torch_ort {
namespace eager {

template <template<class> class V>
void createInplaceOutputValue(OrtValue& input, V<int64_t> shape, OrtValue* p_mlvalue);

template <template<class> class V>
OrtValue reshape_invoke(
  onnxruntime::ORTInvoker& invoker,
  OrtValue& input,
  V<int64_t> shape,
  bool in_place) {
  // the ort reshape kernel already handle the -1 in target shape
  // don't need to invoke at::infer_size here.
  OrtValue shape_tensor;
  // todo: avoid the copy on this small shape vector;
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  onnxruntime::Tensor::InitOrtValue(element_type, onnxruntime::TensorShape({(int64_t)shape.size()}),
                                    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                                    shape_tensor);
  auto* ort_shape_tensor = shape_tensor.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>(invoker, shape.data(), shape.size(), *ort_shape_tensor);
  std::vector<OrtValue> result(1);
  if (in_place) {
    createInplaceOutputValue(input, shape, &result[0]);
  }
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