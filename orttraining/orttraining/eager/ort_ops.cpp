// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_ops.h"
#include "ort_util.h"
#include "ort_log.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace torch_ort {
namespace eager {

void copy(onnxruntime::ORTInvoker& invoker,
          const OrtValue& src, OrtValue& dst){
  auto& ort_ep = invoker.GetCurrentExecutionProvider();
  const auto& src_tensor = src.Get<onnxruntime::Tensor>();
  auto* dst_tensor = dst.GetMutable<onnxruntime::Tensor>();
  if (!dst_tensor)
    throw std::runtime_error("ORT copy: dst is not a tensor");
  ORT_THROW_IF_ERROR(ort_ep.GetDataTransfer()->CopyTensor(src_tensor, *dst_tensor));
}

template <template <class> class V>
void createInplaceOutputValue(OrtValue& input, V<int64_t> shape, OrtValue* p_mlvalue) {
  auto* input_ort_tensor = input.GetMutable<onnxruntime::Tensor>();
  onnxruntime::TensorShapeVector target_shape{shape.begin(), shape.begin() + shape.size()};
  onnxruntime::ReshapeHelper helper(input.Get<onnxruntime::Tensor>().Shape(), target_shape);
  onnxruntime::TensorShape new_shape(target_shape);
  onnxruntime::Tensor::InitOrtValue(input_ort_tensor->DataType(), new_shape, input_ort_tensor->MutableDataRaw(),
                                    input_ort_tensor->Location(), *p_mlvalue);
}

template void createInplaceOutputValue<c10::ArrayRef>(OrtValue& input, c10::ArrayRef<int64_t> shape, OrtValue* p_mlvalue);

template <typename T>
using Vector = std::vector<T, std::allocator<T>>;
template void createInplaceOutputValue<Vector>(OrtValue& input, Vector<int64_t> shape, OrtValue* p_mlvalue);

} // namespace eager
} // namespace torch_ort
