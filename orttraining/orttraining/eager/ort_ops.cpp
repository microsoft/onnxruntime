// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_ops.h"
#include "ort_util.h"
#include "ort_log.h"

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

template <template<class> class V>
void createInplaceOutputValue(OrtValue& input, V<int64_t> shape, OrtValue* p_mlvalue){
  auto* input_ort_tensor = input.GetMutable<onnxruntime::Tensor>();
  // the ort TensorShape class only accept std::vector, so have to conversion.
  std::vector<int64_t> new_shape;
  new_shape.assign(shape.begin(), shape.end());
  CreateMLValue(input_ort_tensor->MutableDataRaw(),
                input_ort_tensor->DataType(), new_shape, p_mlvalue);
}

template <typename T> 
using Vector = std::vector<T, std::allocator<T>>;

template <>
void createInplaceOutputValue<Vector>(OrtValue& input, Vector<int64_t> shape, OrtValue* p_mlvalue){
  auto* input_ort_tensor = input.GetMutable<onnxruntime::Tensor>();
  CreateMLValue(input_ort_tensor->MutableDataRaw(),
                input_ort_tensor->DataType(), shape, p_mlvalue);
}

template void createInplaceOutputValue<c10::ArrayRef>(OrtValue& input, c10::ArrayRef<int64_t> shape, OrtValue* p_mlvalue);

} // namespace eager
} // namespace torch_ort