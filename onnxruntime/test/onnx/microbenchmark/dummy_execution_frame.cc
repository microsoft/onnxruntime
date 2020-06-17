// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dummy_execution_frame.h"

using namespace onnxruntime;
using namespace onnxruntime::common;

Status MyIExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_index,
                                                      const TensorShape* shape, size_t) {
  using T = float;
  if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
    return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
  }
  size_t size;
  int64_t len = shape->Size();
  if (len < 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  if (static_cast<uint64_t>(len) > std::numeric_limits<size_t>::max()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape is too large");
  }

  if (!IAllocator::CalcMemSizeForArrayWithAlignment<0>(static_cast<size_t>(len), sizeof(T), &size)) {
    return Status(ONNXRUNTIME, FAIL, "size overflow");
  }
  auto alloc = a_.GetAllocator(0, OrtMemTypeDefault);
  std::unique_ptr<Tensor> p_tensor = make_unique<Tensor>(DataTypeImpl::GetType<T>(), *shape, alloc);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  return Status::OK();
}
