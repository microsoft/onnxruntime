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

} // namespace eager
} // namespace torch_ort