// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#include "orttraining/training_ops/cpu/triton/triton_op.h"

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel_context_internal.h"
#endif
#include "orttraining/core/framework/triton/triton_op_executor.h"

namespace onnxruntime {
namespace contrib {

InlinedHashSet<size_t> TritonOp::GetBoolOutputs(size_t output_size) const {
  InlinedHashSet<size_t> bool_outputs;
  for (size_t i = 0; i < output_size; ++i) {
    ORT_ENFORCE(i < Node().OutputDefs().size(), "Output index out of range.");
    if (Node().OutputDefs()[i]->Exists() && Node().OutputDefs()[i]->TypeAsProto()->tensor_type().elem_type() ==
                                                ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL) {
      bool_outputs.insert(i);
    }
  }
  return bool_outputs;
}

Status TritonOp::Compute(OpKernelContext* context) const {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  size_t input_size = static_cast<size_t>(p_ctx_internal->InputCount());
  size_t output_size = static_cast<size_t>(p_ctx_internal->OutputCount());
  InlinedVector<const OrtValue*> inputs;
  for (size_t i = 0; i < input_size; ++i) {
    inputs.emplace_back(p_ctx_internal->GetInputMLValue(static_cast<int>(i)));
  }
  InlinedVector<OrtValue> outputs;
  InlinedHashSet<size_t> bool_outputs = GetBoolOutputs(output_size);
  auto& executor = training::framework::triton::TritonOpExecutor::Instance();
  if (func_name_ != "") {
    executor.ExecuteByFuncName(func_name_, inputs, outputs, bool_outputs, kwargs_);
  } else {
    executor.ExecuteByOnnx(onnx_key_, onnx_string_, inputs, outputs, bool_outputs);
  }
  ORT_ENFORCE(output_size == outputs.size());
  for (size_t i = 0; i < output_size; ++i) {
    if (Node().OutputDefs()[i]->Exists()) {
      ORT_THROW_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), outputs[i]));
    }
  }
  return Status::OK();
}

bool IsTritonOpExecutorInitialized() {
  return training::framework::triton::TritonOpExecutor::Instance().IsInitialized();
}

Status ExecuteTritonOpByFuncName(OpKernelContext* p_ctx, const std::string& func_name, size_t input_count,
                                 size_t output_count,
                                 const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs) {
  auto* p_ctx_internal = reinterpret_cast<OpKernelContextInternal*>(p_ctx);
  InlinedVector<const OrtValue*> inputs;
  for (size_t i = 0; i < input_count; ++i) {
    inputs.emplace_back(p_ctx_internal->GetInputMLValue(static_cast<int>(i)));
  }
  for (size_t i = 0; i < output_count; ++i) {
    inputs.emplace_back(p_ctx_internal->GetOutputMLValue(static_cast<int>(i)));
  }
  InlinedVector<OrtValue> outputs;
  training::framework::triton::TritonOpExecutor::Instance().ExecuteByFuncName(func_name, inputs, outputs, {}, kwargs);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
