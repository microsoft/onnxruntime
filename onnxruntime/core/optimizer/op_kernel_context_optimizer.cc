// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "core/optimizer/op_kernel_context_optimizer.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

MLDataType OpKernelContextOptimizer::InputType(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  auto& mlvalue_name = kernel_->Node().InputDefs()[index]->Name();
  const MLValue* p_ml_value = optimizer_.GetNodeInputOrOutputMLValue(mlvalue_name);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

MLDataType OpKernelContextOptimizer::OutputType(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  auto& mlvalue_name = kernel_->Node().OutputDefs()[index]->Name();
  const MLValue* p_ml_value = optimizer_.GetNodeInputOrOutputMLValue(mlvalue_name);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

Tensor* OpKernelContextOptimizer::Output(int index, const TensorShape& shape) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's creating a tensor in the given position with given shape.
  MLValueAllocationParameters parameters{&shape};

  MLValue* p_value = nullptr;
  const NodeArg* node_arg = kernel_->Node().OutputDefs()[index];
  ORT_ENFORCE(optimizer_.GetOrCreateNodeOutputMLValue(node_arg, parameters, p_value).IsOK());
  return p_value ? p_value->GetMutable<Tensor>() : nullptr;
}

Status OpKernelContextOptimizer::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = optimizer_.GetAllocator();
  if (!*output)
    return Status(common::ONNXRUNTIME, common::FAIL, "TempSpace allocator not found");
  return Status::OK();
}

const MLValue* OpKernelContextOptimizer::GetInputMLValue(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  auto& mlvalue_name = kernel_->Node().InputDefs()[index]->Name();
  return optimizer_.GetNodeInputOrOutputMLValue(mlvalue_name);
}

MLValue* OpKernelContextOptimizer::GetOutputMLValue(int index) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  auto& mlvalue_name = kernel_->Node().OutputDefs()[index]->Name();
  return optimizer_.GetMutableNodeInputOrOutputMLValue(mlvalue_name);
}

Status OpKernelContextOptimizer::GetOrCreateOutputMLValue(int index, MLValue*& p_value) {
  const NodeArg* node_arg = kernel_->Node().OutputDefs()[index];
  MLValueAllocationParameters parameters;
  ORT_ENFORCE(optimizer_.GetOrCreateNodeOutputMLValue(node_arg, parameters, p_value).IsOK());
  return Status::OK();
}

}  // namespace onnxruntime
