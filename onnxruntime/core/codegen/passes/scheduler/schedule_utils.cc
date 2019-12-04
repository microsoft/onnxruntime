// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/utils.h"
#include "core/codegen/passes/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

// Check the schedule of tensor
// If it has no compute_root, Insert compute_root to tensor, and record it to ctx.scheduled_tensors
bool InsertRootSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx) {
  auto it = ctx.scheduled_tensors.find(tensor->op.get());
  if (it != ctx.scheduled_tensors.end()) {
    if (it->second == ScheduleType::ScheduleClosure ||
        it->second == ScheduleType::ScheduleRoot) {
      return false;
    }
    it->second = ScheduleType::ScheduleRoot;
  } else {
    ctx.scheduled_tensors.insert(std::make_pair(tensor->op.get(), ScheduleType::ScheduleRoot));
  }
  ctx.schedule[tensor->op].compute_root();
  return true;
}

// Check the schedule of tensor
// If it is not labeled as closure, lable it.
bool InsertClosure(const tvm::Tensor& tensor,
                   ScheduleContext& ctx) {
  auto it = ctx.scheduled_tensors.find(tensor->op.get());
  if (it != ctx.scheduled_tensors.end()) {
    if (it->second == ScheduleType::ScheduleClosure)
      return false;
    it->second = ScheduleType::ScheduleClosure;
  } else {
    ctx.scheduled_tensors.insert(std::make_pair(tensor->op.get(), ScheduleType::ScheduleClosure));
  }
  return true;
}

// Combination of InsertRootSchedule and InsertClosure
bool InsertRootScheduleAndClosure(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx) {
  auto it = ctx.scheduled_tensors.find(tensor->op.get());
  if (it != ctx.scheduled_tensors.end()) {
    if (it->second == ScheduleType::ScheduleClosure) {
      return false;
    }
    it->second = ScheduleType::ScheduleClosure;
  } else {
    ctx.scheduled_tensors.insert(std::make_pair(tensor->op.get(), ScheduleType::ScheduleClosure));
  }
  ctx.schedule[tensor->op].compute_root();
  return true;
}

// Check precondition for vectorize schedule
bool ShouldTryVectorization(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx) {
  auto it = ctx.scheduled_tensors.find(tensor->op.get());
  if (it != ctx.scheduled_tensors.end()) {
    if (it->second > ScheduleType::ScheduleInline) {
      return false;
    }
  }
  return true;
}

// Check the schedule of tensor
// If it is not scheduled, try to vectorize it.
// Note TryVectorization has to use with compute_root.
// Therefore, there is a safty check of tensor's schedule
bool TryVectorization(
    const tvm::Tensor& tensor,
    int64_t natural_vector_size,
    ScheduleContext& ctx) {
  if (!ShouldTryVectorization(tensor, ctx))
    return false;

  auto shape = tensor->shape;
  auto rank = shape.size();
  if (rank < 1) {
    return false;
  }
  const int64_t* tail_dim = as_const_int(shape[rank - 1]);

  if (nullptr != tail_dim) {
    auto extern_op = tensor->op.as<tvm::ExternOpNode>();
    if (nullptr != extern_op) {
      return false;
    }

    auto compute_op = tensor->op.as<tvm::ComputeOpNode>();

    if (nullptr != compute_op) {
      auto axis = compute_op->axis;
      tvm::IterVar x = axis[rank - 1];
      if ((*tail_dim) > natural_vector_size) {
        if ((*tail_dim) % natural_vector_size != 0) {
          natural_vector_size = GCD<int64_t>(natural_vector_size, (*tail_dim));
        }

        if (natural_vector_size > 1) {
          tvm::IterVar xi, xo;
          ctx.schedule[tensor->op].split(x, static_cast<int32_t>(natural_vector_size), &xo, &xi);
          ctx.schedule[tensor->op].vectorize(xi);
          return true;
        }
      } else if (*tail_dim > 0) {
        // don't vectorize if dim is 0
        ctx.schedule[tensor->op].vectorize(x);
        return true;
      }
    }
  }
  return false;
}

// Check the schedule of tensor
// If it is not scheduled, try to add compute_inline on it.
// Note TryInlineSchedule cannot be used with compute_root.
// Therefore, there is a safty check of tensor's schedule.
bool TryInlineSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx) {
  auto it = ctx.scheduled_tensors.find(tensor->op.get());
  if (it != ctx.scheduled_tensors.end()) {
    if ((int)it->second < (int)ScheduleType::ScheduleInline) {
      ctx.schedule[tensor->op].compute_inline();
      it->second = ScheduleType::ScheduleInline;
      return true;
    } else {
      return false;
    }
  }
  ctx.schedule[tensor->op].compute_inline();
  ctx.scheduled_tensors.insert(std::make_pair(tensor->op.get(), ScheduleType::ScheduleInline));
  return true;
}

// Check the schedule of tensor's inputs, and call InsertRootSchedule for each of them
bool InputRootSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx) {
  bool status = false;
  for (auto& t : tensor->op->InputTensors()) {
    if (t->op->InputTensors().size() > 0) {
      bool status_root = InsertRootSchedule(t, ctx);
      status = status || status_root;
    }
  }
  return status;
}

// Check the schedule of tensor's inputs,
// and call InsertRootSchedule and TryVectorization for each of them
bool InputRootScheduleWithVectorization(
    const tvm::Tensor& tensor,
    int64_t natural_vector_size,
    ScheduleContext& ctx) {
  bool status = false;
  for (auto& t : tensor->op->InputTensors()) {
    if (t->op->InputTensors().size() > 0) {
      bool status_vec = TryVectorization(t, natural_vector_size, ctx);
      bool status_root = InsertRootSchedule(t, ctx);
      status = status || status_root || status_vec;
    }
  }
  return status;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
