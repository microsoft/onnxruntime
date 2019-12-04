// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/codegen/passes/scheduler/schedule_utils.h"
#include "core/providers/nuphar/mti_x86/math/reduce_ops.h"
#include <topi/tags.h>

namespace onnxruntime {
namespace nuphar {

bool TVM_SCHEDULER_CLASS(Extern, NupharX86TVMRule)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  bool status = InsertRootScheduleAndClosure(tensor, ctx_sched);
  bool status_input = InputRootScheduleWithVectorizationX86(tensor, ctx_codegen, ctx_sched);
  return status || status_input;
}

static bool ReduceVScheduleNupharX86(
    const tvm::Tensor& tensor,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  InsertRootScheduleAndClosure(tensor, ctx_sched);

  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  if (compute_op->axis.size() > 1) {
    tvm::Expr fuse_dim_expr(compute_op->attrs[nuphar::kNupharVReduceFuseDim].node_);
    const int64_t* fuse_dim = as_const_int(fuse_dim_expr);
    ORT_ENFORCE(nullptr != fuse_dim);

    tvm::Array<tvm::IterVar> fused_axes;
    bool can_vectorize = true;
    for (size_t i = gsl::narrow_cast<size_t>(*fuse_dim); i < compute_op->axis.size(); ++i) {
      fused_axes.push_back(compute_op->axis[i]);
      if (tvm::as_const_int(tensor->shape[i]) == nullptr)
        can_vectorize = false;
    }

    tvm::IterVar fused_x;
    bool has_fused_axis = (fused_axes.size() >= 1 && can_vectorize);
    // currently there's an issue in tvm when fusing two symbolic dims
    // for simplicity, just disable fuse when no vectorize

    if (has_fused_axis) {
      ctx_sched.schedule[tensor->op].fuse(fused_axes, &fused_x);
      if (can_vectorize)
        ctx_sched.schedule[tensor->op].vectorize(fused_x);
    }

    auto shape = tensor->shape;
    const int64_t* head_dim = nullptr;
    if (shape.size() > 0)
      head_dim = as_const_int(shape[0]);

    bool try_parallel = true;

    // unroll packed reduce by checking head dim
    if (nullptr != head_dim) {
      // if head_dim is already fused, don't unroll
      // only unroll 1 < head_dim <=4
      if ((*fuse_dim != 0) && (*head_dim) <= 4 && (*head_dim) > 1) {
        tvm::Array<tvm::IterVar> reorder_axis;
        auto x0 = compute_op->axis[0];

        // handle fused axis if there is
        if (has_fused_axis) {
          for (size_t i = 1; i < gsl::narrow_cast<size_t>(*fuse_dim); ++i) {
            reorder_axis.push_back(compute_op->axis[i]);
          }
          reorder_axis.push_back(fused_x);
        } else {
          for (size_t i = 1; i < compute_op->axis.size(); ++i) {
            reorder_axis.push_back(compute_op->axis[i]);
          }
        }

        for (auto& k : compute_op->reduce_axis)
          reorder_axis.push_back(k);
        reorder_axis.push_back(x0);

        ctx_sched.schedule[tensor->op].reorder(reorder_axis);
        ctx_sched.schedule[tensor->op].unroll(x0);
        try_parallel = false;
      }
    }

    if (try_parallel) {
      TryParallelX86(tensor, *fuse_dim, ctx_codegen, ctx_sched);
    }
  } else if (compute_op->axis.size() > 0 &&
             tvm::as_const_int(tensor->shape[0]) != nullptr) {
    tvm::IterVar x = compute_op->axis[0];
    ctx_sched.schedule[tensor->op].vectorize(x);
  }

  if (compute_op->reduce_axis.size() > 1) {
    tvm::IterVar k;
    ctx_sched.schedule[tensor->op].fuse(compute_op->reduce_axis, &k);
  }

  return true;
}

// For Reduce Compute tvm::Tensor
bool TVM_SCHEDULER_CLASS(Reduce, NupharX86TVMRule)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  // respect topi::kCommReduce
  if (tensor->op->tag == topi::kCommReduce) {
    return InsertRootScheduleAndClosure(tensor, ctx_sched);
  }

  if (tensor->op->tag == nuphar::kNupharVReduce) {
    return ReduceVScheduleNupharX86(tensor, ctx_codegen, ctx_sched);
  }

  // unknown goes to InsertRootScheduleAndClosure
  return InsertRootScheduleAndClosure(tensor, ctx_sched);
}

}  // namespace nuphar
}  // namespace onnxruntime
