// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_16bit.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_8bit.h"
#include "core/codegen/passes/scheduler/schedule_utils.h"
#include "core/framework/op_kernel_info.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

static Status IntMatMulTensorize16bit(const tvm::Tensor& tensor,
                                      const int64_t input_dim,
                                      tvm_codegen::ScheduleContext& ctx) {
  // schedule for imatmul inputs
  InsertRootScheduleAndClosure(tensor, ctx);
  InputRootScheduleWithVectorizationX86(tensor, ctx);

  // decide kernel shape
  std::vector<int32_t> kernel_shape;
  kernel_shape.push_back(1);
  if (input_dim <= 64) {
    kernel_shape.push_back(input_dim);
  } else {
    kernel_shape.push_back(16);
  }

  Gemv16bitTensorization gemv16bit_tensorization("gemv16bit_tensorization", kernel_shape);
  auto shape = gemv16bit_tensorization.Shape();
  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  auto xy = compute_op->axis;
  auto x = xy[0];
  auto y = xy[1];
  auto z = compute_op->reduce_axis[0];
  tvm::IterVar yo, yi;
  ctx.schedule[tensor->op].split(y, shape[0], &yo, &yi);
  tvm::IterVar zo, zi;
  ctx.schedule[tensor->op].split(z, shape[1], &zo, &zi);
  ctx.schedule[tensor->op].reorder({x, yo, zo, yi, zi});
  ctx.schedule[tensor->op].tensorize(yi, gemv16bit_tensorization.CreateTensorIntrin());

  return Status::OK();
}

static Status IntMatMulTensorize8bit(const tvm::Tensor& tensor,
                                     const int64_t input_dim,
                                     tvm_codegen::ScheduleContext& ctx) {
  // schedule for imatmul inputs
  InsertRootScheduleAndClosure(tensor, ctx);
  InputRootScheduleWithVectorizationX86(tensor, ctx);

  // decide kernel shape
  std::vector<int32_t> kernel_shape;
  kernel_shape.push_back(1);
  if (input_dim <= 256) {
    kernel_shape.push_back(input_dim);
  } else if (input_dim % 64 == 0) {
    kernel_shape.push_back(64);
  } else {
    kernel_shape.push_back(32);
  }

  Gemv8bitTensorization gemv8bit_tensorization("gemv8bit_tensorization", kernel_shape);
  auto shape = gemv8bit_tensorization.Shape();
  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  auto xy = compute_op->axis;
  auto x = xy[0];
  auto y = xy[1];
  auto z = compute_op->reduce_axis[0];
  tvm::IterVar yo, yi;
  ctx.schedule[tensor->op].split(y, shape[0], &yo, &yi);
  tvm::IterVar zo, zi;
  ctx.schedule[tensor->op].split(z, shape[1], &zo, &zi);
  ctx.schedule[tensor->op].reorder({x, yo, zo, yi, zi});
  ctx.schedule[tensor->op].tensorize(yi, gemv8bit_tensorization.CreateTensorIntrin());

  return Status::OK();
}

static bool IntMatMulTensorize(
    const tvm::Tensor& tensor,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  // schedule for MatMulInteger root: reshape
  bool reused = Promote<CodeGenUnitStats>(ctx_nuphar->GetGraphStats())->NodeUseCount(node) > 1;

  if (reused) {
    TryVectorizationX86(tensor, ctx_sched);
    InsertRootScheduleAndClosure(tensor, ctx_sched);
  } else {
    TryInlineSchedule(tensor, ctx_sched);
  }

  // schedule for MatMulInteger computation: imatmul
  ORT_ENFORCE(tensor->op->InputTensors().size() > 0);
  auto& imatmul = tensor->op->InputTensors()[0];

  ORT_ENFORCE(imatmul->op->InputTensors().size() == 2);
  tvm::Tensor matrixB = imatmul->op->InputTensors()[0];
  tvm::Tensor matrixA = imatmul->op->InputTensors()[1];

  bool is16bit = (matrixB->dtype == HalideIR::type_of<int16_t>() &&
                  matrixA->dtype == HalideIR::type_of<int16_t>());
  bool is8bit = (matrixB->dtype == HalideIR::type_of<uint8_t>() &&
                 matrixA->dtype == HalideIR::type_of<int8_t>());
  ORT_ENFORCE(is16bit || is8bit);

  ORT_ENFORCE(matrixA->shape.size() == 2);
  const int64_t* input_dim_ptr = tvm::as_const_int(matrixA->shape[1]);
  ORT_ENFORCE(input_dim_ptr != nullptr);
  int64_t input_dim = *input_dim_ptr;

  if (is8bit)
    return IntMatMulTensorize8bit(imatmul, input_dim, ctx_sched).IsOK();

  return IntMatMulTensorize16bit(imatmul, input_dim, ctx_sched).IsOK();
}

bool TVM_SCHEDULER_CLASS(MatMulInteger, NupharX86Tensorize)::Evaluate(
    const tvm::Tensor& tensor,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  return IntMatMulTensorize(tensor, node, ctx_codegen, ctx_sched);
}

bool TVM_SCHEDULER_CLASS(MatMulInteger16, NupharX86Tensorize)::Evaluate(
    const tvm::Tensor& tensor,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  return IntMatMulTensorize(tensor, node, ctx_codegen, ctx_sched);
}

}  // namespace nuphar
}  // namespace onnxruntime
