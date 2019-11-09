// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/utils.h"
#include "core/common/cpuid_info.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_16bit.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_8bit.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemm_8bit.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/codegen/passes/scheduler/schedule_utils.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include <tvm/ir_pass.h>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

constexpr int bits_per_byte = 8;

static Status TensorizeGEMVInteger16(const tvm::Tensor& tensor,
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

  TensorizeIntGemv16bit igemv16bit("igemv16bit", kernel_shape);
  auto shape = igemv16bit.Shape();
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
  ctx.schedule[tensor->op].tensorize(yi, igemv16bit.CreateTensorIntrin());

  return Status::OK();
}

// TODO: refactor below function
static Status TensorizeGEMVInteger(const tvm::Tensor& tensor,
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

  TensorizeIntGemv8bit igemv8bit("igemv8bit", kernel_shape);
  auto shape = igemv8bit.Shape();
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
  ctx.schedule[tensor->op].tensorize(yi, igemv8bit.CreateTensorIntrin());

  return Status::OK();
}

static Status TensorizeIGEMV(const tvm::Tensor& tensor,
                             tvm_codegen::ScheduleContext& ctx,
                             bool tensorize,
                             const std::string& target_str) {
  // Schedule tensor and inputs as root
  bool status_imatmul = InsertRootScheduleAndClosure(tensor, ctx);
  if (status_imatmul == false)
    return Status::OK();
  InputRootScheduleWithVectorizationX86(tensor, ctx);

  // Default tiling size
  // TODO: tuning tiling sizes later
  int tensorize_embed = 1;
  int tensorize_input = (target_str == "avx512-skylake") ? 1024 : (target_str == "avx2") ? 512 : 256;

  // Tensorize kernel shape
  std::vector<int32_t> kernel_shape;
  kernel_shape.push_back(tensorize_embed);
  kernel_shape.push_back(tensorize_input);

  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  auto xy = compute_op->axis;
  auto x = xy[0];
  auto y = xy[1];
  auto z = compute_op->reduce_axis[0];

  // no tiling need for IterVar x
  tvm::IterVar yo, yi;
  ctx.schedule[tensor->op].split(y, kernel_shape[0], &yo, &yi);
  tvm::IterVar zo, zi;
  ctx.schedule[tensor->op].split(z, kernel_shape[1], &zo, &zi);
  ctx.schedule[tensor->op].reorder({x, yo, zo, yi, zi});

  if (tensorize) {
    // TODO: refine tensorize gemv class
    TensorizeIntGemv8bit igemv8bit("igemv8bit", kernel_shape);
    ctx.schedule[tensor->op].tensorize(yi, igemv8bit.CreateTensorIntrin());
  }
  return Status::OK();
}

static Status TensorizeIGEMM(const tvm::Tensor& tensor,
                             tvm_codegen::CodeGenContext& ctx_codegen,
                             tvm_codegen::ScheduleContext& ctx,
                             tvm::Expr batchseq_expr,
                             const std::vector<int64_t> embed_dim_vec,
                             const std::vector<int64_t> input_dim_vec,
                             const std::string& target_str) {
  // Schedule tensor and inputs as root
  bool status_imatmul = InsertRootScheduleAndClosure(tensor, ctx);
  if (status_imatmul == false)
    return Status::OK();
  InputRootScheduleWithVectorizationX86(tensor, ctx);

  // Default tiling size
  int tensorize_batch = 4;
  int tensorize_embed = 8;
  int tensorize_input = 32;
  if (target_str == "avx512-skylake") {
    tensorize_batch = 8;
    tensorize_embed = 16;
    tensorize_input = 64;
  } else if (target_str == "avx2") {
    tensorize_batch = 8;
    tensorize_embed = 16;
    tensorize_input = 32;
  } else if (target_str == "avx") {
    tensorize_batch = 2;
    tensorize_embed = 4;
    tensorize_input = 8;
  }

  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  if (settings.HasOption(kNupharTensorize_IGEMM_Tile_M) &&
      settings.HasOption(kNupharTensorize_IGEMM_Tile_N) &&
      settings.HasOption(kNupharTensorize_IGEMM_Tile_K)) {
    std::string igemm_tile_M = settings.GetOptionValue(kNupharTensorize_IGEMM_Tile_M);
    std::string igemm_tile_N = settings.GetOptionValue(kNupharTensorize_IGEMM_Tile_N);
    std::string igemm_tile_K = settings.GetOptionValue(kNupharTensorize_IGEMM_Tile_K);

    tensorize_batch = std::stoi(igemm_tile_M);
    tensorize_embed = std::stoi(igemm_tile_N);
    tensorize_input = std::stoi(igemm_tile_K);
  }

  const int64_t* p_batchseq_dim = tvm::as_const_int(batchseq_expr);
  int embed_dim = gsl::narrow_cast<int>(embed_dim_vec[0]);
  int embed_dim_padded = gsl::narrow_cast<int>(embed_dim_vec[1]);
  int input_dim_padded = gsl::narrow_cast<int>(input_dim_vec[1]);

  // Guard tiling sizes
  if (p_batchseq_dim != nullptr) {
    tensorize_batch = std::min(tensorize_batch, gsl::narrow_cast<int>(*p_batchseq_dim));
  }
  int embed_min = 8;
  if (target_str == "avx512-skylake") {
    embed_min = 16;
  } else if (target_str == "avx2") {
    embed_min = 8;
  } else if (target_str == "avx") {
    embed_min = 4;
  }
  tensorize_embed = (tensorize_embed % embed_min != 0) ? ((tensorize_embed + embed_min - 1) / embed_min) * embed_min
                                                       : tensorize_embed;
  tensorize_embed = std::min(tensorize_embed, embed_dim_padded);
  tensorize_input = std::pow(2, std::ceil(std::log(tensorize_input) / std::log(2)));
  tensorize_input = std::min(std::max(4, tensorize_input), input_dim_padded);

  // Tensorize kernel shape
  std::vector<int32_t> kernel_shape;
  kernel_shape.push_back(tensorize_batch);
  kernel_shape.push_back(tensorize_embed);
  kernel_shape.push_back(tensorize_input);

  // Loop tiling
  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  auto xy = compute_op->axis;
  auto x = xy[0];
  auto y = xy[1];
  auto z = compute_op->reduce_axis[0];
  tvm::IterVar xo, xi;
  ctx.schedule[tensor->op].split(x, kernel_shape[0], &xo, &xi);
  tvm::IterVar yo, yi;
  ctx.schedule[tensor->op].split(y, kernel_shape[1], &yo, &yi);
  tvm::IterVar zo, zi;
  ctx.schedule[tensor->op].split(z, kernel_shape[2], &zo, &zi);

  // Loop nest permutation
  if (settings.HasOption(kNupharTensorize_IGEMM_Permute) &&
      (settings.OptionMatches(kNupharTensorize_IGEMM_Permute, kNupharTensorize_IGEMM_Permute_All) ||
       settings.OptionMatches(kNupharTensorize_IGEMM_Permute, kNupharTensorize_IGEMM_Permute_Outer))) {
    ctx.schedule[tensor->op].reorder({yo, xo, zo, xi, yi, zi});
  } else {
    // Loop nest default order
    if (target_str == "avx")
      ctx.schedule[tensor->op].reorder({yo, xo, zo, xi, yi, zi});
    else
      ctx.schedule[tensor->op].reorder({xo, yo, zo, xi, yi, zi});
  }

  // Natural vector width
  // AVX:    vector width 16 = 128 bits / 8 bits; 8  = 128 bits / 16bits;
  // AVX2:   vector width 32 = 256 bits / 8 bits; 16 = 256 bits / 16bits;
  // AVX512: vector width 64 = 512 bits / 8 bits; 32 = 512 bits / 16bits;
  CodeGenTargetX86* target = dynamic_cast<CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
  ORT_ENFORCE(target != nullptr, "CodeGen target unknown: not AVX/AVX2/AVX512 !");
  int tensor_bits = tensor->op->InputTensors()[1]->dtype.bits();
  int vector_width = target->NaturalVectorWidth(tensor_bits) / 2;

  // Layout shape
  int layout_tile_row = (sizeof(int32_t) * bits_per_byte) / tensor_bits;
  int layout_tile_col = ((vector_width * bits_per_byte) / tensor_bits) / layout_tile_row;

  // Tensorization configuration
  // IGEMM Tile M dimension config
  tvm::Expr batchseq_iter(xo);
  bool is_symbolic = (p_batchseq_dim == nullptr) ? true : false;
  tvm::Expr batchseq_last = (batchseq_expr + kernel_shape[0] - 1) / kernel_shape[0] - 1;
  TensorizeDimMeta batchseq_meta(batchseq_iter, batchseq_expr, kernel_shape[0], is_symbolic, (batchseq_iter == batchseq_last));

  // IGEMM Tile N dimension config
  tvm::Expr embed_iter(yo);
  tvm::Expr embed_last((embed_dim_padded + kernel_shape[1] - 1) / kernel_shape[1] - 1);
  bool embed_has_tail = (embed_dim % kernel_shape[1] != 0);
  int embed_tail_size = embed_dim % layout_tile_col;
  TensorizeDimMeta embed_meta(embed_iter, embed_dim, kernel_shape[1], layout_tile_col, embed_has_tail, embed_tail_size, (embed_iter == embed_last));

  // IGEMM Tile K dimension config
  tvm::Expr input_iter(zo);
  tvm::Expr load_shift = tvm::ir::Simplify(kernel_shape[2] / layout_tile_row * input_dim_padded - kernel_shape[2]);
  tvm::Expr load_offset = tvm::ir::Simplify((input_iter % (std::max(1, vector_width / kernel_shape[2]))) * load_shift);
  TensorizeDimMeta input_meta(input_iter, kernel_shape[2], layout_tile_row, load_offset);

  TensorizeIntGemm8bit igemm8bit("igemm8bit", kernel_shape, target_str);
  igemm8bit.InsertTensorizeDimInfo("m", batchseq_meta);
  igemm8bit.InsertTensorizeDimInfo("n", embed_meta);
  igemm8bit.InsertTensorizeDimInfo("k", input_meta);
  // Bind tensorization kernel
  ctx.schedule[tensor->op].tensorize(xi, igemm8bit.CreateTensorIntrin());

  return Status::OK();
}

static bool IMatMulTensorizeSchedule(
    const tvm::Tensor& imatmul,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  // marshalled matrix shape
  ORT_ENFORCE(imatmul->op->InputTensors().size() == 2);
  tvm::Tensor matrixA = imatmul->op->InputTensors()[1];
  ORT_ENFORCE(matrixA->shape.size() == 2);
  const int64_t* p_input_dim_padded = tvm::as_const_int(matrixA->shape[1]);
  const int64_t* p_embed_dim_padded = tvm::as_const_int(matrixA->shape[0]);
  ORT_ENFORCE(p_input_dim_padded != nullptr && p_embed_dim_padded != nullptr);

  // symoblic batch seq dimension
  tvm::Tensor matrixB = imatmul->op->InputTensors()[0];
  ORT_ENFORCE(matrixB->shape.size() == 2);
  tvm::Expr batchseq_expr = matrixB->shape[0];
  const int64_t* p_batchseq_dim = tvm::as_const_int(batchseq_expr);
  bool isGEMV = (p_batchseq_dim != nullptr && *p_batchseq_dim == 1);

  // original matrix shape
  ORT_ENFORCE(node->InputDefs().size() == 2);
  auto input1_shape = node->InputDefs()[1]->Shape();
  ORT_ENFORCE(input1_shape->dim_size() == 2);
  tvm::Expr input1_dim0 = ShapeDimToTvmDim(input1_shape->dim(0), ctx_codegen);
  tvm::Expr input1_dim1 = ShapeDimToTvmDim(input1_shape->dim(1), ctx_codegen);
  const int64_t* p_input_dim = tvm::as_const_int(input1_dim0);
  const int64_t* p_embed_dim = tvm::as_const_int(input1_dim1);
  ORT_ENFORCE(p_input_dim != nullptr && p_embed_dim != nullptr);

  // quantization bits
  bool is8bit = (matrixB->dtype == HalideIR::type_of<uint8_t>() &&
                 matrixA->dtype == HalideIR::type_of<int8_t>());
  bool is16bit = (matrixB->dtype == HalideIR::type_of<int16_t>() &&
                  matrixA->dtype == HalideIR::type_of<int16_t>());
  ORT_ENFORCE(is8bit || is16bit);

  // tvm has known issue when handling tensorization of matmul: [1x1] = [1xK]x[Kx1]
  // and this case is not likely happen in real model
  // so add option to fall back to a general reduction
  bool is_scalar = isGEMV && (*p_embed_dim == 1);

  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  TargetFeature feature = GetTargetInfo(settings);

  bool status_tensorize = true;
  if (is8bit) {
    if (feature.hasAVX512) {  // isAVX512
      status_tensorize = is_scalar ? TensorizeIGEMV(imatmul, ctx_sched, /*tensorize=*/false, "avx512-skylake").IsOK()
                                   : TensorizeIGEMM(imatmul, ctx_codegen, ctx_sched, batchseq_expr,
                                                    {*p_embed_dim, *p_embed_dim_padded},
                                                    {*p_input_dim, *p_input_dim_padded},
                                                    "avx512-skylake")
                                         .IsOK();
    } else if (feature.hasAVX2) {  // isAVX2
      ORT_ENFORCE(!is_scalar, "scalar AVX2 is not supported!");
      // TODO: release 8bit tensorize GEMV for AVX2
      status_tensorize = isGEMV ? TensorizeGEMVInteger(imatmul, *p_input_dim, ctx_sched).IsOK()
                                : TensorizeIGEMM(imatmul, ctx_codegen, ctx_sched, batchseq_expr,
                                                 {*p_embed_dim, *p_embed_dim_padded},
                                                 {*p_input_dim, *p_input_dim_padded},
                                                 "avx2")
                                      .IsOK();
    } else if (feature.hasAVX) {  // isAVX
      status_tensorize = is_scalar ? TensorizeIGEMV(imatmul, ctx_sched, /*tensorize=*/false, "avx").IsOK()
                                   : TensorizeIGEMM(imatmul, ctx_codegen, ctx_sched, batchseq_expr,
                                                    {*p_embed_dim, *p_embed_dim_padded},
                                                    {*p_input_dim, *p_input_dim_padded},
                                                    "avx")
                                         .IsOK();
    } else {
      ORT_NOT_IMPLEMENTED("Not supported target in 8bit Tensorization, should be one of avx/avx2/avx512.");
    }
  } else {  // 16bit
    // TODO: add 16bit tensorize GEMV/GEMM for AVX512
    if (feature.hasAVX2) {  //isAVX2
      // TODO: add 16bit tensorize GEMM for AVX2
      ORT_ENFORCE(isGEMV, "16bit GEMM is not supported!");
      // TODO: release 16bit tensorize GEMV for AVX2
      status_tensorize = TensorizeGEMVInteger16(imatmul, *p_input_dim, ctx_sched).IsOK();
    } else {
      ORT_NOT_IMPLEMENTED("Not supported target in 16bit Tensorization.");
    }
  }

  return status_tensorize;
}

bool TVM_SCHEDULER_CLASS(MatMulInteger, NupharX86Tensorize)::Evaluate(
    const tvm::Tensor& tensor,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  // schedule for reshape in MatMulInteger
  bool status_reshape = tvm_codegen::TryInlineSchedule(tensor, ctx_sched);
  // schedule for imatmul tensorization
  ORT_ENFORCE(tensor->op->InputTensors().size() > 0);
  auto imatmul = tensor->op->InputTensors()[0];
  if (imatmul->op->InputTensors().size() != 2) {
    status_reshape = status_reshape || tvm_codegen::TryInlineSchedule(imatmul, ctx_sched);
    imatmul = imatmul->op->InputTensors()[0];
  }
  bool status_tensorize = IMatMulTensorizeSchedule(imatmul, node, ctx_codegen, ctx_sched);

  return status_reshape || status_tensorize;
}

// TODO: enable 16 bit tensorization
bool TVM_SCHEDULER_CLASS(MatMulInteger16, NupharX86Tensorize)::Evaluate(
    const tvm::Tensor& tensor,
    const Node* node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  // schedule for reshape in MatMulInteger
  bool status_reshape = tvm_codegen::TryInlineSchedule(tensor, ctx_sched);
  // schedule for imatmul tensorization
  ORT_ENFORCE(tensor->op->InputTensors().size() > 0);
  auto imatmul = tensor->op->InputTensors()[0];
  if (imatmul->op->InputTensors().size() != 2) {
    status_reshape = status_reshape || tvm_codegen::TryInlineSchedule(imatmul, ctx_sched);
    imatmul = imatmul->op->InputTensors()[0];
  }
  bool status_tensorize = IMatMulTensorizeSchedule(imatmul, node, ctx_codegen, ctx_sched);

  return status_reshape || status_tensorize;
}

}  // namespace nuphar
}  // namespace onnxruntime
