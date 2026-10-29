// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_qlora.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/safeint.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

Status GenerateQLoRADown(ShaderHelper& shader, const ShaderVariableHelper& x, bool acc_f32,
                         bool use_subgroups) {
  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_qlora_down.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32),
                             WGSL_TEMPLATE_PARAMETER(use_subgroups, use_subgroups),
                             WGSL_TEMPLATE_VARIABLE(x, x));
}

Status GenerateQLoRAUp(ShaderHelper& shader, bool acc_f32, bool vector_output) {
  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_qlora_up.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32),
                             WGSL_TEMPLATE_PARAMETER(vector_output, vector_output));
}

namespace {

// Single-row execution stores the base projection separately. This shader computes
// the rank-32 up projection and adds it to that base output in one dispatch.
class QLoRAOutputProgram final : public Program<QLoRAOutputProgram> {
 public:
  explicit QLoRAOutputProgram(bool acc_f32) : Program{"QLoRAOutputVec4"}, acc_f32_(acc_f32) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLoRAOutputProgram);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"M", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("base");
    shader.AddInput("lora_t");
    shader.AddInput("lora_b");
    shader.AddInput("lora_b_scales");
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    ORT_RETURN_IF_ERROR(GenerateQLoRAUp(shader, acc_f32_));
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.size) { return; }\n"
        << "output[global_idx] = base[global_idx] + qlora_delta(global_idx / uniforms.N, global_idx % uniforms.N);\n";
    return Status::OK();
  }

 private:
  bool acc_f32_;
};

}  // namespace

Status ApplyQLoRAOutput(onnxruntime::webgpu::ComputeContext& context, const Tensor* base, const Tensor* t,
                        const Tensor* b, const Tensor* scales, Tensor* y) {
  const auto size = narrow<uint32_t>(y->Shape().Size());
  QLoRAOutputProgram program(context.EnableMatmulFp32Accumulation());
  program
      .AddInputs({{base, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten},
                  {t, ProgramTensorMetadataDependency::Type, 4},
                  {b, ProgramTensorMetadataDependency::Type, 16},
                  {scales, ProgramTensorMetadataDependency::Type}})
      .AddOutput({y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten})
      .AddUniformVariables(
          {{size}, {narrow<uint32_t>(b->Shape()[0])}, {narrow<uint32_t>(y->Shape()[y->Shape().NumDimensions() - 2])}})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize(CeilDiv(size, 64u))
      .CacheHint(context.EnableMatmulFp32Accumulation());
  return context.RunProgram(program);
}

class MatMulNBitsQLoRA final : public WebGpuKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsQLoRA);
  explicit MatMulNBitsQLoRA(const OpKernelInfo& info)
      : WebGpuKernel(info),
        k_(info.GetAttr<int64_t>("K")),
        n_(info.GetAttr<int64_t>("N")),
        accuracy_level_(info.GetAttrOrDefault<int64_t>("accuracy_level", 0)) {
    ORT_ENFORCE(info.GetAttr<int64_t>("bits") == 4 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("lora_bits") == 8 &&
                    info.GetAttr<int64_t>("lora_block_size") == 32,
                "WebGPU MatMulNBitsQLoRA supports only bits=4, block_size=32, lora_bits=8, and lora_block_size=32.");
    ORT_ENFORCE(k_ > 0 && k_ % 32 == 0 && n_ > 0,
                "WebGPU MatMulNBitsQLoRA requires positive K divisible by 32 and positive N.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const auto* x = context.Input(0);
    const auto* w = context.Input(1);
    const auto* s = context.Input(2);
    const auto* aq = context.Input(3);
    const auto* as = context.Input(4);
    const auto* bq = context.Input(5);
    const auto* bs = context.Input(6);
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions() >= 2, "WebGPU MatMulNBitsQLoRA requires X rank at least two.");
    ORT_RETURN_IF_NOT(aq->Shape().NumDimensions() == 3 && bq->Shape().NumDimensions() == 3,
                      "Invalid quantized LoRA input ranks.");
    ORT_RETURN_IF_NOT(
        w->Shape() == TensorShape({n_, k_ / 32, 16}) && s->Shape().Size() == SafeInt<int64_t>(n_) * (k_ / 32),
        "Invalid base weight/scales shape.");
    // TODO: Support arbitrary adapter ranks with a rank uniform, per-block up scales,
    // and vector-tail handling while preserving rank-32 performance.
    const auto rank_a = aq->Shape()[0];
    const auto blocks_b = bq->Shape()[1];
    ORT_RETURN_IF_NOT((rank_a == 0 || rank_a == 32) && (blocks_b == 0 || blocks_b == 1),
                      "WebGPU MatMulNBitsQLoRA supports only adapter rank 32 (or empty adapters).");
    ORT_RETURN_IF_NOT(aq->Shape()[1] == k_ / 32 &&
                          aq->Shape()[2] == 32 && bq->Shape()[0] == n_ && bq->Shape()[2] == 32,
                      "Invalid packed adapter dimensions.");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(x->Shape(), TensorShape({n_, k_}), false, true));
    auto* y = context.Output(0, helper.OutputShape());
    if (y->Shape().Size() == 0) return Status::OK();
    if (rank_a == 0 || blocks_b == 0) {
      return ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, y);
    }
    ORT_RETURN_IF_NOT(as->Shape() == TensorShape({32, k_ / 32}) && bs->Shape() == TensorShape({n_, 1}),
                      "Invalid adapter scales shape.");
    // With logical dequantized weights, Y = X * W + T * B_up, where T = X * A_down.
    // T keeps X's leading dimensions and replaces K with the adapter rank (32).
    auto t_shape = x->Shape().AsShapeVector();
    t_shape.back() = 32;
    auto t = context.CreateGPUTensor(x->DataType(), TensorShape(t_shape));
    const auto m = narrow<uint32_t>(x->Shape()[x->Shape().NumDimensions() - 2]);
    // Prefill fuses T * B_up into the base shader's output stage. Single-row
    // decode uses ApplyQLoRAOutput after computing the base projection and T.
    const bool fuse_up = m > 1;
    // FP32 may select DP4A even for M=1. The joint floating-point down
    // shader must not replace that base projection's activation quantization.
    const bool base_dp4a = CanApplyDP4AMatrixMatMulNBits(context, accuracy_level_, 32, narrow<uint32_t>(n_),
                                                         narrow<uint32_t>(k_), 4, m, false, y);
    std::optional<SubgroupMatrixConfig> base_subgroup_matrix_config;
    const auto subgroup_m = narrow<uint32_t>(helper.M());
    const bool base_subgroup_matrix =
        fuse_up && CanApplySubgroupMatrixMatMulNBits(
                       context, accuracy_level_, 32, narrow<uint32_t>(helper.OutputOffsets().size()),
                       narrow<uint32_t>(n_), narrow<uint32_t>(k_), 4, x->IsDataType<MLFloat16>(),
                       base_subgroup_matrix_config, subgroup_m, false);
    if (base_subgroup_matrix && base_subgroup_matrix_config->Is(8, 16, 16)) {
      const auto down_config = SelectSubgroupMatrixConfig(
          context, {{wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F32,
                     8, 16, 16, 32, true}});
      if (down_config.has_value()) {
        return ApplySubgroupMatrixMatMulNBitsQLoRA(
            x, w, s, aq, as, bq, bs, subgroup_m, narrow<uint32_t>(n_), narrow<uint32_t>(k_),
            *base_subgroup_matrix_config, *down_config, context, &t, y);
      }
    }
    if (!fuse_up && m == 1 && x->IsDataType<float>() && base_dp4a && k_ >= 2048 && k_ <= 8192) {
      auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
      ORT_RETURN_IF_ERROR(ApplyDP4AMatrixMatMulNBits(
          x, w, s, nullptr, nullptr, narrow<uint32_t>(x->Shape().Size() / k_), 1, 1, narrow<uint32_t>(n_),
          narrow<uint32_t>(k_), 32, narrow<uint32_t>(k_ / 32), kMinMForTileOptimization, 4, context, &base, 0, nullptr,
          nullptr, nullptr, nullptr, aq, &t, nullptr, as));
      return ApplyQLoRAOutput(context, &base, &t, bq, bs, y);
    }
    // For one input vector, append four rank-reduction workgroups to the generic
    // base dispatch. GenerateQLoRADown supplies their shader body; DP4A has its own.
    const bool joint_down = !fuse_up && x->Shape().Size() == k_ && k_ >= 2048 && k_ <= 8192 && !base_dp4a;
    if (joint_down) {
      auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
      ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, &base, 0,
                                           nullptr, 0, nullptr, nullptr, nullptr, aq, &t, nullptr, as));
      return ApplyQLoRAOutput(context, &base, &t, bq, bs, y);
    }
    // FP32 small-M prefill can use the existing DP4A down kernel and share
    // quantized X with the base projection when accuracy_level permits it.
    const auto down_accuracy =
        fuse_up &&
                (m >= kMinMForTileOptimization || (x->IsDataType<float>() && base_dp4a))
            ? accuracy_level_
            : 0;
    DP4AQuantizedInput quantized_x;
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, aq, as, nullptr, nullptr, k_, 32, 32, down_accuracy, 8, context, &t, 0,
                                         nullptr, 0, &quantized_x));
    if (fuse_up) {
      return ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, y, 0, nullptr, 0,
                              &quantized_x, &t, bq, nullptr, nullptr, bs);
    }
    auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, &base, 0,
                                         nullptr, 0, &quantized_x));
    return ApplyQLoRAOutput(context, &base, &t, bq, bs, y);
  }

 private:
  int64_t k_;
  int64_t n_;
  int64_t accuracy_level_;
};

ONNX_OPERATOR_KERNEL_EX(MatMulNBitsQLoRA, kMSDomain, 1, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", WebGpuSupportedFloatTypes())
                            .TypeConstraint("TQ", DataTypeImpl::GetTensorType<uint8_t>()),
                        MatMulNBitsQLoRA);

}  // namespace onnxruntime::contrib::webgpu
