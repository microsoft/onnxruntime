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

namespace onnxruntime::contrib::webgpu {

void GenerateQLoRADown(ShaderHelper& shader, const std::string& base_groups, const std::string& x_value, bool acc_f32) {
  shader.AdditionalImplementation() << "alias qlora_acc_t = " << (acc_f32 ? "f32" : "f16") << ";\n"
                                    << R"(
var<workgroup> qlora_x_tile: array<vec4<f16>, 128>;
var<workgroup> qlora_partials: array<array<qlora_acc_t, 32>, 8>;
)";
  shader.MainFunctionBody() << "let qlora_base_groups = " << base_groups << ";\n"
                            << R"(
if (workgroup_idx >= qlora_base_groups) {
  let rank_base = (workgroup_idx - qlora_base_groups) * 8u;
  let lane = local_idx % 32u;
  let rank_local = local_idx / 32u;
  // Match the generic INT8 MatMulNBits tile, dot grouping and reduction order.
  for (var start = 0u; start < uniforms.K; start += 512u) {
    let k4 = start / 4u + local_idx;
    var xv = vec4<f16>(0);
    if (k4 < uniforms.K / 4u) { xv = )"
                            << x_value << R"(; }
    qlora_x_tile[local_idx] = xv;
    workgroupBarrier();
    for (var rank_offset = 0u; rank_offset < 8u; rank_offset += 4u) {
      let rank = rank_base + rank_offset + rank_local;
      let k = start + lane * 16u;
      if (k < uniforms.K) {
        let packed = lora_at[rank * (uniforms.K / 16u) + k / 16u];
        let scale = lora_a_scales[rank * (uniforms.K / 32u) + k / 32u];
        var sum = qlora_acc_t(0);
        for (var i = 0u; i < 4u; i++) {
          let bv = (vec4<f16>(unpack4xU8(packed[i])) - vec4<f16>(128)) * scale;
          sum += dot(vec4<qlora_acc_t>(qlora_x_tile[lane * 4u + i]), vec4<qlora_acc_t>(bv));
        }
        qlora_partials[rank_offset + rank_local][lane] += sum;
      }
    }
    workgroupBarrier();
  }
  if (local_idx < 8u) {
    var value = qlora_acc_t(0);
    for (var i = 0u; i < 32u; i++) { value += qlora_partials[local_idx][i]; }
    lora_down[rank_base + local_idx] = f16(value);
  }
  return;
}
)";
}

void GenerateQLoRAUp(ShaderHelper& shader, bool acc_f32, bool vector_output) {
  shader.AdditionalImplementation() << "alias qlora_acc_t = " << (acc_f32 ? "f32" : "f16") << ";\n"
                                    << R"(
fn qlora_weight(col: u32, r4: u32) -> vec4<f16> {
  let packed = lora_b[col * 2u + r4 / 4u][r4 % 4u];
  return (vec4<f16>(unpack4xU8(packed)) - vec4<f16>(128)) * lora_b_scales[col];
}
fn qlora_delta(row: u32, col: u32) -> f16 {
  var delta = qlora_acc_t(0);
  // The original K=32 INT8 up projection uses wide tiles at M>=4.
  if (uniforms.M >= 4u) {
    for (var r4 = 0u; r4 < 8u; r4 += 2u) {
      let a0 = lora_t[row * 8u + r4];
      let a1 = lora_t[row * 8u + r4 + 1u];
      delta += dot(vec4<qlora_acc_t>(a0), vec4<qlora_acc_t>(qlora_weight(col, r4))) +
               dot(vec4<qlora_acc_t>(a1), vec4<qlora_acc_t>(qlora_weight(col, r4 + 1u)));
    }
  } else {
    for (var block = 0u; block < 2u; block++) {
      var sum = qlora_acc_t(0);
      for (var i = 0u; i < 4u; i++) {
        let r4 = block * 4u + i;
        sum += dot(vec4<qlora_acc_t>(lora_t[row * 8u + r4]), vec4<qlora_acc_t>(qlora_weight(col, r4)));
      }
      delta += sum;
    }
    // The remaining generic-kernel partials are zero, including their sign.
    delta += qlora_acc_t(0);
  }
  return f16(delta);
}
)";
  if (vector_output) {
    shader.AdditionalImplementation() << R"(
fn qlora_delta4(row: u32, col: u32) -> vec4<f16> {
  var delta = vec4<qlora_acc_t>(0);
  for (var r4 = 0u; r4 < 8u; r4 += 2u) {
    let a0 = vec4<qlora_acc_t>(lora_t[row * 8u + r4]);
    let a1 = vec4<qlora_acc_t>(lora_t[row * 8u + r4 + 1u]);
    delta += vec4<qlora_acc_t>(
        dot(a0, vec4<qlora_acc_t>(qlora_weight(col, r4))),
        dot(a0, vec4<qlora_acc_t>(qlora_weight(col + 1u, r4))),
        dot(a0, vec4<qlora_acc_t>(qlora_weight(col + 2u, r4))),
        dot(a0, vec4<qlora_acc_t>(qlora_weight(col + 3u, r4)))) +
        vec4<qlora_acc_t>(
        dot(a1, vec4<qlora_acc_t>(qlora_weight(col, r4 + 1u))),
        dot(a1, vec4<qlora_acc_t>(qlora_weight(col + 1u, r4 + 1u))),
        dot(a1, vec4<qlora_acc_t>(qlora_weight(col + 2u, r4 + 1u))),
        dot(a1, vec4<qlora_acc_t>(qlora_weight(col + 3u, r4 + 1u))));
  }
  return vec4<f16>(delta);
}
)";
  }
}

namespace {

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
    shader.AddOutput("output");
    GenerateQLoRAUp(shader, acc_f32_);
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.size) { return; }\n"
        << "output[global_idx] = base[global_idx] + qlora_delta(global_idx / uniforms.N, global_idx % uniforms.N);\n";
    return Status::OK();
  }

 private:
  bool acc_f32_;
};

class QLoRAAddProgram final : public Program<QLoRAAddProgram> {
 public:
  QLoRAAddProgram() : Program{"QLoRAAdd"} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLoRAAddProgram);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("base");
    shader.AddInput("delta");
    shader.AddOutput("output");
    shader.MainFunctionBody()
        << "if (global_idx < uniforms.size) { output[global_idx] = base[global_idx] + delta[global_idx]; }\n";
    return Status::OK();
  }
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
    const auto mode = info.GetConfigOptions().GetConfigOrDefault("session.webgpu.lora_fusion_mode", "auto");
    ORT_ENFORCE(mode == "auto" || mode == "fuse_up" || mode == "fuse_down",
                "session.webgpu.lora_fusion_mode must be auto, fuse_down or fuse_up.");
    automatic_ = mode == "auto";
    fuse_up_ = mode == "fuse_up";
    ORT_ENFORCE(k_ > 0 && k_ % 32 == 0 && n_ > 0 && n_ % 4 == 0 && info.GetAttr<int64_t>("bits") == 4 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("lora_bits") == 8 &&
                    info.GetAttr<int64_t>("lora_block_size") == 32,
                "MatMulNBitsQLoRA requires INT4 base and INT8 adapters with block size 32.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const auto* x = context.Input(0);
    const auto* w = context.Input(1);
    const auto* s = context.Input(2);
    const auto* aq = context.Input(3);
    const auto* as = context.Input(4);
    const auto* bq = context.Input(5);
    const auto* bs = context.Input(6);
    ORT_RETURN_IF_NOT(
        x->Shape().NumDimensions() >= 2 && aq->Shape().NumDimensions() == 3 && bq->Shape().NumDimensions() == 3,
        "Invalid quantized LoRA input ranks.");
    ORT_RETURN_IF_NOT(
        w->Shape() == TensorShape({n_, k_ / 32, 16}) && s->Shape().Size() == SafeInt<int64_t>(n_) * (k_ / 32),
        "Invalid base weight/scales shape.");
    const auto rank_a = aq->Shape()[0];
    const auto blocks_b = bq->Shape()[1];
    ORT_RETURN_IF_NOT((rank_a == 0 || rank_a == 32) && (blocks_b == 0 || blocks_b == 1) && aq->Shape()[1] == k_ / 32 &&
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
    auto t_shape = x->Shape().AsShapeVector();
    t_shape.back() = 32;
    auto t = context.CreateGPUTensor(x->DataType(), TensorShape(t_shape));
    const auto m = narrow<uint32_t>(x->Shape()[x->Shape().NumDimensions() - 2]);
    const bool fuse_up = fuse_up_ || (automatic_ && m > 1);
    const bool joint_down = !fuse_up && x->Shape().Size() == k_ && k_ >= 2048 && k_ <= 8192 &&
                            context.AdapterInfo().vendor == std::string_view{"nvidia"};
    if (joint_down) {
      auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
      ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, &base, 0,
                                           nullptr, 0, nullptr, nullptr, aq, &t, true, nullptr, as));
      return ApplyQLoRAOutput(context, &base, &t, bq, bs, y);
    }
    // Allow activation quantization for prefill only when the model opts into it.
    const auto down_accuracy = fuse_up && m >= kMinMForTileOptimization ? accuracy_level_ : 0;
    DP4AQuantizedInput quantized_x;
    // Rank 32 exposes too few output tiles for the tiled DP4A kernel at these sizes.
    const bool small_m_down = fuse_up && m <= 4096 && k_ >= 2048 &&
                              context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
                              CanApplyDP4AMatrixMatMulNBits(context, down_accuracy, 32, 32,
                                                            narrow<uint32_t>(k_), 4, m, false, &t);
    if (small_m_down) {
      ORT_RETURN_IF_ERROR(ApplyDP4AMatrixMatMulNBits(
          x, aq, as, nullptr, nullptr, narrow<uint32_t>(x->Shape().Size() / k_ / m), m, m, 32,
          narrow<uint32_t>(k_), 32, narrow<uint32_t>(k_ / 32), kMinMForTileOptimization, 8, context, &t, 0,
          nullptr, nullptr, nullptr, nullptr, nullptr, false, nullptr, &quantized_x, 32));
    } else {
      ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, aq, as, nullptr, nullptr, k_, 32, 32, down_accuracy, 8, context, &t, 0,
                                           nullptr, 0, nullptr, nullptr, nullptr, nullptr, false, nullptr, nullptr, &quantized_x));
    }
    std::optional<SubgroupMatrixConfig> up_config;
    const bool subgroup_up =
        CanApplySubgroupMatrixMatMulNBits(context, 0, 32, narrow<uint32_t>(helper.OutputOffsets().size()),
                                          narrow<uint32_t>(n_), 32, 8, true, up_config, m);
    if (fuse_up && !subgroup_up) {
      return ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, y, 0, nullptr, 0, &t,
                              bq, nullptr, nullptr, true, bs, nullptr, &quantized_x);
    }
    auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, w, s, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, &base, 0,
                                         nullptr, 0, nullptr, nullptr, nullptr, nullptr, false, nullptr, nullptr, &quantized_x));
    if (!subgroup_up) return ApplyQLoRAOutput(context, &base, &t, bq, bs, y);
    // Keep the original up kernel on devices with a different matrix-accumulation path.
    auto delta = context.CreateGPUTensor(y->DataType(), y->Shape());
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&t, bq, bs, nullptr, nullptr, 32, n_, 32, 0, 8, context, &delta));
    const auto size = narrow<uint32_t>(y->Shape().Size());
    QLoRAAddProgram add;
    add.AddInputs({{&base, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten},
                   {&delta, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten}})
        .AddOutput({y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten})
        .AddUniformVariables({{size}})
        .SetWorkgroupSize(64)
        .SetDispatchGroupSize(CeilDiv(size, 64u));
    return context.RunProgram(add);
  }

 private:
  int64_t k_;
  int64_t n_;
  int64_t accuracy_level_;
  bool automatic_;
  bool fuse_up_;
};

ONNX_OPERATOR_KERNEL_EX(MatMulNBitsQLoRA, kMSDomain, 1, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
                            .TypeConstraint("TQ", DataTypeImpl::GetTensorType<uint8_t>()),
                        MatMulNBitsQLoRA);

}  // namespace onnxruntime::contrib::webgpu
