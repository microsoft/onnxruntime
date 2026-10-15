// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_qlora.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/safeint.h"
#include "core/providers/webgpu/math/matmul.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

void GenerateLoRADown(ShaderHelper& shader, const std::string& base_groups) {
  shader.AddInput("lora_x");
  shader.AddInput("lora_at");
  shader.AddOutput("lora_down", ShaderUsage::UseValueTypeAlias);
  shader.AdditionalImplementation() << R"(
var<workgroup> lora_x_tile: array<lora_down_value_t, 256>;
var<workgroup> lora_a_tile: array<lora_down_value_t, 256>;
)";
  shader.MainFunctionBody() << "let lora_base_groups = " << base_groups << ";\n"
                            << R"(
if (workgroup_idx >= lora_base_groups) {
  let group = workgroup_idx - lora_base_groups;
  let row_local = local_idx / 8u;
  let k4_local = local_idx % 8u;
  let row = group * 32u + row_local;
  let rank_base = k4_local * 4u;
  var acc = lora_down_value_t(0);
  for (var start = 0u; start < uniforms.K / 4u; start += 8u) {
    var xv = lora_down_value_t(0);
    if (row < uniforms.M * uniforms.batch_count) {
      xv = lora_x[row * (uniforms.K / 4u) + start + k4_local];
    }
    lora_x_tile[local_idx] = xv;
    lora_a_tile[local_idx] = lora_at[row_local * (uniforms.K / 4u) + start + k4_local];
    workgroupBarrier();
    for (var kk = 0u; kk < 8u; kk++) {
      let x = lora_x_tile[row_local * 8u + kk];
      let a0 = lora_a_tile[rank_base * 8u + kk];
      let a1 = lora_a_tile[(rank_base + 1u) * 8u + kk];
      let a2 = lora_a_tile[(rank_base + 2u) * 8u + kk];
      let a3 = lora_a_tile[(rank_base + 3u) * 8u + kk];
      acc = lora_down_value_t(a0.x, a1.x, a2.x, a3.x) * x.x + acc;
      acc = lora_down_value_t(a0.y, a1.y, a2.y, a3.y) * x.y + acc;
      acc = lora_down_value_t(a0.z, a1.z, a2.z, a3.z) * x.z + acc;
      acc = lora_down_value_t(a0.w, a1.w, a2.w, a3.w) * x.w + acc;
    }
    workgroupBarrier();
  }
  if (row < uniforms.M * uniforms.batch_count) {
    lora_down[row * 8u + k4_local] = acc;
  }
  return;
}
)";
}

void GenerateLoRAGemv(ShaderHelper& shader, const std::string& base_groups, const std::string& x_value) {
  shader.AdditionalImplementation() << "var<workgroup> lora_partials: array<vec4<f32>, 128>;\n";
  shader.MainFunctionBody() << "let lora_base_groups = " << base_groups << ";\n"
                            << R"(
  // Extra workgroups compute each rank output once, alongside independent base
  // tiles. The next dispatch supplies the global dependency before consuming T.
  if (workgroup_idx >= lora_base_groups) {
    let lane = local_idx % 32u;
    let rank = (workgroup_idx - lora_base_groups) * 4u + local_idx / 32u;
    var acc = vec4<f32>(0);
    for (var k4 = lane; k4 < uniforms.K / 4u; k4 += 32u) {
      acc = fma(vec4<f32>()" << x_value
                            << R"(), vec4<f32>(lora_at[rank * (uniforms.K / 4u) + k4]), acc);
    }
    lora_partials[local_idx] = acc;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride /= 2u) {
      if (lane < stride) { lora_partials[local_idx] += lora_partials[local_idx + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let v = lora_partials[local_idx];
      lora_partials[local_idx] = vec4<f32>(v.x + v.z, v.y + v.w, 0, 0);
    }
    // Materialize the penultimate reduction to prevent reassociation of the
    // final four terms, matching the original GEMV's workgroup barriers.
    workgroupBarrier();
    if (lane == 0u) {
      let v = lora_partials[local_idx];
      lora_down[rank] = lora_down_element_t(v.x + v.y);
    }
    return;
  }
)";
}

class LoRAGemvProgram final : public Program<LoRAGemvProgram> {
 public:
  LoRAGemvProgram() : Program{"LoRADownGemv"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"K", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("lora_x");
    shader.AddInput("lora_at");
    shader.AddOutput("lora_down", ShaderUsage::UseElementTypeAlias);
    GenerateLoRAGemv(shader, "0u", "lora_x[k4]");
    return Status::OK();
  }
};

class LoRADownProgram final : public Program<LoRADownProgram> {
 public:
  LoRADownProgram() : Program{"LoRADownVec4"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"batch_count", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    GenerateLoRADown(shader, "0u");
    return Status::OK();
  }
};

// Fallback epilogue for base kernels without an integrated adapter output path.
class LoRAOutputProgram final : public Program<LoRAOutputProgram> {
 public:
  explicit LoRAOutputProgram(bool transposed = false)
      : Program{transposed ? "LoRAOutputVec4" : "LoRAOutput"}, transposed_(transposed) {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("base", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("t");
    shader.AddInput("b");
    shader.AddOutput("output");
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.size) { return; }\n"
        << "let row = global_idx / uniforms.N; let col = global_idx % uniforms.N;\n"
        << "var delta = base_element_t(0);\n";
    if (transposed_) {
      shader.MainFunctionBody()
          << "for (var r4 = 0u; r4 < 8u; r4++) {\n"
          << "  let bv = b[col * 8u + r4]; let tv = t[row * 8u + r4];\n"
          << "  delta = bv.x * tv.x + delta; delta = bv.y * tv.y + delta;\n"
          << "  delta = bv.z * tv.z + delta; delta = bv.w * tv.w + delta;\n}\n";
    } else {
      shader.MainFunctionBody()
          << "for (var r = 0u; r < 32u; r++) { delta = b[r * uniforms.N + col] * t[row * 32u + r] + delta; }\n";
    }
    shader.MainFunctionBody() << "output[global_idx] = base[global_idx] + delta;\n";
    return Status::OK();
  }

 private:
  bool transposed_;
};

Status ApplyLoRAOutput(onnxruntime::webgpu::ComputeContext& context, const Tensor* base, const Tensor* t, const Tensor* b, Tensor* y, bool transposed, const Tensor* quantized_scales) {
  if (quantized_scales) return ApplyQLoRAOutput(context, base, t, b, quantized_scales, y);
  const auto size = narrow<uint32_t>(y->Shape().Size());
  LoRAOutputProgram program(transposed);
  program.AddInputs({{base, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten},
                     {t, ProgramTensorMetadataDependency::Type, transposed ? 4 : 1},
                     {b, ProgramTensorMetadataDependency::Type, transposed ? 4 : 1}})
      .AddOutput({y, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten})
      .AddUniformVariables({{size}, {narrow<uint32_t>(b->Shape()[transposed ? 0 : 1])}})
      .SetDispatchGroupSize(CeilDiv(size, 64u));
  return context.RunProgram(program);
}

Status ApplyTransposedLoRA(onnxruntime::webgpu::ComputeContext& context, const Tensor* x, const Tensor* weight,
                           const Tensor* scales, const Tensor* at, const Tensor* bt, Tensor* y,
                           int64_t k, int64_t n, int64_t accuracy_level, bool fuse_up) {
  const auto rows = narrow<uint32_t>(x->Shape().Size() / k);
  const auto m = narrow<uint32_t>(x->Shape()[x->Shape().NumDimensions() - 2]);
  const bool use_gemv = rows == 1 && k >= 2048 && k <= 8192 &&
                        context.AdapterInfo().vendor == std::string_view{"nvidia"};
  auto down_shape = x->Shape().AsShapeVector();
  down_shape.back() = 32;
  auto t = context.CreateGPUTensor(x->DataType(), TensorShape(down_shape));
  const auto compute_down = [&]() -> Status {
    if (use_gemv) {
      LoRAGemvProgram down;
      down.AddInputs({{x, ProgramTensorMetadataDependency::Type, 4},
                      {at, ProgramTensorMetadataDependency::Type, 4}})
          .AddOutput({&t, ProgramTensorMetadataDependency::Type})
          .SetWorkgroupSize(128)
          .SetDispatchGroupSize(8)
          .AddUniformVariables({{narrow<uint32_t>(k)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(down));
    } else {
      LoRADownProgram down;
      down.AddInputs({{x, ProgramTensorMetadataDependency::Type, 4},
                      {at, ProgramTensorMetadataDependency::Type, 4}})
          .AddOutput({&t, ProgramTensorMetadataDependency::Type, 4})
          .SetWorkgroupSize(256)
          .SetDispatchGroupSize(CeilDiv(rows, 32u))
          .AddUniformVariables({{rows}, {narrow<uint32_t>(k)}, {1u}});
      ORT_RETURN_IF_ERROR(context.RunProgram(down));
    }
    return Status::OK();
  };
  if (fuse_up) {
    ORT_RETURN_IF_ERROR(compute_down());
    return ApplyMatMulNBits(x, weight, scales, nullptr, nullptr, k, n, 32, accuracy_level, 4,
                            context, y, 0, nullptr, 0, &t, bt, nullptr, nullptr, true);
  }
  auto base = context.CreateGPUTensor(y->DataType(), y->Shape());
  // FP32 can select DP4A even at M=1. Keep that base policy when the
  // floating joint down kernel would otherwise replace it.
  const bool base_dp4a = CanApplyDP4AMatrixMatMulNBits(
      context, accuracy_level, 32, narrow<uint32_t>(n), narrow<uint32_t>(k), 4, m, false, y);
  if (use_gemv && !base_dp4a) {
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, weight, scales, nullptr, nullptr, k, n, 32, accuracy_level, 4,
                                         context, &base, 0, nullptr, 0, nullptr, nullptr, at, &t));
  } else if (m >= kMinMForTileOptimization && context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
             base_dp4a) {
    // Preserve DP4A's activation quantization and compute T from original X
    // in extra workgroups of the base dispatch.
    ORT_RETURN_IF_ERROR(ApplyDP4AMatrixMatMulNBits(x, weight, scales, nullptr, nullptr, rows / m, m, m,
                                                   narrow<uint32_t>(n), narrow<uint32_t>(k), 32, 0,
                                                   kMinMForTileOptimization, 4, context, &base, 0,
                                                   nullptr, nullptr, nullptr, at, &t));
  } else {
    ORT_RETURN_IF_ERROR(compute_down());
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(x, weight, scales, nullptr, nullptr, k, n, 32, accuracy_level, 4,
                                         context, &base));
  }
  return ApplyLoRAOutput(context, &base, &t, bt, y, true);
}

class MatMulNBitsLoRA final : public WebGpuKernel {
 public:
  explicit MatMulNBitsLoRA(const OpKernelInfo& info)
      : WebGpuKernel(info), k_(info.GetAttr<int64_t>("K")), n_(info.GetAttr<int64_t>("N")), accuracy_level_(info.GetAttrOrDefault<int64_t>("accuracy_level", 0)), transposed_(info.GetAttrOrDefault<int64_t>("adapter_transposed", 0)) {
    const auto path = info.GetConfigOptions().GetConfigOrDefault("session.webgpu.lora_fusion_mode", "auto");
    ORT_ENFORCE(path == "auto" || path == "fuse_down" || path == "fuse_up",
                "session.webgpu.lora_fusion_mode must be auto, fuse_down or fuse_up.");
    fusion_mode_ = path == "auto" ? LoRAFusionMode::Auto : path == "fuse_up" ? LoRAFusionMode::FuseUp
                                                                             : LoRAFusionMode::FuseDown;
    ORT_ENFORCE(transposed_ == 0 || transposed_ == 1, "Invalid adapter_transposed value.");
    ORT_ENFORCE(info.GetAttr<int64_t>("bits") == 4 && info.GetAttr<int64_t>("block_size") == 32,
                "MatMulNBitsLoRA currently requires INT4 block size 32.");
    ORT_ENFORCE(k_ > 0 && k_ % 32 == 0 && n_ > 0 && n_ % 4 == 0, "Unsupported projection dimensions.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const auto* x = context.Input(0);
    const auto* weight = context.Input(1);
    const auto* scales = context.Input(2);
    const auto* a = context.Input(3);
    const auto* b = context.Input(4);
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions() >= 2 && a->Shape().NumDimensions() == 2 &&
                          b->Shape().NumDimensions() == 2,
                      "Invalid LoRA input ranks.");
    const auto adapter_axis = static_cast<size_t>(transposed_);
    ORT_RETURN_IF_NOT(a->Shape()[adapter_axis] == k_ && b->Shape()[1 - adapter_axis] == n_, "Invalid LoRA K/N dimensions.");
    const int64_t rank_a = a->Shape()[1 - adapter_axis];
    const int64_t rank_b = b->Shape()[adapter_axis];
    ORT_RETURN_IF_NOT((rank_a == 0 || rank_a == 32) && (rank_b == 0 || rank_b == 32),
                      "MatMulNBitsLoRA currently supports rank 32 or an empty adapter.");
    ORT_RETURN_IF_NOT(weight->Shape() == TensorShape({n_, k_ / 32, 16}) &&
                          scales->Shape().Size() == SafeInt<int64_t>(n_) * (k_ / 32),
                      "Invalid packed weight/scales shape.");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(x->Shape(), TensorShape({n_, k_}), false, true));
    auto* y = context.Output(0, helper.OutputShape());
    if (y->Shape().Size() == 0) return Status::OK();
    if (rank_a == 0 || rank_b == 0) {
      return ApplyMatMulNBits(x, weight, scales, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4, context, y);
    }
    if (transposed_) {
      // Use the per-matrix row count so batched one-token decode still selects fuse_down.
      const auto m = x->Shape()[x->Shape().NumDimensions() - 2];
      const bool fuse_up = fusion_mode_ == LoRAFusionMode::FuseUp ||
                           (fusion_mode_ == LoRAFusionMode::Auto && m > 1);
      return ApplyTransposedLoRA(context, x, weight, scales, a, b, y, k_, n_, accuracy_level_, fuse_up);
    }
    MatMulComputeHelper lora_helper;
    ORT_RETURN_IF_ERROR(lora_helper.Compute(x->Shape(), a->Shape()));
    auto t = context.CreateGPUTensor(x->DataType(), lora_helper.OutputShape());
    std::vector<const Tensor*> inputs{x, a};
    ORT_RETURN_IF_ERROR(compute_dispatcher_.Compute(context, Activation(), inputs, &t,
                                                    /*is_channels_last=*/true, /*b_is_constant=*/false));
    return ApplyMatMulNBits(x, weight, scales, nullptr, nullptr, k_, n_, 32, accuracy_level_, 4,
                            context, y, 0, nullptr, 0, &t, b);
  }

 private:
  enum class LoRAFusionMode { Auto,
                              FuseDown,
                              FuseUp };

  int64_t k_;
  int64_t n_;
  int64_t accuracy_level_;
  int64_t transposed_;
  LoRAFusionMode fusion_mode_;
  mutable MatMulComputeDispatcher compute_dispatcher_;
};

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBitsLoRA, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TQ", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBitsLoRA);

}  // namespace onnxruntime::contrib::webgpu
