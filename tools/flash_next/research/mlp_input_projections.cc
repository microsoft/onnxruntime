#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "quickgelu_mul_overlay.h"
#include "static_node_replay.h"

namespace onnxruntime::contrib::webgpu {
using namespace onnxruntime::webgpu;

class MlpInputProjectionsProgram final : public Program<MlpInputProjectionsProgram> {
 public:
  MlpInputProjectionsProgram() : Program{"MlpInputProjectionsF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& rw = shader.AddInput("rw", ShaderUsage::UseValueTypeAlias);
    const auto& rs = shader.AddInput("rs", ShaderUsage::UseElementTypeAlias);
    const auto& gw = shader.AddInput("gw", ShaderUsage::UseValueTypeAlias);
    const auto& gs = shader.AddInput("gs", ShaderUsage::UseElementTypeAlias);
    const auto& uw = shader.AddInput("uw", ShaderUsage::UseValueTypeAlias);
    const auto& us = shader.AddInput("us", ShaderUsage::UseElementTypeAlias);
    const auto& sw = shader.AddInput("sw", ShaderUsage::UseValueTypeAlias);
    const auto& ss = shader.AddInput("ss", ShaderUsage::UseElementTypeAlias);
    const auto& router = shader.AddOutput("router", ShaderUsage::UseElementTypeAlias);
    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);
    const auto& shared = shader.AddOutput("shared_gate", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() <<
        "var<workgroup> partials: array<f16, 256>;\n"
        "var<workgroup> up_partials: array<f16, 256>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let is_router = workgroup_idx < 64u;\n"
         << "let is_shared = workgroup_idx == 144u;\n"
         << "var col = workgroup_idx * 8u + local_idx / 32u;\n"
         << "if (!is_router) { col = (workgroup_idx - 64u) * 8u + local_idx / 32u; }\n"
         << "if (is_shared) { col = local_idx / 32u; }\n"
         << "var total = f16(0);\n"
         << "var up_total = f16(0);\n"
         << "if (!is_shared || col == 0u) {\n"
         << "  for (var tile = 0u; tile < 3u; tile++) {\n"
         << "    let block = tile * 32u + lane;\n"
         << "    if (block < 80u) {\n"
         << "      let offset = col * 80u + block;\n"
         << "      var packed: vec4<u32>;\n"
         << "      var scale: f16;\n"
         << "      if (is_router) {\n"
         << "        packed = " << rw.GetByOffset("offset") << ";\n"
         << "        scale = " << rs.GetByOffset("offset") << ";\n"
         << "      } else if (is_shared) {\n"
         << "        packed = " << sw.GetByOffset("block") << ";\n"
         << "        scale = " << ss.GetByOffset("block") << ";\n"
         << "      } else {\n"
         << "        packed = " << gw.GetByOffset("offset") << ";\n"
         << "        scale = " << gs.GetByOffset("offset") << ";\n"
         << "      }\n"
         << "      var sum = f16(0);\n"
         << "      for (var i = 0u; i < 4u; i++) {\n"
         << "        let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "        let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "        let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "        let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "        sum += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n"
         << "      total += sum;\n"
         << "      if (!is_router && !is_shared) {\n"
         << "        let up_packed = " << uw.GetByOffset("offset") << ";\n"
         << "        let up_scale = " << us.GetByOffset("offset") << ";\n"
         << "        var up_sum = f16(0);\n"
         << "        for (var i = 0u; i < 4u; i++) {\n"
         << "          let low = vec4<f16>(unpack4xU8(up_packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "          let high = vec4<f16>(unpack4xU8((up_packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "          let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * up_scale;\n"
         << "          let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * up_scale;\n"
         << "          let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "          let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "          up_sum += dot(x0, w0) + dot(x1, w1);\n"
         << "        }\n"
         << "        up_total += up_sum;\n"
         << "      }\n"
         << "    }\n"
         << "  }\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "up_partials[local_idx] = up_total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u && (!is_shared || col == 0u)) {\n"
         << "  var value = f16(0);\n"
         << "  var up_value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) {\n"
         << "    value += partials[local_idx + i];\n"
         << "    up_value += up_partials[local_idx + i];\n"
         << "  }\n"
         << "  if (is_router) {\n"
         << router.SetByOffset("col", "value") << "\n"
         << "  } else if (is_shared) {\n"
         << shared.SetByOffset("0u", "value") << "\n"
         << "  } else {\n"
         << "    let v = value * f16(1.0);\n"
         << "    var sigmoid_value: f16;\n"
         << "    if (v >= f16(0.0)) { sigmoid_value = f16(1.0) / (f16(1.0) + exp(-v)); }\n"
         << "    else { sigmoid_value = f16(1.0) - f16(1.0) / (f16(1.0) + exp(v)); }\n"
         << "    let silu = f16(value * sigmoid_value);\n"
         << activated.SetByOffset("col", "silu * up_value") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }
};

class MlpInputProjectionsF16 final : public WebGpuKernel {
 public:
  explicit MlpInputProjectionsF16(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MLP_INPUT_PROJECTIONS", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_MLP_INPUT_PROJECTIONS must be 0 or 1");
    const auto* a = context.Input(0);
    ORT_RETURN_IF_NOT(a->IsDataType<MLFloat16>() && a->Shape().NumDimensions() > 0 &&
                          a->Shape().GetDims().back() == 2560,
                      "MlpInputProjectionsF16 requires FP16 activations with K2560");
    const int64_t widths[] = {512, 640, 640, 1};
    for (size_t i = 0; i < 4; ++i) {
      const auto* weight = context.Input(1 + i * 2);
      const auto* scale = context.Input(2 + i * 2);
      ORT_RETURN_IF_NOT(weight->IsDataType<uint8_t>() && scale->IsDataType<MLFloat16>() &&
                            weight->Shape() == TensorShape({widths[i], 80, 16}) &&
                            scale->Shape() == TensorShape({widths[i], 80}),
                        "MlpInputProjectionsF16 received incompatible symmetric block32 INT4 weights/scales");
    }
    TensorShapeVector shape(a->Shape().GetDims().begin(), a->Shape().GetDims().end());
    shape.back() = 512;
    auto* router = context.Output(0, TensorShape(shape));
    shape.back() = 640;
    auto* activated = context.Output(1, TensorShape(shape));
    shape.back() = 1;
    auto* shared = context.Output(2, TensorShape(shape));
    if (a->Shape().Size() == 0) return Status::OK();
    if (enabled == 1 && a->Shape().Size() == 2560 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        !context.EnableMatmulFp32Accumulation() &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0 &&
        context.DeviceLimits().maxStorageBuffersPerShaderStage >= 12 &&
        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
        context.DeviceLimits().maxComputeWorkgroupSizeX >= 256) {
      return replay_.Run(context, {router, activated, shared}, [&]() -> Status {
        MlpInputProjectionsProgram program;
        program.SetWorkgroupSize(256).SetDispatchGroupSize(145);
        program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4});
        for (size_t i = 0; i < 4; ++i) {
          program.AddInput({context.Input(1 + i * 2), ProgramTensorMetadataDependency::TypeAndRank, 16})
              .AddInput({context.Input(2 + i * 2), ProgramTensorMetadataDependency::TypeAndRank});
        }
        program.AddOutput({router, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({activated, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({shared, ProgramTensorMetadataDependency::TypeAndRank});
        return context.RunProgram(program);
      });
    }
    shape.back() = 640;
    auto gate = context.CreateGPUTensor(a->DataType(), TensorShape(shape));
    auto up = context.CreateGPUTensor(a->DataType(), TensorShape(shape));
    Tensor* targets[] = {router, &gate, &up, shared};
    for (size_t i = 0; i < 4; ++i) {
      ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, context.Input(1 + i * 2), context.Input(2 + i * 2),
          nullptr, nullptr, 2560, widths[i], 32, 4, 4, context, targets[i]));
    }
    const auto count = onnxruntime::narrow<uint32_t>(activated->Shape().Size() / 4);
    QuickGeluMulF16Program program;
    program.AddInput({&gate, ProgramTensorMetadataDependency::Type, {count}, 4})
        .AddInput({&up, ProgramTensorMetadataDependency::Type, {count}, 4})
        .AddOutput({activated, ProgramTensorMetadataDependency::None, {count}, 4})
        .AddUniformVariables({{count}})
        .SetDispatchGroupSize((count + 63) / 64)
        .SetWorkgroupSize(64);
    return context.RunProgram(program);
  }

 private:
  mutable StaticNodeReplay replay_{"MlpInputProjectionsF16", {0, 1, 2, 3, 4, 5, 6, 7, 8}};
};

ONNX_OPERATOR_KERNEL_EX(
    MlpInputProjectionsF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()), MlpInputProjectionsF16);
}
