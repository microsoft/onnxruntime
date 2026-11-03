#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/bert/scaled_silu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "static_node_replay.h"

namespace onnxruntime::contrib::webgpu {
class HCProjectionSiLUProgram final : public Program<HCProjectionSiLUProgram> {
 public:
  HCProjectionSiLUProgram() : Program{"HCProjectionSiLUF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);
    const auto& tail = shader.AddOutput("tail");
    shader.AdditionalImplementation() <<
        "fn stable_sigmoid(v: f32) -> f32 {\n"
        "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
        "  let e = exp(v); return e / (1.0 + e);\n}\n";
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 2u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < 10u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  let packed = " << b.GetByOffset("col * 320u + block") << ";\n"
         << "  let scale = " << scales.GetByOffset("col * 320u + block") << ";\n"
         << "  var sum = f16(0);\n"
         << "  for (var i = 0u; i < 4u; i++) {\n"
         << "    let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "    let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "    let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "    let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "    sum += dot(x0, w0) + dot(x1, w1);\n"
         << "  }\n"
         << "  total += sum;\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << "  if (col < uniforms.prefix) {\n"
         << "    let z_t = activated_element_t(f32(value) * uniforms.alpha);\n"
         << "    let sigmoid_t = activated_element_t(stable_sigmoid(f32(z_t)));\n"
         << activated.SetByOffset("col", "activated_element_t(f32(z_t) * f32(sigmoid_t))") << "\n"
         << "  } else {\n"
         << tail.SetByOffset("col - uniforms.prefix", "value") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"prefix", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});
};

class HCProjectionSiLUF16 final : public WebGpuKernel {
 public:
  explicit HCProjectionSiLUF16(const OpKernelInfo& info)
      : WebGpuKernel(info),
        alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)),
        accuracy_(info.GetAttrOrDefault<int64_t>("accuracy_level", 4)) {
    ORT_ENFORCE(info.GetAttr<int64_t>("K") == 10240 && info.GetAttr<int64_t>("N") == 324 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("bits") == 4,
                "HCProjectionSiLUF16 requires K10240/N324 symmetric block32 INT4");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_SILU", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_HC_PROJECTION_SILU must be 0 or 1");
    const auto* a = context.Input(0);
    const auto* sizes = context.Input(1);
    const auto* b = context.Input(2);
    const auto* scales = context.Input(3);
    ORT_RETURN_IF_NOT(a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
                          b->IsDataType<uint8_t>() && sizes->IsDataType<int64_t>(),
                      "HCProjectionSiLUF16 requires FP16 activations/scales, UINT8 weights and INT64 sizes");
    ORT_RETURN_IF_NOT(a->Shape().NumDimensions() > 0 && a->Shape().GetDims().back() == 10240 &&
                          b->Shape() == TensorShape({324, 320, 16}) && scales->Shape().Size() == 324 * 320 &&
                          sizes->Shape() == TensorShape({2}),
                      "HCProjectionSiLUF16 received incompatible matrix or split shapes");
    const auto* split = sizes->Data<int64_t>();
    ORT_RETURN_IF_NOT(split[0] > 0 && split[0] < 324 && split[1] == 324 - split[0],
                      "HCProjectionSiLUF16 requires positive split sizes summing to 324");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), TensorShape({324, 10240}), false, true));
    TensorShapeVector shape(helper.OutputShape().GetDims().begin(), helper.OutputShape().GetDims().end());
    auto prefix_shape = shape;
    auto suffix_shape = shape;
    prefix_shape.back() = split[0];
    suffix_shape.back() = split[1];
    auto* activated = context.Output(0, TensorShape(prefix_shape));
    auto* tail = context.Output(1, TensorShape(suffix_shape));
    if (activated->Shape().Size() == 0) return Status::OK();
    const bool fused = enabled == 1 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        a->Shape().Size() == 10240 && !context.EnableMatmulFp32Accumulation() &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0;
    if (fused) {
      return static_replay_.Run(context, {activated, tail}, [&]() -> Status {
        HCProjectionSiLUProgram program;
        program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
        program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({activated, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({tail, ProgramTensorMetadataDependency::TypeAndRank})
            .AddUniformVariables({{onnxruntime::narrow<uint32_t>(split[0])}, {alpha_}});
        return context.RunProgram(program);
      });
    }
    auto intermediate = context.CreateGPUTensor(a->DataType(), TensorShape(shape));
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, b, scales, nullptr, nullptr, 10240, 324, 32,
                                        accuracy_, 4, context, &intermediate));
    SplitScaledSiLUF16Program program;
    const auto count = onnxruntime::narrow<uint32_t>(intermediate.Shape().Size());
    program.AddInput({&intermediate, ProgramTensorMetadataDependency::Type})
        .AddOutput({activated, ProgramTensorMetadataDependency::None})
        .AddOutput({tail, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{count},
                              {324u}, {onnxruntime::narrow<uint32_t>(split[0])},
                              {onnxruntime::narrow<uint32_t>(split[1])}, {alpha_}})
        .SetDispatchGroupSize((count + 255) / 256)
        .SetWorkgroupSize(256);
    return context.RunProgram(program);
  }

 private:
  float alpha_;
  int64_t accuracy_;
  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HCProjectionSiLUF16", {0, 2, 3}};
};

ONNX_OPERATOR_KERNEL_EX(
    HCProjectionSiLUF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 1),
    HCProjectionSiLUF16);

}
