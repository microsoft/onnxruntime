#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "contrib_ops/webgpu/bert/hyper_connection_pre_mix.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "static_node_replay.h"

namespace onnxruntime::contrib::webgpu {

class HCUpPreMixProgram final : public Program<HCUpPreMixProgram> {
 public:
  HCUpPreMixProgram() : Program{"HCUpPreMixF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& streams = shader.AddInput("streams");
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales");
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation()
        << "var<workgroup> partials: array<f16, 128>;\n"
        << "var<workgroup> gates: array<f16, 4>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let branch = local_idx / 32u;\n"
         << "let col = workgroup_idx + branch * 2560u;\n"
         << "var total = f16(0);\n"
         << "if (lane < 10u) {\n"
         << "  let packed = " << b.GetByOffset("col * 10u + lane") << ";\n"
         << "  let scale = " << scales.GetByOffset("col * 10u + lane") << ";\n"
         << "  var sum = f16(0);\n"
         << "  for (var i = 0u; i < 4u; i++) {\n"
         << "    let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "    let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "    let x0 = " << a.GetByOffset("lane * 8u + i * 2u") << ";\n"
         << "    let x1 = " << a.GetByOffset("lane * 8u + i * 2u + 1u") << ";\n"
         << "    sum += dot(x0, w0) + dot(x1, w1);\n"
         << "  }\n"
         << "  total += sum;\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << "  gates[branch] = f16(1.0 / (1.0 + exp(-value)));\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx == 0u) {\n"
         << "  var sum = 0.0;\n"
         << "  for (var c = 0u; c < uniforms.branches; c++) {\n"
         << "    sum += f32(" << streams.GetByOffset("c * 2560u + workgroup_idx")
         << ") * f32(gates[c]);\n"
         << "  }\n"
         << y.SetByOffset("workgroup_idx", "y_element_t(sum * uniforms.reduction_scale)") << "\n"
         << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"reduction_scale", ProgramUniformVariableDataType::Float32});
};

class HCUpPreMixF16 final : public WebGpuKernel {
 public:
  explicit HCUpPreMixF16(const OpKernelInfo& info)
      : WebGpuKernel(info),
        reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 1.0f)),
        accuracy_(info.GetAttrOrDefault<int64_t>("accuracy_level", 4)) {
    ORT_ENFORCE(info.GetAttr<int64_t>("K") == 320 && info.GetAttr<int64_t>("N") == 10240 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("bits") == 4 &&
                    info.GetAttr<int64_t>("num_branches") == 4,
                "HCUpPreMixF16 requires K320/N10240 symmetric block32 INT4 and four flattened streams");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_UP_PREMIX", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_HC_UP_PREMIX must be 0 or 1");
    const auto* streams = context.Input(0);
    const auto* a = context.Input(1);
    const auto* b = context.Input(2);
    const auto* scales = context.Input(3);
    ORT_RETURN_IF_NOT(streams->IsDataType<MLFloat16>() && a->IsDataType<MLFloat16>() &&
                          scales->IsDataType<MLFloat16>() && b->IsDataType<uint8_t>(),
                      "HCUpPreMixF16 requires FP16 streams/activations/scales and UINT8 weights");
    hyper_connection::StreamShape params;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), 4, params));
    ORT_RETURN_IF_NOT(params.hidden == 2560 && b->Shape() == TensorShape({10240, 10, 16}) &&
                          scales->Shape().Size() == 10240 * 10 && a->Shape().NumDimensions() > 0,
                      "HCUpPreMixF16 received incompatible stream or weight shapes");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), TensorShape({10240, 320}), false, true));
    ORT_RETURN_IF_NOT(helper.OutputShape() == streams->Shape() &&
                          a->Shape().GetDims().back() == 320,
                      "HCUpPreMixF16 projection rows must match flattened streams");
    auto* y = context.Output(0, TensorShape(params.reduced_shape));
    const auto count = onnxruntime::narrow<uint32_t>(y->Shape().Size());
    if (count == 0) return Status::OK();
    if (enabled == 1 && params.rows == 1 && !context.EnableMatmulFp32Accumulation() &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) == 0) {
      return static_replay_.Run(context, {y}, [&]() -> Status {
        HCUpPreMixProgram program;
        program.SetWorkgroupSize(128).SetDispatchGroupSize(2560);
        program.AddInput({streams, ProgramTensorMetadataDependency::TypeAndRank})
            .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
            .AddUniformVariables({{4u}, {reduction_scale_}});
        return context.RunProgram(program);
      });
    }
    auto projected = context.CreateGPUTensor(a->DataType(), helper.OutputShape());
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, b, scales, nullptr, nullptr, 320, 10240, 32,
                                        accuracy_, 4, context, &projected));
    hyper_connection::GateLayout layout;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
        projected.Shape(), streams->Shape(), params, false, layout, true));
    HyperConnectionPreMixProgram program{static_cast<int>(layout), true};
    program.CacheHint(static_cast<int>(layout), true)
        .AddInput({streams, ProgramTensorMetadataDependency::Type})
        .AddInput({&projected, ProgramTensorMetadataDependency::Type})
        .AddOutput({y, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{count}, {4u}, {2560u}, {reduction_scale_}})
        .SetDispatchGroupSize((count + 255) / 256).SetWorkgroupSize(256);
    return context.RunProgram(program);
  }

 private:
  float reduction_scale_;
  int64_t accuracy_;
  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HCUpPreMixF16", {0, 1, 2, 3}};
};

ONNX_OPERATOR_KERNEL_EX(
    HCUpPreMixF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()), HCUpPreMixF16);

}  // namespace onnxruntime::contrib::webgpu
