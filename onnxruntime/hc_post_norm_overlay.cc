#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "static_node_replay.h"

namespace onnxruntime::contrib::webgpu {
using namespace onnxruntime::webgpu;

class HCPostNormProgram final : public Program<HCPostNormProgram> {
 public:
  HCPostNormProgram(bool shared, bool shared_scale, uint32_t hidden)
      : Program{"HCPostNormF16"}, shared_{shared}, shared_scale_{shared_scale}, hidden_{hidden} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& streams = shader.AddInput("streams");
    const auto& branch = shader.AddInput("branch");
    const auto& gate = shader.AddInput("gate");
    const auto& scale = shader.AddInput("scale");
    const ShaderVariableHelper* shared = nullptr;
    const ShaderVariableHelper* shared_gate = nullptr;
    if (shared_) {
      shared = &shader.AddInput("shared_output");
      shared_gate = &shader.AddInput("shared_gate");
    }
    const auto& y = shader.AddOutput("y");
    const auto& normalized = shader.AddOutput("normalized");
    shader.AdditionalImplementation() << "var<workgroup> sums: array<f32, 256>;\n";
    shader.AdditionalImplementation() << "var<workgroup> post_values: array<f16, " << hidden_ << ">;\n";
    auto& code = shader.MainFunctionBody();
    code << "let group = workgroup_idx;\n"
         << "let row = group / 4u;\n"
         << "let base = group * uniforms.hidden;\n"
         << "let divided_gate = f16(" << gate.GetByOffset("group") << " / f16(4.0));\n"
         << "let sigmoid_gate = f16(1.0 / (1.0 + exp(-divided_gate)));\n"
         << "let gate_value = f16(sigmoid_gate * f16(2.0));\n";
    if (shared_) {
      code << "let shared_sigmoid = f16(1.0 / (1.0 + exp(-"
           << shared_gate->GetByOffset("row") << ")));\n";
    }
    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let branch_offset = row * uniforms.hidden + h;\n";
    if (shared_) {
      code << "let branch_value = f16(" << branch.GetByOffset("branch_offset") << " + ("
           << shared->GetByOffset("branch_offset") << " * shared_sigmoid));\n";
    } else {
      code << "let branch_value = " << branch.GetByOffset("branch_offset") << ";\n";
    }
    code << "  var value = f32(" << streams.GetByOffset("base + h") << ");\n"
         << "  value += f32(gate_value) * f32(branch_value);\n"
         << "  let stored = f16(value);\n"
         << "  post_values[h] = stored;\n"
         << y.SetByOffset("base + h", "stored") << "\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "var sum = 0.0;\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let v = f32(post_values[h]);\n"
         << "  sum += v * v;\n"
         << "}\n"
         << "sums[local_idx] = sum;\n"
         << "workgroupBarrier();\n"
         << "for (var width = 128u; width > 0u; width >>= 1u) {\n"
         << "  if (local_idx < width) { sums[local_idx] += sums[local_idx + width]; }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "let inv_rms = inverseSqrt(sums[0] / f32(uniforms.hidden) + uniforms.epsilon);\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let weight = f32(" << scale.GetByOffset(
                shared_scale_ ? "h" : "(group % 4u) * uniforms.hidden + h") << ");\n"
         << normalized.SetByOffset("base + h", "f16(f32(post_values[h]) * inv_rms * weight)") << "\n"
         << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool shared_;
  bool shared_scale_;
  uint32_t hidden_;
};

class HCPostNormF16 : public WebGpuKernel {
 public:
  explicit HCPostNormF16(const OpKernelInfo& info) : HCPostNormF16(info, false) {}
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    for (int i = 0; i < (shared_ ? 6 : 4); ++i) {
      ORT_RETURN_IF_NOT(context.Input(i)->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                        "HCPostNormF16 requires FP16 inputs");
    }
    const auto* streams = context.Input(0);
    const auto* branch = context.Input(1);
    const auto* gate = context.Input(2);
    const auto* scale = context.Input(3);
    hyper_connection::StreamShape params;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), 4, params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(branch->Shape(), params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
    hyper_connection::GateLayout layout;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
        gate->Shape(), streams->Shape(), params, false, layout));
    ORT_RETURN_IF_NOT(layout == hyper_connection::GateLayout::Branch,
                      "HCPostNormF16 requires one gate per branch");
    if (shared_) {
      ORT_RETURN_IF_NOT(context.Input(4)->Shape() == branch->Shape(),
                        "HCPostNormF16 requires shared and branch outputs of identical shape");
      auto expected_gate = params.reduced_shape;
      expected_gate.back() = 1;
      ORT_RETURN_IF_NOT(context.Input(5)->Shape() == TensorShape(expected_gate),
                        "HCPostNormF16 requires one shared gate per row");
    }
    auto* y = context.Output(0, streams->Shape());
    auto* normalized = context.Output(1, streams->Shape());
    const auto groups = onnxruntime::narrow<uint32_t>(streams->Shape().Size() / params.hidden);
    if (groups == 0) return Status::OK();
    const auto hidden = onnxruntime::narrow<uint32_t>(params.hidden);
    const auto workgroup_bytes = context.DeviceLimits().maxComputeWorkgroupStorageSize;
    ORT_RETURN_IF_NOT(workgroup_bytes >= 1024 && hidden <= (workgroup_bytes - 1024) / sizeof(MLFloat16),
                      "HCPostNormF16 hidden dimension exceeds workgroup storage limit");
    return static_replay_.Run(context, {y, normalized}, [&]() -> Status {
    HCPostNormProgram program{shared_, scale->Shape().Size() == params.hidden, hidden};
    program.CacheHint(shared_, scale->Shape().Size() == params.hidden, hidden)
        .AddInput({streams, ProgramTensorMetadataDependency::Type})
        .AddInput({branch, ProgramTensorMetadataDependency::Type})
        .AddInput({gate, ProgramTensorMetadataDependency::Type})
        .AddInput({scale, ProgramTensorMetadataDependency::Type});
    if (shared_) {
      program.AddInput({context.Input(4), ProgramTensorMetadataDependency::Type})
          .AddInput({context.Input(5), ProgramTensorMetadataDependency::Type});
    }
    program.AddOutput({y, ProgramTensorMetadataDependency::Type})
        .AddOutput({normalized, ProgramTensorMetadataDependency::Type})
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(params.hidden)}, {epsilon_}})
        .SetWorkgroupSize(256).SetDispatchGroupSize(groups);
    return context.RunProgram(program);
    }, params.branches * params.hidden);
  }

 protected:
  HCPostNormF16(const OpKernelInfo& info, bool shared)
      : WebGpuKernel(info), shared_{shared}, epsilon_{info.GetAttrOrDefault<float>("epsilon", 1e-6f)},
        static_replay_{shared ? "HCSharedPostNormF16" : "HCPostNormF16",
                       shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5} :
                                std::initializer_list<int>{0, 1, 2, 3}} {}

 private:
  bool shared_;
  float epsilon_;
  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_;
};

class HCSharedPostNormF16 final : public HCPostNormF16 {
 public:
  explicit HCSharedPostNormF16(const OpKernelInfo& info) : HCPostNormF16(info, true) {}
};

ONNX_OPERATOR_KERNEL_EX(
    HCPostNormF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()),
    HCPostNormF16);

ONNX_OPERATOR_KERNEL_EX(
    HCSharedPostNormF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()),
    HCSharedPostNormF16);
}  // namespace onnxruntime::contrib::webgpu
