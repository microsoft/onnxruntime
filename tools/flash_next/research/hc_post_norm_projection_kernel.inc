class HCPostNormProjectionF16 : public WebGpuKernel {
 public:
  explicit HCPostNormProjectionF16(const OpKernelInfo& info) : HCPostNormProjectionF16(info, false) {}

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_POST_NORM_PROJECTION", 0);
    const int projection_mode = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_SILU", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_HC_POST_NORM_PROJECTION must be 0 or 1");
    ORT_RETURN_IF_NOT(projection_mode == 0 || projection_mode == 1, "ORT_WEBGPU_HC_PROJECTION_SILU must be 0 or 1");
    const int norm_inputs = shared_ ? 6 : 4;
    for (int index = 0; index < norm_inputs; ++index) {
      ORT_RETURN_IF_NOT(context.Input(index)->IsDataType<MLFloat16>(),
                        "HCPostNormProjectionF16 requires FP16 normalization inputs");
    }
    const auto* streams = context.Input(0);
    const auto* scale = context.Input(3);
    const auto* sizes = context.Input(norm_inputs);
    const auto* weights = context.Input(norm_inputs + 1);
    const auto* scales = context.Input(norm_inputs + 2);
    ORT_RETURN_IF_NOT(sizes->IsDataType<int64_t>() && weights->IsDataType<uint8_t>() &&
                          scales->IsDataType<MLFloat16>() && sizes->Shape() == TensorShape({2}) &&
                          weights->Shape() == TensorShape({324, 320, 16}) &&
                          scales->Shape().Size() == 324 * 320,
                      "HCPostNormProjectionF16 requires canonical INT4 weights/scales and INT64 split sizes");
    hyper_connection::StreamShape params;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), 4, params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(context.Input(1)->Shape(), params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
    hyper_connection::GateLayout layout;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
        context.Input(2)->Shape(), streams->Shape(), params, false, layout));
    ORT_RETURN_IF_NOT(layout == hyper_connection::GateLayout::Branch &&
                          streams->Shape().NumDimensions() > 0 && streams->Shape().GetDims().back() == 10240,
                      "HCPostNormProjectionF16 requires four flattened branches and one gate per branch");
    if (shared_) {
      auto expected_gate = params.reduced_shape;
      expected_gate.back() = 1;
      ORT_RETURN_IF_NOT(context.Input(4)->Shape() == context.Input(1)->Shape() &&
                            context.Input(5)->Shape() == TensorShape(expected_gate),
                        "HCPostNormProjectionF16 received incompatible shared branch/gate shapes");
    }
    const auto* split = sizes->Data<int64_t>();
    ORT_RETURN_IF_NOT(split[0] > 0 && split[0] < 324 && split[1] == 324 - split[0],
                      "HCPostNormProjectionF16 requires positive split sizes summing to 324");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(streams->Shape(), TensorShape({324, 10240}), false, true));
    TensorShapeVector shape(helper.OutputShape().GetDims().begin(), helper.OutputShape().GetDims().end());
    auto prefix_shape = shape;
    auto suffix_shape = shape;
    prefix_shape.back() = split[0];
    suffix_shape.back() = split[1];
    auto* y = context.Output(0, streams->Shape());
    auto* normalized = context.Output(1, streams->Shape());
    auto* activated = context.Output(2, TensorShape(prefix_shape));
    auto* tail = context.Output(3, TensorShape(suffix_shape));
    if (activated->Shape().Size() == 0) return Status::OK();
    const auto hidden = onnxruntime::narrow<uint32_t>(params.hidden);
    const auto groups = onnxruntime::narrow<uint32_t>(streams->Shape().Size() / params.hidden);
    const auto prefix = onnxruntime::narrow<uint32_t>(split[0]);
    const auto& limits = context.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxComputeWorkgroupStorageSize >= 1024 &&
                          hidden <= (limits.maxComputeWorkgroupStorageSize - 1024) / sizeof(MLFloat16),
                      "HCPostNormProjectionF16 normalization exceeds workgroup storage limit");
    const bool shared_scale = scale->Shape().Size() == params.hidden;
    const bool projection_fused = projection_mode == 1 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} && streams->Shape().Size() == 10240 &&
        !context.EnableMatmulFp32Accumulation() &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0;
    const bool fused = enabled == 1 && projection_fused && hidden == 2560 &&
        limits.maxComputeWorkgroupSizeX >= 256 && limits.maxComputeInvocationsPerWorkgroup >= 256 &&
        limits.maxComputeWorkgroupStorageSize >= 27648 &&
        limits.maxStorageBuffersPerShaderStage >= (shared_ ? 12u : 10u);
    if (fused) {
      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {
        HCPostNormProjectionProgram program{shared_, shared_scale};
        program.CacheHint(shared_, shared_scale)
            .AddInput({streams, ProgramTensorMetadataDependency::Type})
            .AddInput({context.Input(1), ProgramTensorMetadataDependency::Type})
            .AddInput({context.Input(2), ProgramTensorMetadataDependency::Type})
            .AddInput({scale, ProgramTensorMetadataDependency::Type});
        if (shared_) {
          program.AddInput({context.Input(4), ProgramTensorMetadataDependency::Type})
              .AddInput({context.Input(5), ProgramTensorMetadataDependency::Type});
        }
        program.AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::Type})
            .AddOutput({normalized, ProgramTensorMetadataDependency::Type})
            .AddOutput({activated, ProgramTensorMetadataDependency::Type})
            .AddOutput({tail, ProgramTensorMetadataDependency::Type})
            .AddUniformVariables({{hidden}, {epsilon_}, {prefix}, {alpha_}})
            .SetWorkgroupSize(256).SetDispatchGroupSize(41);
        return context.RunProgram(program);
      });
    }
    HCPostNormProjectionBaselineNormProgram norm{shared_, shared_scale, hidden};
    norm.CacheHint(shared_, shared_scale, hidden)
        .AddInput({streams, ProgramTensorMetadataDependency::Type})
        .AddInput({context.Input(1), ProgramTensorMetadataDependency::Type})
        .AddInput({context.Input(2), ProgramTensorMetadataDependency::Type})
        .AddInput({scale, ProgramTensorMetadataDependency::Type});
    if (shared_) {
      norm.AddInput({context.Input(4), ProgramTensorMetadataDependency::Type})
          .AddInput({context.Input(5), ProgramTensorMetadataDependency::Type});
    }
    norm.AddOutput({y, ProgramTensorMetadataDependency::Type})
        .AddOutput({normalized, ProgramTensorMetadataDependency::Type})
        .AddUniformVariables({{hidden}, {epsilon_}})
        .SetWorkgroupSize(256).SetDispatchGroupSize(groups);
    ORT_RETURN_IF_ERROR(context.RunProgram(norm));
    if (projection_fused) {
      HCPostNormProjectionBaselineMatMulProgram program;
      program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
      program.AddInput({normalized, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
          .AddOutput({activated, ProgramTensorMetadataDependency::TypeAndRank})
          .AddOutput({tail, ProgramTensorMetadataDependency::TypeAndRank})
          .AddUniformVariables({{prefix}, {alpha_}});
      return context.RunProgram(program);
    }
    auto intermediate = context.CreateGPUTensor(streams->DataType(), TensorShape(shape));
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(normalized, weights, scales, nullptr, nullptr, 10240, 324, 32,
                                        accuracy_, 4, context, &intermediate));
    SplitScaledSiLUF16Program program;
    const auto count = onnxruntime::narrow<uint32_t>(intermediate.Shape().Size());
    program.AddInput({&intermediate, ProgramTensorMetadataDependency::Type})
        .AddOutput({activated, ProgramTensorMetadataDependency::None})
        .AddOutput({tail, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{count}, {324u}, {prefix},
                              {onnxruntime::narrow<uint32_t>(split[1])}, {alpha_}})
        .SetDispatchGroupSize((count + 255) / 256).SetWorkgroupSize(256);
    return context.RunProgram(program);
  }

 protected:
  HCPostNormProjectionF16(const OpKernelInfo& info, bool shared)
      : WebGpuKernel(info), shared_{shared}, epsilon_{info.GetAttrOrDefault<float>("epsilon", 1e-6f)},
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        accuracy_{info.GetAttrOrDefault<int64_t>("accuracy_level", 4)},
        replay_{shared ? "HCSharedPostNormProjectionF16" : "HCPostNormProjectionF16",
                shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5, 7, 8}
                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {
    ORT_ENFORCE(info.GetAttr<int64_t>("K") == 10240 && info.GetAttr<int64_t>("N") == 324 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("bits") == 4,
                "HCPostNormProjectionF16 requires K10240/N324 symmetric block32 INT4");
  }

 private:
  bool shared_;
  float epsilon_;
  float alpha_;
  int64_t accuracy_;
  mutable StaticNodeReplay replay_;
};

class HCSharedPostNormProjectionF16 final : public HCPostNormProjectionF16 {
 public:
  explicit HCSharedPostNormProjectionF16(const OpKernelInfo& info) : HCPostNormProjectionF16(info, true) {}
};

ONNX_OPERATOR_KERNEL_EX(
    HCPostNormProjectionF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 4), HCPostNormProjectionF16);
ONNX_OPERATOR_KERNEL_EX(
    HCSharedPostNormProjectionF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 6), HCSharedPostNormProjectionF16);
