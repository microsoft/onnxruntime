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
