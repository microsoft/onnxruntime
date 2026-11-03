from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
GPU_COPY_SELECTOR = r'''
static bool WebGpuGreedyArgMaxTraceEnabled();
static bool WebGpuGpuSliceCopyEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GPU_SLICE_COPY", value);
    if (value && !WebGpuGreedyArgMaxEnabled())
      throw std::runtime_error("ORT_GENAI_WEBGPU_GPU_SLICE_COPY=1 requires ORT_GENAI_WEBGPU_GREEDY_ARGMAX=1");
    return value;
  }();
  return enabled;
}
'''
GPU_COPY_BRANCH = r'''    } else if (source.GetType() == device_label && begin_dest == 0 && begin_source != 0 &&
               dynamic_cast<WebGPUMemory*>(&source) != nullptr &&
               WebGpuGpuSliceCopyEnabled() && size_in_bytes != 0 &&
               p_device_ != source.p_device_ && begin_source % 4 == 0 && size_in_bytes % 4 == 0 &&
               source.size_in_bytes_ % 4 == 0 &&
               source.size_in_bytes_ / 4 <= std::numeric_limits<uint32_t>::max()) {
      if (size_in_bytes > size_in_bytes_ || begin_source > source.size_in_bytes_ ||
          size_in_bytes > source.size_in_bytes_ - begin_source)
        throw std::runtime_error("WebGPU sliced copy exceeds source or destination buffer bounds");
      const int64_t source_words = static_cast<int64_t>(source.size_in_bytes_ / 4);
      const int64_t output_words = static_cast<int64_t>(size_in_bytes / 4);
      int64_t start = static_cast<int64_t>(begin_source / 4);
      int64_t end = start + output_words;
      ModelConfig config("WebGpuSliceCopyU32");
      config.inputs.push_back(TensorConfig("source", Ort::TypeToTensorType<uint32_t>, {-1}));
      config.inputs.push_back(TensorConfig("starts", Ort::TypeToTensorType<int64_t>, {1}));
      config.inputs.push_back(TensorConfig("ends", Ort::TypeToTensorType<int64_t>, {1}));
      config.outputs.push_back(TensorConfig("output", Ort::TypeToTensorType<uint32_t>, {-1}));
      ExecutionParams params(DeviceType::WEBGPU, "WebGpuExecutionProvider", ort_memory_info_);
      params.inputs.push_back(TensorSpec(source.p_device_, Ort::TypeToTensorType<uint32_t>,
                                        {source_words}, source.size_in_bytes_));
      params.inputs.push_back(TensorSpec(&start, Ort::TypeToTensorType<int64_t>, {1}, sizeof(start)));
      params.inputs.push_back(TensorSpec(&end, Ort::TypeToTensorType<int64_t>, {1}, sizeof(end)));
      params.outputs.push_back(TensorSpec(p_device_, Ort::TypeToTensorType<uint32_t>,
                                         {output_words}, size_in_bytes));
      params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64", "session.disable_cpu_ep_fallback"};
      params.session_config_values = {"1", "1"};
      GraphExecutor::Execute(config, params);
      if (WebGpuGreedyArgMaxTraceEnabled())
        std::cerr << "WEBGPU_GPU_SLICE_COPY source_bytes=" << source.size_in_bytes_
                  << " offset_bytes=" << begin_source << " copied_bytes=" << size_in_bytes << std::endl;
'''


if __name__ == "__main__":
    source = (ROOT / "build_gpu_greedy_argmax_v5.py").read_text()
    source = replace_once(source, "gpu-greedy-argmax/build-v5", "gpu-greedy-argmax/build-v6-r2")
    source = replace_once(source, "\ndef interface_transform(source):",
        "\nGPU_COPY_SELECTOR = " + repr(GPU_COPY_SELECTOR) +
        "\nGPU_COPY_BRANCH = " + repr(GPU_COPY_BRANCH) + "\n\ndef interface_transform(source):")
    source = replace_once(source,
        '    source = replace_once(source, "namespace WebGPU {", "namespace WebGPU {\\n" + SELECTOR)',
        '    source = replace_once(source, "namespace WebGPU {", "namespace WebGPU {\\n" + SELECTOR)\n'
        '    source = replace_once(source, "struct WebGPUMemory final : DeviceBuffer {",\n'
        '                          GPU_COPY_SELECTOR + "\\nstruct WebGPUMemory final : DeviceBuffer {")\n'
        '    source = replace_once(source, "    } else if (strcmp(source.GetType(), label_cpu) == 0",\n'
        '                          GPU_COPY_BRANCH + "    } else if (strcmp(source.GetType(), label_cpu) == 0")')
    source = replace_once(source,
        "    return replace_once(source, \"  auto session_options = OrtSessionOptions::Create();\",",
        '    source = replace_once(source, "  // Bind inputs\\n",\n'
        '        "  std::unique_ptr<OrtMemoryInfo> cpu_memory;\\n"\n'
        '        "  if (model_config.op_type == \\"WebGpuSliceCopyU32\\")\\n"\n'
        '        "    cpu_memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);\\n"\n'
        '        "  // Bind inputs\\n")\n'
        '    source = replace_once(source, "    auto input_tensor = OrtValue::CreateTensor(\\n        *exec_params.memory_info,",\n'
        '        "    const OrtMemoryInfo* input_memory = cpu_memory && i != 0 ? cpu_memory.get() : exec_params.memory_info;\\n"\n'
        '        "    auto input_tensor = OrtValue::CreateTensor(\\n        *input_memory,")\n'
        '    return replace_once(source, "  auto session_options = OrtSessionOptions::Create();",')
    source = replace_once(source,
        'original.replace("  auto node = ", "  node = ")',
        'original.replace("  auto node = ", "  node = ").replace(\n'
        '    "      config.op_type.c_str(),",\n'
        '    \'      config.op_type == "WebGpuSliceCopyU32" ? "Slice" : config.op_type.c_str(),\')')
    source = source.replace("supported_v4", "supported_v5").replace("supported-v4-relink", "supported-v5-relink")
    source = replace_once(source,
        'Path("/workspace/qwen38-int4-gpu-greedy-argmax/build-v4")',
        'Path("/workspace/qwen38-int4-gpu-greedy-argmax/build-v5")')
    source = replace_once(source,
        'for relative in ("models/model.cpp", "ep/webgpu/interface.cpp", "models/graph_executor.cpp"):',
        'for relative in ("models/model.cpp", "ep/webgpu/interface.cpp", "models/graph_executor.cpp", "models/graph_builder.cpp"):')
    with (ROOT / "build_gpu_greedy_argmax_v6_r2.py").open("x") as stream:
        stream.write(source)
