from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"
SELECTOR = f'''
static bool WebGpuGreedyFusedMinLengthEnabled() {{
  static const bool enabled = []() {{
    bool value = false;
    GetEnv("{FLAG}", value);
    return value;
  }}();
  return enabled;
}}
'''
GRAPH_TRANSFORM = r'''

def graph_builder_transform(source):
    begin = source.index("  // Create node using RAII wrapper")
    end = source.index("  // Add node to graph", begin)
    original = source[begin:end]
    replacement = """  std::unique_ptr<OrtNode> node;
  if (config.op_type == "WebGpuGreedyWhereArgMax") {
    if (config.inputs.size() != 3 || config.outputs.size() != 2)
      throw std::runtime_error("Invalid WebGPU greedy Where/ArgMax graph signature");
    const char* masked[] = {config.outputs[0].name.c_str()};
    const char* tokens[] = {config.outputs[1].name.c_str()};
    auto where = OrtNode::Create("Where", "", "greedy_where",
                                input_names.data(), input_names.size(), masked, 1, nullptr, 0);
    graph->AddNode(where.get());
    where.release();
    node = OrtNode::Create("ArgMax", "", "greedy_argmax", masked, 1, tokens, 1,
                          node_attr_ptrs.data(), node_attr_ptrs.size());
  } else {
""" + original.replace("  auto node = ", "  node = ") + """  }

"""
    return source[:begin] + replacement + source[end:]

'''
RELINK = r'''
    supported_v4 = Path("/workspace/qwen38-int4-gpu-greedy-argmax/build-v4")
    supported_manifest = supported_v4 / "build-manifest.json"
    supported = json.loads(supported_manifest.read_text())
    supported_link = link.copy()
    supported_runtime = OUTPUT / "supported-v4-relink.so"
    supported_link[supported_link.index("-o") + 1] = str(supported_runtime)
    for relative in ("models/model.cpp", "ep/webgpu/interface.cpp", "models/graph_executor.cpp"):
        old_obj = f"CMakeFiles/onnxruntime-genai-obj.dir/src/{relative}.o"
        if supported_link.count(old_obj) != 1:
            raise RuntimeError(f"Nonunique v4 original object: {relative}")
        supported_link[supported_link.index(old_obj)] = str(supported_v4 / (relative.replace("/", "_") + ".o"))
    subprocess.run(supported_link, cwd=WORKING, check=True)
    if sha(supported_runtime) != supported["runtime_sha256"]:
        raise RuntimeError("Supported v4 runtime relink mismatch")
    sources.extend((supported_manifest, supported_runtime))
'''


if __name__ == "__main__":
    source = (ROOT / "build_gpu_greedy_argmax_v4.py").read_text()
    source = replace_once(source, "gpu-greedy-argmax/build-v4", "gpu-greedy-argmax/build-v5")
    source = replace_once(source, "struct LazyWebGpuLogits final", SELECTOR + "\nstruct LazyWebGpuLogits final")
    source = replace_once(source, "    gpu_min_length_applied_ = false;",
                          "    gpu_min_length_applied_ = false;\n    masked_tokens_ready_ = false;")
    source = replace_once(source, '    ModelConfig config("Where");',
        "    const bool fused = WebGpuGreedyFusedMinLengthEnabled() &&\n"
        "                       rows <= 65535 / ((vocab + 2047) / 2048);\n"
        '    ModelConfig config(fused ? "WebGpuGreedyWhereArgMax" : "Where");')
    source = replace_once(source,
        '    config.outputs.push_back(TensorConfig("masked", Ort::TypeToTensorType<float>, {-1, -1}));',
        '    config.outputs.push_back(TensorConfig("masked", Ort::TypeToTensorType<float>, {-1, -1}));\n'
        "    if (fused) {\n"
        "      if (masked_tokens_.size() != static_cast<size_t>(rows))\n"
        "        masked_tokens_ = device_.Allocate<int64_t>(static_cast<size_t>(rows));\n"
        '      config.outputs.push_back(TensorConfig("tokens", Ort::TypeToTensorType<int64_t>, {-1}));\n'
        '      config.attributes.push_back(AttributeValue::Int("axis", -1));\n'
        '      config.attributes.push_back(AttributeValue::Int("keepdims", 0));\n'
        '      config.attributes.push_back(AttributeValue::Int("select_last_index", 0));\n'
        "    }")
    source = replace_once(source,
        "    GraphExecutor::Execute(config, params);\n    lazy_ =",
        "    if (fused) {\n"
        "      params.outputs.push_back(TensorSpec(masked_tokens_.Span().data(), Ort::TypeToTensorType<int64_t>,\n"
        "                                         {rows}, static_cast<size_t>(rows) * sizeof(int64_t)));\n"
        '      params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64"};\n'
        '      params.session_config_values = {"1"};\n'
        "    }\n"
        "    GraphExecutor::Execute(config, params);\n"
        "    masked_tokens_ready_ = fused;\n"
        "    if (fused && WebGpuGreedyArgMaxTraceEnabled())\n"
        '      std::cerr << "WEBGPU_GREEDY_MIN_LENGTH_FUSED" << std::endl;\n'
        "    lazy_ =")
    source = replace_once(source,
        "    if (!device_.ArgMax(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,\n"
        "                        rows, vocab, tokens.data()))\n"
        '      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");',
        "    if (masked_tokens_ready_) {\n"
        "      const auto cached = masked_tokens_.CopyDeviceToCpu();\n"
        "      for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {\n"
        "        if (cached[row] < 0 || cached[row] >= vocab)\n"
        '          throw std::runtime_error("WebGPU cached ArgMax returned an invalid token ID");\n'
        "        tokens[row] = static_cast<int32_t>(cached[row]);\n"
        "      }\n"
        "      if (WebGpuGreedyArgMaxTraceEnabled())\n"
        '        std::cerr << "WEBGPU_GREEDY_ARGMAX_CACHED" << std::endl;\n'
        "    } else if (!device_.ArgMax(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,\n"
        "                               rows, vocab, tokens.data())) {\n"
        '      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");\n'
        "    }")
    source = replace_once(source, "  bool gpu_min_length_applied_{};",
                          "  DeviceSpan<int64_t> masked_tokens_;\n  bool gpu_min_length_applied_{}, masked_tokens_ready_{};")
    source = replace_once(source, "\ndef main():", GRAPH_TRANSFORM + "\ndef main():")
    source = replace_once(source, "    rebuilt = []", RELINK + "\n    rebuilt = []")
    source = replace_once(source, '("models/graph_executor.cpp", graph_transform)):',
                          '("models/graph_executor.cpp", graph_transform), ("models/graph_builder.cpp", graph_builder_transform)):')
    with (ROOT / "build_gpu_greedy_argmax_v5.py").open("x") as stream:
        stream.write(source)
