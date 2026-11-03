import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_QMOE_FAST_FLAGS"
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill512/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-fast-flags/build-v1")
COMPARATOR = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
HELPER = '''
int ReadQMoEExperimentFlag(const char* name, int default_value, bool fast) {
  if (fast) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\\0') return default_value;
    if ((value[0] == '0' || value[0] == '1') && value[1] == '\\0') return value[0] - '0';
  }
  return ParseEnvironmentVariableWithDefault<int>(name, default_value);
}
'''


def transform(source):
    start = source.index("Status QMoE::ComputeInternal(")
    prefix, body = source[:start], source[start:]
    assert body.count("ParseEnvironmentVariableWithDefault<int>") == 24
    body = body.replace("ParseEnvironmentVariableWithDefault<int>", "read_flag")
    setup = f'''
  const int fast_flags = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fast_flags == 0 || fast_flags == 1, "{FLAG} must be 0 or 1");
  auto read_flag = [fast_flags](const char* name, int default_value) {{
    return ReadQMoEExperimentFlag(name, default_value, fast_flags == 1);
  }};
'''
    body = replace_once(body, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
                        "Status QMoE::ComputeInternal(ComputeContext& context) const {" + setup)
    body = replace_once(body, "std::array<std::optional<Tensor>, 5>* scratch) const {",
                        "std::array<std::optional<Tensor>, 5>* scratch) const {" + setup)
    prefix = replace_once(prefix, "namespace {\nstd::atomic<uint64_t> block_fp8_matrix_dispatch_count",
                          "namespace {\n" + HELPER + "std::atomic<uint64_t> block_fp8_matrix_dispatch_count")
    return prefix + body


def transform_compare(unused):
    source = (COMPARATOR / "native_compare.cpp").read_text()
    assert source.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    source = source.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"')
    marker = f'      set_flag("{FLAG}", "2");'
    checks = r'''
      if (file == 2) {
        size_t parser_cases = 0;
        for (const char* name : {
            "ORT_WEBGPU_EXPERT_BITCAST_UNPACK", "ORT_WEBGPU_QMOE_EXTENDED_PREFILL",
            "ORT_WEBGPU_QMOE_PREFILL256", "ORT_WEBGPU_QMOE_PREFILL512",
            "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX",
            "ORT_WEBGPU_QMOE_FC1_ACTIVATION", "ORT_WEBGPU_QMOE_COMMAND_REPLAY",
            "ORT_WEBGPU_QMOE_INT4_COOPERATIVE", "ORT_WEBGPU_QMOE_SUBGROUP_GATE",
            "ORT_WEBGPU_QMOE_INT4_PREFILL", "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL",
            "ORT_WEBGPU_MATMUL_NBITS_SKINNY", "ORT_WEBGPU_MATMUL_GEMV_N96"}) {
          const char* original = std::getenv(name);
          const bool was_set = original != nullptr;
          const std::string saved = was_set ? original : "";
          for (const char* value : {"", "0", "1", "+1", "01", " 1", "1 ", "-0",
                                    "2", "-1", "no", "1x", "2147483648", static_cast<const char*>(nullptr)}) {
            if (value == nullptr) {
              if (unsetenv(name) != 0) throw std::runtime_error("Cannot unset parser fixture flag");
            } else {
              set_flag(name, value);
            }
            std::string reference_error, candidate_error;
            std::vector<Ort::Float16_t> expected;
            size_t expected_size = 0;
            set_flag("ORT_WEBGPU_QMOE_FAST_FLAGS", "0");
            try {
              auto result = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
              expected_size = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
              const auto* begin = result[0].GetTensorData<Ort::Float16_t>();
              expected.assign(begin, begin + expected_size);
            } catch (const Ort::Exception& error) {
              reference_error = error.what();
            }
            set_flag("ORT_WEBGPU_QMOE_FAST_FLAGS", "1");
            try {
              auto result = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);
              if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != expected_size ||
                  std::memcmp(expected.data(), result[0].GetTensorData<Ort::Float16_t>(),
                              expected.size() * sizeof(Ort::Float16_t)) != 0)
                throw std::runtime_error("Parser edge fixture changed output");
            } catch (const Ort::Exception& error) {
              candidate_error = error.what();
            }
            if (reference_error != candidate_error)
              throw std::runtime_error(std::string("Parser error mismatch: ") + name);
            ++parser_cases;
          }
          if (was_set) set_flag(name, saved.c_str());
          else if (unsetenv(name) != 0) throw std::runtime_error("Cannot restore parser fixture flag");
        }
        std::cout << "PASS parser_edge_cases=" << parser_cases << std::endl;
      }
'''
    return replace_once(source, marker, checks + marker)


if __name__ == "__main__":
    frozen = json.loads((COMPARATOR / "build-manifest.json").read_text())
    assert sha(COMPARATOR / "native_compare.cpp") == frozen["source_sha256"]["native_compare.cpp"]
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="qmoe.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL512", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"),))
