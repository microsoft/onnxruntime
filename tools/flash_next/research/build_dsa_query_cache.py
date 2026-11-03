import json
from pathlib import Path

from build_dsa_partition32 import HEADER, dispatch_private_attention, private_program
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DSA_QUERY_CACHE"
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")


NATIVE_COMPARE = r'''
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("Usage: native_compare provider fixture...");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dsa-query-cache"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[1]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    const char* output_names[] = {"y"};
    const char* pending = std::getenv("ORT_TEST_MAX_PENDING_DISPATCHES");
    if (pending == nullptr || (std::strcmp(pending, "1") != 0 && std::strcmp(pending, "64") != 0))
      throw std::runtime_error("Pending must be 1 or 64");
    auto flag = [](const char* value) {
      if (setenv("ORT_WEBGPU_DSA_QUERY_CACHE", value, 1) != 0)
        throw std::runtime_error("Cannot set query-cache selector");
    };
    size_t comparisons = 0;
    for (int file = 2; file < argc; ++file) {
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      options.AppendExecutionProvider_V2(
          env, devices, {{"enableInt64", "1"}, {"validationMode", "full"},
                         {"maxNumPendingDispatches", pending}, {"enableGraphCapture", "0"}});
      Ort::Session session{env, argv[file], options};
      if (session.GetInputCount() != 0 || session.GetOutputCount() != 1)
        throw std::runtime_error("Expected constant-input attention fixture");
      for (int round = 0; round < 9; ++round) {
        flag("0");
        auto reference = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, output_names, 1);
        const auto info = reference[0].GetTensorTypeAndShapeInfo();
        const auto type = info.GetElementType();
        const size_t count = info.GetElementCount();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          throw std::runtime_error("Expected float16 or float32 attention output");
        const size_t bytes = count * (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 : 4);
        std::vector<uint8_t> expected(bytes);
        std::memcpy(expected.data(), reference[0].GetTensorRawData(), bytes);
        flag("1");
        auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, output_names, 1);
        const auto actual_info = actual[0].GetTensorTypeAndShapeInfo();
        if (actual_info.GetShape() != info.GetShape() || actual_info.GetElementType() != type)
          throw std::runtime_error("Output shape/type changed");
        if (std::memcmp(expected.data(), actual[0].GetTensorRawData(), bytes) != 0)
          throw std::runtime_error(std::string(argv[file]) + " bitwise mismatch round=" + std::to_string(round));
        for (size_t i = 0; i < count; ++i) {
          const float value = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ?
              actual[0].GetTensorData<Ort::Float16_t>()[i].ToFloat() : actual[0].GetTensorData<float>()[i];
          if (!std::isfinite(value)) throw std::runtime_error("Non-finite attention output");
          ++comparisons;
        }
      }
      flag("2");
      bool rejected = false;
      try {
        session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, output_names, 1);
      } catch (const Ort::Exception& error) {
        rejected = std::string(error.what()).find("ORT_WEBGPU_DSA_QUERY_CACHE must be 0 or 1") != std::string::npos;
      }
      if (!rejected) throw std::runtime_error("Invalid query-cache selector was not rejected");
      flag("1");
      std::cout << "PASS finite_bitwise " << std::filesystem::path(argv[file]).stem().string()
                << " rounds=9 invalid_flag_rejected=1" << std::endl;
    }
    std::cout << "PASS total_exact_elements=" << comparisons << "; CPU fallback disabled" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
'''


def cached_program(source, header):
    program = private_program(source, header).replace(
        "DynamicSparseAttentionPartition32Program", "DynamicSparseAttentionQueryCacheProgram").replace(
        'Program{"DynamicSparseAttentionPartition32"}', 'Program{"DynamicSparseAttentionQueryCache"}')
    anchor = "  if (!local_plus_selected_) {\n"
    preload = '''
  for (uint32_t slot = 0; slot < 8; ++slot) {
    body << "  let cached_query" << slot << " = f32("
         << query.GetByOffset("q_base + local_idx + " + std::to_string(slot * 32) + "u") << ");\\n";
  }
'''
    program = replace_once(program, anchor, preload + anchor)
    start = program.index('         << "        for (var c = local_idx;')
    stop = program.index("    if (use_subgroups_) {", start)
    unrolled = '''         ;
    for (uint32_t slot = 0; slot < 8; ++slot) {
      body << "        dot_product += cached_query" << slot << " * f32("
           << candidate_key.GetByOffset("candidate_base + local_idx + " +
                                       std::to_string(slot * 32) + "u") << ");\\n";
    }
'''
    return program[:start] + unrolled + program[stop:]


def transform(source, header, *, fp16_only=False):
    program = cached_program(source, header)
    source = replace_once(source, "Status DynamicSparseAttentionProgram::GenerateShaderCode",
                          program + "Status DynamicSparseAttentionProgram::GenerateShaderCode")
    signature = "Status DynamicSparseAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int query_cache_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(query_cache_mode == 0 || query_cache_mode == 1, "{FLAG} must be 0 or 1");
''')
    anchor = "  constexpr uint32_t split_partition_size = 64;"
    admission = "query_cache_mode == 1 && split_kv && parameters.head_size == 256"
    if fp16_only:
        admission += " && query->IsDataType<MLFloat16>()"
    source = replace_once(source, anchor,
        f"  const bool query_cache = {admission};\n" + anchor)
    return dispatch_private_attention(source, "DynamicSparseAttentionQueryCacheProgram", "query_cache")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=lambda source: transform(source, HEADER.read_text(), fp16_only=True),
          component_dir=COMPONENT, source_name="dynamic_sparse_attention.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024", comparator_transform=lambda unused: NATIVE_COMPARE,
          parent_dir=COMPONENT)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(original_attention_header_sha256=sha(HEADER), original_partition_size_preserved=64,
                  original_dot_order_preserved=True, original_merge_unchanged=True,
                  finite_bitwise_comparator_required=True, fp16_only=True,
                  fp32_original_fallback_preserved=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
