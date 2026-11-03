import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
SOURCE = Path("/workspace/onnxruntime-genai-webgpu-fp8-checkpoint")
WORKING = SOURCE / "build-webgpu-fp8/Release"
CPU = Path("/workspace/qwen38-int4-cpu-greedy-argmax/build-v1")
DENSE = Path("/workspace/qwen38-int4-dense-columns8-double-buffer/build-v1")
ARGMAX = Path("/workspace/qwen38-int4-argmax-parallel/build-v4")
OUTPUT = Path("/workspace/qwen38-int4-gpu-greedy-argmax/build-v4")
FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX"
SELECTOR = f'''
namespace {{
bool WebGpuGreedyArgMaxEnabled() {{
  static const bool enabled = []() {{
    bool value = false;
    GetEnv("{FLAG}", value);
    if (value) {{
      bool shader = false;
      GetEnv("ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE", shader);
      if (!shader)
        throw std::runtime_error("{FLAG}=1 requires ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1");
    }}
    return value;
  }}();
  return enabled;
}}
}}
'''
LAZY_SEARCH = r'''

static bool WebGpuGreedyArgMaxTraceEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE", value);
    return value;
  }();
  return enabled;
}

struct LazyWebGpuLogits final : DeviceBuffer {
  explicit LazyWebGpuLogits(DeviceSpan<float> logits) : logits_{logits} {
    p_device_ = reinterpret_cast<uint8_t*>(logits_.Span().data());
    size_in_bytes_ = logits_.size() * sizeof(float);
  }
  const char* GetType() const override { return device_label; }
  void AllocateCpu() override {
    if (!cpu_ready_) {
      const auto cpu = logits_.CopyDeviceToCpu();
      p_cpu_ = reinterpret_cast<uint8_t*>(cpu.data());
      cpu_ready_ = true;
    }
    cpu_accessed_ = true;
  }
  void CopyDeviceToCpu() override {
    // Materialized CPU scores are authoritative for this search step.
    AllocateCpu();
  }
  void CopyCpuToDevice() override {
    if (!cpu_ready_)
      throw std::runtime_error("WebGPU greedy CPU scores have not been materialized");
    logits_.CopyCpuToDevice();
    cpu_accessed_ = true;
  }
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, size);
  }
  void Zero() override {
    logits_.Zero();
    cpu_ready_ = cpu_accessed_ = false;
    p_cpu_ = nullptr;
  }
  bool CpuAccessed() const { return cpu_accessed_; }
 private:
  DeviceSpan<float> logits_;
  bool cpu_ready_{}, cpu_accessed_{};
};

struct GreedySearch_WebGpu final : GreedySearch_Cpu {
  GreedySearch_WebGpu(const GeneratorParams& params, DeviceInterface& device)
      : GreedySearch_Cpu{params}, device_{device} {}
  void SetLogits(DeviceSpan<float> logits) override {
    lazy_.reset();
    gpu_min_length_applied_ = false;
    if (logits.empty())
      throw std::runtime_error("WebGPU greedy requires nonempty logits");
    if (logits.CpuSpan().data() == logits.Span().data()) {
      GreedySearch_Cpu::SetLogits(logits);
      return;
    }
    lazy_ = std::make_shared<LazyWebGpuLogits>(logits);
    next_token_scores_ = DeviceSpan<float>{std::shared_ptr<DeviceBuffer>{lazy_}};
  }

  void ApplyMinLength(int min_length, std::optional<int> sequence_length = std::nullopt) override {
    const int length = sequence_length.value_or(sequences_.GetSequenceLength());
    if (length >= min_length)
      return;
    const int vocab = params_->config.model.vocab_size;
    const int rows = params_->search.batch_size;
    const auto& eos = params_->config.model.eos_token_id;
    if (!lazy_ || lazy_->CpuAccessed() || vocab < 32 || vocab > 1048576) {
      GreedySearch_Cpu::ApplyMinLength(min_length, sequence_length);
      return;
    }
    if (gpu_min_length_applied_) {
      if (cached_eos_ != eos)
        GreedySearch_Cpu::ApplyMinLength(min_length, sequence_length);
      return;
    }
    if (next_token_scores_.size() != static_cast<size_t>(rows) * vocab)
      throw std::runtime_error("WebGPU min-length logits shape mismatch");
    if (eos_mask_.size() != static_cast<size_t>(vocab) || cached_eos_ != eos) {
      std::vector<uint8_t> mask(vocab, 0);
      for (auto token : eos) {
        if (token >= 0 && token < vocab)
          mask[token] = 1;
      }
      eos_mask_ = device_.Allocate<uint8_t>(static_cast<size_t>(vocab));
      eos_mask_.CopyFromCpu(mask);
      cached_eos_ = eos;
    }
    if (lowest_.empty()) {
      const float lowest = std::numeric_limits<float>::lowest();
      lowest_ = device_.Allocate<float>(1);
      lowest_.CopyFromCpu(std::span<const float>{&lowest, 1});
    }
    if (masked_.size() != static_cast<size_t>(rows) * vocab)
      masked_ = device_.Allocate<float>(static_cast<size_t>(rows) * vocab);
    ModelConfig config("Where");
    config.inputs.push_back(TensorConfig("condition", Ort::TypeToTensorType<bool>, {-1}));
    config.inputs.push_back(TensorConfig("lowest", Ort::TypeToTensorType<float>, {1}));
    config.inputs.push_back(TensorConfig("scores", Ort::TypeToTensorType<float>, {-1, -1}));
    config.outputs.push_back(TensorConfig("masked", Ort::TypeToTensorType<float>, {-1, -1}));
    auto memory = device_.GetMemoryInfo();
    ExecutionParams params(DeviceType::WEBGPU, "WebGpuExecutionProvider", memory.get());
    params.inputs.push_back(TensorSpec(eos_mask_.Span().data(), Ort::TypeToTensorType<bool>,
                                      {vocab}, static_cast<size_t>(vocab)));
    params.inputs.push_back(TensorSpec(lowest_.Span().data(), Ort::TypeToTensorType<float>, {1}, sizeof(float)));
    params.inputs.push_back(TensorSpec(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,
                                      {rows, vocab}, static_cast<size_t>(rows) * vocab * sizeof(float)));
    params.outputs.push_back(TensorSpec(masked_.Span().data(), Ort::TypeToTensorType<float>,
                                       {rows, vocab}, static_cast<size_t>(rows) * vocab * sizeof(float)));
    GraphExecutor::Execute(config, params);
    lazy_ = std::make_shared<LazyWebGpuLogits>(masked_);
    next_token_scores_ = DeviceSpan<float>{std::shared_ptr<DeviceBuffer>{lazy_}};
    gpu_min_length_applied_ = true;
    if (WebGpuGreedyArgMaxTraceEnabled())
      std::cerr << "WEBGPU_GREEDY_MIN_LENGTH rows=" << rows << " vocabulary=" << vocab << std::endl;
  }

  void SelectTop() override {
    const int vocab = params_->config.model.vocab_size;
    const int rows = params_->search.batch_size;
    if (!lazy_ || lazy_->CpuAccessed() || vocab < 32 || vocab > 1048576 ||
        rows > 65535 / ((vocab + 2047) / 2048)) {
      if (WebGpuGreedyArgMaxTraceEnabled())
        std::cerr << "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" << std::endl;
      GreedySearch_Cpu::SelectTop();
      return;
    }
    if (next_token_scores_.size() != static_cast<size_t>(rows) * vocab)
      throw std::runtime_error("WebGPU greedy logits shape does not match batch/vocabulary");
    std::vector<int32_t> tokens(rows);
    if (!device_.ArgMax(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,
                        rows, vocab, tokens.data()))
      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");
    if (WebGpuGreedyArgMaxTraceEnabled())
      std::cerr << "WEBGPU_GREEDY_ARGMAX rows=" << rows << " vocabulary=" << vocab
                << " token_bytes=" << rows * sizeof(int64_t) << std::endl;
    for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
      if (!PadIfAlreadyEOS(row))
        SetNextToken(row, tokens[row]);
    }
    if (!done_)
      AppendNextTokensToSequences();
  }
 private:
  DeviceInterface& device_;
  std::shared_ptr<LazyWebGpuLogits> lazy_;
  DeviceSpan<uint8_t> eos_mask_;
  DeviceSpan<float> lowest_, masked_;
  std::vector<int32_t> cached_eos_;
  bool gpu_min_length_applied_{};
};
'''
ARGMAX_METHOD = r'''
  bool ArgMax(const void* logits, ONNXTensorElementDataType type, int rows, int vocab,
              int32_t* out_tokens) override {
    if (!WebGpuGreedyArgMaxEnabled())
      return false;
    if (type != Ort::TypeToTensorType<float> && type != Ort::TypeToTensorType<Ort::Float16_t>)
      return false;
    if (!logits || !out_tokens || rows <= 0 || vocab <= 0)
      throw std::runtime_error("Invalid WebGPU ArgMax input/output dimensions");
    if (vocab < 32 || vocab > 1048576 || rows > 65535 / ((vocab + 2047) / 2048))
      return false;
    if (!ort_allocator_)
      throw std::runtime_error("WebGPU allocator not initialized");
    ModelConfig config("ArgMax");
    config.inputs.push_back(TensorConfig("input", type, {-1, -1}));
    config.outputs.push_back(TensorConfig("output", Ort::TypeToTensorType<int64_t>, {-1}));
    config.attributes.push_back(AttributeValue::Int("axis", -1));
    config.attributes.push_back(AttributeValue::Int("keepdims", 0));
    config.attributes.push_back(AttributeValue::Int("select_last_index", 0));
    auto tokens = Allocate<int64_t>(static_cast<size_t>(rows));
    ExecutionParams params(GetType(), "WebGpuExecutionProvider", ort_memory_info_);
    params.inputs.push_back(TensorSpec(const_cast<void*>(logits), type, {rows, vocab},
                                      static_cast<size_t>(rows) * vocab * Ort::SizeOf(type)));
    params.outputs.push_back(TensorSpec(tokens.Span().data(), Ort::TypeToTensorType<int64_t>,
                                       {rows}, static_cast<size_t>(rows) * sizeof(int64_t)));
    params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64"};
    params.session_config_values = {"1"};
    GraphExecutor::Execute(config, params);
    const auto cpu_tokens = tokens.CopyDeviceToCpu();
    for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
      if (cpu_tokens[row] < 0 || cpu_tokens[row] >= vocab)
        throw std::runtime_error("WebGPU ArgMax returned an invalid token ID");
      out_tokens[row] = static_cast<int32_t>(cpu_tokens[row]);
    }
    return true;
  }
'''


def model_transform(source):
    source = replace_once(source, '#include "model.h"', '#include "model.h"\n#include "models/env_utils.h"')
    source = replace_once(source, "namespace Generators {", "namespace Generators {\n" + SELECTOR)
    source = replace_once(source,
        "  p_device_logits_ = (p_device_->GetType() == DeviceType::AMDGPU) ? GetDeviceInterface(DeviceType::CPU) : p_device_inputs_;",
        "  p_device_logits_ = (p_device_->GetType() == DeviceType::AMDGPU) ? GetDeviceInterface(DeviceType::CPU) : p_device_inputs_;\n"
        "  if (p_device_->GetType() == DeviceType::WEBGPU && WebGpuGreedyArgMaxEnabled())\n"
        "    p_device_logits_ = p_device_;")
    begin = source.index("Model::Model(std::unique_ptr<Config>")
    before, constructor = source[:begin], source[begin:]
    constructor = replace_once(constructor,
        "      p_device_->GetType() == DeviceType::NvTensorRtRtx)\n    p_device_scoring_ = p_device_;",
        "      p_device_->GetType() == DeviceType::NvTensorRtRtx ||\n"
        "      (p_device_->GetType() == DeviceType::WEBGPU && WebGpuGreedyArgMaxEnabled()))\n"
        "    p_device_scoring_ = p_device_;")
    return before + constructor


def interface_transform(source):
    source = replace_once(source, '#include "interface.h"',
                          '#include "interface.h"\n#include "models/env_utils.h"\n#include "models/graph_builder.h"\n#include <iostream>\n#include <limits>')
    source = replace_once(source, "namespace WebGPU {", "namespace WebGPU {\n" + SELECTOR)
    source = replace_once(source, "struct InterfaceImpl : DeviceInterface {",
                          LAZY_SEARCH + "\nstruct InterfaceImpl : DeviceInterface {")
    source = replace_once(source,
        "  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }",
        "  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {\n"
        "    if (WebGpuGreedyArgMaxEnabled()) return std::make_unique<GreedySearch_WebGpu>(params, *this);\n"
        "    return std::make_unique<GreedySearch_Cpu>(params);\n  }")
    return replace_once(source, "  bool Cast(void* input, void* output,", ARGMAX_METHOD + "\n  bool Cast(void* input, void* output,")



def graph_transform(source):
    source = replace_once(source, '#include "session_options.h"',
                          '#include "session_options.h"\n#include "models/env_utils.h"')
    source = replace_once(source, "namespace Generators {", "namespace Generators {\n" + SELECTOR)
    return replace_once(source, "  auto session_options = OrtSessionOptions::Create();",
        "  auto session_options = OrtSessionOptions::Create();\n"
        "  if (device_type == DeviceType::WEBGPU && WebGpuGreedyArgMaxEnabled())\n"
        "    session_options->SetIntraOpNumThreads(1);")


def main():
    preflight_path, cpu_manifest_path = CPU / "preflight.json", CPU / "build-manifest.json"
    preflight, cpu_manifest = [json.loads(p.read_text()) for p in (preflight_path, cpu_manifest_path)]
    if not preflight["baseline_relink_byte_identical"] or not preflight["baseline_search_rebuild_byte_identical"]:
        raise RuntimeError("GenAI baseline provenance missing")
    if sha(CPU / "libonnxruntime-genai.so") != cpu_manifest["runtime_sha256"]:
        raise RuntimeError("Changed supported CPU lanes8 runtime")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    runtime = OUTPUT / "libonnxruntime-genai.so"
    link = preflight["link_command"].copy()
    link[link.index("-o") + 1] = str(runtime)
    search_baseline = str(CPU / "search.baseline.cpp.o")
    if link.count(search_baseline) != 1:
        raise RuntimeError("Nonunique original search link")
    link[link.index(search_baseline)] = str(CPU / "search.cpp.o")
    subprocess.run(link, cwd=WORKING, check=True)
    if sha(runtime) != cpu_manifest["runtime_sha256"]:
        raise RuntimeError("Supported lanes8 baseline relink mismatch")
    sources = [Path(__file__), preflight_path, cpu_manifest_path, CPU / "search.cpp.o",
               CPU / "libonnxruntime-genai.so", SOURCE / "src/smartptrs.h", SOURCE / "src/search.h",
               SOURCE / "src/models/env_utils.h", SOURCE / "src/models/graph_executor.h"]
    rebuilt = []
    for relative, transform in (("models/model.cpp", model_transform), ("ep/webgpu/interface.cpp", interface_transform),
                                ("models/graph_executor.cpp", graph_transform)):
        original = SOURCE / "src" / relative
        old_obj = f"CMakeFiles/onnxruntime-genai-obj.dir/src/{relative}.o"
        if link.count(old_obj) != 1:
            raise RuntimeError(f"Original TU link not unique: {relative}")
        obj = OUTPUT / (relative.replace("/", "_") + ".o")
        command = [*preflight["compile_command"], "-c", str(original), "-o", str(obj)]
        subprocess.run(command, cwd=WORKING, check=True)
        if sha(obj) != sha(WORKING / old_obj):
            raise RuntimeError(f"Original GenAI TU did not rebuild identically: {relative}")
        candidate = OUTPUT / relative.replace("/", "_")
        candidate.write_text(transform(original.read_text()))
        command = [*preflight["compile_command"], "-iquote", str(original.parent),
                   f"-fmacro-prefix-map={candidate}={original}", "-c", str(candidate), "-o", str(obj)]
        subprocess.run(command, cwd=WORKING, check=True)
        link[link.index(old_obj)] = str(obj)
        sources.extend((original, candidate, obj))
        rebuilt.append(relative)
    subprocess.run([*link, "-Wl,--no-undefined"], cwd=WORKING, check=True)
    dense_manifest_path, argmax_manifest_path = DENSE / "build-manifest.json", ARGMAX / "build-manifest.json"
    dense, argmax = [json.loads(p.read_text()) for p in (dense_manifest_path, argmax_manifest_path)]
    for directory, manifest in ((DENSE, dense), (ARGMAX, argmax)):
        if sha(directory / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"]:
            raise RuntimeError("Changed qualified provider")
        for path, digest in manifest["source_sha256"].items():
            candidate = Path(path) if Path(path).is_absolute() else directory / path
            if sha(candidate) != digest:
                raise RuntimeError(f"Changed provider input: {path}")
    provider = OUTPUT / "libonnxruntime_providers_webgpu.so"
    provider_link = dense["link_command"].copy()
    provider_link[provider_link.index("-o") + 1] = str(provider)
    provider_working = Path(json.loads(Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/build-manifest.json").read_text())["working_directory"])
    subprocess.run(provider_link, cwd=provider_working, check=True)
    if sha(provider) != dense["provider_sha256"]:
        raise RuntimeError("Supported double-buffer provider relink mismatch")
    reduction, = [x for x in provider_link if x.endswith("core__providers__webgpu__reduction__reduction_ops.cc.o")]
    provider_link[provider_link.index(reduction)] = str(ARGMAX / "reduction_ops.cc.o")
    subprocess.run(provider_link, cwd=provider_working, check=True)
    sources.extend((dense_manifest_path, argmax_manifest_path, ARGMAX / "reduction_ops.cc.o"))
    report = dict(status="built_not_qualified", runtime_sha256=sha(runtime), provider_sha256=sha(provider),
        parent_runtime_sha256=cpu_manifest["runtime_sha256"], parent_provider_sha256=dense["provider_sha256"],
        flag=FLAG, shader_flag="ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE", default_enabled=False,
        public_headers_and_layouts_unchanged=True, private_leaf_search_and_buffer_only=True,
        original_genai_translation_units_rebuilt_identically=rebuilt,
        supported_runtime_and_provider_relinked_identically=True,
        cpu_sampling_penalties_and_api_access_materialize_scores=True,
        model_eligible=False, link_command=link, provider_link_command=provider_link,
        source_sha256={str(path): sha(path) for path in sources})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], runtime_sha256=report["runtime_sha256"],
                         provider_sha256=report["provider_sha256"])), flush=True)


if __name__ == "__main__":
    main()
