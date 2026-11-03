from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
MIN_LENGTH = r'''
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
'''


if __name__ == "__main__":
    source = (ROOT / "build_gpu_greedy_argmax_v3.py").read_text()
    source = replace_once(source, "gpu-greedy-argmax/build-v3", "gpu-greedy-argmax/build-v4")
    source = replace_once(source, "    lazy_.reset();", "    lazy_.reset();\n    gpu_min_length_applied_ = false;")
    source = replace_once(source, "  void SelectTop() override {", MIN_LENGTH + "\n  void SelectTop() override {")
    source = replace_once(source, "  std::shared_ptr<LazyWebGpuLogits> lazy_;",
        "  std::shared_ptr<LazyWebGpuLogits> lazy_;\n"
        "  DeviceSpan<uint8_t> eos_mask_;\n"
        "  DeviceSpan<float> lowest_, masked_;\n"
        "  std::vector<int32_t> cached_eos_;\n"
        "  bool gpu_min_length_applied_{};")
    source = replace_once(source, "#include <iostream>", "#include <iostream>\\n#include <limits>")
    with (ROOT / "build_gpu_greedy_argmax_v4.py").open("x") as stream:
        stream.write(source)
    comparator = (ROOT / "argmax_parallel_compare_bitwise.cpp").read_text()
    comparator = replace_once(comparator, "bool timing, bool exclude_subnormals) {",
                               "bool timing, bool exclude_subnormals, bool masked) {")
    comparator = replace_once(comparator, "x[n - 1] = T(std::is_same_v<T, float>",
                               "x[masked ? n / 2 : n - 1] = T(std::is_same_v<T, float>")
    comparator = replace_once(comparator, "      float best = as_float(input[row * n]);",
        "      auto score = [&](size_t i) {\n"
        "        const bool eos = i == 0 || i == std::min<size_t>(127, n - 1) ||\n"
        "                         i == std::min<size_t>(2048, n - 1) || i == n - 1;\n"
        "        return masked && eos ? std::numeric_limits<float>::lowest() : as_float(input[row * n + i]);\n"
        "      };\n"
        "      float best = score(0);")
    comparator = replace_once(comparator, "const float candidate = as_float(input[row * n + i]);",
                               "const float candidate = score(i);")
    comparator = comparator.replace('std::string(argv[1]) != "compare-normal"))',
                                     'std::string(argv[1]) != "compare-normal" && std::string(argv[1]) != "compare-masked"))')
    comparator = comparator.replace('std::string(argv[1]) == "compare-normal");',
                                     'std::string(argv[1]) == "compare-normal", std::string(argv[1]) == "compare-masked");')
    with (ROOT / "gpu_greedy_mask_compare.cpp").open("x") as stream:
        stream.write(comparator)
