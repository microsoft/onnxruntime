import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


DIRECTORY = Path("/workspace/qwen38-int4-gdn-value2/build-v1")
OUTPUT = DIRECTORY.parent / "capture-compare-v1"


def transform(source):
    source = replace_once(source, "      const bool toggle_mid =", '''
      Ort::IoBinding baseline_binding{baseline};
      Ort::IoBinding fused_binding{fused};
      std::vector<Ort::Value> expected_buffers;
      std::vector<Ort::Value> actual_buffers;
      expected_buffers.reserve(names.size());
      actual_buffers.reserve(names.size());
      for (size_t output = 0; output < names.size(); ++output) {
        const auto type_info = fused.GetOutputTypeInfo(output);
        const auto info = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        for (auto dimension : shape)
          if (dimension < 0) throw std::runtime_error("Capture fixture requires fixed output dimensions");
        expected_buffers.emplace_back(
            Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType()));
        actual_buffers.emplace_back(
            Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType()));
        baseline_binding.BindOutput(names[output], expected_buffers.back());
        fused_binding.BindOutput(names[output], actual_buffers.back());
      }
      auto bound_run = [&](Ort::Session& session, Ort::IoBinding& binding) {
        session.Run(Ort::RunOptions{}, binding);
        binding.SynchronizeOutputs();
        return binding.GetOutputValues();
      };
      const bool toggle_mid =''')
    source = replace_once(source,
        "auto expected = baseline.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names.data(), names.size());",
        "auto expected = bound_run(baseline, baseline_binding);")
    return replace_once(source,
        "auto actual = fused.Run(Ort::RunOptions{}, nullptr, nullptr, 0, names.data(), names.size());",
        "auto actual = bound_run(fused, fused_binding);")


def main(directory=DIRECTORY):
    output = directory.parent / "capture-compare-v1"
    original = directory / "native_compare.cpp"
    parent = json.loads((directory / "build-manifest.json").read_text())
    assert sha(original) == parent["source_sha256"][original.name]
    output.mkdir(parents=True, exist_ok=False)
    source = output / "native_compare.cpp"
    source.write_text(transform(original.read_text()))
    binary = output / "native_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(source),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report = dict(status="capture_comparator_built_not_qualified", comparator_sha256=sha(binary),
        source_sha256={source.name: sha(source)}, original_comparator_source_sha256=sha(original),
        provider_sha256=parent["provider_sha256"], fixed_output_io_binding=True)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
