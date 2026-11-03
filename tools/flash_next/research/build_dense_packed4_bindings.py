import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


PARENT = Path("/workspace/qwen38-int4-dense-packed4/build-v4")
OUTPUT = PARENT.parent / "binding-proof-v1"


GPU_INPUTS = r'''
      if (inputs.size() != 1 ||
          inputs[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        throw std::runtime_error("Binding proof requires one FP16 activation input");
      auto gpu_memory = devices.front().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
      Ort::Allocator gpu_allocator{session, gpu_memory};
      const auto activation_shape = inputs[0].GetTensorTypeAndShapeInfo().GetShape();
      std::vector<std::vector<Ort::Value>> gpu_inputs(3);
      for (auto& bank : gpu_inputs) {
        bank.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            gpu_allocator, activation_shape.data(), activation_shape.size()));
        if (bank[0].GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_GPU)
          throw std::runtime_error("Binding proof activation allocation is not GPU memory");
      }
      for (size_t bank = 0; bank < gpu_inputs.size(); ++bank)
        for (size_t other = 0; other < bank; ++other)
          if (gpu_inputs[bank][0].GetTensorRawData() == gpu_inputs[other][0].GetTensorRawData())
            throw std::runtime_error("Binding proof requires three distinct live GPU buffers");
      std::cout << "Binding proof distinct_live_gpu_inputs=3\n";
'''


def transform(source):
    source = replace_once(source, "      for (int round = 0; round < 128; ++round) {",
                          GPU_INPUTS + "      for (int round = 0; round < 128; ++round) {")
    source = replace_once(source, '        flag("0");',
                          '        auto& bank = gpu_inputs[round % gpu_inputs.size()];\n'
                          '        auto copied = env.CopyTensors(inputs, bank, nullptr);\n'
                          '        if (!copied.IsOK()) throw std::runtime_error(copied.GetErrorMessage());\n'
                          '        flag("0");')
    old = "names.data(), inputs.data(), inputs.size(),"
    assert source.count(old) == 3
    source = source.replace(old, "names.data(), bank.data(), bank.size(),", 2)
    return source


def build_bindings(parent_directory, output_directory, transform_source=transform):
    parent_directory, output_directory = Path(parent_directory), Path(output_directory)
    parent = json.loads((parent_directory / "build-manifest.json").read_text())
    assert parent["baseline_byte_identical"]
    for name, digest in parent["source_sha256"].items():
        assert sha(parent_directory / name) == digest, name
    provider = parent_directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"]
    output_directory.mkdir(parents=True, exist_ok=False)
    source = output_directory / "native_compare.cpp"
    source.write_text(transform_source((parent_directory / "native_compare.cpp").read_text()))
    command = ["c++", "-O3", "-std=c++20", str(source),
               "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
               "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
               "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
               "-lonnxruntime", "-o", str(output_directory / "native_compare")]
    subprocess.run(command, check=True)
    relative_parent = os.path.relpath(parent_directory, output_directory)
    (output_directory / provider.name).symlink_to(Path(relative_parent) / provider.name)
    report = dict(parent)
    report.update(status="built_not_qualified", distinct_live_gpu_inputs=3,
                  binding_proof_compile_command=command,
                  comparator_sha256=sha(output_directory / "native_compare"))
    report["source_sha256"] = {
        relative_parent + "/" + name: digest for name, digest in parent["source_sha256"].items()
        if name != "native_compare.cpp"}
    report["source_sha256"]["native_compare.cpp"] = sha(source)
    (output_directory / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "comparator_sha256", "distinct_live_gpu_inputs")}))

def main():
    build_bindings(PARENT, OUTPUT)



if __name__ == "__main__":
    main()
