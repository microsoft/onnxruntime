import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-pair-accumulator/stress-compare-v1")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def transform(source):
    source = replace_once(source, 'if (argc != 12) throw std::runtime_error("Usage: comparator capture|compare provider reference eight-fixtures...");',
                          'if (argc != 5) throw std::runtime_error("Usage: comparator capture|compare provider reference fixture-list");')
    source = replace_once(source, "std::vector<Ort::Float16_t> reference(2048);", """std::ifstream list(argv[4]);
    if (!list) throw std::runtime_error("Cannot open fixture list");
    std::vector<std::string> fixtures;
    for (std::string path; std::getline(list, path);) {
      if (path.empty()) throw std::runtime_error("Empty fixture path");
      fixtures.push_back(std::move(path));
    }
    if (fixtures.empty()) throw std::runtime_error("Empty fixture list");
    std::vector<Ort::Float16_t> reference(fixtures.size() * 256);""")
    source = replace_once(source, "for (int file = 4; file < argc; ++file)",
                          "for (size_t file = 0; file < fixtures.size(); ++file)")
    source = replace_once(source, "Ort::Session session{env, argv[file], options};",
                          "Ort::Session session{env, fixtures[file].c_str(), options};")
    return source.replace("(file - 4) * 256", "file * 256").replace('<< file - 4', '<< file')


if __name__ == "__main__":
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed SDK input: {name}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / "sweep_compare.cpp"
    source.write_text(transform((ROOT / "fp16_dot4_cross_shader_compare.cpp").read_text()))
    binary = OUTPUT / "sweep_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(source),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report = dict(status="built_not_qualified", sdk_provenance_checked=True,
                  source_sha256=sha(source), comparator_sha256=sha(binary),
                  provider_changed=False, model_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
