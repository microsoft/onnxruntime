import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    root = Path(__file__).parent
    prior = json.loads(Path("/workspace/qwen38-int4-quickgelu-next/build/build-manifest.json").read_text())
    runtime = json.loads((repo / "build-int4-dynamic-uniforms/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-large-binding-cache/build")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    original = repo / "onnxruntime/core/providers/webgpu/webgpu_context.cc"
    assert sha(original) == runtime["source_sha256"][str(original)], "Runtime source has changed"
    link = prior["link_command"].copy()
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(library) == prior["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    text = original.read_text()
    anchor = """  constexpr uint64_t kMaxEntryBytes = 1024 * 1024;
  constexpr uint64_t kMaxCacheBytes = 256 * 1024 * 1024;"""
    assert text.count(anchor) == 1
    text = text.replace(anchor, """  const int large = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_BIND_GROUP_CACHE_LARGE", 0);
  ORT_ENFORCE(large == 0 || large == 1, "ORT_WEBGPU_BIND_GROUP_CACHE_LARGE must be 0 or 1.");
  const uint64_t kMaxEntryBytes = large ? uint64_t{4} * 1024 * 1024 * 1024 : 1024 * 1024;
  const uint64_t kMaxCacheBytes = large ? uint64_t{96} * 1024 * 1024 * 1024 : 256 * 1024 * 1024;""")
    anchor = "  if (cache.entries.size() > capacity) {"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "  if (cache.entries.size() > capacity || cache.referenced_bytes > kMaxCacheBytes) {")
    source = output / "webgpu_context.cc"
    source.write_text(text)
    command = next(c.copy() for c in runtime["compile_commands"] if str(original) in c)
    obj = output / "webgpu_context.cc.o"
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=runtime["working_directory"], env=environment, check=True)
    matches = [p for p in link if p.endswith("onnxruntime__core__providers__webgpu__webgpu_context.cc.o")]
    assert len(matches) == 1
    link[link.index(matches[0])] = str(obj)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native = Path("/workspace/qwen38-int4-dense-next/native_compare.cpp").read_text()
    native = native.replace("ORT_WEBGPU_HC_EXACT_DENSE", "ORT_WEBGPU_BIND_GROUP_CACHE_LARGE")
    native = native.replace("round < 42", "round < 8").replace("times[0][20]", "times[0][3]").replace(
        "times[1][20]", "times[1][3]").replace("samples_per_mode=40", "samples_per_mode=6")
    native_source = output / "native_compare.cpp"
    native_source.write_text(native)
    subprocess.run(["c++", "-O3", "-std=c++20", str(native_source),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_command=command, link_command=link, working_directory=prior["working_directory"],
                  source_sha256={str(source): sha(source)}, quality_qualified=False, headline_eligible=False,
                  large_entry_bytes=4 * 1024**3, large_cache_bytes=96 * 1024**3,
                  scope="Default-off H200-scale resource-retention experiment; no new weight copies")
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
