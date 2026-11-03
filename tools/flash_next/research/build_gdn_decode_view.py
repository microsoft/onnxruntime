import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(text):
    anchor = "  std::optional<Tensor> packed_qkv;\n  if (use_packed_qkv) {"
    assert text.count(anchor) == 1
    text = text.replace('#include <cmath>', '#include <cmath>\n#include "core/platform/env_var_utils.h"')
    text = text.replace("&*packed_qkv", "packed_qkv_input")
    text = text.replace(".AddOutput({packed_qkv_input,", ".AddOutput({&*packed_qkv,")
    text = text.replace("ProgramOutput::BufferView(packed_qkv_input,",
                        "ProgramOutput::BufferView(&*packed_qkv,")
    text = text.replace(
        anchor, """  const int decode_view_flag =
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW", 0);
  ORT_RETURN_IF_NOT(decode_view_flag == 0 || decode_view_flag == 1,
                    "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW must be 0 or 1");
  std::optional<Tensor> packed_qkv;
  const Tensor* packed_qkv_input = nullptr;
  if (use_packed_qkv && input_is_packed_qkv && total_tokens == 1 && decode_view_flag == 1) {
    packed_qkv_input = query;
  } else if (use_packed_qkv) {""")
    anchor = "TensorShape{onnxruntime::narrow<int64_t>(qkv_element_count)}));"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "\n    packed_qkv_input = &*packed_qkv;")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=("quickgelu", "postnorm"), default="quickgelu")
    args = parser.parse_args()
    root = Path(__file__).parent
    prior_path = ("/workspace/qwen38-int4-quickgelu-next/build/build-manifest.json" if args.base == "quickgelu"
                  else "/workspace/qwen38-int4-hc-post-norm/build-v5/build-manifest.json")
    prior = json.loads(Path(prior_path).read_text())
    output = Path("/workspace/qwen38-int4-gdn-decode-view/" +
                  ("build-original" if args.base == "quickgelu" else "build"))
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    link = prior["link_command"].copy()
    library = output / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(library) == prior["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    original = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.cc")
    source = output / original.name
    source.write_text(transform(original.read_text()))
    obj = output / "gated_delta_net.cc.o"
    command = prior["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=prior["working_directory"], env=environment, check=True)
    matches = [p for p in link if p.endswith("contrib_ops__webgpu__bert__gated_delta_net.cc.o")]
    assert len(matches) == 1, matches
    link[link.index(matches[0])] = str(obj)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native = (root / "hc_fusion_compare.cpp").read_text().replace(
        "ORT_TEST_TOGGLE_EXACT_MID", "ORT_TEST_TOGGLE_GDN_VIEW").replace(
        "ORT_WEBGPU_EXACT_DENSE_MID", "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW").replace(
        'set_mid(run == 1 ? "1" : "0")', 'set_mid("1")')
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
                  source_sha256={str(source): sha(source), str(original): sha(original)},
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
