import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_qmoe_prefill512 import FLAG


ORIGINAL_BOUND = '''  if (parameters.sequence_length > static_cast<int>(kMaxSequenceLength)) {
    return NotImplementedBound("sequence_length", parameters.sequence_length, kMaxSequenceLength);
  }'''
EXTENDED_BOUND = f'''  const int prefill512 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, "{FLAG} must be 0 or 1");
  const bool extended_sequence =
      prefill512 == 1 && context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
      parameters.batch_size == 1 && parameters.sequence_length == 512 &&
      static_cast<uint64_t>(parameters.sequence_length) * parameters.num_heads <=
          context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint32_t max_sequence_length = extended_sequence ? 512u : kMaxSequenceLength;
  if (parameters.sequence_length > static_cast<int>(max_sequence_length)) {{
    return NotImplementedBound("sequence_length", parameters.sequence_length, max_sequence_length);
  }}'''


def transform(source):
    return replace_once(source, ORIGINAL_BOUND, EXTENDED_BOUND)


def main():
    parent_dir = Path("/workspace/qwen38-int4-qmoe-prefill512/build-v2")
    output = Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3")
    original = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/dynamic_sparse_attention.cc")
    parent = json.loads((parent_dir / "build-manifest.json").read_text())
    assert sha(parent_dir / "libonnxruntime_providers_webgpu.so") == parent["provider_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    command = parent["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(original)
    baseline_object = output / "original-dsa.cc.o"
    command[command.index("-o") + 1] = str(baseline_object)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    old_objects = [value for value in parent["link_command"] if value.endswith(
        "build-int4-dsa-split-kv/contrib_ops__webgpu__bert__dynamic_sparse_attention.cc.o")]
    assert len(old_objects) == 1
    assert sha(baseline_object) == sha(Path(old_objects[0])), "Original DSA object does not reproduce exactly"
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Unmodified combined-parent relink mismatch"
    library.unlink()
    original_copy = output / "unmodified_dynamic_sparse_attention.cc"
    original_copy.write_text(original.read_text())
    source = output / "dynamic_sparse_attention.cc"
    source.write_text(transform(original_copy.read_text()))
    obj = output / "dynamic_sparse_attention.cc.o"
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    link[link.index(old_objects[0])] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = output / "native_compare.cpp"
    comparator.write_text(replace_once((parent_dir / comparator.name).read_text(),
                                      "prefill512-v2-native-", "prefill512-v3-native-"))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["baseline_provider_sha256"], baseline_byte_identical=True,
        intermediate_provider_sha256=parent["provider_sha256"], original_dsa_object_byte_identical=True,
        original_dsa_object_sha256=sha(baseline_object),
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        comparator_sha256=sha(output / "native_compare"),
        source_sha256={path.name: sha(path) for path in (source, original_copy, comparator)},
        parent_build_manifest_sha256=sha(parent_dir / "build-manifest.json"),
        no_shader_or_reduction_changes=True, default_sequence_bound_preserved=True,
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256",
          "original_dsa_object_byte_identical")}), flush=True)


if __name__ == "__main__":
    main()
