import json
from pathlib import Path
import subprocess

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha
from build_qmoe_prefill1280 import FLAG


COMPONENT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill1280/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill1280/build-v2")
ANCHOR = '''  const uint32_t max_sequence_length =
      extended_sequence1024 ? 1024u : extended_sequence ? 512u : kMaxSequenceLength;'''
EXTENSION = f'''  const int prefill1280 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill1280 == 0 || prefill1280 == 1, "{FLAG} must be 0 or 1");
  const bool extended_sequence1280 =
      prefill1280 == 1 && context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
      parameters.batch_size == 1 &&
      (parameters.sequence_length == 768 || parameters.sequence_length == 1280) &&
      static_cast<uint64_t>(parameters.sequence_length) * parameters.num_heads <=
          context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  const uint32_t max_sequence_length =
      extended_sequence1280 ? 1280u : extended_sequence1024 ? 1024u :
      extended_sequence ? 512u : kMaxSequenceLength;'''


def transform(source):
    return replace_once(source, ANCHOR, EXTENSION)


def transform_compare(unused):
    source = PARENT / "native_compare.cpp"
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    assert sha(source) == parent["source_sha256"][source.name]
    return source.read_text()


if __name__ == "__main__":
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    proof = OUTPUT.parent / "dsa-proof-v1"
    proof.mkdir(parents=True, exist_ok=False)
    command = component["compile_commands"][0].copy()
    obj = proof / "original-dsa.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=component["working_directory"], check=True)
    assert sha(obj) == sha(COMPONENT / "dynamic_sparse_attention.cc.o")
    build(output=OUTPUT, flag=FLAG, source_transform=transform, component_dir=COMPONENT,
          source_name="dynamic_sparse_attention.cc", comparator_flag="ORT_WEBGPU_DSA_QUERY_CACHE",
          comparator_transform=transform_compare, parent_dir=PARENT)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    report.update(baseline_provider_sha256=parent["baseline_provider_sha256"],
                  intermediate_provider_sha256=parent["provider_sha256"],
                  original_dsa_object_byte_identical=True, original_dsa_object_sha256=sha(obj),
                  no_shader_or_reduction_changes=True, default_sequence_bound_preserved=True,
                  existing_512_and_1024_sequence_bounds_preserved=True,
                  parent_build_manifest_sha256=sha(PARENT / "build-manifest.json"))
    path.write_text(json.dumps(report, indent=2) + "\n")
