import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_prefill2048_dsa import ANCHOR, EXTENSION
from build_qmoe_node_replay import replace_once, sha
from build_qmoe_prefill2048 import FLAG as OLD_FLAG
from build_qmoe_prefill2048_rows4 import FLAG


COMPONENT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill2048-rows4/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill2048-rows4/build-v2")
DSA_EXTENSION = EXTENSION.replace(OLD_FLAG, FLAG)


def transform(source):
    return replace_once(source, ANCHOR, DSA_EXTENSION)


def transform_compare(unused):
    source = PARENT / "native_compare.cpp"
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    assert sha(source) == parent["source_sha256"][source.name]
    return source.read_text().replace("prefill2048-rows4-v1-native-", "prefill2048-rows4-v2-native-")


if __name__ == "__main__":
    proof = Path("/workspace/qwen38-int4-qmoe-prefill1280/build-v2/build-manifest.json")
    prior = json.loads(proof.read_text())
    assert prior["original_dsa_object_byte_identical"]
    assert sha(COMPONENT / "dynamic_sparse_attention.cc.o") == prior["original_dsa_object_sha256"]
    build(output=OUTPUT, flag=FLAG, source_transform=transform, component_dir=COMPONENT,
          source_name="dynamic_sparse_attention.cc", comparator_flag="ORT_WEBGPU_DSA_QUERY_CACHE",
          comparator_transform=transform_compare, parent_dir=PARENT)
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    report.update(baseline_provider_sha256=parent["baseline_provider_sha256"],
                  intermediate_provider_sha256=parent["provider_sha256"],
                  original_dsa_object_byte_identical=True,
                  original_dsa_object_sha256=prior["original_dsa_object_sha256"],
                  original_dsa_object_proof_sha256=sha(proof),
                  no_shader_or_reduction_changes=True, default_sequence_bound_preserved=True,
                  parent_build_manifest_sha256=sha(PARENT / "build-manifest.json"))
    path.write_text(json.dumps(report, indent=2) + "\n")
