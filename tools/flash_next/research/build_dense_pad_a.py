import json
from pathlib import Path
import re

from build_dense_direct_a import BITCAST_TEMPLATE, BLOCK32_TEMPLATE, COMPONENT, transform as clone
from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_PAD_A"
OUTPUT = Path("/workspace/qwen38-int4-dense-pad-a/build-v1")
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")


def padded_template(source):
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//")) + "\n"
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDensePadATemplate(")
    source = replace_once(source, '''ss << "var<workgroup> tile_A : array<input_a_value_t, ";
ss << __param_a_length_per_tile;
ss << ">;\\n";''', '''ss << "var<workgroup> tile_A : array<input_a_value_t, ";
ss << ((__param_a_length_per_tile + 7) / 8) * 9;
ss << ">;\\n";
ss << "fn dense_padded_a_index(index: u32) -> u32 { return (index / 8u) * 9u + index % 8u; }\\n";''')
    references = source.count("tile_A[")
    source, count = re.subn(r"tile_A\[(col|a_offset(?: \+ \d+)?)\]",
                           r"tile_A[dense_padded_a_index(\1)]", source)
    assert count == references and count > 30
    return source


def transform(source):
    return clone(source, flag=FLAG, name="PadA", suffix="pad_a", replay_variant=7)


if __name__ == "__main__":
    original = BLOCK32_TEMPLATE / "dense_block32_template.inc"
    frozen = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(original) == frozen["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    includes = OUTPUT.parent / "pad-a-template-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_pad_a_template.inc"
    header.write_text(padded_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, FLAG),
          parent_dir=PARENT, extra_include_dirs=(BITCAST_TEMPLATE, BLOCK32_TEMPLATE, includes))
    manifest = OUTPUT / "build-manifest.json"
    report = json.loads(manifest.read_text())
    report["source_sha256"]["../pad-a-template-v1/dense_pad_a_template.inc"] = sha(header)
    report["original_dot_order_activation_staging_and_serial_fold_preserved"] = True
    report["activation_padding_vector_block"] = 8
    report["activation_padding_vector_pitch"] = 9
    report["fp16_canonical_only"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
