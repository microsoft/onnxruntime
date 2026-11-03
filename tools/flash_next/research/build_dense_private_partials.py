import json
from pathlib import Path

from build_dense_direct_a import BITCAST_TEMPLATE, BLOCK32_TEMPLATE, COMPONENT, PARENT, transform as clone
from build_dense_double_buffer import transform_compare
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_PRIVATE_PARTIALS"
OUTPUT = Path("/workspace/qwen38-int4-dense-private-partials/build-v3")


def private_partials_template(source):
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//")) + "\n"
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDensePrivatePartialsTemplate(")
    source = replace_once(source, "{ auto& ss = shader_helper.MainFunctionBody();\n",
        '{ auto& ss = shader_helper.MainFunctionBody();\n'
        'ss << "  var private_partial = acc_element_t(0);\\n";\n')
    source = replace_once(source,
        'ss << "        inter_results[local_row_offset + idy][idx] += sum;\\n";',
        'ss << "        private_partial += sum;\\n";')
    return replace_once(source,
        'ss << "    workgroupBarrier();\\n";\nss << "  }\\n";',
        '''ss << "    if (kidx + " << __param_tile_size_k << "u >= uniforms.K) {\\n";
ss << "      inter_results[idy][idx] = private_partial;\\n";
ss << "    }\\n";
ss << "    workgroupBarrier();\\n";
ss << "  }\\n";''')


def transform(source):
    return clone(source, flag=FLAG, name="PrivatePartials", suffix="private_partials", replay_variant=6)


if __name__ == "__main__":
    original = BLOCK32_TEMPLATE / "dense_block32_template.inc"
    frozen = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(original) == frozen["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    includes = OUTPUT.parent / "private-partials-template-v3"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "dense_private_partials_template.inc"
    header.write_text(private_partials_template(original.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, FLAG),
          parent_dir=PARENT, extra_include_dirs=(BITCAST_TEMPLATE, BLOCK32_TEMPLATE, includes))
    manifest = OUTPUT / "build-manifest.json"
    report = json.loads(manifest.read_text())
    report["source_sha256"]["../private-partials-template-v3/dense_private_partials_template.inc"] = sha(header)
    report["fp16_canonical_private_partials"] = True
    report["original_dot_order_activation_staging_and_serial_fold_preserved"] = True
    report["typed_fp16_partial_updates_every_k_tile"] = True
    report["workgroup_threads"] = 128
    report["k_lanes_per_column"] = 32
    report["one_scalar_partial_and_unique_final_shared_slot_per_thread"] = True
    report["private_fp16_rounding_not_yet_gpu_qualified"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
