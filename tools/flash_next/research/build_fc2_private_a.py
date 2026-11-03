import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
PARENT = Path("/workspace/qwen38-int4-gpu-slice-copy/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1")
INCLUDE = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")
FLAG = "ORT_WEBGPU_QMOE_FC2_PRIVATE_A"


def transform(source):
    source = replace_once(source, "class QmoeFC2FinalMixProgram final", f'''namespace {{
bool Fc2PrivateAEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    return value == 1;
  }}();
  return enabled;
}}
}}

class QmoeFC2FinalMixProgram final''')
    source = replace_once(source, 'Program{bitcast ? "QmoeFC2FinalMixBitcastF16"',
        'Program{Fc2PrivateAEnabled() && bitcast ? "QmoeFC2PrivateABitcastF16" : bitcast ? "QmoeFC2FinalMixBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, private_a_{Fc2PrivateAEnabled() && bitcast} {}")
    source = replace_once(source,
        '    shader.AdditionalImplementation()\n        << "var<workgroup> tile_a: array<vec4<f16>, 1600>;\\n"',
        '    if (!private_a_)\n      shader.AdditionalImplementation() << "var<workgroup> tile_a: array<vec4<f16>, 1600>;\\n";\n'
        '    shader.AdditionalImplementation()')
    start = source.index('         << "for (var id = local_idx; id < 1600u;')
    stop = source.index('         << "  for (var c = 0u;', start)
    original = source[start:stop]
    source = source[:start] + ''';\n
    if (!private_a_) {
      code ''' + original[:original.index('         << "if (lane < 20u)')] + ''';\n
    }
    code << "if (lane < 20u) {\\n";
    if (private_a_) {
      code << "  var cached_a: array<vec4<f16>, 8>;\\n"
           << "  for (var i = 0u; i < 8u; i++) {\\n"
           << "    cached_a[i] = " << a.GetByOffset("rank * 160u + lane * 8u + i") << ";\\n"
           << "  }\\n";
    }
    code ''' + source[stop:]
    source = replace_once(source,
        '"      sum += dot(tile_a[rank * 160u + lane * 8u + i * 2u], w0) + "\n'
        '            "dot(tile_a[rank * 160u + lane * 8u + i * 2u + 1u], w1);\\n"',
        '(private_a_ ? "      sum += dot(cached_a[i * 2u], w0) + dot(cached_a[i * 2u + 1u], w1);\\n" :\n'
        '          "      sum += dot(tile_a[rank * 160u + lane * 8u + i * 2u], w0) + "\n'
        '          "dot(tile_a[rank * 160u + lane * 8u + i * 2u + 1u], w1);\\n")')
    return replace_once(source, "  bool bitcast_;\n", "  bool bitcast_;\n  bool private_a_;\n")


if __name__ == "__main__":
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    working = Path(json.loads(Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/build-manifest.json").read_text())["working_directory"])
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed parent provider")
    original = COMPONENT / "qmoe.cc"
    if sha(original) != component["source_sha256"]["qmoe.cc"]:
        raise RuntimeError("Changed QMoE source")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=working, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Composite parent relink mismatch")
    command = next(c.copy() for c in component["compile_commands"]
                   if c[c.index("-c") + 1] == str(original))
    baseline = OUTPUT / "qmoe.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    subprocess.run(command, cwd=working, check=True)
    if sha(baseline) != sha(COMPONENT / "qmoe.cc.o"):
        raise RuntimeError("Original QMoE object not byte-identical")
    include_source = INCLUDE / "qmoe_fc2_final_mix.inc"
    candidate_include = OUTPUT / "qmoe_fc2_final_mix.inc"
    candidate_include.write_text(transform(include_source.read_text()))
    candidate = OUTPUT / "qmoe.cc"
    candidate.write_text(original.read_text())
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-c") + 1] = str(candidate)
    command[command.index("-o") + 1] = str(obj)
    command.insert(1, f"-fmacro-prefix-map={candidate}={original}")
    subprocess.run(command, cwd=working, check=True)
    old = str(COMPONENT / "qmoe.cc.o")
    if link.count(old) != 1:
        raise RuntimeError("Nonunique QMoE link object")
    link[link.index(old)] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    comparator = Path("/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2/native_compare")
    sources = (Path(__file__), PARENT / "build-manifest.json", COMPONENT / "build-manifest.json",
               original, include_source, candidate, candidate_include, obj, baseline, comparator)
    result = dict(status="fc2_private_a_built", flag=FLAG, default_enabled=False,
        provider_sha256=sha(library), parent_provider_sha256=parent["provider_sha256"],
        original_qmoe_object_byte_identical=True, parent_relink_byte_identical=True,
        process_static_selector=True, no_public_kernel_layout_changes=True,
        no_arithmetic_or_reduction_changes=True, declared_workgroup_bytes=1280,
        baseline_declared_workgroup_bytes=14080, native_comparator=str(comparator),
        compile_command=command, link_command=link, working_directory=str(working),
        source_sha256={str(p): sha(p) for p in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "provider_sha256", "declared_workgroup_bytes")}), flush=True)
