import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-fc1-private-a/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
INCLUDE = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")
FLAG = "ORT_WEBGPU_QMOE_FC1_PRIVATE_A"


def transform(source):
    source = replace_once(source, "class QmoeFC1SwiGLUProgram final", f'''namespace {{
bool Fc1PrivateAEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    return value == 1;
  }}();
  return enabled;
}}
}}

class QmoeFC1SwiGLUProgram final''')
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
        'Program{Fc1PrivateAEnabled() && bitcast ? "QmoeFC1PrivateABitcastF16" : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
        "bias_{bias}, bitcast_{bitcast}, private_a_{Fc1PrivateAEnabled() && bitcast} {}")
    source = replace_once(source,
        '    shader.AdditionalImplementation()\n        << "var<workgroup> tile_a: array<vec4<f16>, 256>;\\n"',
        '    if (!private_a_)\n      shader.AdditionalImplementation() << "var<workgroup> tile_a: array<vec4<f16>, 256>;\\n";\n'
        '    shader.AdditionalImplementation()')
    begin = source.index('         << "  for (var id = local_idx;')
    end = source.index('         << "  for (var offset =', begin)
    original = source[begin:end]
    source = source[:begin] + ''';\n
    if (private_a_) {
      code << "  var cached_a: array<vec4<f16>, 8>;\\n"
           << "  for (var i = 0u; i < 8u; i++) {\\n"
           << "    let offset = kidx / 4u + lane * 8u + i;\\n"
           << "    cached_a[i] = vec4<f16>(0);\\n"
           << "    if (offset < uniforms.K / 4u) {\\n"
           << "      cached_a[i] = " << a.GetByOffset("offset") << ";\\n"
           << "    }\\n"
           << "  }\\n";
    } else {
      code ''' + original + ''';\n
    }
    code ''' + source[end:]
    source = replace_once(source,
        '         << "        sum += dot(tile_a[lane * 8u + i * 2u], w0) + dot(tile_a[lane * 8u + i * 2u + 1u], w1);\\n"',
        '         << (private_a_ ? "        sum += dot(cached_a[i * 2u], w0) + dot(cached_a[i * 2u + 1u], w1);\\n" :\n'
        '             "        sum += dot(tile_a[lane * 8u + i * 2u], w0) + dot(tile_a[lane * 8u + i * 2u + 1u], w1);\\n")')
    source = replace_once(source,
        '         << "  workgroupBarrier();\\n"\n         << "}\\n"\n         << "if (local_idx < 8u',
        '         << (private_a_ ? "" : "  workgroupBarrier();\\n")\n         << "}\\n"\n'
        '         << (private_a_ ? "workgroupBarrier();\\n" : "")\n         << "if (local_idx < 8u')
    return replace_once(source, "  bool bitcast_;\n", "  bool bitcast_;\n  bool private_a_;\n")


if __name__ == "__main__":
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed composite parent")
    for p, digest in parent["source_sha256"].items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"Changed parent input: {p}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    working = Path(parent["working_directory"])
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=working, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Parent relink mismatch")
    command = parent["compile_command"].copy()
    baseline = OUTPUT / "qmoe.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    subprocess.run(command, cwd=working, check=True)
    if sha(baseline) != sha(PARENT / "qmoe.cc.o"):
        raise RuntimeError("Original QMoE object mismatch")
    fc1 = INCLUDE / "qmoe_fc1_activation.inc"
    candidate_include = OUTPUT / "qmoe_fc1_activation.inc"
    candidate_include.write_text(transform(fc1.read_text()))
    fc2 = OUTPUT / "qmoe_fc2_final_mix.inc"
    fc2.write_text((PARENT / fc2.name).read_text())
    candidate = OUTPUT / "qmoe.cc"
    candidate.write_text((PARENT / "qmoe.cc").read_text())
    original_macro = next(c.split("=", 2)[2] for c in command if c.startswith("-fmacro-prefix-map="))
    command.insert(1, f"-fmacro-prefix-map={candidate}={original_macro}")
    command[command.index("-c") + 1] = str(candidate)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=working, check=True)
    old = str(PARENT / "qmoe.cc.o")
    if link.count(old) != 1:
        raise RuntimeError("Nonunique QMoE object replacement")
    link[link.index(old)] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    comparator = Path("/workspace/qwen38-int4-qmoe-fc1-activation/build-v3/native_compare")
    sources = [Path(__file__), PARENT / "build-manifest.json", PARENT / "qmoe.cc",
        PARENT / "qmoe_fc2_final_mix.inc", fc1, candidate, candidate_include, fc2, obj, baseline, comparator]
    result = dict(status="fc1_private_a_built", flag=FLAG, default_enabled=False,
        provider_sha256=sha(library), parent_provider_sha256=parent["provider_sha256"],
        parent_relink_byte_identical=True, baseline_qmoe_object_byte_identical=True,
        process_static_selector=True, workgroup_threads=128, declared_workgroup_bytes=512,
        baseline_declared_workgroup_bytes=2560, no_public_kernel_layout_changes=True,
        original_word_tile_fold_and_swiglu_arithmetic_preserved=True,
        native_comparator=str(comparator), compile_command=command, link_command=link, working_directory=str(working),
        source_sha256={str(p): sha(p) for p in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "provider_sha256", "declared_workgroup_bytes")}), flush=True)
