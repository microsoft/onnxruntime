import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-fc2-rank-waves/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
FLAG = "ORT_WEBGPU_QMOE_FC2_RANK_WAVES"


def transform(source):
    source = replace_once(source, "class QmoeFC2FinalMixProgram final", f'''namespace {{
bool Fc2RankWavesEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    ORT_ENFORCE(value == 0 || Fc2PrivateAEnabled(),
                "{FLAG} requires ORT_WEBGPU_QMOE_FC2_PRIVATE_A=1");
    return value == 1;
  }}();
  return enabled;
}}
}}

class QmoeFC2FinalMixProgram final''')
    source = replace_once(source, 'Program{Fc2PrivateAEnabled() && bitcast ?',
        'Program{Fc2RankWavesEnabled() && bitcast ? "QmoeFC2RankWavesBitcastF16" : Fc2PrivateAEnabled() && bitcast ?')
    source = replace_once(source, "private_a_{Fc2PrivateAEnabled() && bitcast} {}",
        "private_a_{Fc2PrivateAEnabled() && bitcast}, rank_waves_{Fc2RankWavesEnabled() && bitcast} {}")
    source = replace_once(source, '         << "let rank = local_idx / 32u;\\n"',
        '         << (rank_waves_ ? "for (var wave = 0u; wave < 3u; wave++) {\\n"\n'
        '             "let rank = local_idx / 32u + wave * 4u;\\n"\n'
        '             "if (rank < uniforms.k) {\\n" : "let rank = local_idx / 32u;\\n")')
    source = replace_once(source,
        '         << "}\\n"\n         << "workgroupBarrier();\\n"\n         << "if (local_idx < 20u)',
        '         << "}\\n"\n         << (rank_waves_ ? "}\\n}\\n" : "")\n'
        '         << "workgroupBarrier();\\n"\n         << "if (local_idx < 20u)')
    return replace_once(source, "  bool private_a_;\n", "  bool private_a_;\n  bool rank_waves_;\n")


if __name__ == "__main__":
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed private-A parent provider")
    for p, digest in parent["source_sha256"].items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"Changed parent evidence: {p}")
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
        raise RuntimeError("Original private-A object mismatch")
    candidate_include = OUTPUT / "qmoe_fc2_final_mix.inc"
    candidate_include.write_text(transform((PARENT / "qmoe_fc2_final_mix.inc").read_text()))
    candidate = OUTPUT / "qmoe.cc"
    candidate.write_text(replace_once((PARENT / "qmoe.cc").read_text(),
        ".SetWorkgroupSize(320).SetDispatchGroupSize(1280)",
        ".SetWorkgroupSize(expert_bitcast == 1 && Fc2RankWavesEnabled() ? 128 : 320).SetDispatchGroupSize(1280)"))
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
    sources = [Path(__file__), PARENT / "build-manifest.json", PARENT / "qmoe.cc",
               PARENT / "qmoe_fc2_final_mix.inc", candidate, candidate_include, obj, baseline]
    result = dict(status="fc2_rank_waves_built", provider_sha256=sha(library),
        parent_provider_sha256=parent["provider_sha256"], flag=FLAG, default_enabled=False,
        process_static_selector=True, workgroup_threads=128, rank_waves=3, ranks_per_wave=4,
        parent_relink_byte_identical=True, baseline_qmoe_object_byte_identical=True,
        declared_workgroup_bytes=1280, original_32_lane_fold_and_rank_mix_order_preserved=True,
        no_public_kernel_layout_changes=True, native_comparator=parent["native_comparator"],
        compile_command=command, link_command=link, working_directory=str(working),
        source_sha256={str(p): sha(p) for p in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "provider_sha256", "workgroup_threads")}), flush=True)
