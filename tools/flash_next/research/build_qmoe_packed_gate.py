import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-expert-bitcast-unpack/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-packed-gate/build-v3")
FLAG = "ORT_WEBGPU_QMOE_PACKED_GATE"


def transform(source):
    source = replace_once(source, '#include "qmoe_hierarchical_gate.inc"',
        '#include "qmoe_hierarchical_gate.inc"\n#include "qmoe_packed_gate.inc"')
    original = '  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "ORT_WEBGPU_EXPERT_BITCAST_UNPACK must be 0 or 1");'
    assert source.count(original) == 2
    source = source.replace(original, original + f'''
  const int packed_gate = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(packed_gate == 0 || packed_gate == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(expert_bitcast);",
        "  key.push_back(expert_bitcast);\n  key.push_back(packed_gate);")
    return replace_once(source,
        """    if (use_hierarchical) {
      QmoeHierarchicalGateProgram gate;
      ORT_RETURN_IF_ERROR(run_gate(gate));
    } else {""",
        """    if (use_hierarchical && packed_gate == 1 && router_logits->IsDataType<MLFloat16>()) {
      QmoePackedGateProgram gate;
      ORT_RETURN_IF_ERROR(run_gate(gate));
    } else if (use_hierarchical) {
      QmoeHierarchicalGateProgram gate;
      ORT_RETURN_IF_ERROR(run_gate(gate));
    } else {""")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    shutil.copytree(PARENT / "overlay/contrib_ops/webgpu/moe", overlay)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    source.write_text(transform(source.read_text()))
    shutil.copy2(ROOT / "qmoe_packed_gate.inc", overlay)
    command = parent["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(PARENT / obj.name)) == 1
    link[link.index(str(PARENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    text = (PARENT / comparator.name).read_text()
    assert text.count('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"') == 4
    comparator.write_text(text.replace('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', f'"{FLAG}"'))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in (*sorted(overlay.iterdir()), comparator)
                       if p.is_file()},
        comparator_sha256=sha(OUTPUT / "native_compare"), no_existing_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
