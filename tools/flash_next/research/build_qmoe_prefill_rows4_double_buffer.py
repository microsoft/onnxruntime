import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build, PARENT, PROOF


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_QMOE_PREFILL_ROWS4_DOUBLE_BUFFER"
CLASS = "GroupedPrefillRows4DoubleBufferProgram"
NAME = "MatMulNBitsGroupedRows4DoubleBufferF16"
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4-double-buffer/build-v2")
SOURCE = PARENT / "matmul_nbits.cc"
QMOE = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc")
STORAGE_BYTES = 2 * 1024 * 8 + 4 * 8 * 32 * 2 + 4 * 4


def make_program(source):
    start = source.index("class GroupedPrefillRows4Program final")
    stop = source.index("class MatMulInt3Program final", start)
    program = source[start:stop].replace("GroupedPrefillRows4Program", CLASS)
    program = replace_once(program, "MatMulNBitsGroupedRows4F16", NAME)
    program = replace_once(program, "tile_a: array<vec4<f16>, 1024>",
                           "tile_a: array<array<vec4<f16>, 1024>, 2>")
    program = replace_once(program,
        '<< "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\\n"',
        '<< "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\\n"\n'
        '         << "  let buffer = (kidx / 1024u) % 2u;\\n"')
    for address in ("i", "offset", "offset + 1u"):
        program = program.replace(f"tile_a[{address}]", f"tile_a[buffer][{address}]")
    return replace_once(program,
        '<< "  workgroupBarrier();\\n"\n         << "}\\n"\n         << "if (local_idx < 8u)',
        '<< "}\\n"\n         << "workgroupBarrier();\\n"\n         << "if (local_idx < 8u)')


def transform(source):
    program = make_program(source)
    source = replace_once(source, "class GroupedPrefillRows4Program final",
                          program + "class GroupedPrefillRows4Program final")
    start = source.index("Status ApplyMatMulNBits(")
    branch_start = source.index("  if (grouped_prefill_rows == 1", start)
    branch_end = source.index("\n  if (nbits == 3) {", branch_start)
    branch = source[branch_start:branch_end]
    branch = replace_once(branch, "  if (grouped_prefill_rows == 1",
        f"  if (prefill_double_buffer == 1 && K == 2560 && N == 1280 &&\n"
        f"      context.DeviceLimits().maxComputeWorkgroupStorageSize >= {STORAGE_BYTES} &&\n"
        "      context.DeviceLimits().maxComputeWorkgroupSizeX >= 128 &&\n"
        "      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 128 &&\n"
        "      grouped_prefill_rows == 1")
    branch = branch.replace("GroupedPrefillRows4Program", CLASS)
    validation = (
        f'  static const int prefill_double_buffer = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'  ORT_RETURN_IF_NOT(prefill_double_buffer == 0 || prefill_double_buffer == 1,\n'
        f'                    "{FLAG} must be 0 or 1");\n')
    return source[:branch_start] + validation + branch + source[branch_start:]


def transform_replay(source):
    validation = (
        '  ORT_RETURN_IF_NOT(grouped_prefill == 0 || grouped_prefill == 1, '
        '"ORT_WEBGPU_QMOE_GROUPED_PREFILL must be 0 or 1");')
    if source.count(validation) != 2:
        raise ValueError("Unexpected QMoE replay/compute validation surfaces")
    source = source.replace(validation, validation + (
        f'\n  static const int prefill_double_buffer = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'  ORT_RETURN_IF_NOT(prefill_double_buffer == 0 || prefill_double_buffer == 1,\n'
        f'                    "{FLAG} must be 0 or 1");'))
    return replace_once(source, "  key.push_back(grouped_prefill);",
                        "  key.push_back(grouped_prefill);\n  key.push_back(prefill_double_buffer);")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, PARENT / "native_compare.cpp", FLAG,
                   dict(prefill_only=True, target_k=2560, target_n=1280,
                        rows_per_group=4, columns_per_group=8, physical_threads=128,
                        shared_storage_bytes=STORAGE_BYTES, model_eligible=False,
                        builder_sha256=sha(Path(__file__))))
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    recovered = json.loads((PROOF / "recovered-commands.json").read_text())
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    component_path = QMOE.parent / "build-manifest.json"
    component = json.loads(component_path.read_text())
    assert sha(QMOE) == component["source_sha256"][QMOE.name]
    original, = [name for name, command in recovered.items()
                 if command[command.index("-c") + 1] == str(QMOE)]
    command = proof["compile_commands"][list(recovered).index(original)].copy()
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in command:
            command[command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    original_obj = Path(original)
    if not original_obj.is_absolute():
        original_obj = Path(parent["working_directory"]) / original_obj
    assert sha(obj) == sha(original_obj), "Original QMoE translation-unit rebuild mismatch"
    report["original_qmoe_translation_unit_sha256"] = sha(obj)
    candidate = OUTPUT / QMOE.name
    candidate.write_text(transform_replay(QMOE.read_text()))
    command[command.index("-c") + 1] = str(candidate)
    command.insert(1, f"-fmacro-prefix-map={candidate}={QMOE}")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    link = report["link_command"].copy()
    assert link.count(original) == 1
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    report.update(provider_sha256=sha(OUTPUT / "libonnxruntime_providers_webgpu.so"),
                  link_command=link, compile_commands=[report["compile_command"], command],
                  qmoe_replay_selector_keyed=True)
    for path in (candidate, QMOE, component_path):
        report["source_sha256"][os.path.relpath(path, OUTPUT)] = sha(path)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], provider_sha256=report["provider_sha256"])), flush=True)
