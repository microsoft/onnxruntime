import json
import os
from pathlib import Path
import subprocess

from build_hc_bitcast_unpack import COMPONENT, DENSE, main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_HC_POST_NORM_SMALL_CTA"
OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-small-cta/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
STORAGE_BYTES = 4096 + 11520 * 2 + 128 * 2


def small_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    original = source[start:stop]
    sums_start = original.index('    for (int stream = 0; stream < 4; ++stream) code << "var sum"')
    reduce_start = original.index('    code << "workgroupBarrier();\\n"\n         << "for (var width', sums_start)
    inverse_start = original.index('    for (int stream = 0; stream < 4; ++stream) {\n      code << "let inv_rms', reduce_start)
    prefix = replace_once(original[:sums_start], "h += 256u", "h += 128u")
    sums = original[sums_start:reduce_start]
    sums = replace_once(sums, "var h = local_idx;", "var h = norm_lane;")
    sums = replace_once(sums, '<< "u + local_idx] = sum"', '<< "u + norm_lane] = sum"')
    sums = ('    code << "for (var norm_lane = local_idx; norm_lane < 256u; norm_lane += 128u) {\\n";\n'
            + sums + '    code << "}\\n";\n')
    reduction = original[reduce_start:inverse_start]
    reduction = replace_once(reduction, "if (local_idx < width) {",
                             "for (var reduce_lane = local_idx; reduce_lane < width; reduce_lane += 128u) {")
    reduction = replace_once(reduction, "u + local_idx] += sums[", "u + reduce_lane] += sums[")
    reduction = replace_once(reduction, "u + local_idx + width]", "u + reduce_lane + width]")
    suffix = replace_once(original[inverse_start:], "h += 256u", "h += 128u")
    suffix = replace_once(suffix, "workgroup_idx * 8u", "workgroup_idx * 4u")
    program = prefix + sums + reduction + suffix
    program = program.replace("HCPostNormProjectionProgram", "HCPostNormProjectionSmallCtaProgram")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{"HCPostNormProjectionSmallCtaF16"}')
    return replace_once(program, "partials: array<f16, 256>", "partials: array<f16, 128>")


def transform(source):
    source = replace_once(source, "class HCPostNormProjectionF16",
                          small_program(source) + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int cta_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(cta_mode == 0 || cta_mode == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    setup = replace_once(setup, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
                         ".SetWorkgroupSize(small_cta ? 128 : 256).SetDispatchGroupSize(small_cta ? 81 : 41);")
    replacement = (
        "      const bool small_cta = cta_mode == 1;\n"
        "      auto run = [&]<typename P>(P& program) -> Status {\n" + setup +
        "      };\n"
        "      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {\n"
        "        if (small_cta) {\n"
        "          HCPostNormProjectionSmallCtaProgram program{shared_, shared_scale};\n"
        "          return run(program);\n"
        "        }\n"
        "        HCPostNormProjectionProgram program{shared_, shared_scale};\n"
        "        return run(program);\n"
        "      }, 0, small_cta);")
    return source[:start] + replacement + source[stop:]


def main():
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    source = COMPONENT / "hc_post_norm_projection.cc"
    if sha(source) != component["source_sha256"][source.name]:
        raise RuntimeError("Frozen HC source changed")
    proof = OUTPUT.parent / "baseline-proof-v1"
    proof.mkdir(parents=True, exist_ok=False)
    scratch = proof / "scratch"
    scratch.mkdir()
    command = next(command.copy() for command in component["compile_commands"]
                   if command[command.index("-c") + 1] == str(source))
    command[1:1] = ["-I" + str(DENSE / "overlay")]
    original_object = COMPONENT / (source.name + ".o")
    proof_object = proof / original_object.name
    command[command.index("-o") + 1] = str(proof_object)
    subprocess.run(command, cwd=component["working_directory"],
                   env=dict(os.environ, TMPDIR=str(scratch)), check=True)
    if sha(proof_object) != sha(original_object):
        raise RuntimeError("Unchanged HC object did not reproduce; no candidate build admitted")
    build(output=OUTPUT, flag=FLAG, source_transform=transform, parent_dir=PARENT)
    report = dict(status="original_hc_object_byte_identical",
                  object_sha256=sha(original_object), compile_command=command,
                  original_workgroup_size=256, candidate_workgroup_size=128,
                  original_groups=41, candidate_groups=81, declared_workgroup_bytes=STORAGE_BYTES,
                  source_sha256={str(Path(__file__)): sha(Path(__file__))})
    (proof / "proof.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
