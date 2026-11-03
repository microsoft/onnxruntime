import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-generic-matmul-replay/build-v1"
FROZEN = REPOS / "qwen38-int4-static-node-replay/build-v8"
OUTPUT = REPOS / "qwen38-int4-direct-a-matmul/build-v2"
FLAG = "ORT_WEBGPU_MATMUL_NBITS_DIRECT_A"
VALIDATE = f'''
  const int direct_a = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(direct_a == 0 || direct_a == 1, "{FLAG} must be 0 or 1");
'''
ROUTE = '''
  if (direct_a == 1 && context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      batch_count == 1 && (M == 1 || (has_weight_idx_indirect && dispatch_M == 10)) &&
      nbits == 4 && block_size == 32 && K % 32 == 0 && !has_zero_points &&
      !single_scale_weights && components_a == 4 && components_b == 4 &&
      !acc_f32 && a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      (bias == nullptr || bias->IsDataType<MLFloat16>()) &&
      (weight_index_indirect == nullptr || weight_index_indirect->IsDataType<uint32_t>())) {
    MatMulNBitsDirectAProgram program{has_bias, has_weight_idx_indirect, broadcast_a};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(CeilDiv(N, 8u), dispatch_M);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
        .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
        .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_bias) program.AddInput({bias, ProgramTensorMetadataDependency::None});
    if (has_weight_idx_indirect) program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({N, K, CeilDiv(N, 8u), weight_index})
        .CacheHint(has_bias, has_weight_idx_indirect, broadcast_a);
    return context.RunProgram(program);
  }

'''


def transform_matmul(source):
    source = replace_once(source, "namespace {\n", 'namespace {\n#include "direct_a_matmul.inc"\n')
    source = replace_once(source, "  const Tensor* a = context.Input(0);",
                          VALIDATE + "  const Tensor* a = context.Input(0);")
    source = replace_once(source, "  TensorShape b_shape({N_op, K_op});",
                          VALIDATE + "  TensorShape b_shape({N_op, K_op});")
    source = replace_once(source, "  // Use tile_size_k_vec=32 by default", ROUTE +
                          "  // Use tile_size_k_vec=32 by default")
    return replace_once(source, "return static_replay_.Run(context, {y}, compute);",
                        "return static_replay_.Run(context, {y}, compute, 0, direct_a);")


def transform_helper(source):
    source = replace_once(source, "int64_t row_elements = 0) {",
                          "int64_t row_elements = 0, uint64_t shader_variant = 0) {")
    return replace_once(source, "key.push_back(context.EnableMatmulFp32Accumulation());",
                        "key.push_back(context.EnableMatmulFp32Accumulation());\n"
                        "    key.push_back(shader_variant);")


def transform_qmoe(source):
    source = replace_once(source, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
                          "Status QMoE::ComputeInternal(ComputeContext& context) const {" + VALIDATE)
    return replace_once(source, "key.push_back(context.EnableMatmulFp32Accumulation());",
                        "key.push_back(context.EnableMatmulFp32Accumulation());\n"
                        "  key.push_back(direct_a);")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay"
    overlay.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    helper = overlay / "static_node_replay.h"
    helper.write_text(transform_helper((FROZEN / "overlay/static_node_replay.h").read_text()))
    shutil.copy2(ROOT / "direct_a_matmul.inc", overlay / "direct_a_matmul.inc")
    commands = []
    sources = {}
    manifests = {}
    for token in link.copy():
        if not token.endswith(".cc.o") or "/qwen38-int4-" not in token:
            continue
        obj = Path(token)
        name = obj.name[:-2]
        if name in ("webgpu_context.cc", "wgsl_gen.cc", "gated_delta_net.cc"):
            continue
        manifest_path = obj.parent / "build-manifest.json"
        if manifest_path not in manifests:
            manifests[manifest_path] = json.loads(manifest_path.read_text())
        manifest = manifests[manifest_path]
        command = next(c.copy() for c in manifest["compile_commands"]
                       if c[c.index("-o") + 1] == token)
        original = Path(command[command.index("-c") + 1])
        text = original.read_text()
        if name == "matmul_nbits.cc":
            text = transform_matmul(text)
        elif name == "qmoe.cc":
            text = transform_qmoe(text)
        source = OUTPUT / name
        source.write_text(text)
        sources[str(original)] = sha(original)
        command[1:1] = ["-I" + str(overlay)]
        command[command.index("-c") + 1] = str(source)
        replacement = OUTPUT / obj.name
        command[command.index("-o") + 1] = str(replacement)
        subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
        link[link.index(token)] = str(replacement)
        commands.append(command)
    assert len(commands) == 10, [c[c.index("-c") + 1] for c in commands]
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator_sources = []
    for name in ("static_node_replay_compare", "qmoe_node_replay_compare"):
        text = (ROOT / (name + ".cpp")).read_text()
        if name.startswith("static"):
            text = replace_once(text, 'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1)',
                                f'setenv("{FLAG}", value, 1)')
        else:
            text = text.replace('"ORT_WEBGPU_QMOE_COMMAND_REPLAY"', f'"{FLAG}"')
            text = replace_once(text,
                'set_flag("ORT_WEBGPU_MATMUL_NBITS_SKINNY", (round / 4) % 2 ? "1" : "0");',
                'set_flag("ORT_WEBGPU_MATMUL_NBITS_SKINNY", "0");')
        source = OUTPUT / (name + ".cpp")
        source.write_text(text)
        subprocess.run(["c++", "-O3", "-std=c++20", str(source),
            "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
            "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-lonnxruntime", "-o", str(OUTPUT / name)], env=environment, check=True)
        comparator_sources.append(source)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=commands, link_command=link, working_directory=parent["working_directory"],
        original_source_sha256=sources, frozen_helper_sha256=frozen["helper_sha256"],
        source_sha256={str(p.relative_to(OUTPUT)): sha(p) for p in
                       [helper, overlay / "direct_a_matmul.inc", *comparator_sources,
                        *(Path(c[c.index("-c") + 1]) for c in commands)]},
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
