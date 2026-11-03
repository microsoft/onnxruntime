import json
import os
from pathlib import Path
import subprocess

from build_mlp_bitcast_unpack import DECODE_IMPLEMENTATION
from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-prefill-dense-stack/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
DENSE = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-hc-bitcast-unpack/build-v1")
FLAG = "ORT_WEBGPU_HC_POST_NORM_BITCAST"


def transform(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop]
    program = replace_once(program, "HCPostNormProjectionProgram(bool shared, bool shared_scale)",
                           "HCPostNormProjectionProgram(bool shared, bool shared_scale, bool bitcast)")
    program = replace_once(program,
        ': Program{"HCPostNormProjectionF16"}, shared_{shared}, shared_scale_{shared_scale} {}',
        ': Program{bitcast ? "HCPostNormProjectionBitcastF16" : "HCPostNormProjectionF16"}, '
        'shared_{shared}, shared_scale_{shared_scale}, bitcast_{bitcast} {}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();", DECODE_IMPLEMENTATION)
    for name, argument, original in (
            ("low", "packed[i]", "packed[i] & 0x0f0f0f0fu"),
            ("high", "packed[i] >> 4u", "(packed[i] >> 4u) & 0x0f0f0f0fu")):
        literal = f'"    let {name} = vec4<f16>(unpack4xU8({original})) - vec4<f16>(8);\\n"'
        program = replace_once(program, literal,
            f'(bitcast_ ? "    let {name} = decode_low_nibbles({argument});\\n" : {literal})')
    program = replace_once(program, "  bool shared_scale_;\n", "  bool shared_scale_;\n  bool bitcast_;\n")
    kernel = source[stop:]
    kernel = replace_once(kernel,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int bitcast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(bitcast == 0 || bitcast == 1, "{FLAG} must be 0 or 1");''')
    kernel = replace_once(kernel, "HCPostNormProjectionProgram program{shared_, shared_scale};",
                          "HCPostNormProjectionProgram program{shared_, shared_scale, bitcast == 1};")
    kernel = replace_once(kernel, "return context.RunProgram(program);\n      });",
                          "return context.RunProgram(program);\n      }, 0, bitcast);")
    return source[:start] + program + kernel


def main(output=OUTPUT, flag=FLAG, source_transform=transform, component_dir=COMPONENT,
         source_name="hc_post_norm_projection.cc", comparator_flag="ORT_WEBGPU_HC_POST_NORM_PROJECTION",
         parent_dir=PARENT, extra_include_dirs=(), comparator_transform=None, replay_variant_keyed=True,
         preserves_reduction=True, comparator_dir=None, additional_components=()):
    parent = json.loads((parent_dir / "build-manifest.json").read_text())
    component = json.loads((component_dir / "build-manifest.json").read_text())
    original = component_dir / source_name
    assert sha(original) == component["source_sha256"][source_name]
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Combined parent relink mismatch"
    library.unlink()
    sources, commands = [], []
    components = ((component_dir, source_name, source_transform, extra_include_dirs), *additional_components)
    for directory, name, transform_source, includes in components:
        manifest = component if directory == component_dir else json.loads(
            (directory / "build-manifest.json").read_text())
        component_source = directory / name
        assert sha(component_source) == manifest["source_sha256"][name]
        source = output / name
        if source.exists():
            raise FileExistsError(source)
        source.write_text(transform_source(component_source.read_text()))
        obj = output / (source.name + ".o")
        command = next(c.copy() for c in manifest["compile_commands"]
                       if c[c.index("-c") + 1] == str(component_source))
        command[1:1] = ["-I" + str(DENSE / "overlay")]
        command[1:1] = ["-I" + str(path) for path in includes]
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
        assert link.count(str(directory / obj.name)) == 1
        link[link.index(str(directory / obj.name))] = str(obj)
        sources.append(source)
        commands.append(command)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = output / "native_compare.cpp"
    compare_directory = component_dir if comparator_dir is None else comparator_dir
    if comparator_dir is not None:
        compare_manifest = json.loads((compare_directory / "build-manifest.json").read_text())
        assert sha(compare_directory / "native_compare.cpp") == compare_manifest["source_sha256"]["native_compare.cpp"]
    comparator_source = replace_once((compare_directory / "native_compare.cpp").read_text(),
        f'setenv("{comparator_flag}", value, 1)', f'setenv("{flag}", value, 1)')
    comparator.write_text(comparator_transform(comparator_source) if comparator_transform is not None else replace_once(comparator_source,
        'std::to_string(round) + " element=" + std::to_string(i)',
        '''std::to_string(round) + " output=" + std::to_string(output) +
        " element=" + std::to_string(i) +
        " expected_bits=" + std::to_string(half ? reinterpret_cast<const uint16_t*>(expected)[i] :
                                                 reinterpret_cast<const uint32_t*>(expected)[i]) +
        " actual_bits=" + std::to_string(half ? reinterpret_cast<const uint16_t*>(observed)[i] :
                                               reinterpret_cast<const uint32_t*>(observed)[i])'''))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=commands, link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (*sources, comparator)},
        comparator_sha256=sha(output / "native_compare"), no_existing_kernel_object_layout_changes=True,
        no_reduction_or_normalization_changes=preserves_reduction, replay_variant_keyed=replay_variant_keyed,
        frozen_replay_header_sha256=sha(DENSE / "overlay/static_node_replay.h"),
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
