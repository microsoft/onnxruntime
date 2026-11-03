import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


PARENT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v3")
FLAG = "ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS"


def transform(source):
    source = replace_once(source,
        'HCPostNormProjectionProgram(bool shared, bool shared_scale)\n'
        '      : Program{"HCPostNormProjectionF16"}, shared_{shared}, shared_scale_{shared_scale} {}',
        '''HCPostNormProjectionProgram(bool shared, bool shared_scale, bool four_columns)
      : Program{four_columns ? "HCPostNormProjectionFourColumnsF16" : "HCPostNormProjectionF16"},
        shared_{shared}, shared_scale_{shared_scale}, four_columns_{four_columns} {}''')
    source = replace_once(source,
        '<< "let col = workgroup_idx * 8u + local_idx / 32u;\\n"',
        '<< "let col = workgroup_idx * " << (four_columns_ ? 4 : 8) << "u + local_idx / 32u;\\n"')
    source = replace_once(source, '<< "if (col < 324u) {\\n"',
        '<< (four_columns_ ? "if (local_idx < 128u && col < 324u) {\\n" : "if (col < 324u) {\\n")')
    source = replace_once(source, '<< "if (lane == 0u && col < 324u) {\\n"',
        '<< (four_columns_ ? "if (local_idx < 128u && lane == 0u && col < 324u) {\\n" : "if (lane == 0u && col < 324u) {\\n")')
    source = replace_once(source, "  bool shared_scale_;\n};",
        "  bool shared_scale_;\n  bool four_columns_;\n};")
    source = replace_once(source,
        '    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_POST_NORM_PROJECTION", 0);',
        f'''    const int column_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(column_mode == 0 || column_mode == 1, "{FLAG} must be 0 or 1");
    const bool four_columns = column_mode == 1;
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_POST_NORM_PROJECTION", 0);''')
    source = replace_once(source, "return replay_.Run(context, {y, normalized, activated, tail},",
        "auto& replay = four_columns ? replay_four_ : replay_;\n"
        "      return replay.Run(context, {y, normalized, activated, tail},")
    source = replace_once(source, "HCPostNormProjectionProgram program{shared_, shared_scale};",
        "HCPostNormProjectionProgram program{shared_, shared_scale, four_columns};")
    source = replace_once(source, "program.CacheHint(shared_, shared_scale)",
        "program.CacheHint(shared_, shared_scale, four_columns)")
    source = replace_once(source, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
        ".SetWorkgroupSize(256).SetDispatchGroupSize(four_columns ? 81 : 41);")
    source = replace_once(source,
        ': std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {',
        ''': std::initializer_list<int>{0, 1, 2, 3, 5, 6}},
        replay_four_{shared ? "HCSharedPostNormProjectionFourColumnsF16" : "HCPostNormProjectionFourColumnsF16",
                     shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5, 7, 8}
                            : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {''')
    return replace_once(source, "  mutable StaticNodeReplay replay_;",
        "  mutable StaticNodeReplay replay_;\n  mutable StaticNodeReplay replay_four_;")


def comparator_text(source):
    anchor = '      flag("1");\n      std::cout << "PASS "'
    replacement = f'''      flag("1");
      if (setenv("{FLAG}", "2", 1) != 0)
        throw std::runtime_error("Cannot set column selector");
      rejected = false;
      try {{
        session.Run(Ort::RunOptions{{}}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());
      }} catch (const Ort::Exception& error) {{
        rejected = std::string(error.what()).find("{FLAG} must be 0 or 1") != std::string::npos;
      }}
      if (!rejected) throw std::runtime_error("Invalid column selector was not rejected");
      if (setenv("{FLAG}", "1", 1) != 0)
        throw std::runtime_error("Cannot restore column selector");
      std::cout << "PASS "'''
    return replace_once(source, anchor, replacement)


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = OUTPUT / "hc_post_norm_projection.cc"
    source.write_text(transform((PARENT / source.name).read_text()))
    command = parent["compile_commands"][0].copy()
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / (source.name + ".o")
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(PARENT / obj.name)) == 1
    link[link.index(str(PARENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    comparator.write_text(comparator_text((PARENT / comparator.name).read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={p.name: sha(p) for p in (source, comparator)},
        comparator_sha256=sha(OUTPUT / "native_compare"),
        no_runtime_object_layout_changes=True, private_kernel_layout_changed=True,
        independent_eight_and_four_column_replay_caches=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
