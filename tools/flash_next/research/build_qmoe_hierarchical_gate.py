import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-hierarchical-gate/build-v2")
FLAG = "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE"


def transform(source):
    source = replace_once(source, '#include "qmoe_fc2_final_mix.inc"',
        '#include "qmoe_fc2_final_mix.inc"\n#include "qmoe_hierarchical_gate.inc"')
    source = replace_once(source, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
        f'''Status QMoE::ComputeInternal(ComputeContext& context) const {{
  const int hierarchical_gate = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(hierarchical_gate == 0 || hierarchical_gate == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(fc2_mix);",
        "  key.push_back(fc2_mix);\n  key.push_back(hierarchical_gate);")
    start = source.index("    Gate1TokenProgram gate{k_,")
    end = source.index("    ORT_RETURN_IF_ERROR(context.RunProgram(gate));", start)
    stop = end + len("    ORT_RETURN_IF_ERROR(context.RunProgram(gate));")
    original = source[start:stop]
    binding_start = original.index("    gate\n")
    bindings = original[binding_start:].replace(
        "    ORT_RETURN_IF_ERROR(context.RunProgram(gate));", "    return context.RunProgram(gate);")
    replacement = f'''    const bool use_hierarchical = use_subgroups &&
        context.AdapterInfo().subgroupMinSize >= 32 &&
        ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0) == 1;
    auto run_gate = [&](auto& gate) -> Status {{
{bindings}
    }};
    if (use_hierarchical) {{
      QmoeHierarchicalGateProgram gate;
      ORT_RETURN_IF_ERROR(run_gate(gate));
    }} else {{
      Gate1TokenProgram gate{{k_, is_fp16, router_weights != nullptr, normalize_routing_weights_, sort_experts, use_subgroups}};
      ORT_RETURN_IF_ERROR(run_gate(gate));
    }}'''
    return source[:start] + replacement + source[stop:]


def comparator_text(text):
    assert text.count('"ORT_WEBGPU_QMOE_FC2_FINAL_MIX"') == 4
    text = text.replace('"ORT_WEBGPU_QMOE_FC2_FINAL_MIX"', f'"{FLAG}"')
    text = "#include <algorithm>\n#include <limits>\n" + text
    text = replace_once(text,
        '        const char* gate_mode = (round / 16) % 2 ? "1" : "0";',
        '''        const bool identity = std::filesystem::path(argv[file]).stem().string() == "gate_identity_rows1";
        if (identity) {
          std::fill(data[1].begin(), data[1].end(), Ort::Float16_t(-std::numeric_limits<float>::infinity()));
          const int pattern = round % 6;
          if (pattern == 0 || pattern == 4) {
            for (size_t i = 0; i < 512; ++i)
              data[1][i] = Ort::Float16_t(pattern == 4 && i % 2 ? -0.0f : 0.0f);
          } else {
            for (size_t i = 0; i < 10; ++i) {
              const size_t expert = pattern == 2 ? 502 + i : pattern == 5 ? 2 + i * 32 : i;
              data[1][expert] = Ort::Float16_t(pattern == 3 ? 128.0f : 1.0f);
            }
          }
        }
        const char* gate_mode = (round / 16) % 2 ? "1" : "0";''')
    text = replace_once(text, "          ++comparisons;\n        }\n      }",
        '''          ++comparisons;
        }
        if (identity) {
          std::vector<size_t> expected;
          for (size_t i = 0; i < 512; ++i) expected.push_back(i);
          std::sort(expected.begin(), expected.end(), [&](size_t left, size_t right) {
            const auto a = data[1][left].ToFloat(), b = data[1][right].ToFloat();
            return a > b || (a == b && left < right);
          });
          expected.resize(10);
          for (size_t i = 0; i < 2560; ++i) {
            const Ort::Float16_t weight(std::find(expected.begin(), expected.end(), i) != expected.end() ? 0.1f : 0.0f);
            if (std::memcmp(&weight, actual[0].GetTensorData<Ort::Float16_t>() + i, sizeof(weight)) != 0)
              throw std::runtime_error("Identity fixture failed direct selected-expert/routing-weight check");
          }
        }
      }''')
    return text


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/contrib_ops/webgpu/moe"
    shutil.copytree(COMPONENT / "overlay/contrib_ops/webgpu/moe", overlay)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    source = overlay / "qmoe.cc"
    source.write_text(transform(source.read_text()))
    shutil.copy2(ROOT / "qmoe_hierarchical_gate.inc", overlay)
    command = component["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(OUTPUT / "overlay")]
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(COMPONENT / obj.name)) == 1
    link[link.index(str(COMPONENT / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = OUTPUT / "native_compare.cpp"
    text = (COMPONENT / comparator.name).read_text()
    comparator.write_text(comparator_text(text))
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
        comparator_sha256=sha(OUTPUT / "native_compare"), no_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
