import json
from pathlib import Path
import subprocess

from build_fp16_dot4_pair_matrix_materialized import PROGRAM as OLD_PROGRAM, FLAG as OLD_FLAG, CLASS as OLD_CLASS, SOURCE, transform as old_transform
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_PAIR_COMPONENTS"
CLASS = "Fp16Dot4PairComponentsProgram"
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-pair-components/build-v1")
PROGRAM = OLD_PROGRAM.replace("Fp16Dot4PairMatrixMaterialized", "Fp16Dot4PairComponents")
PROGRAM = replace_once(PROGRAM, "var<workgroup> masks: array<u32, 16>;", """var<workgroup> masks: array<u32, 16>;
var<workgroup> scalar_pairs: array<vec2<f16>, 16>;
var<workgroup> fp32_pairs: array<vec2<f16>, 16>;""")
PROGRAM = replace_once(PROGRAM, "  masks[col] = mask;", """  masks[col] = mask;
  scalar_pairs[col] = vec2<f16>(fma(xv.w, wv.w, rhs[col]), fma(xv.z, wv.z, rhs[32u + col]));
  fp32_pairs[col] = vec2<f16>(
    f16(fma(f32(xv.w), f32(wv.w), f32(rhs[col]))),
    f16(fma(f32(xv.z), f32(wv.z), f32(rhs[32u + col]))));""")
PROGRAM = replace_once(PROGRAM, "  let mask = masks[col];", """  let mask = masks[col];
  let scalar_tree = scalar_pairs[col]; let fp32_tree = fp32_pairs[col];""")
PROGRAM = PROGRAM.replace("row_base * 64u", "row_base * 160u")
PROGRAM = replace_once(PROGRAM, '<< " " << y.SetByOffset("row_base * 160u + 48u + col", "f16(mask)") << "\\n}\\n";',
    '''<< " " << y.SetByOffset("row_base * 160u + 48u + col", "f16(mask)") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 64u + col", "rounded_native.x") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 80u + col", "rounded_native.y") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 96u + col", "scalar_tree.x") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 112u + col", "scalar_tree.y") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 128u + col", "scalar_tree.x + scalar_tree.y") << "\\n"
         << " " << y.SetByOffset("row_base * 160u + 144u + col", "fp32_tree.x + fp32_tree.y") << "\\n}\\n";''')


def transform(source):
    result = replace_once(old_transform(source), OLD_PROGRAM, PROGRAM)
    result = result.replace(OLD_FLAG, FLAG)
    result = replace_once(result, OLD_CLASS + " program", CLASS + " program")
    result = replace_once(result, "N == 64 && K == 4", "N == 160 && K == 4")
    return result.replace("M16 K4 N64 fixture", "M16 K4 N160 fixture")


def comparator(source):
    source = source.replace("{16, 64}", "{16, 160}").replace("row * 64", "row * 160")
    source = replace_once(source, "std::array<size_t, 2> mismatches{};",
                          "std::array<size_t, 2> mismatches{}, native_pair_mismatches{}, sum_mismatches{};")
    source = replace_once(source, "            ++comparisons;", """            for (int pair = 0; pair < 2; ++pair) {
              const auto& native = values[row * 160 + 64 + pair * 16 + col];
              const auto& scalar_pair = values[row * 160 + 96 + pair * 16 + col];
              const auto& tree_sum = values[row * 160 + 128 + pair * 16 + col];
              if (!std::isfinite(native.ToFloat()) || !std::isfinite(scalar_pair.ToFloat()) ||
                  !std::isfinite(tree_sum.ToFloat())) throw std::runtime_error("Non-finite component");
              if (std::memcmp(&native, &scalar_pair, sizeof(native)) != 0) {
                if (native_pair_mismatches[pair] < 4)
                  std::cerr << "COMPONENT pair=" << pair << " row=" << row << " col=" << col
                            << " native=" << native.ToFloat() << " scalar=" << scalar_pair.ToFloat() << '\\n';
                ++native_pair_mismatches[pair];
              }
              sum_mismatches[pair] += std::memcmp(&tree_sum, &scalar, sizeof(tree_sum)) != 0;
            }
            ++comparisons;""")
    return replace_once(source, "    return mismatches[1] == 0 ? 0 : 2;", """    for (int pair = 0; pair < 2; ++pair) {
      std::cout << "COMPONENT pair=" << pair << " comparisons=" << comparisons
                << " mismatches=" << native_pair_mismatches[pair] << std::endl;
      std::cout << "TREE kind=" << pair << " comparisons=" << comparisons
                << " mismatches=" << sum_mismatches[pair] << std::endl;
    }
    return sum_mismatches[0] == 0 ? 0 : 2;""")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, component_characterization=True))
    runner = OUTPUT / "pair_matrix_compare.cpp"
    runner.write_text(comparator((ROOT / "fp16_dot4_pair_matrix_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "pair_matrix_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "pair_matrix_compare"))
    report["source_sha256"]["pair_matrix_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
