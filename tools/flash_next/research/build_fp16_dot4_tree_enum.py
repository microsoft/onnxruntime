import json
from pathlib import Path
import subprocess

from build_fp16_dot4_tree_probe import PROGRAM as ORIGINAL_PROGRAM, transform as original_transform
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_TREE_ENUM"
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-tree-enum/build-v1")
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
PATTERNS = 96
COLUMNS = (PATTERNS + 1) * 16
PAIR_HELPER = r'''
var<workgroup> pairs: array<vec2<f16>, 256>;
fn pair_value(kind: u32, xv: vec4<f16>, wv: vec4<f16>, p: vec4<f16>, a: u32, b: u32) -> f16 {
  switch kind {
    case 0u: { return p[a] + p[b]; }
    case 1u: { return f16(f32(xv[a]) * f32(wv[a]) + f32(xv[b]) * f32(wv[b])); }
    case 2u: { return fma(xv[a], wv[a], p[b]); }
    default: { return fma(xv[b], wv[b], p[a]); }
  }
}
'''


def enum_program():
    stop = ORIGINAL_PROGRAM.index('         << "for (var i = local_idx; i < 256u; i += 32u) {\\n"',
                                  ORIGINAL_PROGRAM.index('weights[i] = wv; products[i] = xv * wv;'))
    prefix = ORIGINAL_PROGRAM[:stop].replace("Fp16Dot4TreeProbe", "Fp16Dot4TreeEnum")
    prefix = replace_once(prefix, "var<workgroup> products: array<vec4<f16>, 256>;",
                          "var<workgroup> products: array<vec4<f16>, 256>;\n" + PAIR_HELPER)
    return prefix + r'''
         << R"(
for (var tree = 0u; tree < 3u; tree++) {
  let first_order = select(vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 2u, 1u, 3u), tree == 1u);
  let order = select(first_order, vec4<u32>(0u, 3u, 1u, 2u), tree == 2u);
  for (var first = 0u; first < 4u; first++) {
    for (var second = 0u; second < 4u; second++) {
      let pattern = tree * 16u + first * 4u + second;
      for (var i = local_idx; i < 256u; i += 32u) {
        let row = i / 16u; let col = i % 16u;
)"
         << " let xv = vec4<f16>(" << a.GetByOffset("row * 4u")
         << ", " << a.GetByOffset("row * 4u + 1u")
         << ", " << a.GetByOffset("row * 4u + 2u")
         << ", " << a.GetByOffset("row * 4u + 3u") << ");\n"
         << R"(
        let wv = weights[i]; let p = products[i];
        let left = pair_value(first, xv, wv, p, order.x, order.y);
        let right = pair_value(second, xv, wv, p, order.z, order.w);
        pairs[i] = vec2<f16>(left, right);
)"
         << " " << y.SetByOffset("row * 1552u + pattern * 16u + col", "left + right") << "\n"
         << R"(
      }
      workgroupBarrier();
      for (var i = local_idx; i < 256u; i += 32u) {
        let row = i / 16u; let col = i % 16u;
        let stored = pairs[i];
)"
         << " " << y.SetByOffset("row * 1552u + (pattern + 48u) * 16u + col", "stored.x + stored.y") << "\n"
         << R"(
      }
      workgroupBarrier();
    }
  }
}
for (var i = local_idx; i < 256u; i += 32u) {
  let row = i / 16u; let col = i % 16u;
)"
         << " let xv = vec4<f16>(" << a.GetByOffset("row * 4u")
         << ", " << a.GetByOffset("row * 4u + 1u")
         << ", " << a.GetByOffset("row * 4u + 2u")
         << ", " << a.GetByOffset("row * 4u + 3u") << ");\n"
         << " " << y.SetByOffset("row * 1552u + 1536u + col", "dot(xv, weights[i])") << "\n}\n";
    return Status::OK();
  }
};
'''


PROGRAM = enum_program()


def transform(source):
    result = original_transform(source)
    result = replace_once(result, ORIGINAL_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_FP16_DOT4_TREE_PROBE", FLAG).replace(
        "Fp16Dot4TreeProbeProgram program", "Fp16Dot4TreeEnumProgram program")
    return result.replace("N == 80", "N == 1552").replace("M16 K4 N80 fixture", "M16 K4 N1552 fixture")


def comparator_transform(source):
    return (source.replace("std::array<size_t, 4>", "std::array<size_t, 96>")
            .replace("pattern < 4", "pattern < 96")
            .replace("{16, 80}", "{16, 1552}")
            .replace("row * 80 + 64 + col", "row * 1552 + 1536 + col")
            .replace("row * 80 + pattern", "row * 1552 + pattern"))


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, pairings=3,
                        pair_rounding_choices=4, pair_materialization_modes=2, patterns=PATTERNS))
    runner = OUTPUT / "dot4_tree_compare.cpp"
    runner.write_text(comparator_transform((ROOT / "fp16_dot4_tree_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "dot4_tree_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "dot4_tree_compare"))
    report["source_sha256"]["dot4_tree_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
