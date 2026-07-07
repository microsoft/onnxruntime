#!/usr/bin/env python3
"""
verify_f16_fix.py - Static verification for the fp16 overflow fix (issue #26732).

Checks that WebGPU matmul shader generators/templates accumulate in f32 instead
of the (overflow-prone) f16 output type. No build or GPU required.

Usage: python3 scripts/verify_f16_fix.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSEP = ROOT / "js/web/lib/wasm/jsep/webgpu/ops"
NATIVE = ROOT / "onnxruntime/contrib_ops/webgpu/quantization"

checks = []


def check(name, path, bad_patterns, good_patterns, description):
    if not path.exists():
        checks.append((name, "SKIP", f"file not found: {path}"))
        return
    content = path.read_text(encoding="utf-8")
    bad = [p for p in bad_patterns if re.search(p, content)]
    good = [p for p in good_patterns if re.search(p, content)]
    if bad:
        checks.append((name, "FAIL", f"{description} - f16 accumulator pattern still present: {bad}"))
    elif len(good) == len(good_patterns):
        checks.append((name, "PASS", description))
    else:
        missing = [p for p in good_patterns if p not in good]
        checks.append((name, "UNKNOWN", f"{description} - expected pattern(s) not found: {missing}"))


check(
    "matmulnbits.ts (default kernel)",
    JSEP / "matmulnbits.ts",
    [r"var<workgroup> workgroup_shared: array<\$\{output\.type\.value\}"],
    [
        r"var<workgroup> workgroup_shared: array<\$\{accType\}",
        r"vec\$\{aComponents\}<f32>\(b_dequantized_values",
    ],
    "MatMulNBits default kernel must accumulate in f32",
)

check(
    "matmulnbits.ts (blockSize32 kernel)",
    JSEP / "matmulnbits.ts",
    [r"inter_results: array<array<\$\{output\.type\.value\}"],
    [
        r"inter_results: array<array<f32",
        r"dot\(vec4<f32>\(a_data\$\{i\}\), vec4<f32>\(b_dequantized_values\[\$\{i\}\]\)\)",
    ],
    "BlockwiseMatMulNBits32 must accumulate in f32",
)

check(
    "matmul-shaders.ts (naive matmul)",
    JSEP / "matmul-shaders.ts",
    [r"var values: array<\$\{output\.type\.value\}"],
    [r"var values: array<\$\{accType\}", r"fma\(\$\{accType\}\("],
    "Naive MatMul must accumulate in f32",
)

check(
    "3rd-party/matmul_packed_webgpu.ts (tiled matmul)",
    JSEP / "3rd-party/matmul_packed_webgpu.ts",
    [
        r"var acc: array<vec4<\$\{type\}>, rowPerThread>",
        r"var acc : array<array<\$\{type\}, colPerThread>",
    ],
    [
        r"var acc: array<vec4<f32>, rowPerThread>",
        r"var acc : array<array<f32, colPerThread>",
        r"vec4<f32>\(BCached0\)",
        r"f32\(ACached\)",
    ],
    "Packed MatMul (vec4 + scalar variants) must accumulate in f32",
)

check(
    "native matmul_nbits.wgsl.template",
    NATIVE / "matmul_nbits.wgsl.template",
    [r"inter_results: array<array<output_element_t"],
    [
        r"inter_results: array<array<f32",
        r"inter_results\[local_row_offset \+ idy\]\[idx\] \+= f32\(sum\)",
        r"output\.setByOffset\(output_idx, output_element_t\(output_value\)\)",
    ],
    "Native MatMulNBits template must accumulate in f32 along K",
)

check(
    "native matmul_nbits_wide_tile.wgsl.template (pre-existing f32)",
    NATIVE / "matmul_nbits_wide_tile.wgsl.template",
    [],
    [r"var results : array<f32, kTileM>"],
    "Wide-tile kernel already accumulates in f32 (reference pattern)",
)

print("\n=== Static verification: fix for issue #26732 ===\n")
all_pass = True
for name, status, desc in checks:
    icon = {"PASS": "[PASS]", "SKIP": "[SKIP]", "FAIL": "[FAIL]", "UNKNOWN": "[????]"}[status]
    print(f"{icon} {name}: {desc}")
    if status in ("FAIL", "UNKNOWN"):
        all_pass = False

print()
if all_pass:
    print("All checks passed - the fix appears to be applied correctly.")
    sys.exit(0)
print("Some checks failed - review the patches.")
sys.exit(1)
