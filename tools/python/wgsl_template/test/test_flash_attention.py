# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Execute the real FlashAttention template's C++ shader generator."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent.parent))

from wgsl_template.code_generator import resolve_code_generator  # noqa: E402
from wgsl_template.generator import generate  # noqa: E402
from wgsl_template.loader import load_from_directory  # noqa: E402
from wgsl_template.parser import parse  # noqa: E402

_SOURCE_ROOT = _THIS_DIR.parents[3] / "onnxruntime" / "contrib_ops" / "webgpu"
_COMPILER = shutil.which("clang++") or shutil.which("g++") or shutil.which("cl")


class FlashAttentionTemplateTest(unittest.TestCase):
    @unittest.skipUnless(_COMPILER, "C++ compiler unavailable")
    def test_reduction_expansion_and_fallbacks(self) -> None:
        repo = parse(load_from_directory(_SOURCE_ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("static-cpp", "static-cpp-literal"):
                cg = resolve_code_generator(mode)
                result = generate("bert/flash_attention.wgsl.template", repo, cg)
                strings = "\n".join(
                    f"const char* __str_{index} = {json.dumps(value)};"
                    for value, index in (cg.string_table or {}).items()
                )
                params = "\n".join(f"int __param_{name} = 0;" for name in result.params)
                source = (
                    """#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#define ORT_RETURN_IF_NOT(cond, message) if (!(cond)) return 1;
#define MainFunctionStart() { ss << "// MAIN\\n";
#define MainFunctionEnd() }
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  std::ostringstream ss;
"""
                    + params
                    + "\n"
                    + strings
                    + """
  __param_qkv_head_size = std::atoi(argv[1]);
  __param_head_size_vec_param = __param_qkv_head_size / 4;
  __param_use_shm_path = std::atoi(argv[2]);
  __param_unroll_shm_inner = std::atoi(argv[3]);
  __param_max_k_step_param = std::atoi(argv[4]);
  __param_qkv_num_heads = 2;
"""
                    + result.code
                    + """
  std::cout << ss.str();
  return 0;
}
"""
                )
                cpp = root / "flash.cc"
                exe = root / "flash.exe"
                cpp.write_text(source, encoding="utf-8")
                command = (
                    [_COMPILER, "/nologo", "/EHsc", "/std:c++17", str(cpp), f"/Fe:{exe}"]
                    if Path(_COMPILER).stem.lower() == "cl"
                    else [_COMPILER, "-std=c++17", str(cpp), "-o", str(exe)]
                )
                compiled = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                for head_size in (64, 128, 192, 256, 512):
                    for use_shm, is_nvidia in ((True, True), (True, False), (False, True)):
                        for max_k_step in (16, 32):
                            expand = is_nvidia and use_shm and head_size // 4 <= 32
                            with self.subTest(
                                mode=mode, head_size=head_size, shm=use_shm, nvidia=is_nvidia, max_k_step=max_k_step
                            ):
                                shader = subprocess.run(
                                    [str(exe), str(head_size), str(int(use_shm)), str(int(expand)), str(max_k_step)],
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                ).stdout
                                self.assertIn("alias qk_precision = f32;", shader)
                                self.assertNotRegex(shader, r"\bloop\s*\(")
                                self.assertNotIn("flash_qk_unrolled", shader)
                                self.assertNotIn("flash_pv_unrolled", shader)
                                if not expand:
                                    self.assertNotIn("fn flash_qk_unrolled", shader)
                                    self.assertNotIn("fn flash_pv_unrolled", shader)
                                    if use_shm:
                                        self.assertIn("for (var i = 0u; i < head_size_vec; i++)", shader)
                                        self.assertIn("for (var k = 0u; k < max_k_step; k++)", shader)
                                    continue
                                main = shader.split("// MAIN\n", 1)[1]
                                qk = main.split("var score = qk_precision(0);", 1)[1].split("qk_scores[k] =", 1)[0]
                                pv = main.split("var acc = q_value_t(0);", 1)[1].split("o_tile[i] =", 1)[0]
                                self.assertNotRegex(qk, r"[{}]")
                                self.assertNotRegex(pv, r"[{}]")
                                # Compare every accumulation with the ordered formulas emitted by #33395.
                                qk_statements = re.findall(r"score \+= ([^;]+);", qk)
                                pv_statements = re.findall(r"acc \+= ([^;]+);", pv)
                                self.assertEqual(
                                    qk_statements,
                                    [
                                        f"dot(vec4<qk_precision>(q_tile[({i})]), vec4<qk_precision>(k_tile[k][({i})]))"
                                        for i in range(head_size // 4)
                                    ],
                                )
                                self.assertEqual(
                                    pv_statements,
                                    [f"v_tile[({k})][i] * q_element_t(qk_scores[({k})])" for k in range(max_k_step)],
                                )
                                self.assertIn("var score = qk_precision(0);", main)
                                self.assertIn("var acc = q_value_t(0);", main)
