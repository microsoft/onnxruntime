import ast
import json
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

from build_hc_projection_inner_unroll import HELPER, program_parts, transform_program, transform_source


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")


def render(program, mode, shared, shared_scale, helper=HELPER,
           flag="ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL",
           class_name="HCPostNormProjectionProgram", constructor_args=None):
    program = re.sub(r"  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES\(.*?\);", "", program, flags=re.S)
    fixture = r'''
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#define ORT_ENFORCE(condition, ...) do { if (!(condition)) throw std::runtime_error("invalid mode"); } while (false)
template<typename T> T ParseEnvironmentVariableWithDefault(const char* name, T fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::stoi(value);
}
struct Status { static Status OK() { return {}; } };
enum class ShaderUsage { UseValueTypeAlias, UseElementTypeAlias };
struct ShaderVariableHelper {
  std::string name;
  std::string GetByOffset(const std::string& index) const { return name + "[" + index + "]"; }
  std::string SetByOffset(const std::string& index, const std::string& value) const {
    return name + "[" + index + "] = " + value;
  }
};
struct ShaderHelper {
  std::map<std::string, ShaderVariableHelper> variables;
  std::ostringstream additional, main;
  auto& AdditionalImplementation() { return additional; }
  auto& MainFunctionBody() { return main; }
  const auto& AddInput(const std::string& name, ShaderUsage = ShaderUsage::UseValueTypeAlias) {
    return variables.try_emplace(name, ShaderVariableHelper{name}).first->second;
  }
  const auto& AddOutput(const std::string& name, ShaderUsage usage = ShaderUsage::UseValueTypeAlias) {
    return AddInput(name, usage);
  }
};
template<typename T> struct Program {
  explicit Program(const char*) {}
  virtual Status GenerateShaderCode(ShaderHelper&) const = 0;
};
''' + helper + program + '''
int main() {
  ShaderHelper shader;
  PROGRAM_CLASS program{CONSTRUCTOR_ARGS};
  program.GenerateShaderCode(shader);
  std::cout << shader.additional.str() << shader.main.str();
}
'''
    arguments = (shared, shared_scale) if constructor_args is None else constructor_args
    fixture = fixture.replace("PROGRAM_CLASS", class_name).replace(
        "CONSTRUCTOR_ARGS", ", ".join(str(value).lower() for value in arguments))
    environment = dict(os.environ)
    environment[flag] = str(mode)
    environment.pop(flag + "_STATS_FILE", None)
    with tempfile.TemporaryDirectory(prefix="hc-inner-unroll-render-") as directory:
        cpp, exe = Path(directory) / "render.cpp", Path(directory) / "render"
        cpp.write_text(fixture)
        subprocess.run(["c++", "-std=c++20", "-O0", str(cpp), "-o", str(exe)], check=True,
                       capture_output=True, text=True)
        return subprocess.check_output([str(exe)], env=environment, text=True)


class HCProjectionInnerUnrollTests(unittest.TestCase):
    def test_original_compute_and_fallbacks_unchanged(self):
        source = SOURCE.read_text()
        start, end, _ = program_parts(source)
        candidate = transform_source(source)
        self.assertEqual(source[end:], candidate[candidate.index("class HCPostNormProjectionF16"):])
        self.assertEqual(source[:start], candidate[ len("#include <cstdlib>\n#include <fstream>\n"):
                                                   candidate.index(HELPER)])
        self.assertIn('"HCPostNormProjectionInnerUnrollF16" : "HCPostNormProjectionF16"', candidate)
        self.assertIn("static const int mode", candidate)
        self.assertIn("value == 0 || value == 1", candidate)

    def test_actual_renderer_preserves_default_and_component_order(self):
        program = program_parts(SOURCE.read_text())[2]
        candidate_program = transform_program(program)
        for shared, shared_scale in ((False, False), (False, True), (True, False), (True, True)):
            original = render(program, 0, shared, shared_scale)
            self.assertEqual(render(candidate_program, 0, shared, shared_scale), original)
            candidate = render(candidate_program, 1, shared, shared_scale)
            start = original.index("  for (var i = 0u; i < 4u; i++)")
            body_start = original.index("    let low", start)
            body_end = original.index("  }\n  total += sum;", body_start)
            body = original[body_start:body_end]
            expected = "".join(f"  {{\n    let i = {i}u;\n" + body + "  }\n" for i in range(4))
            self.assertEqual(candidate, original[:start] + expected + original[body_end + len("  }\n"):])
            self.assertEqual(original.count("workgroupBarrier();"), candidate.count("workgroupBarrier();"))
            self.assertEqual(original.count("sum += dot(x0, w0) + dot(x1, w1);"), 1)
            self.assertEqual(candidate.count("sum += dot(x0, w0) + dot(x1, w1);"), 4)

    def test_generated_native_entrypoint_order(self):
        from prepare_hc_projection_inner_unroll_proofs import ROOT, native_source

        source = native_source((ROOT / "run_hc_bitcast_native.py").read_text())
        tree = ast.parse(source)
        entrypoints = [node for node in tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(entrypoints), 1)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.assertLess(main.lineno, entrypoints[0].lineno)
        actual = ast.parse((ROOT / "run_hc_projection_inner_unroll_native.py").read_text())
        self.assertEqual(ast.dump(tree), ast.dump(actual))

    def test_model_statistics_only_during_coverage(self):
        from run_hc_projection_inner_unroll_model import DIRECTORY, FLAG, ROOT, TEMPLATE, model_command

        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                for mode in ("control", "unroll"):
                    command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                            ROOT / "hc-inner-command-test", context, full, mode, "standard")
                    self.assertIn(f"--contexts {context}", command[-1])
                    self.assertIn("--tokens 256 --repetitions 3", command[-1])
                    self.assertIn(f"export {FLAG}={int(mode == 'unroll')}; ", command[-1])
                    self.assertIn("unset ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE; ", command[-1])
                    self.assertNotIn("export ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE=", command[-1])
                    self.assertNotIn("--profile-prefix", command[-1])


if __name__ == "__main__":
    unittest.main()
