# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_test_exports import Symbols, read_symbols, select_exports, write_exports


class TestTestExports(unittest.TestCase):
    def test_parse_coff_symbols(self):
        symbols = read_symbols(
            """
Dump of file tests.obj
00A 00000000 UNDEF  notype ()    External     | ?Run@Session@@QEAAHXZ (public: int __cdecl Session::Run(void))
00B 00000000 SECT2  notype ()    External     | ?Local@@YAHXZ
00C 00000000 SECT10 notype       External     | ?data@@3HA
00D 00000004 UNDEF  notype       External     | common_data
00E 00000000 UNDEF  notype       External     | __imp_?data@@3HA
00F 00000000 SECT3  notype ()    Static       | local_function
010 00000000 ABS    notype       External     | @feat.00
011 00000000 UNDEF  notype       WeakExternal | weak_fallback
""".splitlines()
        )
        self.assertEqual(symbols.undefined, {"?Run@Session@@QEAAHXZ", "__imp_?data@@3HA"})
        self.assertEqual(symbols.defined, {"?Local@@YAHXZ", "?data@@3HA", "common_data"})
        self.assertEqual(symbols.functions, {"?Local@@YAHXZ"})

    def test_static_library_definitions_and_duplicate_references(self):
        module = read_symbols(
            """
001 00000000 UNDEF notype () External | required
002 00000000 UNDEF notype () External | required
003 00000000 UNDEF notype () External | module_local
004 00000000 UNDEF notype () External | system_function
Archive member name at 123: helpers.obj
005 00000000 SECT1 notype () External | module_local
""".splitlines()
        )
        host = read_symbols(
            """
Archive member name at 12: session.obj
001 00000000 SECT1 notype () External | required
Archive member name at 34: helpers.obj
002 00000000 SECT2 notype () External | module_local
003 00000000 SECT3 notype () External | unrelated
""".splitlines()
        )
        self.assertEqual(select_exports(host, module), {"required": False})

    def test_dllimport_functions_and_data(self):
        host = Symbols(defined={"function", "data"}, functions={"function"})
        module = Symbols(undefined={"__imp_function", "__imp_data"})
        self.assertEqual(select_exports(host, module), {"data": True, "function": False})

    def test_direct_data_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without __declspec"):
            select_exports(Symbols(defined={"data"}), Symbols(undefined={"data"}))

    def test_unrelated_symbols_do_not_exhaust_import_library(self):
        functions = {f"unrelated_{i}" for i in range(70000)} | {"required"}
        self.assertEqual(
            select_exports(Symbols(defined=functions, functions=functions), Symbols(undefined={"required"})),
            {"required": False},
        )

    def test_excess_required_exports_are_rejected(self):
        functions = {f"required_{i}" for i in range(65536)}
        with self.assertRaisesRegex(ValueError, "exceed the Windows limit"):
            select_exports(Symbols(defined=functions, functions=functions), Symbols(undefined=functions))

    def test_empty_or_unrecognized_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No COFF external symbols"):
            read_symbols(["Microsoft linker: unknown object format"])
        with self.assertRaisesRegex(ValueError, "No host symbols"):
            select_exports(Symbols(defined={"unrelated"}), Symbols(undefined={"required"}))

    def test_definition_file_is_sorted_and_preserves_decorated_names(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exports.def"
            write_exports(output, {"?Run@Session@@QEAAHXZ": False, "?Data@@3HA": True})
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                'EXPORTS\n  "?Data@@3HA" DATA\n  "?Run@Session@@QEAAHXZ"\n',
            )

    @unittest.skipUnless(sys.platform == "win32", "Requires the Windows MSVC toolchain")
    def test_windows_targeted_build_and_incremental_exports(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="ort test exports ") as directory:
            source = Path(directory)
            build = source / "build"
            (source / "host.cc").write_text(
                "namespace test_runtime {\nint required(int value) { return value + 1; }\n"
                + "\n".join(f"int unused_{i}(int value) {{ return value + {i}; }}" for i in range(70000))
                + "\n}\n",
                encoding="utf-8",
            )
            (source / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
            module = source / "module.cc"
            module.write_text(
                "namespace test_runtime { int required(int); }\n"
                'extern "C" __declspec(dllexport) int run_tests() { return test_runtime::required(41); }\n',
                encoding="utf-8",
            )
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(TestExports LANGUAGES CXX)
set(REPO_ROOT "{repo_root.as_posix()}")
include("${{REPO_ROOT}}/cmake/onnxruntime_test_exports.cmake")
add_library(runtime STATIC host.cc)
target_compile_options(runtime PRIVATE /bigobj)
add_library(test_objects OBJECT module.cc)
add_executable(provider_test_executable main.cc)
set_target_properties(provider_test_executable PROPERTIES OUTPUT_NAME provider_test)
target_link_libraries(provider_test_executable PRIVATE runtime)
onnxruntime_export_test_symbols(provider_test_executable OBJECT_TARGET test_objects HOST_LIBS runtime)
add_library(test_module MODULE $<TARGET_OBJECTS:test_objects>)
target_link_libraries(test_module PRIVATE provider_test_executable)
add_custom_target(provider_test ALL DEPENDS provider_test_executable test_module)
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/linker.txt" CONTENT "${{CMAKE_LINKER}}")
""",
                encoding="utf-8",
            )
            subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64"],
                check=True,
                timeout=120,
            )
            command = ["cmake", "--build", str(build), "--config", "Release", "--target", "provider_test", "--parallel"]
            subprocess.run(command, check=True, timeout=240)
            output_dir = build / "Release"
            for artifact in ("provider_test.exe", "provider_test.lib", "test_module.dll"):
                self.assertTrue((output_dir / artifact).is_file(), artifact)
            definition = build / "provider_test_executable_exports/Release/exports.def"
            self.assertEqual(
                definition.read_text(encoding="utf-8").splitlines(),
                [
                    "EXPORTS",
                    '  "?required@test_runtime@@YAHH@Z"',
                ],
            )
            linker = (build / "linker.txt").read_text(encoding="utf-8")
            imports = subprocess.check_output(
                [linker, "/dump", "/nologo", "/imports", str(output_dir / "test_module.dll")],
                text=True,
                timeout=30,
            )
            self.assertIn("provider_test.exe", imports)
            self.assertIn("?required@test_runtime@@YAHH@Z", imports)

            module.write_text(
                "namespace test_runtime { int required(int); int unused_0(int); }\n"
                'extern "C" __declspec(dllexport) int run_tests() {\n'
                "  return test_runtime::required(41) + test_runtime::unused_0(1);\n}\n",
                encoding="utf-8",
            )
            subprocess.run(command, check=True, timeout=120)
            self.assertEqual(len(definition.read_text(encoding="utf-8").splitlines()), 3)
            self.assertIn("?unused_0@test_runtime@@YAHH@Z", definition.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
