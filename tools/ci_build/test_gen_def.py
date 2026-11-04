# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPublicExports(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires an ELF linker and dynamic loader")
    def test_embedded_curl_is_not_exported_or_interposed(self):
        repo_root = Path(__file__).resolve().parents[2]
        public_symbols = {
            line.strip()
            for line in (repo_root / "onnxruntime" / "core" / "providers" / "cpu" / "symbols.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        with tempfile.TemporaryDirectory(prefix="ort curl exports ") as directory:
            root = Path(directory)
            symbol_file = root / "onnxruntime.lds"
            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools" / "ci_build" / "gen_def.py"),
                    "--src_root",
                    str(repo_root / "onnxruntime"),
                    "--version_file",
                    str(repo_root / "VERSION_NUMBER"),
                    "--output",
                    str(symbol_file),
                    "--output_source",
                    str(root / "generated_source.c"),
                    "--style",
                    "gcc",
                    "--config",
                    "cpu",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            (root / "curl.c").write_text(
                "int curl_easy_init(void) { return 42; }\n"
                "int Curl_private(void) { return 1; }\n"
                "int mbedtls_private(void) { return 2; }\n",
                encoding="utf-8",
            )
            (root / "host.c").write_text("int curl_easy_init(void) { return -1; }\n", encoding="utf-8")
            (root / "runtime.c").write_text(
                "extern int curl_easy_init(void);\n"
                "int OrtGetApiBase(void) { return curl_easy_init(); }\n"
                + "".join(
                    f"int {symbol}(void) {{ return 0; }}\n" for symbol in sorted(public_symbols - {"OrtGetApiBase"})
                ),
                encoding="utf-8",
            )
            for command in (
                ["cc", "-fPIC", "-c", "curl.c", "-o", "curl.o"],
                ["ar", "rcs", "libcustom-transport.a", "curl.o"],
                ["cc", "-shared", "-fPIC", "host.c", "-o", "libhost.so"],
                ["cc", "-shared", "-fPIC", "runtime.c", "libcustom-transport.a", "-o", "libunrestricted.so"],
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "runtime.c",
                    "libcustom-transport.a",
                    f"-Wl,--version-script={symbol_file}",
                    "-o",
                    "libonnxruntime.so",
                ],
            ):
                subprocess.run(command, cwd=root, check=True, capture_output=True, text=True, timeout=30)
            symbols = subprocess.run(
                ["nm", "-D", "--defined-only", "libonnxruntime.so"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            exported_functions = {
                fields[2].split("@")[0]
                for line in symbols.splitlines()
                if len(fields := line.split()) == 3 and fields[1] == "T"
            }
            self.assertEqual(exported_functions, public_symbols)
            self.assertNotIn("curl_easy_init", symbols)
            self.assertNotIn("Curl_private", symbols)
            self.assertNotIn("mbedtls_private", symbols)
            interposition_test = (
                "import ctypes; "
                "host = ctypes.CDLL('./libhost.so', mode=ctypes.RTLD_GLOBAL); "
                "unrestricted = ctypes.CDLL('./libunrestricted.so'); "
                "runtime = ctypes.CDLL('./libonnxruntime.so'); "
                "assert host.curl_easy_init() == -1; "
                "assert unrestricted.OrtGetApiBase() == -1; "
                "assert runtime.OrtGetApiBase() == 42"
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    interposition_test,
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def test_public_exports_exclude_private_dependencies(self):
        repo_root = Path(__file__).resolve().parents[2]
        providers_root = repo_root / "onnxruntime" / "core" / "providers"
        providers = sorted(path.parent.name for path in providers_root.glob("*/symbols.txt"))
        public_symbols = set()
        for provider in providers:
            public_symbols.update(
                line.strip()
                for line in (providers_root / provider / "symbols.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        android_symbols = {
            line.strip()
            for line in (repo_root / "onnxruntime" / "core" / "platform" / "posix" / "android_telemetry_symbols.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        self.assertIn("OrtGetApiBase", public_symbols)
        self.assertTrue(android_symbols)
        self.assertFalse(any(symbol.startswith(("curl_", "Curl_", "mbedtls_")) for symbol in public_symbols))

        with tempfile.TemporaryDirectory(prefix="ort public exports ") as directory:
            output = Path(directory) / "exports"
            source = Path(directory) / "generated_source.c"
            for style in ("gcc", "vc", "xcode", "aix"):
                for android in (False, True):
                    with self.subTest(style=style, android=android):
                        command = [
                            sys.executable,
                            str(repo_root / "tools" / "ci_build" / "gen_def.py"),
                            "--src_root",
                            str(repo_root / "onnxruntime"),
                            "--version_file",
                            str(repo_root / "VERSION_NUMBER"),
                            "--output",
                            str(output),
                            "--output_source",
                            str(source),
                            "--style",
                            style,
                            "--config",
                            *providers,
                        ]
                        if android:
                            command.extend(
                                [
                                    "--extra_symbol_file",
                                    str(
                                        repo_root
                                        / "onnxruntime"
                                        / "core"
                                        / "platform"
                                        / "posix"
                                        / "android_telemetry_symbols.txt"
                                    ),
                                ]
                            )
                        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                        contents = output.read_text(encoding="utf-8")
                        expected_symbols = public_symbols | android_symbols if android else public_symbols
                        if style == "gcc":
                            global_section, local_section = contents.split(" local:\n")
                            actual_symbols = {
                                line.strip().removesuffix(";")
                                for line in global_section.split(" global:\n")[1].splitlines()
                            }
                            self.assertEqual(local_section, "    *;\n};   \n")
                        elif style == "vc":
                            actual_symbols = {line.split()[0] for line in contents.splitlines()[2:]}
                        elif style == "xcode":
                            actual_symbols = {line.removeprefix("_") for line in contents.splitlines()}
                        else:
                            actual_symbols = set(contents.splitlines())
                        self.assertEqual(actual_symbols, expected_symbols)
                        for symbol in android_symbols:
                            self.assertNotIn(symbol, source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
