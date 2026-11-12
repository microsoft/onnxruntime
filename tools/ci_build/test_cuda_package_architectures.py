# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the CUDA architectures that the packages are built for.

They run the production CMake code (cuda_configuration.cmake, onnxruntime_cuda_source_filters.cmake and the
FlashAttention blocks of the CUDA provider and the CUDA plugin) in a throw-away CMake project that enables no
language, so neither a CUDA toolkit nor a compiler is needed. They also check that the four places that spell the
Linux aarch64 wheel architecture list stay in sync, and that the list covers every GPU the wheel targets.
"""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_DIR = REPO_ROOT / "cmake"
GITHUB_DIR = REPO_ROOT / "tools" / "ci_build" / "github"

# Compute capabilities of the GPUs that the Linux aarch64 CUDA 13 wheel supports.
AARCH64_WHEEL_GPUS = {
    (8, 7): "Jetson Orin",
    (8, 9): "L4/L40S/RTX 6000 Ada",
    (9, 0): "GH200",
    (10, 0): "B200/GB200",
    (10, 3): "B300/GB300",
    (11, 0): "Jetson Thor",
    (12, 0): "RTX 50/RTX PRO",
    (12, 1): "DGX Spark/GB10",
}

# GPUs that do not follow the CUDA cubin minor-version compatibility rule (Tegra).
TEGRA_GPUS = {(8, 7), (11, 0)}


def _entry_runs_on(entry, gpu):
    """Return True if SASS generated for the nvcc architecture `entry` loads on `gpu` = (major, minor).

    `entry` is a target as normalized by setup_cuda_architectures(), for example "90a-real" or "120f-real".
    "<cc>a": exactly that compute capability. "<cc>f": the same major version and an equal or higher minor
    version. "<cc>": the same, but Tegra GPUs only run their own cubin.
    """
    match = re.fullmatch(r"(\d+)([af]?)(?:-(real|virtual))?", entry)
    if match is None or match.group(3) == "virtual":
        return False
    major, minor = divmod(int(match.group(1)), 10)
    suffix = match.group(2)
    if suffix == "a" or (suffix == "" and ((major, minor) in TEGRA_GPUS or gpu in TEGRA_GPUS)):
        return gpu == (major, minor)
    return gpu[0] == major and gpu[1] >= minor


class CudaPackageArchitecturesTest(unittest.TestCase):
    def _run_cmake(self, architectures, body, prelude="", before_setup=""):
        """Configure a project that runs `before_setup`, setup_cuda_architectures(), `prelude` and `body`."""
        with tempfile.TemporaryDirectory(prefix="ort cuda arch ") as directory:
            source = Path(directory)
            (source / "CMakeLists.txt").write_text(
                f"""
cmake_minimum_required(VERSION 3.28)
project(TestCudaPackageArchitectures LANGUAGES NONE)
include("{CMAKE_DIR.as_posix()}/external/cuda_configuration.cmake")
include("{CMAKE_DIR.as_posix()}/onnxruntime_cuda_source_filters.cmake")
set(CMAKE_CUDA_COMPILER_VERSION 13.0)
set(CMAKE_CUDA_ARCHITECTURES "{architectures}")
{before_setup}
setup_cuda_architectures()
{prelude}
{body}
""",
                encoding="utf-8",
            )
            return subprocess.run(
                ["cmake", "-S", str(source), "-B", str(source / "build")],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def _configure(self, architectures, body, prelude=""):
        """Run `body` after setup_cuda_architectures() and return the values printed as `RESULT <name>=<value>`."""
        result = self._run_cmake(architectures, body, prelude)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(re.findall(r"RESULT (\w+)=(.*)", result.stdout))

    def _flash_architectures(self, architectures, reuse_sm80=None, plugin=False):
        """Return the CUDA_ARCHITECTURES of the FlashAttention library, as computed by the production CMake code.

        `reuse_sm80` sets onnxruntime_FLASH_ATTENTION_REUSE_SM80; None keeps the default from CMakeLists.txt.
        """
        if plugin:
            text = (CMAKE_DIR / "onnxruntime_providers_cuda_plugin.cmake").read_text(encoding="utf-8")
            start = text.index("if(_cuda_plugin_flash_attention_srcs)")
            block = text[start : text.index("if(_cuda_plugin_xqa_srcs)", start)]
            prelude = """
function(onnxruntime_add_cuda_plugin_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES;COMPILE_OPTIONS")
  message(STATUS "RESULT FLASH_ARCHS=${_ARG_CUDA_ARCHITECTURES}")
endfunction()
set(_cuda_plugin_flash_attention_srcs flash.cu)
set(_cuda_plugin_shared_compile_options "")
"""
        else:
            text = (CMAKE_DIR / "onnxruntime_providers_cuda.cmake").read_text(encoding="utf-8")
            start = text.index("    if(onnxruntime_cuda_flash_attention_srcs)")
            block = text[start : text.index("    # XQA kernels require SM80+.", start)]
            prelude = """
function(onnxruntime_add_cuda_object_library)
  cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;PARENT;CUDA_ARCHITECTURES;NVCC_THREADS" "SOURCES")
  message(STATUS "RESULT FLASH_ARCHS=${_ARG_CUDA_ARCHITECTURES}")
endfunction()
set(onnxruntime_cuda_flash_attention_srcs flash.cu)
"""
        prelude += "set(onnxruntime_FLASH_NVCC_THREADS 1)\n"
        if reuse_sm80 is None:
            text = (CMAKE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
            option = re.search(r"^option\(onnxruntime_FLASH_ATTENTION_REUSE_SM80 .*\)$", text, re.MULTILINE)
            self.assertIsNotNone(option)
            prelude += option.group(0)
        else:
            prelude += f"set(onnxruntime_FLASH_ATTENTION_REUSE_SM80 {'ON' if reuse_sm80 else 'OFF'})"
        # Every case has an SM80+ entry, so the block never falls back to compiling the sources in the parent
        # target, which this project does not define.
        results = self._configure(architectures, block, prelude)
        return results.get("FLASH_ARCHS", "")

    def test_flash_attention_reuses_sm80_sass_on_sm86_and_sm89_only_when_requested(self):
        cases = (
            # (architectures, expected without the option, expected with the option)
            (
                "75-real;80-real;86-real;89-real;90-real;120-real",
                "80-real;86-real;89-real;90a-real;120a-real",
                "80-real;90a-real;120a-real",
            ),
            (
                "60-real;70-real;75-real;80-real;86-real;89-real;90-real;120-real",
                "80-real;86-real;89-real;90a-real;120a-real",
                "80-real;90a-real;120a-real",
            ),
            # Without SM80 SASS there is nothing to reuse.
            ("86-real;89-real", "86-real;89-real", "86-real;89-real"),
            ("75;86;89", "86-real;89-real", "86-real;89-real"),
            # SM87 (Tegra) is never dropped, and PTX entries are kept.
            ("80-real;87-real;89-real", "80-real;87-real;89-real", "80-real;87-real"),
            ("80-real;86-real;89-virtual", "80-real;86-real;89-virtual", "80-real;89-virtual"),
            ("86-real;89-real;80-virtual", "86-real;89-real;80-virtual", "86-real;89-real;80-virtual"),
            # The Linux aarch64 wheel has no SM80 SASS to reuse, so its SM89 entry is kept.
            (
                self._aarch64_wheel_architectures(),
                "87-real;89-real;90a-real;100f-real;110a-real;120f-real",
                "87-real;89-real;90a-real;100f-real;110a-real;120f-real",
            ),
        )
        for plugin in (False, True):
            for architectures, without_option, with_option in cases:
                with self.subTest(architectures=architectures, plugin=plugin):
                    self.assertEqual(self._flash_architectures(architectures, False, plugin), without_option)
                    self.assertEqual(self._flash_architectures(architectures, True, plugin), with_option)

    def test_x64_packages_reuse_sm80_flash_attention_by_default(self):
        text = (GITHUB_DIR / "azure-pipelines" / "py-cuda13-packaging-pipeline.yml").read_text(encoding="utf-8")
        for platform, expected in (("linux", "80-real;90a-real;120a-real"), ("windows", "80-real;120a-real")):
            match = re.search(rf"cmake_{platform}_x64_cuda_archs:\s*'([^']+)'", text)
            self.assertIsNotNone(match)
            for plugin in (False, True):
                with self.subTest(platform=platform, plugin=plugin):
                    self.assertEqual(self._flash_architectures(match.group(1), plugin=plugin), expected)

    def test_reuse_sm80_sass_keeps_sm86_and_sm89_when_min_sm_excludes_sm80(self):
        results = self._configure(
            "80-real;86-real;89-real",
            """
onnxruntime_filter_cuda_archs(min80 MIN_SM 80 REUSE_SM80_SASS)
onnxruntime_filter_cuda_archs(min86 MIN_SM 86 REUSE_SM80_SASS)
message(STATUS "RESULT MIN80=${min80}")
message(STATUS "RESULT MIN86=${min86}")
""",
        )
        self.assertEqual(results["MIN80"], "80-real")
        # MIN_SM drops the SM80 entry, so there is no SM80 cubin for SM86/SM89 GPUs to run.
        self.assertEqual(results["MIN86"], "86-real;89-real")

    def _aarch64_wheel_architectures(self):
        text = (GITHUB_DIR / "azure-pipelines" / "py-cuda13-packaging-pipeline.yml").read_text(encoding="utf-8")
        match = re.search(r"cmake_linux_aarch64_cuda_archs:\s*'([^']+)'", text)
        self.assertIsNotNone(match)
        return match.group(1)

    def _normalized_architectures(self, architectures):
        """Return the targets that nvcc builds for `architectures`, as normalized by setup_cuda_architectures()."""
        results = self._configure(architectures, 'message(STATUS "RESULT ARCHS=${CMAKE_CUDA_ARCHITECTURES}")')
        return results["ARCHS"]

    def test_linux_aarch64_wheel_architectures_keep_family_targets(self):
        results = self._configure(
            self._aarch64_wheel_architectures(),
            """
get_directory_property(defs COMPILE_DEFINITIONS)
message(STATUS "RESULT ARCHS=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "RESULT DEFS=${defs}")
message(STATUS "RESULT SM90=${ORT_HAS_SM90}")
message(STATUS "RESULT SM12X=${ORT_HAS_SM12X}")
""",
        )
        # One cubin per family: no PTX is added, and 100f/120f are not rewritten to the arch-specific 100a/120a.
        self.assertEqual(results["ARCHS"], "87-real;89-real;90a-real;100f-real;110a-real;120f-real")
        self.assertEqual(results["SM90"], "ON")
        self.assertEqual(results["SM12X"], "ON")
        definitions = results["DEFS"].split(";")
        self.assertIn("HAS_SM80_OR_LATER", definitions)
        self.assertIn("HAS_SM100_OR_LATER", definitions)
        # Kernels for SM80/SM86/SM89 are compiled for the SM87 and later entries, so nothing is excluded.
        self.assertEqual([d for d in definitions if d.startswith("EXCLUDE_SM_")], [])

    def test_lowest_cc_per_major_keeps_one_compute_capability_per_major_version(self):
        results = self._configure(
            "90",
            """
onnxruntime_cuda_lowest_cc_per_major(none)
onnxruntime_cuda_lowest_cc_per_major(one 121)
onnxruntime_cuda_lowest_cc_per_major(several 121 103 120 100 121)
message(STATUS "RESULT NONE=${none}")
message(STATUS "RESULT ONE=${one}")
message(STATUS "RESULT SEVERAL=${several}")
""",
        )
        self.assertEqual(results["NONE"], "")
        self.assertEqual(results["ONE"], "121")
        self.assertEqual(results["SEVERAL"], "100;120")

    def test_family_specific_architectures_report_the_required_cmake_version(self):
        # CMake rejects the "f" suffix in CMAKE_CUDA_ARCHITECTURES before 3.31.8 and 4.0.2, when CUDA is enabled.
        # setup_cuda_architectures() reports that up front; pretend to be another CMake version to check it.
        cases = (
            ("3.31.7", "90-real;120f-real", False),
            ("3.31.7", "120f", False),
            ("3.31.7", "90-real;121f-virtual", False),
            ("4.0.1", "100f-real", False),
            ("3.31.7", "90-real;120a-real", True),
            ("3.31.8", "120f-real", True),
            ("4.0.2", "100f-real;120f-real", True),
        )
        for version, architectures, supported in cases:
            with self.subTest(version=version, architectures=architectures):
                result = self._run_cmake(architectures, "", before_setup=f"set(CMAKE_VERSION {version})")
                output = " ".join((result.stdout + result.stderr).split())
                if supported:
                    self.assertEqual(result.returncode, 0, output)
                else:
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("requires CMake 3.31.8 or later (4.0.2 or later in the 4.0 series)", output)

    def test_linux_aarch64_cuda13_architectures_are_in_sync(self):
        pipelines = GITHUB_DIR / "azure-pipelines"
        found = {}
        for name in ("py-cuda13-packaging-pipeline.yml", "plugin-cuda13-pipeline.yml"):
            text = (pipelines / name).read_text(encoding="utf-8")
            found[name] = re.search(r"cmake_linux_aarch64_cuda_archs:\s*'([^']+)'", text).group(1)
        text = (pipelines / "stages" / "plugin-cuda-packaging-stage.yml").read_text(encoding="utf-8")
        found["plugin-cuda-packaging-stage.yml"] = re.search(
            r"name: cmake_linux_aarch64_cuda_archs\s+type: string\s+displayName: '[^']*'\s+default: '([^']+)'", text
        ).group(1)
        text = (GITHUB_DIR / "linux" / "build_linux_python_package.sh").read_text(encoding="utf-8")
        found["build_linux_python_package.sh"] = re.search(
            r'"\$ARCH" == "aarch64" \]; then\s+(?:#[^\n]*\n\s+)*CUDA_ARCHS="([^"]+)"', text
        ).group(1)
        self.assertEqual(len(set(found.values())), 1, found)

    def test_linux_aarch64_wheel_architectures_cover_target_gpus_without_redundant_entries(self):
        # Check the targets that nvcc builds, not the spelling in the pipeline: setup_cuda_architectures() turns
        # "100-real" into the arch-specific "100a-real", which does not run on 10.3.
        entries = self._normalized_architectures(self._aarch64_wheel_architectures()).split(";")
        for gpu, name in AARCH64_WHEEL_GPUS.items():
            with self.subTest(gpu=gpu, name=name):
                self.assertTrue(any(_entry_runs_on(entry, gpu) for entry in entries), f"{name} {gpu} is not covered")
        # Every entry must be the only one that covers some target GPU. A second entry for the same family
        # (e.g. 103 next to 100f, or 121 next to 120f) only adds ~40 MB of cubin to the wheel.
        for entry in entries:
            with self.subTest(entry=entry):
                others = [e for e in entries if e != entry]
                self.assertTrue(
                    any(
                        _entry_runs_on(entry, gpu) and not any(_entry_runs_on(other, gpu) for other in others)
                        for gpu in AARCH64_WHEEL_GPUS
                    ),
                    f"{entry} is redundant",
                )


if __name__ == "__main__":
    unittest.main()
