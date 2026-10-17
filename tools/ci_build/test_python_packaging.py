#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
import unittest
from pathlib import Path

import yaml
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
AZURE = "tools/ci_build/github/azure-pipelines/"
REQUIREMENTS = (
    "tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt",
    "tools/ci_build/github/linux/docker/inference/aarch64/python/cpu/scripts/requirements.txt",
    "tools/ci_build/github/linux/docker/scripts/lort/requirements.txt",
    "tools/ci_build/requirements/transformers-test/requirements.txt",
)
EXCLUDED = {
    REQUIREMENTS[0]: {"triton", "neural-compressor"},
    REQUIREMENTS[1]: set(),
    REQUIREMENTS[2]: {"h5py"},
    REQUIREMENTS[3]: {"torch", "torchvision", "sentencepiece"},
}


def marker_environment(version):
    return {
        **default_environment(),
        "python_version": version,
        "python_full_version": version + ".0",
        "implementation_name": "cpython",
        "implementation_version": version + ".0",
        "platform_python_implementation": "CPython",
    }


def read_requirements(path):
    return [
        Requirement(line)
        for line in (ROOT / path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


class PythonPackagingRequirementsTest(unittest.TestCase):
    def test_numpy_selection_preserves_older_versions(self):
        for path in REQUIREMENTS:
            requirements = read_requirements(path)
            self.assertIn('numpy==2.2.6; python_version < "3.11"', (ROOT / path).read_text().splitlines())
            for version, pin in (("3.11", "==2.4.2"), ("3.14", "==2.4.2"), ("3.15", "==2.5.4")):
                with self.subTest(path=path, version=version):
                    selected = [
                        str(req.specifier)
                        for req in requirements
                        if req.name == "numpy"
                        and (req.marker is None or req.marker.evaluate(marker_environment(version)))
                    ]
                    self.assertEqual(selected, [pin])

    def test_unsupported_dependencies_are_selected_only_before_python315(self):
        for path, excluded in EXCLUDED.items():
            requirements = read_requirements(path)
            for version in ("3.11", "3.14", "3.15"):
                with self.subTest(path=path, version=version):
                    active = {
                        req.name
                        for req in requirements
                        if req.marker is None or req.marker.evaluate(marker_environment(version))
                    }
                    self.assertEqual(active & excluded, excluded if version != "3.15" else set())


class PythonPackagingCpuJobsTest(unittest.TestCase):
    def test_windows_matrix_includes_unconditional_python315_for_both_architectures(self):
        path = ROOT / "tools/ci_build/github/azure-pipelines/templates/py-win-cpu.yml"
        template = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(
            [p["name"] for p in template["parameters"]],
            ["architecture", "build_py_parameters", "cmake_build_type"],
        )
        matrix = template["jobs"][0]["strategy"]["matrix"]
        self.assertEqual(
            list(matrix), [f"Python{v}_${{{{parameters.architecture}}}}" for v in (311, 312, 313, 314, 315)]
        )
        for architecture in ("x64", "arm64"):
            with self.subTest(architecture=architecture):
                expanded = {
                    key.replace("${{parameters.architecture}}", architecture): value for key, value in matrix.items()
                }
                self.assertEqual(expanded[f"Python315_{architecture}"], {"PythonVersion": "3.15"})

    def test_macos_python315_is_inside_existing_platform_switch(self):
        path = ROOT / "tools/ci_build/github/azure-pipelines/stages/py-cpu-packaging-stage.yml"
        template = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(
            [p["name"] for p in template["parameters"]],
            [
                "build_py_parameters",
                "enable_linux_cpu",
                "enable_windows_cpu",
                "enable_mac_cpu",
                "enable_linux_arm",
                "cmake_build_type",
            ],
        )
        self.assertEqual(
            [next(iter(branch)) for branch in template["stages"]],
            [
                f"${{{{ if eq(parameters.{name}, true) }}}}"
                for name in ("enable_windows_cpu", "enable_mac_cpu", "enable_linux_arm", "enable_linux_cpu")
            ],
        )
        mac = template["stages"][1]["${{ if eq(parameters.enable_mac_cpu, true) }}"][0]
        self.assertEqual(
            [job["parameters"]["python_version"] for job in mac["jobs"]], ["3.11", "3.12", "3.13", "3.14", "3.15"]
        )
        self.assertTrue(all(job["parameters"]["arch"] == "arm64" for job in mac["jobs"]))


def read_pipeline(path):
    return yaml.load((ROOT / AZURE / path).read_text(), Loader=yaml.BaseLoader)


def parameter_defaults(template):
    return {item["name"]: item["default"] for item in template["parameters"] if "default" in item}


def gpu_name(value, version):
    return (
        value.replace("${{ parameters.EP_NAME }}", "gpu")
        .replace("${{ replace(parameters.PYTHON_VERSION,'.','_') }}", version.replace(".", "_"))
        .replace("${{ parameters.PYTHON_VERSION }}", version)
    )


class PythonPackagingGpuJobsTest(unittest.TestCase):
    def test_gpu_version_lists_preserve_existing_entries_and_add_ordinary_python315(self):
        shared = read_pipeline("stages/py-gpu-packaging-stage.yml")
        defaults = parameter_defaults(shared)
        self.assertEqual(defaults["PythonVersions"], ["3.11", "3.12", "3.13", "3.14", "3.15"])
        versions = ["3.11", "3.12", "3.13", "3.13t", "3.14", "3.14t", "3.15"]
        cuda13 = read_pipeline("py-cuda13-packaging-pipeline.yml")
        override = cuda13["extends"]["parameters"]["stages"][0]["parameters"]
        for configurations in (defaults["LinuxPythonConfigurations"], override["AArch64LinuxPythonConfigurations"]):
            with self.subTest(configurations=configurations):
                self.assertEqual([config["python_version"] for config in configurations], versions)
                for config in configurations:
                    version = config["python_version"]
                    digits = version.rstrip("t").replace(".", "")
                    abi = "t" if version.endswith("t") else ""
                    self.assertEqual(
                        config["docker_python_exe_path"],
                        f"/opt/python/cp{digits}-cp{digits}{abi}/bin/python{version.rstrip('t')}",
                    )

    def test_windows_stage_names_publication_and_test_condition(self):
        shared = parameter_defaults(read_pipeline("stages/py-gpu-packaging-stage.yml"))
        build, tests = read_pipeline("stages/py-win-gpu-stage.yml")["stages"]
        names = []
        for version in shared["PythonVersions"]:
            build_name = gpu_name(build["stage"], version)
            test_name = gpu_name(tests["stage"], version)
            names.extend((build_name, test_name))
            self.assertEqual(build_name, f"Win_py_gpu_Wheels_{version.replace('.', '_')}_Build")
            self.assertEqual(test_name, f"Win_py_gpu_Wheels_{version.replace('.', '_')}_Tests")
            self.assertEqual(gpu_name(tests["dependsOn"], version), build_name)
            self.assertEqual(gpu_name(build["jobs"][0]["job"], version), build_name)
            self.assertEqual(gpu_name(tests["jobs"][0]["job"], version), test_name)
            output = build["jobs"][0]["templateContext"]["outputs"][0]
            self.assertEqual(gpu_name(output["artifactName"], version), f"win_gpu_wheel_{version}")
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("Win_py_gpu_Wheels_3_15_Build", names)
        self.assertIn("Win_py_gpu_Wheels_3_15_Tests", names)
        self.assertNotIn("condition", build)
        self.assertNotIn("condition", build["jobs"][0])
        self.assertNotIn("condition", tests)
        self.assertNotIn("condition", tests["jobs"][0])
        conditions = {
            version: gpu_name(tests.get("condition", "succeeded()"), version) for version in shared["PythonVersions"]
        }
        self.assertEqual(conditions["3.15"], conditions["3.14"])
        for version in ("3.11", "3.14", "3.15"):
            for succeeded in (False, True):
                self.assertEqual(conditions[version].replace("succeeded()", str(succeeded)), str(succeeded))

    def test_windows_republish_downloads_have_matching_producers(self):
        versions = parameter_defaults(read_pipeline("stages/py-gpu-packaging-stage.yml"))["PythonVersions"]
        steps = read_pipeline("py-cuda-package-test-pipeline.yml")["stages"][1]["jobs"][0]["steps"]
        downloads = [step for step in steps if step.get("artifact", "").startswith("win_gpu_wheel_")]
        self.assertEqual([step["artifact"] for step in downloads], [f"win_gpu_wheel_{v}" for v in versions])
        self.assertIn("win_gpu_wheel_3.15", [step["artifact"] for step in downloads])
        for step, version in zip(downloads, versions, strict=True):
            self.assertEqual(step["displayName"], f"Download Pipeline Artifact - Win GPU {version}")
            self.assertEqual(step["patterns"], "*.whl")
            self.assertEqual(step["download"], "build")

    def test_linux_python315_producers_match_merge_dependencies_and_downloads(self):
        template = read_pipeline("stages/py-gpu-packaging-stage.yml")
        self.assertIn(
            "3.15", [config["python_version"] for config in parameter_defaults(template)["LinuxPythonConfigurations"]]
        )
        for architecture, parameter in (
            ("x86_64", "LinuxPythonConfigurations"),
            ("aarch64", "AArch64LinuxPythonConfigurations"),
        ):
            loop = "${{ each config in parameters." + parameter + " }}"
            producer = next(stage[loop][0] for stage in template["stages"] if loop in stage)
            self.assertEqual(producer["template"], "py-linux-gpu-stage.yml")
            parameters = producer["parameters"]
            self.assertEqual(parameters["arch"], architecture)
            self.assertEqual(parameters["python_version"], "${{ config.python_version }}")
            self.assertEqual(parameters["docker_python_exe_path"], "${{ config.docker_python_exe_path }}")
            stage_name = parameters["stage_name"].replace("${{ replace(config.python_version, '.', '_') }}", "3_15")
            artifact = parameters["wheel_artifact_name"].replace(
                "${{ replace(config.python_version, '.', '_') }}", "3_15"
            )
            self.assertEqual(stage_name, f"Linux_py_GPU_Wheels_{architecture}_3_15")
            self.assertEqual(artifact, "onnxruntime_gpu_" + ("aarch64_" if architecture == "aarch64" else "") + "3_15")
            merge = template["stages"][-1]
            dependencies = next(item[loop] for item in merge["dependsOn"] if loop in item)
            self.assertIn(parameters["stage_name"], dependencies)
            steps = merge["jobs"][0]["steps"]
            download = next(item[loop][0] for item in steps if loop in item)
            self.assertEqual(download["inputs"]["artifact"], parameters["wheel_artifact_name"])
            self.assertNotIn("build_intermediates_artifact_name", parameters)
            self.assertEqual(
                [key for key in parameters if key.startswith("${{ if")],
                ["${{ if eq(config.python_version, '3.12') }}"],
            )
        linux = read_pipeline("stages/py-linux-gpu-stage.yml")
        self.assertEqual(linux["stages"][0]["stage"], "${{ parameters.stage_name }}")
        self.assertIn("${{ parameters.wheel_artifact_name }}", str(linux["stages"][0]["jobs"]))

    def test_smoke_matrix_and_package_classifier_include_python315(self):
        matrix = read_pipeline("templates/py-package-smoking-test.yml")["jobs"][0]["strategy"]["matrix"]
        self.assertEqual(list(matrix), [f"Python{version}" for version in (311, 312, 313, 314, 315)])
        self.assertEqual(list(matrix.values()), [{"PythonVersion": f"3.{version}"} for version in (11, 12, 13, 14, 15)])
        tree = ast.parse((ROOT / "setup.py").read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "classifiers" for target in node.targets)
        )
        classifiers = ast.literal_eval(assignment.value)
        python_classifiers = [
            value for value in classifiers if value.startswith("Programming Language :: Python :: 3.")
        ]
        self.assertEqual(python_classifiers, [f"Programming Language :: Python :: 3.{v}" for v in (11, 12, 13, 14, 15)])
        self.assertEqual(len(classifiers), len(set(classifiers)))


if __name__ == "__main__":
    unittest.main()
