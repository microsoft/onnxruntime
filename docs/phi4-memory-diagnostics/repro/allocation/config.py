# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Explicit external prerequisites; checking never imports a GPU library."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import support as s

LIBRARIES = (
    "libonnxruntime.so",
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime-genai.so",
    "libonnxruntime-genai-cuda.so",
)


def absolute_path(value, label):
    s.require(isinstance(value, str) and Path(value).is_absolute(), f"{label} must be an absolute external path")
    path = Path(value).resolve()
    s.require(not path.is_relative_to(s.HERE), f"{label} must be external to this artifact")
    return path


def validate_config(config):
    required = {
        "schema_version",
        "source_pins",
        "model_dir",
        "ort_home",
        "cuda_toolkit",
        "cudnn_library_dir",
        "inputs",
        "output_dir",
        "gpu_uuid",
        "gpu_index",
        "runtimes",
        "phase_b",
        "resource_limits",
    }
    s.require(set(config) == required, f"Config keys must be exactly {sorted(required)}")
    s.require(config["schema_version"] == 1, "Unsupported config schema")
    s.require(config["source_pins"] == {"genai": s.GENAI_COMMIT, "ort": s.ORT_COMMIT}, "Source pins changed")
    s.require(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", config["gpu_uuid"]) is not None, "Explicit GPU UUID required")
    s.require(type(config["gpu_index"]) is int and config["gpu_index"] >= 0, "Invalid physical GPU index")
    paths = [absolute_path(config[key], key) for key in ("model_dir", "ort_home", "cuda_toolkit", "cudnn_library_dir")]
    s.require(set(config["inputs"]) == {"2048", "32768"}, "Both external saved input files are required")
    paths.extend(absolute_path(path, "input") for path in config["inputs"].values())
    s.require(set(config["runtimes"]) == set(s.VARIANTS), "Exactly baseline and patched runtimes are required")
    for variant in s.VARIANTS:
        runtime = config["runtimes"][variant]
        s.require(
            set(runtime)
            == {
                "genai_library_dir",
                "test_models_dir",
                "runtime_sha256",
                "phase_a",
                "allocation_tests",
                "existing_tests",
            },
            f"Unexpected {variant} runtime keys",
        )
        paths.append(absolute_path(runtime["genai_library_dir"], "genai_library_dir"))
        paths.append(absolute_path(runtime["test_models_dir"], "test_models_dir"))
        s.require(set(runtime["runtime_sha256"]) == set(LIBRARIES), "Five explicit runtime digests required")
        for digest in runtime["runtime_sha256"].values():
            validate_digest(digest)
        for name in ("phase_a", "allocation_tests", "existing_tests"):
            validate_executable(runtime[name])
            paths.append(absolute_path(runtime[name]["path"], name))
        validate_test_layout(runtime)
    validate_executable(config["phase_b"])
    paths.append(absolute_path(config["phase_b"]["path"], "phase_b"))
    output = absolute_path(config["output_dir"], "output_dir")
    for path in paths:
        s.require(
            not output.is_relative_to(path) and not path.is_relative_to(output),
            "Output must not overlap external input/runtime/model/toolkit paths",
        )
    baseline = config["runtimes"]["baseline"]
    patched = config["runtimes"]["patched"]
    s.require(
        Path(baseline["genai_library_dir"]).resolve() != Path(patched["genai_library_dir"]).resolve(),
        "Baseline and patched library directories must differ",
    )
    s.require(
        baseline["runtime_sha256"]["libonnxruntime-genai.so"] != patched["runtime_sha256"]["libonnxruntime-genai.so"],
        "Baseline and patched GenAI must differ",
    )
    for name in LIBRARIES[:3]:
        s.require(
            baseline["runtime_sha256"][name] == patched["runtime_sha256"][name],
            "Both variants must use the identical ORT build",
        )
    limits = config["resource_limits"]
    s.require(
        set(limits)
        == {"min_host_available_bytes", "max_swap_used_bytes", "min_output_free_bytes", "min_gpu_free_bytes"},
        "Explicit resource limits required",
    )
    s.require(all(type(value) is int and value >= 0 for value in limits.values()), "Invalid resource limit")
    s.require(all(limits[key] > 0 for key in limits if key.startswith("min_")), "Resource gates cannot be disabled")
    return config


def validate_digest(digest):
    s.require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Explicit SHA-256 required")


def verify_fixtures(models_dir):
    root = absolute_path(str(models_dir), "test_models_dir")
    manifest = s.load(s.HERE / "fixture-manifest.json")
    s.require(manifest["source_revision"] == s.GENAI_COMMIT, "Fixture revision changed")
    identities = {}
    for entry in manifest["files"]:
        path = root / entry["path"]
        s.require(
            path.is_file() and path.stat().st_size == entry["bytes"],
            f"Missing/incomplete pinned regression fixture: {path}",
        )
        s.require(s.sha256(path) == entry["sha256"], f"Pinned regression fixture hash mismatch: {path}")
        identities[entry["path"]] = entry["sha256"]
    return {"fixture_check_passed": True, "models_dir": str(root), "sha256": identities}


def validate_executable(value):
    s.require(set(value) == {"path", "sha256"}, "Executable requires path and SHA-256")
    absolute_path(value["path"], "executable")
    validate_digest(value["sha256"])


def validate_test_layout(runtime):
    library_dir = Path(runtime["genai_library_dir"]).resolve()
    for name in ("allocation_tests", "existing_tests"):
        s.require(
            Path(runtime[name]["path"]).resolve().parent == library_dir,
            f"{name} must remain in its corresponding genai_library_dir build directory; "
            "the pinned loader requires an adjacent libonnxruntime-genai-cuda.so",
        )


def runtime_hashes(config, variant):
    validate_test_layout(config["runtimes"][variant])
    result = {}
    for name in LIBRARIES:
        base = (
            Path(config["runtimes"][variant]["genai_library_dir"])
            if name.startswith("libonnxruntime-genai")
            else Path(config["ort_home"]) / "lib"
        )
        path = (base / name).resolve(strict=True)
        expected = config["runtimes"][variant]["runtime_sha256"][name]
        s.require(s.sha256(path) == expected, f"Runtime hash changed: {path}")
        result[str(path)] = expected
    return result


def host_resources(config):
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) * 1024
    limits = config["resource_limits"]
    output = Path(config["output_dir"])
    while not output.exists():
        output = output.parent
    result = {
        "host_available_bytes": fields["MemAvailable"],
        "swap_used_bytes": fields["SwapTotal"] - fields["SwapFree"],
        "output_free_bytes": shutil.disk_usage(output).free,
    }
    s.require(result["host_available_bytes"] >= limits["min_host_available_bytes"], "Host memory resource gate failed")
    s.require(result["swap_used_bytes"] <= limits["max_swap_used_bytes"], "Swap resource gate failed")
    s.require(result["output_free_bytes"] >= limits["min_output_free_bytes"], "Output disk resource gate failed")
    return result


def verify_native(executable, phase):
    path = Path(executable["path"])
    s.require(path.is_file() and os.access(path, os.X_OK), f"Missing executable: {path}")
    s.require(s.sha256(path) == executable["sha256"], f"Executable hash changed: {path}")
    if phase not in ("a", "b"):
        return
    dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
    s.require("(RUNPATH)" in dynamic and "(RPATH)" not in dynamic, "Native app must use overridable RUNPATH")
    s.require("Shared library: [libonnxruntime-genai.so]" in dynamic, "Unexpected GenAI linkage")
    symbols = subprocess.check_output(["nm", "-u", str(path)], text=True)
    if phase == "b":
        s.require(
            "OgaGenerator_GetLogits" not in symbols and "cudaDeviceSynchronize" in symbols,
            "Phase B must have completion barriers and no diagnostic GetLogits",
        )
    else:
        s.require("OgaGenerator_GetLogits" in symbols, "Phase A must capture full logits")


def check(config):
    """Validate local files and ELF metadata only; never start native code."""
    validate_config(config)
    s.require(os.name == "posix" and Path("/proc/self/maps").is_file(), "Linux /proc is required")
    for key in ("model_dir", "ort_home", "cuda_toolkit", "cudnn_library_dir"):
        s.require(Path(config[key]).is_dir(), f"Missing external prerequisite: {key}={config[key]}")
    for tool in ("readelf", "nm"):
        s.require(shutil.which(tool) is not None, f"Missing host prerequisite: {tool}")
    s.require(
        importlib.util.find_spec("pynvml") is not None,
        "Missing external Python dependency: nvidia-ml-py (pynvml); nothing was installed",
    )
    for path in (
        Path(config["cuda_toolkit"]) / "lib64/libcudart.so",
        Path(config["cudnn_library_dir"]) / "libcudnn.so",
    ):
        s.require(path.is_file(), f"Missing toolkit library: {path}")
    header = Path(config["ort_home"]) / "include/onnxruntime_experimental_c_api.inc"
    s.require(
        header.is_file() and "OrtApi_DebugLogAndShrinkGpuArenas" in header.read_text(),
        "BLOCKED: matching diagnostic ORT headers/API required; stock ORT is not a substitute",
    )
    manifest = s.load(s.MANIFEST)
    for context in s.COUNTS:
        workload = manifest["workloads"][str(context)]
        s.require(
            workload["input_tokens"] == s.COUNTS[context] and workload["input_ids_sha256"] == s.INPUT_HASHES[context],
            "Evidence input contract changed",
        )
        s.validate_input(Path(config["inputs"][str(context)]).read_bytes(), context)
        s.verify_config(workload["effective_config"], context)
        s.require(
            s.sha256(Path(config["model_dir"]) / "genai_config.json") == workload["source_config_sha256"],
            "Model source configuration differs from captured workload",
        )
    for artifact in manifest["model"]["artifacts"]:
        # The workload's locally edited config, not the download's original config, is authoritative.
        if artifact["name"] == "genai_config.json":
            continue
        path = Path(config["model_dir"]) / artifact["name"]
        s.require(
            path.is_file() and path.stat().st_size == artifact["bytes"] and s.sha256(path) == artifact["sha256"],
            f"Model artifact identity mismatch: {path}",
        )
    identities = {}
    fixtures = {}
    for variant in s.VARIANTS:
        fixtures[variant] = verify_fixtures(config["runtimes"][variant]["test_models_dir"])
        identities[variant] = runtime_hashes(config, variant)
        for key in ("phase_a", "allocation_tests", "existing_tests"):
            verify_native(config["runtimes"][variant][key], "a" if key == "phase_a" else "tests")
    verify_native(config["phase_b"], "b")
    return {
        "host_check_passed": True,
        "gpu_execution_validated": False,
        "runtime_sha256": identities,
        "regression_fixtures": fixtures,
        "model_manifest_sha256": s.sha256(s.MANIFEST),
        "resources": host_resources(config),
    }


def environment(config, variant, phase_b):
    overrides = {
        "CUDA_VISIBLE_DEVICES": config["gpu_uuid"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "LD_LIBRARY_PATH": ":".join(
            (
                config["runtimes"][variant]["genai_library_dir"],
                str(Path(config["ort_home"]) / "lib"),
                str(Path(config["cuda_toolkit"]) / "lib64"),
                config["cudnn_library_dir"],
            )
        ),
        "ORT_ARENA_DIAGNOSTICS": "1" if phase_b else "0",
        "ORTGENAI_LOGITS_ALLOC_TRACE": "0",
        "ORTGENAI_LOGITS_ALLOCATION_TRACE": "0",
        "ORTGENAI_ORT_VERBOSE_LOGGING": "0",
        "ORTGENAI_LOG_ORT_LIB": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    removed = ("LD_PRELOAD", "LD_AUDIT", "LD_TRACE_LOADED_OBJECTS", "ORT_LOGGING_LEVEL")
    env = {key: value for key, value in os.environ.items() if key not in removed}
    env.update(overrides)
    return env, overrides, removed
