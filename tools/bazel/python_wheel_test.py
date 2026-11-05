# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate native wheel payload/metadata and inference after a real install."""

import base64
import csv
import hashlib
import io
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


def run_inference(directory, model, version, distribution):
    import numpy as np  # noqa: PLC0415

    import onnxruntime as ort  # noqa: PLC0415

    if not Path(ort.__file__).resolve().is_relative_to(Path(directory).resolve()):
        raise AssertionError(f"Expected the wheel-installed package in {directory}, imported {ort.__file__}.")
    if ort.__version__ != version:
        raise AssertionError(f"Imported wheel reports version {ort.__version__!r}; expected {version!r}.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    provider = "CUDAExecutionProvider" if distribution == "onnxruntime-gpu" else "CPUExecutionProvider"
    if distribution not in {"onnxruntime", "onnxruntime-gpu"}:
        raise AssertionError(f"Unsupported wheel distribution for inference: {distribution!r}.")
    if distribution == "onnxruntime-gpu":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, sess_options=options, providers=[provider])
    if distribution == "onnxruntime-gpu":
        session.disable_fallback()
    selected_providers = session.get_providers()
    if not selected_providers or selected_providers[0] != provider:
        raise AssertionError(f"Expected {provider} to execute the wheel test, selected {selected_providers}.")
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b = a * 10
    output = session.run(["C"], {"A": a, "B": b})[0]
    np.testing.assert_array_equal(output, a + b)
    if output.dtype != np.float32:
        raise AssertionError(f"Wheel inference output dtype is {output.dtype}; expected float32.")
    print("Native wheel high-level Python Add inference passed.")


def _platform_tag():
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    if sys.platform == "linux" and machine in {"amd64", "x86_64"}:
        return "linux_x86_64"
    if sys.platform == "linux" and machine in {"aarch64", "arm64"}:
        return "linux_aarch64"
    if sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
        return "macosx_11_0_x86_64"
    if sys.platform == "darwin" and machine in {"aarch64", "arm64"}:
        return "macosx_11_0_arm64"
    raise AssertionError(f"Unsupported wheel test platform: {sys.platform}/{platform.machine()}.")


def _subprocess_pythonpath(install_directory, dependency_paths=None):
    paths = [str(Path(install_directory).resolve())]
    for dependency_path in sys.path if dependency_paths is None else dependency_paths:
        if not dependency_path:
            continue
        resolved_path = str(Path(dependency_path).resolve())
        if resolved_path not in paths:
            paths.append(resolved_path)
    return os.pathsep.join(paths)


def main():
    wheel_directory, version_path, model_path = map(Path, sys.argv[1:4])
    distribution = sys.argv[4]
    version = version_path.read_text(encoding="utf-8").strip()
    wheels = list(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"Expected one wheel in {wheel_directory}, found {wheels}.")
    wheel_path = wheels[0]
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        if len(set(names)) != len(names):
            raise AssertionError("Native wheel contains duplicate archive members.")
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        expected_tag = f"{python_tag}-{python_tag}-{_platform_tag()}"
        filename_distribution = distribution.replace("-", "_")
        if wheel_path.name != f"{filename_distribution}-{version}-{expected_tag}.whl":
            raise AssertionError(f"Wheel filename {wheel_path.name!r} does not match version/platform {expected_tag}.")
        distinfo = f"{filename_distribution}-{version}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{distinfo}/METADATA").decode("utf-8"))
        if metadata["Name"] != distribution or metadata["Version"] != version:
            raise AssertionError("Wheel METADATA name/version does not match VERSION_NUMBER.")
        requirements = metadata.get_all("Requires-Dist")
        for expected in ["flatbuffers", "numpy >= 1.21.6", "packaging", "protobuf>=4.25.8"]:
            if expected not in requirements:
                raise AssertionError(f"Wheel dependency metadata is missing {expected!r}: {requirements}")
        if metadata["Requires-Python"] != ">=3.11":
            raise AssertionError("Wheel must retain the release Python compatibility floor.")
        wheel_metadata = Parser().parsestr(wheel.read(f"{distinfo}/WHEEL").decode("utf-8"))
        if wheel_metadata["Root-Is-Purelib"] != "false" or wheel_metadata["Tag"] != expected_tag:
            raise AssertionError("Native extension wheel must declare a platform-specific, non-purelib tag.")
        extension = "pyd" if sys.platform == "win32" else "so"
        required = {
            "onnxruntime/__init__.py",
            "onnxruntime/capi/_pybind_state.py",
            "onnxruntime/capi/onnxruntime_inference_collection.py",
            "onnxruntime/backend/backend.py",
            "onnxruntime/datasets/mul_1.onnx",
            "onnxruntime/py.typed",
            "onnxruntime/quantization/quantize.py",
            "onnxruntime/tools/mobile_helpers/nnapi_supported_ops.md",
            "onnxruntime/tools/ort_format_model/ort_flatbuffers_py/fbs/Model.py",
            "onnxruntime/tools/qdq_helpers/optimize_qdq_model.py",
            "onnxruntime/tools/symbolic_shape_infer.py",
            "onnxruntime/transformers/optimizer.py",
            f"onnxruntime/capi/onnxruntime_pybind11_state.{extension}",
            f"{distinfo}/LICENSE",
            f"{distinfo}/ThirdPartyNotices.txt",
            f"{distinfo}/entry_points.txt",
        }
        bridge = (
            "onnxruntime_providers_shared.dll"
            if sys.platform == "win32"
            else "libonnxruntime_providers_shared.dylib"
            if sys.platform == "darwin"
            else "libonnxruntime_providers_shared.so"
        )
        required.add("onnxruntime/capi/" + bridge)
        if distribution == "onnxruntime-gpu":
            required.add("onnxruntime/capi/libonnxruntime_providers_cuda.so")
        missing = required.difference(names)
        if missing:
            raise AssertionError(f"Native wheel is missing required package files: {sorted(missing)}.")
        if any(name.startswith("onnxruntime/training/") for name in names):
            raise AssertionError("Inference release wheels must not contain training packages.")
        if any(not name.startswith(("onnxruntime/", distinfo + "/")) for name in names):
            raise AssertionError("Native wheel must not bundle unrelated packages such as NumPy or test sources.")
        records = list(csv.reader(io.StringIO(wheel.read(f"{distinfo}/RECORD").decode("utf-8"))))
        if len(records) != len(names) or {row[0] for row in records} != set(names):
            raise AssertionError("Wheel RECORD must cover every archive member exactly once.")
        for name, digest, size in records:
            if name == f"{distinfo}/RECORD":
                if digest or size:
                    raise AssertionError("Wheel RECORD must not include a hash or size for itself.")
                continue
            data = wheel.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            if digest != expected or size != str(len(data)):
                raise AssertionError(f"Wheel RECORD hash/size does not match archive member {name!r}.")
        with tempfile.TemporaryDirectory(prefix="ort-bazel-wheel-") as directory:
            install_directory = Path(directory) / "installed"
            install_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    str(install_directory),
                    str(wheel_path.resolve()),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            sys.stdout.buffer.write(install_result.stdout)
            install_result.check_returncode()
            env = dict(os.environ)
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONPATH"] = _subprocess_pythonpath(install_directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--inference",
                    str(install_directory),
                    str(model_path.resolve()),
                    version,
                    distribution,
                ],
                cwd=install_directory,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            sys.stdout.buffer.write(result.stdout)
            result.check_returncode()
            if b"Init provider bridge failed" in result.stdout.replace(b"\x00", b""):
                raise AssertionError("Wheel import could not initialize its packaged provider bridge.")
    print("Native wheel payload, metadata, RECORD, pip install, and inference passed.")


if __name__ == "__main__":
    if sys.argv[1] == "--inference":
        run_inference(*sys.argv[2:])
    else:
        main()
