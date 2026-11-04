# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate native wheel metadata/RECORD and inference from its extracted files."""

import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


def run_inference(directory, model, version):
    import numpy as np  # noqa: PLC0415

    import onnxruntime as ort  # noqa: PLC0415

    if not Path(ort.__file__).resolve().is_relative_to(Path(directory).resolve()):
        raise AssertionError(f"Expected the wheel-installed package in {directory}, imported {ort.__file__}.")
    if ort.__version__ != version:
        raise AssertionError(f"Imported wheel reports version {ort.__version__!r}; expected {version!r}.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b = a * 10
    output = session.run(["C"], {"A": a, "B": b})[0]
    np.testing.assert_array_equal(output, a + b)
    if output.dtype != np.float32:
        raise AssertionError(f"Wheel inference output dtype is {output.dtype}; expected float32.")
    print("Native wheel high-level Python Add inference passed.")


def main():
    wheel_path, version_path, model_path = map(Path, sys.argv[1:])
    version = version_path.read_text(encoding="utf-8").strip()
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        if len(set(names)) != len(names):
            raise AssertionError("Native wheel contains duplicate archive members.")
        expected_tag = "cp312-cp312-" + ("win_amd64" if sys.platform == "win32" else "linux_x86_64")
        if wheel_path.name != f"onnxruntime-{version}-{expected_tag}.whl":
            raise AssertionError(f"Wheel filename {wheel_path.name!r} does not match version/platform {expected_tag}.")
        distinfo = f"onnxruntime-{version}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{distinfo}/METADATA").decode("utf-8"))
        if metadata["Name"] != "onnxruntime" or metadata["Version"] != version:
            raise AssertionError("Wheel METADATA name/version does not match VERSION_NUMBER.")
        if metadata.get_all("Requires-Dist") != ["numpy==2.4.0"]:
            raise AssertionError("Wheel dependency metadata must declare the configured external NumPy dependency.")
        wheel_metadata = Parser().parsestr(wheel.read(f"{distinfo}/WHEEL").decode("utf-8"))
        if wheel_metadata["Root-Is-Purelib"] != "false" or wheel_metadata["Tag"] != expected_tag:
            raise AssertionError("Native extension wheel must declare a platform-specific, non-purelib tag.")
        extension = "pyd" if sys.platform == "win32" else "so"
        required = {
            "onnxruntime/__init__.py",
            "onnxruntime/capi/_pybind_state.py",
            "onnxruntime/capi/onnxruntime_inference_collection.py",
            f"onnxruntime/capi/onnxruntime_pybind11_state.{extension}",
            f"{distinfo}/LICENSE",
            f"{distinfo}/ThirdPartyNotices.txt",
        }
        bridge = "onnxruntime_providers_shared.dll" if sys.platform == "win32" else "libonnxruntime_providers_shared.so"
        required.add("onnxruntime/capi/" + bridge)
        missing = required.difference(names)
        if missing:
            raise AssertionError(f"Native wheel is missing required package files: {sorted(missing)}.")
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
            wheel.extractall(directory)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([directory, *sys.path])
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--inference",
                    directory,
                    str(model_path.resolve()),
                    version,
                ],
                cwd=directory,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            sys.stdout.buffer.write(result.stdout)
            result.check_returncode()
            if b"Init provider bridge failed" in result.stdout.replace(b"\x00", b""):
                raise AssertionError("Wheel import could not initialize its packaged provider bridge.")
    print("Native wheel metadata, RECORD, isolated import, and inference passed.")


if __name__ == "__main__":
    if sys.argv[1] == "--inference":
        run_inference(*sys.argv[2:])
    else:
        main()
