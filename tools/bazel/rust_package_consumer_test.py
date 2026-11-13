# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build and run an ordinary offline Cargo consumer outside the source bundle."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

MAIN = r"""
use onnxruntime::{environment::Environment, ndarray::Array};

fn main() {
    let runtime = std::env::var("ORT_CONSUMER_RUNTIME").unwrap();
    let model = std::fs::read(std::env::var("ORT_CONSUMER_MODEL").unwrap()).unwrap();
    let environment = Environment::builder().with_library_path(runtime).build().unwrap();
    let session = environment.new_session_builder().unwrap().with_model_from_memory(&model).unwrap();
    let left = Array::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Array::from_shape_vec((2, 3), vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let outputs = session.run(vec![left.into(), right.into()]).unwrap();
    let values = outputs[0].float_array().unwrap().view().iter().copied().collect::<Vec<_>>();
    assert_eq!(values, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}
"""


def main():
    cargo, package, model = map(Path, sys.argv[1:4])
    if not cargo.is_file():
        raise RuntimeError(f"Explicit Cargo executable does not exist: {cargo}")
    cargo = cargo.resolve()
    cargo_home = os.environ.get("ORT_RELEASE_CARGO_HOME")
    if not cargo_home:
        raise RuntimeError(
            "ORT_RELEASE_CARGO_HOME must name a cache populated with "
            "`cargo fetch --locked --manifest-path rust/Cargo.toml`."
        )
    work = Path.cwd() / "_rust_release_consumer"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extractall(work / "bundle")
    bundle = (work / "bundle" / "onnxruntime-rust").resolve()
    headers = list((bundle / "native" / "include").rglob("onnxruntime_c_api.h"))
    if len(headers) != 1:
        raise RuntimeError(f"Expected exactly one packaged onnxruntime_c_api.h, found {headers}")
    api_header = headers[0].resolve()
    consumer = work / "consumer"
    (consumer / "src").mkdir(parents=True)
    crate = (bundle / "source" / "onnxruntime").as_posix()
    (consumer / "Cargo.toml").write_text(
        f'[package]\nname = "ort-release-consumer"\nversion = "0.0.0"\nedition = "2021"\n'
        f'[dependencies]\nonnxruntime = {{ path = "{crate}" }}\n',
        encoding="utf-8",
    )
    (consumer / "src" / "main.rs").write_text(MAIN, encoding="utf-8")
    native = (bundle / "native").resolve()
    lib = native / "lib"
    runtime_name = (
        "onnxruntime.dll"
        if os.name == "nt"
        else ("libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so")
    )
    env = os.environ.copy()
    env.update(
        {
            "CARGO_BUILD_JOBS": "1",
            "CARGO_HOME": str(Path(cargo_home).resolve()),
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": str((work / "target").resolve()),
            "ORT_RUST_STRATEGY": "system",
            "ORT_RUST_LIB_LOCATION": str(native),
            "ORT_RUST_HEADER_LOCATION": str(api_header),
            "ORT_CONSUMER_RUNTIME": str(lib / runtime_name),
            "ORT_CONSUMER_MODEL": str(model.resolve()),
        }
    )
    loader_variable = (
        "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    env[loader_variable] = str(lib) + os.pathsep + env.get(loader_variable, "")
    subprocess.run([str(cargo), "run", "--offline", "--quiet"], cwd=consumer, env=env, check=True)
    print("Outside-source offline Cargo Add inference passed.")


if __name__ == "__main__":
    main()
