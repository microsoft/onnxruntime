# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build and run a Go consumer through the generated local module proxy."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

MAIN = r"""
package main

import (
    "context"
    "fmt"
    "math"
    "os"
    ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

func main() {
    ort.SetSharedLibraryPath(os.Getenv("ORT_CONSUMER_NATIVE"))
    if err := ort.Init(); err != nil { panic(err) }
    defer ort.Shutdown()
    session, err := ort.NewSession(os.Getenv("ORT_CONSUMER_MODEL"), nil)
    if err != nil { panic(err) }
    defer session.Close()
    a, _ := ort.CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := ort.CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
    defer a.Close()
    defer b.Close()
    results, err := session.Run(context.Background(), map[string]*ort.Tensor{"A": a, "B": b}, nil)
    if err != nil { panic(err) }
    defer results["C"].Close()
    values, err := ort.TensorData[float32](results["C"])
    if err != nil { panic(err) }
    expected := []float32{11, 22, 33, 44, 55, 66}
    for i, value := range values {
        if math.Abs(float64(value-expected[i])) > 1e-6 { panic(fmt.Sprintf("output %d: %v", i, value)) }
    }
}
"""


def main():
    go, proxy, package, model = map(Path, sys.argv[1:5])
    if not go.is_file():
        raise RuntimeError(f"Explicit Go executable does not exist: {go}")
    go = go.resolve()
    work = Path.cwd() / "_go_proxy_consumer"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extractall(work / "bundle")
    bundle = (work / "bundle" / "onnxruntime-go").resolve()
    consumer = work / "consumer"
    consumer.mkdir()
    versions = list((proxy / "github.com" / "microsoft" / "onnxruntime" / "go" / "@v").glob("*.mod"))
    if len(versions) != 1:
        raise RuntimeError(f"Expected one Go module proxy version, found {versions}")
    version = versions[0].stem
    (consumer / "go.mod").write_text(
        f"module ort-release-consumer\n\ngo 1.26\n\nrequire github.com/microsoft/onnxruntime/go {version}\n",
        encoding="utf-8",
    )
    (consumer / "main.go").write_text(MAIN, encoding="utf-8")
    home = work / "home"
    cache = work / "go-cache"
    module_cache = work / "go-module-cache"
    home.mkdir()
    cache.mkdir()
    module_cache.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "GOFLAGS": "-p=1",
            "GOMAXPROCS": "1",
            "GOCACHE": str(cache.resolve()),
            "GOMODCACHE": str(module_cache.resolve()),
            "GOPROXY": proxy.resolve().as_uri(),
            "GOSUMDB": "off",
            "HOME": str(home.resolve()),
            "XDG_CACHE_HOME": str(cache.resolve()),
            "ORT_CONSUMER_NATIVE": str((bundle / "native" / "lib").resolve()),
            "ORT_CONSUMER_MODEL": str(model.resolve()),
        }
    )
    subprocess.run(
        [str(go), "mod", "download", f"github.com/microsoft/onnxruntime/go@{version}"],
        cwd=consumer,
        env=env,
        check=True,
    )
    subprocess.run([str(go), "run", "."], cwd=consumer, env=env, check=True)
    print("Local Go module proxy consumer with externally supplied native SDK passed.")


if __name__ == "__main__":
    main()
