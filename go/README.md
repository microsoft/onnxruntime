# Go API

This directory contains the Go binding for ONNX Runtime.

## Tested Linux CPU setup

The following source and runtime pairing has been exercised together:

- Go binding source at commit `2a43ec07e67df9263c7c56c15da76cc7f74d7b32`.
- ONNX Runtime archive `onnxruntime-linux-x64-1.29.0.tgz`, SHA-256
  `c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba`.
- Linux x86-64 CPU, Go 1.26.0, `CGO_ENABLED=1`, and GCC.

The source commit is currently untagged for this module. Pin that commit when
using the binding; do not infer a released Go module version from this
directory. This page records the tested pairing above. It does not define a
minimum ONNX Runtime version or a compatibility matrix for other runtimes,
platforms, execution providers, or toolchains.

The binding loads the native library at runtime. Call
`SetSharedLibraryPath` with the extracted archive's `lib` directory before
calling `Init`. The directory contains `libonnxruntime.so`, which the
binding resolves on Linux.

## Reproduce the pinned setup

The commands below use a detached checkout of the exact source commit and
verify the runtime archive before extraction:

```sh
export SDK_COMMIT=2a43ec07e67df9263c7c56c15da76cc7f74d7b32
export SDK_ROOT="$PWD/onnxruntime"

git clone --filter=blob:none --no-checkout https://github.com/microsoft/onnxruntime.git "$SDK_ROOT"
git -C "$SDK_ROOT" sparse-checkout init --cone
git -C "$SDK_ROOT" sparse-checkout set go
git -C "$SDK_ROOT" checkout --detach "$SDK_COMMIT"

export ORT_ARCHIVE=onnxruntime-linux-x64-1.29.0.tgz
export ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/$ORT_ARCHIVE
export ORT_SHA256=c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba
curl -fL "$ORT_URL" -o "$ORT_ARCHIVE"
printf '%s  %s\n' "$ORT_SHA256" "$ORT_ARCHIVE" | sha256sum -c -
mkdir -p runtime
tar -xzf "$ORT_ARCHIVE" -C runtime

export ORT_LIB_PATH="$PWD/runtime/onnxruntime-linux-x64-1.29.0/lib"
export ORT_GO_ROOT="$SDK_ROOT/go"
```

For a consumer module outside the SDK checkout, use a local replacement while
the source remains untagged:

```sh
mkdir consumer
cd consumer
go mod init example.com/onnxruntime-consumer
go mod edit -go=1.26.0
go mod edit -require=github.com/microsoft/onnxruntime/go@v0.0.0
go mod edit -replace=github.com/microsoft/onnxruntime/go="$ORT_GO_ROOT"
```

The replacement points at the commit checked out above; `v0.0.0` is only a
local module requirement and is not a published release.

## Minimal CPU example

Copy the small model already used by the Go package tests:

```sh
cp "$ORT_GO_ROOT/testdata/add_f32.onnx" ./add_f32.onnx
```

Save this as `main.go` in the consumer module:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

func main() {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		log.Fatal("ORT_LIB_PATH is required")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.Init(); err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := ort.Shutdown(); err != nil {
			log.Fatal(err)
		}
	}()

	version, err := ort.GetVersion()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("ONNX Runtime:", version)

	session, err := ort.NewSession("add_f32.onnx", nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	a, err := ort.CreateTensor[float32]([]int64{2, 3},
		[]float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	b, err := ort.CreateTensor[float32]([]int64{2, 3},
		[]float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		log.Fatal(err)
	}
	defer b.Close()

	results, err := session.Run(context.Background(), map[string]*ort.Tensor{
		"A": a,
		"B": b,
	}, []string{"C"})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		for _, result := range results {
			result.Close()
		}
	}()

	output, ok := results["C"]
	if !ok {
		log.Fatal("output C not found")
	}
	values, err := ort.TensorData[float32](output)
	if err != nil {
		log.Fatal(err)
	}

	expected := []float32{11, 22, 33, 44, 55, 66}
	if len(values) != len(expected) {
		log.Fatalf("output has %d values, want %d", len(values), len(expected))
	}
	for i, value := range values {
		if math.Abs(float64(value-expected[i])) > 1e-6 {
			log.Fatalf("output[%d] = %v, want %v", i, value, expected[i])
		}
	}
	fmt.Println("output C:", values)
}
```

Build and run it with the compiler and native library selected explicitly:

```sh
CGO_ENABLED=1 CC=gcc ORT_LIB_PATH="$ORT_LIB_PATH" go run .
```

The example uses the CPU execution provider supplied by the tested Linux
archive. Close sessions before calling `Shutdown`.
