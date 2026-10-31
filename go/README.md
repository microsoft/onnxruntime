# Go API

This directory contains the Go binding for ONNX Runtime.

## Tested Linux CPU setup

The following source and runtime pairing has been exercised together:

- Go binding source at commit `402979ac1bef89dcd55add41eaa7277c407599eb`
  (the repository's `v1.31.0` tag).
- ONNX Runtime archive `onnxruntime-linux-x64-1.31.0.tgz`, SHA-256
  `cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57`.
- Linux x86-64 CPU, Go 1.26.0, `CGO_ENABLED=1`, and GCC.

## Prerequisites

- Import path: `github.com/microsoft/onnxruntime/go/onnxruntime`.
- Go 1.26 and cgo enabled (`CGO_ENABLED=1`).
- A C compiler; the commands below use GCC (`CC=gcc`).
- `curl`, `tar`, and `sha256sum` for the runtime download.

The repository tag does not publish a `go/v1.31.0` module version. Use the
commit pin below to install the binding. This page records only the tested
pairing above; it does not define a minimum ONNX Runtime version or a
compatibility matrix for other runtimes, platforms, execution providers,
or toolchains.

The binding loads the native library at runtime. Call
`SetSharedLibraryPath` with the extracted archive's `lib` directory before
calling `Init`. The directory contains `libonnxruntime.so`, which the
binding resolves on Linux.

## Reproduce the pinned setup

Verify the runtime archive before extraction:

```sh
export ORT_ARCHIVE=onnxruntime-linux-x64-1.31.0.tgz
export ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/$ORT_ARCHIVE
export ORT_SHA256=cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57
curl -fL "$ORT_URL" -o "$ORT_ARCHIVE"
printf '%s  %s\n' "$ORT_SHA256" "$ORT_ARCHIVE" | sha256sum -c -
mkdir -p runtime
tar -xzf "$ORT_ARCHIVE" -C runtime

export ORT_LIB_PATH="$PWD/runtime/onnxruntime-linux-x64-1.31.0/lib"
```

Install the binding in a standalone consumer module:

```sh
mkdir consumer
cd consumer
go mod init example.com/onnxruntime-consumer
export SDK_COMMIT=402979ac1bef89dcd55add41eaa7277c407599eb
go get github.com/microsoft/onnxruntime/go/onnxruntime@"$SDK_COMMIT"
```

Go records the commit as a pseudo-version in `go.mod` and verifies the module
through `go.sum`; no SDK checkout or local replacement is needed.

## Minimal CPU example

Copy the small model already used by the Go package tests:

```sh
ORT_GO_ROOT=$(go list -m -f '{{.Dir}}' github.com/microsoft/onnxruntime/go)
cp "$ORT_GO_ROOT/testdata/add_f32.onnx" ./add_f32.onnx
```

Save this as `main.go` in the consumer module:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"os"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		return fmt.Errorf("ORT_LIB_PATH is required")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.Init(); err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, ort.Shutdown())
	}()

	version, err := ort.GetVersion()
	if err != nil {
		return err
	}
	fmt.Println("ONNX Runtime:", version)

	session, err := ort.NewSession("add_f32.onnx", nil)
	if err != nil {
		return err
	}
	defer session.Close()

	a, err := ort.CreateTensor[float32]([]int64{2, 3},
		[]float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		return err
	}
	defer a.Close()

	b, err := ort.CreateTensor[float32]([]int64{2, 3},
		[]float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		return err
	}
	defer b.Close()

	results, err := session.Run(context.Background(), map[string]*ort.Tensor{
		"A": a,
		"B": b,
	}, []string{"C"})
	if err != nil {
		return err
	}
	defer func() {
		for _, result := range results {
			result.Close()
		}
	}()

	output, ok := results["C"]
	if !ok {
		return fmt.Errorf("output C not found")
	}
	values, err := ort.TensorData[float32](output)
	if err != nil {
		return err
	}

	expected := []float32{11, 22, 33, 44, 55, 66}
	if len(values) != len(expected) {
		return fmt.Errorf("output has %d values, want %d", len(values), len(expected))
	}
	for i, value := range values {
		if math.Abs(float64(value-expected[i])) > 1e-6 {
			return fmt.Errorf("output[%d] = %v, want %v", i, value, expected[i])
		}
	}
	fmt.Println("output C:", values)
	return nil
}
```

Build and run it with the compiler and native library selected explicitly:

```sh
CGO_ENABLED=1 CC=gcc ORT_LIB_PATH="$ORT_LIB_PATH" go run .
```

The example uses the CPU execution provider supplied by the tested Linux
archive. The helper returns errors so deferred cleanup runs before `main`
exits. Close output and input tensors and sessions before calling `Shutdown`.
