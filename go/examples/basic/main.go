package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime/debug"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

const modulePath = "github.com/microsoft/onnxruntime/go"

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("basic", flag.ContinueOnError)
	flags.SetOutput(stderr)
	expectedVersion := flags.String("expect-runtime", "", "expected initialized ORT runtime version")
	expectedAPI := flags.Int("expect-api", 0, "expected negotiated ORT API version (0 disables the check)")
	expectedModule := flags.String("expect-module-version", "", "expected SDK dependency version; omit when the SDK is the main module")
	flags.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: basic [flags] MODEL_PATH\nRequires ORT_LIB_PATH. In-checkout runs omit -expect-module-version because the SDK is the main module.")
		flags.PrintDefaults()
	}
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if *expectedAPI < 0 {
		return fmt.Errorf("expect-api must be nonnegative")
	}
	if flags.NArg() != 1 || flags.Arg(0) == "" {
		return fmt.Errorf("require one model path")
	}
	path := os.Getenv("ORT_LIB_PATH")
	if path == "" {
		return fmt.Errorf("ORT_LIB_PATH must be nonempty")
	}
	if *expectedModule != "" {
		info, ok := debug.ReadBuildInfo()
		if !ok {
			return fmt.Errorf("module build info unavailable")
		}
		if err := checkBuildInfo(info, *expectedModule); err != nil {
			return err
		}
	}
	ort.SetSharedLibraryPath(path)
	var version string
	var api int
	var values []float32
	err := func() (err error) {
		if err = ort.Init(); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, ort.Shutdown()) }()
		closeResource := func(resource io.Closer, name string) {
			if closeErr := resource.Close(); closeErr != nil {
				err = errors.Join(err, fmt.Errorf("close %s: %w", name, closeErr))
			}
		}
		version, err = ort.GetVersion()
		if err != nil {
			return err
		}
		api = ort.APIVersion()
		if err = checkRuntime(version, api, *expectedVersion, *expectedAPI); err != nil {
			return err
		}
		session, err := ort.NewSession(flags.Arg(0), nil)
		if err != nil {
			return err
		}
		defer closeResource(session, "session")
		a, err := ort.CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			return err
		}
		defer closeResource(a, "input A")
		b, err := ort.CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
		if err != nil {
			return err
		}
		defer closeResource(b, "input B")
		outputs, err := session.Run(context.Background(), map[string]*ort.Tensor{"A": a, "B": b}, nil)
		if err != nil {
			return err
		}
		defer func() {
			for name, output := range outputs {
				closeResource(output, "output "+name)
			}
		}()
		output, ok := outputs["C"]
		if len(outputs) != 1 || !ok {
			return fmt.Errorf("require exactly output C, got %d outputs", len(outputs))
		}
		data, err := ort.TensorData[float32](output)
		if err != nil {
			return err
		}
		if err = checkOutput(output.Shape(), data); err != nil {
			return err
		}
		// Copy before closing the output tensor, which owns the returned data.
		values = append([]float32(nil), data...)
		return nil
	}()
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "ORT version: %s, API version: %d\nC: %v\nShutdown successful\n", version, api, values)
	return err
}

func checkOutput(shape []int64, values []float32) error {
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		return fmt.Errorf("output C shape: expected [2 3], got %v", shape)
	}
	expected := []float32{11, 22, 33, 44, 55, 66}
	if len(values) != len(expected) {
		return fmt.Errorf("output C length: expected %d, got %d", len(expected), len(values))
	}
	for i, value := range values {
		v := float64(value)
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-float64(expected[i])) > 1e-6 {
			return fmt.Errorf("output C[%d]: expected %g, got %g", i, expected[i], value)
		}
	}
	return nil
}

func checkRuntime(version string, api int, expectedVersion string, expectedAPI int) error {
	if expectedVersion != "" && version != expectedVersion {
		return fmt.Errorf("runtime mismatch: expected %s, got %s", expectedVersion, version)
	}
	if expectedAPI != 0 && api != expectedAPI {
		return fmt.Errorf("API mismatch: expected %d, got %d", expectedAPI, api)
	}
	return nil
}

func checkBuildInfo(info *debug.BuildInfo, expectedVersion string) error {
	if info == nil {
		return fmt.Errorf("module build info unavailable")
	}
	for _, dependency := range info.Deps {
		if dependency.Path != modulePath {
			continue
		}
		if dependency.Replace != nil {
			return fmt.Errorf("module %s has a replacement", modulePath)
		}
		if dependency.Version != expectedVersion {
			return fmt.Errorf("module version mismatch: expected %s, got %s", expectedVersion, dependency.Version)
		}
		return nil
	}
	return fmt.Errorf("module dependency %s missing from build info", modulePath)
}
