// Example: Basic ONNX Runtime inference with the Go binding.
//
// Build requirements:
//   1. ONNX Runtime shared library installed (libonnxruntime.so / onnxruntime.dll)
//   2. An ONNX model file (e.g., model.onnx)
//
// Install:
//   go get github.com/microsoft/onnxruntime/go@latest
//
// Run:
//   CGO_ENABLED=1 go run main.go model.onnx
//
// Import:
//   import onnxruntime "github.com/microsoft/onnxruntime/go"
//   The import path is "github.com/microsoft/onnxruntime/go" (matching the go/ subdirectory).
//   The package name is "onnxruntime" — use onnxruntime.Initialize(), onnxruntime.NewEnv(), etc.

package main

import (
	"fmt"
	"os"

	onnxruntime "github.com/microsoft/onnxruntime/go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <model.onnx>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Set ORT_LIB_PATH to the ONNX Runtime shared library path if not in system path.\n")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	libPath := os.Getenv("ORT_LIB_PATH")

	// Initialize the ONNX Runtime library
	if err := onnxruntime.Initialize(libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize ONNX Runtime: %v\n", err)
		os.Exit(1)
	}

	// Create the environment (singleton)
	env, err := onnxruntime.NewEnv("basic-example", onnxruntime.LogWarning)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create environment: %v\n", err)
		os.Exit(1)
	}
	defer onnxruntime.ReleaseEnv(env)

	// Configure session options
	opts, err := onnxruntime.NewSessionOptionsBuilder()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create session options: %v\n", err)
		os.Exit(1)
	}
	defer opts.Close()

	opts.WithGraphOptimizationLevel(onnxruntime.GraphOptAll).
		WithIntraOpNumThreads(4)

	// Load the model
	session, err := onnxruntime.NewSessionFromFile(env, modelPath, opts.Ptr())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer session.Close()

	// Inspect model inputs
	inputCount, err := session.InputCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get input count: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model has %d input(s):\n", inputCount)
	for i := 0; i < inputCount; i++ {
		name, err := session.InputName(i)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to get input name %d: %v\n", i, err)
			os.Exit(1)
		}
		fmt.Printf("  Input %d: %s\n", i, name)
	}

	// Inspect model outputs
	outputCount, err := session.OutputCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get output count: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model has %d output(s):\n", outputCount)
	for i := 0; i < outputCount; i++ {
		name, err := session.OutputName(i)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to get output name %d: %v\n", i, err)
			os.Exit(1)
		}
		fmt.Printf("  Output %d: %s\n", i, name)
	}

	fmt.Println("\nONNX Runtime Go binding works! Ready for inference.")
}
