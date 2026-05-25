//go:build linux && amd64 && cgo

package onnxruntime

import (
	"os"
	"testing"
)

func TestSqueezeNet(t *testing.T) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		libPath = "libonnxruntime.so"
	}
	modelPath := os.Getenv("ORT_TEST_MODEL")
	if modelPath == "" {
		modelPath = "testdata/squeezenet.onnx"
	}

	Initialize(libPath)
	env, _ := NewEnv("test", LogWarning)
	defer ReleaseEnv(env)

	opts, _ := NewSessionOptionsBuilder()
	defer opts.Close()

	session, err := NewSessionFromFile(env, modelPath, opts.Ptr())
	if err != nil {
		t.Fatalf("NewSessionFromFile: %v", err)
	}
	defer session.Close()

	n, _ := session.InputCount()
	t.Logf("Inputs: %d", n)
	for i := 0; i < n; i++ {
		name, _ := session.InputName(i)
		t.Logf("  input[%d] = %s", i, name)
	}

	n, _ = session.OutputCount()
	t.Logf("Outputs: %d", n)
	for i := 0; i < n; i++ {
		name, _ := session.OutputName(i)
		t.Logf("  output[%d] = %s", i, name)
	}

	memInfo, _ := DefaultCPUAllocatorMemoryInfo()
	defer ReleaseMemoryInfo(memInfo)

	inputTensor, _ := NewTensor(memInfo, make([]float32, 1*3*224*224), []int64{1, 3, 224, 224})
	defer ReleaseValue(inputTensor)

	outputs, err := session.Run(
		[]string{"data_0"},
		[]OrtValue{inputTensor},
		[]string{"softmaxout_1"},
	)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	var outFloats []float32
	GetTensorData(outputs[0], &outFloats)
	ReleaseValue(outputs[0])

	t.Logf("Inference: %d classes, top score=%f", len(outFloats), outFloats[0])
}
