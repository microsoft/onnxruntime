package onnxruntime

import (
	"testing"
)

func TestIOBindingBasic(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	binding, err := NewIOBinding(sess)
	if err != nil {
		t.Fatal(err)
	}
	defer binding.Close()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	defer a.Close()
	defer b.Close()

	if err := binding.BindInput("A", a); err != nil {
		t.Fatal(err)
	}
	if err := binding.BindInput("B", b); err != nil {
		t.Fatal(err)
	}

	memInfo, err := NewCPUMemoryInfo()
	if err != nil {
		t.Fatal(err)
	}
	defer memInfo.Close()

	if err := binding.BindOutputToDevice("C", memInfo); err != nil {
		t.Fatal(err)
	}

	if err := binding.Run(nil); err != nil {
		t.Fatal(err)
	}

	outputs, err := binding.OutputValues()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, o := range outputs {
			o.Close()
		}
	}()

	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}

	data, err := TensorData[float32](outputs[0])
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != 11 {
		t.Errorf("expected 11, got %f", data[0])
	}
}

func TestMemoryInfo(t *testing.T) {
	mem, err := NewCPUMemoryInfo()
	if err != nil {
		t.Fatal(err)
	}
	mem.Close()
	mem.Close()
}

func TestNewMemoryInfo(t *testing.T) {
	mem, err := NewMemoryInfo("Cpu", AllocatorTypeDevice, 0, MemTypeDefault)
	if err != nil {
		t.Fatal(err)
	}
	defer mem.Close()
}

func TestIOBindingClear(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	binding, err := NewIOBinding(sess)
	if err != nil {
		t.Fatal(err)
	}
	defer binding.Close()

	binding.ClearInputs()
	binding.ClearOutputs()
}
