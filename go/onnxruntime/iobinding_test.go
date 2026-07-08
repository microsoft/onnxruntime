package onnxruntime

import (
	"math"
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

func TestIOBindingOnClosedSession(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()

	_, err = NewIOBinding(sess)
	if err == nil {
		t.Fatal("expected error creating IOBinding on closed session")
	}
}

func TestIOBindingBindOutput(t *testing.T) {
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

	outTensor, err := CreateTensor[float32]([]int64{2, 3}, []float32{0, 0, 0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	defer outTensor.Close()

	if err := binding.BindOutput("C", outTensor); err != nil {
		t.Fatal(err)
	}

	if err := binding.Run(nil); err != nil {
		t.Fatal(err)
	}

	data, err := TensorData[float32](outTensor)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{11, 22, 33, 44, 55, 66}
	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 1e-6 {
			t.Errorf("output[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestIOBindingOutputNames(t *testing.T) {
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

	names, err := binding.OutputNames()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 1 {
		t.Fatalf("expected 1 output name, got %d", len(names))
	}
	if names[0] != "C" {
		t.Errorf("expected output name 'C', got %q", names[0])
	}
}
