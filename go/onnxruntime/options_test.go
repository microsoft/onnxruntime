package onnxruntime

import (
	"testing"
)

func TestCloneSessionOptions(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	opts.SetIntraOpNumThreads(2)

	clone, err := opts.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()

	sess, err := NewSession(testdataPath("add_f32.onnx"), clone)
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
}

func TestSessionOptionsMemory(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.DisableMemPattern(); err != nil {
		t.Fatal(err)
	}
	if err := opts.EnableMemPattern(); err != nil {
		t.Fatal(err)
	}
	if err := opts.DisableCpuMemArena(); err != nil {
		t.Fatal(err)
	}
	if err := opts.EnableCpuMemArena(); err != nil {
		t.Fatal(err)
	}
}

func TestSessionOptionsExecutionMode(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetExecutionMode(ExecutionModeParallel); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetExecutionMode(ExecutionModeSequential); err != nil {
		t.Fatal(err)
	}
}

func TestSessionOptionsProfiling(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.EnableProfiling("/tmp/ort_test_profile"); err != nil {
		t.Fatal(err)
	}

	sess, err := NewSession(testdataPath("add_f32.onnx"), opts)
	if err != nil {
		t.Fatal(err)
	}

	path, err := sess.EndProfiling()
	if err != nil {
		t.Fatal(err)
	}
	if path == "" {
		t.Error("expected non-empty profiling path")
	}
	t.Logf("profiling output: %s", path)
	sess.Close()
}

func TestSessionOptionsFreeDimension(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.AddFreeDimensionOverrideByName("batch", 4); err != nil {
		t.Fatal(err)
	}

	sess, err := NewSession(testdataPath("matmul_dynamic.onnx"), opts)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	inputs := sess.Inputs()
	for _, in := range inputs {
		for _, d := range in.Shape {
			if d == -1 {
				t.Errorf("expected no dynamic dims after override, but input %s has shape %v", in.Name, in.Shape)
			}
		}
	}
}

func TestTensorIsTensor(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if !tensor.IsTensor() {
		t.Error("expected IsTensor() = true")
	}
	if tensor.IsSequence() {
		t.Error("expected IsSequence() = false")
	}
	if tensor.IsMap() {
		t.Error("expected IsMap() = false")
	}
}
