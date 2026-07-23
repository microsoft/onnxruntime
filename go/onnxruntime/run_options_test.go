package onnxruntime

import (
	"context"
	"testing"
)

func TestRunOptions(t *testing.T) {
	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetLogVerbosityLevel(1); err != nil {
		t.Errorf("SetLogVerbosityLevel: %v", err)
	}
	if err := opts.SetLogSeverityLevel(2); err != nil {
		t.Errorf("SetLogSeverityLevel: %v", err)
	}
	if err := opts.SetTag("test-tag"); err != nil {
		t.Errorf("SetTag: %v", err)
	}
	if err := opts.AddConfigEntry("test.key", "test.value"); err != nil {
		t.Errorf("AddConfigEntry: %v", err)
	}
}

func TestRunWithOptions(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	defer a.Close()
	defer b.Close()

	results, err := sess.RunWithOptions(context.Background(), opts, map[string]*Tensor{
		"A": a, "B": b,
	}, []string{"C"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, r := range results {
			r.Close()
		}
	}()

	out := results["C"]
	data, _ := TensorData[float32](out)
	if data[0] != 11 {
		t.Errorf("expected 11, got %f", data[0])
	}
}

func TestRunOptionsTerminate(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetTerminate(); err != nil {
		t.Fatal(err)
	}

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	defer a.Close()
	defer b.Close()

	_, err = sess.RunWithOptions(context.Background(), opts, map[string]*Tensor{
		"A": a, "B": b,
	}, []string{"C"})
	if err == nil {
		t.Fatal("expected error from terminated run options")
	}
}
