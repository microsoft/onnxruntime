package onnxruntime

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func testdataPath(name string) string {
	return filepath.Join("..", "testdata", name)
}

func TestNewSessionAddModel(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	inputs := sess.Inputs()
	if len(inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(inputs))
	}
	for _, in := range inputs {
		if in.DataType != TensorElementDataTypeFloat32 {
			t.Errorf("input %s: expected Float32, got %s", in.Name, in.DataType)
		}
		if len(in.Shape) != 2 || in.Shape[0] != 2 || in.Shape[1] != 3 {
			t.Errorf("input %s: expected shape [2,3], got %v", in.Name, in.Shape)
		}
	}

	outputs := sess.Outputs()
	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}
	if outputs[0].DataType != TensorElementDataTypeFloat32 {
		t.Errorf("output dtype: expected Float32, got %s", outputs[0].DataType)
	}
}

func TestRunAddModel(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{10, 20, 30, 40, 50, 60}
	expected := []float32{11, 22, 33, 44, 55, 66}

	tensorA, err := CreateTensor[float32]([]int64{2, 3}, a)
	if err != nil {
		t.Fatal(err)
	}
	defer tensorA.Close()

	tensorB, err := CreateTensor[float32]([]int64{2, 3}, b)
	if err != nil {
		t.Fatal(err)
	}
	defer tensorB.Close()

	results, err := sess.Run(context.Background(), map[string]*Tensor{
		"A": tensorA,
		"B": tensorB,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, r := range results {
			r.Close()
		}
	}()

	out, ok := results["C"]
	if !ok {
		t.Fatal("output C not found")
	}

	data, err := TensorData[float32](out)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range data {
		if math.Abs(float64(v-expected[i])) > 1e-6 {
			t.Errorf("output[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestNewSessionFromBytes(t *testing.T) {
	data, err := os.ReadFile(testdataPath("add_f32.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	sess, err := NewSessionFromBytes(data, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	if len(sess.Inputs()) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(sess.Inputs()))
	}
}

func TestDynamicShapeModel(t *testing.T) {
	sess, err := NewSession(testdataPath("matmul_dynamic.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	inputs := sess.Inputs()
	aInfo := inputs[0]
	hasDynamic := false
	for _, d := range aInfo.Shape {
		if d == -1 {
			hasDynamic = true
		}
	}
	if !hasDynamic {
		t.Errorf("expected dynamic dim (-1) in input shape, got %v", aInfo.Shape)
	}

	a := []float32{1, 0, 0, 1, 0, 1, 1, 0}
	b := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	tensorA, err := CreateTensor[float32]([]int64{2, 4}, a)
	if err != nil {
		t.Fatal(err)
	}
	defer tensorA.Close()

	tensorB, err := CreateTensor[float32]([]int64{4, 2}, b)
	if err != nil {
		t.Fatal(err)
	}
	defer tensorB.Close()

	results, err := sess.Run(context.Background(), map[string]*Tensor{
		"A": tensorA,
		"B": tensorB,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, r := range results {
			r.Close()
		}
	}()

	out := results["C"]
	if out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("expected output shape [2,2], got %v", out.Shape())
	}
}

func TestZeroLengthDimension(t *testing.T) {
	sess, err := NewSession(testdataPath("kvconcat.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	past, err := CreateTensor[float32]([]int64{1, 2, 0, 4}, []float32{})
	if err != nil {
		t.Fatal(err)
	}
	defer past.Close()

	newVal, err := CreateTensor[float32]([]int64{1, 2, 1, 4}, []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer newVal.Close()

	results, err := sess.Run(context.Background(), map[string]*Tensor{
		"past":    past,
		"new_val": newVal,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, r := range results {
			r.Close()
		}
	}()

	out := results["out"]
	if out == nil {
		t.Fatalf("output 'out' not found in results (keys: %v)", mapKeys(results))
	}
	if out.Shape()[2] != 1 {
		t.Errorf("expected concat output seq dim = 1 (0+1), got %v", out.Shape())
	}
}

func TestSubsetOutputNames(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	defer a.Close()
	defer b.Close()

	results, err := sess.Run(context.Background(), map[string]*Tensor{
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

	if len(results) != 1 {
		t.Errorf("expected 1 output, got %d", len(results))
	}
}

func TestConcurrentRun(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	var wg sync.WaitGroup
	errs := make(chan error, 16*50)

	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
				b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 1, 1, 1, 1, 1})

				results, err := sess.Run(context.Background(), map[string]*Tensor{
					"A": a, "B": b,
				}, nil)
				a.Close()
				b.Close()
				if err != nil {
					errs <- err
					continue
				}
				for _, r := range results {
					r.Close()
				}
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestCancelledContext(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	defer a.Close()
	defer b.Close()

	_, err = sess.Run(ctx, map[string]*Tensor{"A": a, "B": b}, nil)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestSessionOptions(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetIntraOpNumThreads(1); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetGraphOptimizationLevel(GraphOptimizationLevelAll); err != nil {
		t.Fatal(err)
	}

	sess, err := NewSession(testdataPath("add_f32.onnx"), opts)
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
}

func TestRunEmptyOutputNames(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	defer a.Close()
	defer b.Close()

	_, err = sess.Run(context.Background(), map[string]*Tensor{
		"A": a, "B": b,
	}, []string{})
	if err == nil {
		t.Fatal("expected error for empty output names")
	}
}

func TestRunAfterClose(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()

	a, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := CreateTensor[float32]([]int64{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	defer a.Close()
	defer b.Close()

	_, err = sess.Run(context.Background(), map[string]*Tensor{
		"A": a, "B": b,
	}, nil)
	if err == nil {
		t.Fatal("expected error running on closed session")
	}
	if got := err.Error(); got != "ort: run: session is closed" {
		t.Errorf("unexpected error message: %s", got)
	}
}

func TestNewSessionFromBytesCorrupt(t *testing.T) {
	garbage := []byte("this is not a valid onnx model at all")
	_, err := NewSessionFromBytes(garbage, nil)
	if err == nil {
		t.Fatal("expected error for corrupt model bytes")
	}
}

func TestNewSessionFromBytesEmpty(t *testing.T) {
	_, err := NewSessionFromBytes(nil, nil)
	if err == nil {
		t.Fatal("expected error for nil model bytes")
	}

	_, err = NewSessionFromBytes([]byte{}, nil)
	if err == nil {
		t.Fatal("expected error for empty model bytes")
	}
}

func mapKeys(m map[string]*Tensor) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
