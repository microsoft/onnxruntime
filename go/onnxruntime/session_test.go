package onnxruntime

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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

func TestRunNilTensor(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	tensorA, err := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer tensorA.Close()

	_, err = sess.Run(context.Background(), map[string]*Tensor{
		"A": tensorA,
		"B": nil,
	}, nil)
	if err == nil {
		t.Fatal("expected error for nil input tensor")
	}
}

func TestRunClosedTensor(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	tensorA, err := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer tensorA.Close()

	tensorB, err := CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		t.Fatal(err)
	}
	_ = tensorB.Close()

	_, err = sess.Run(context.Background(), map[string]*Tensor{
		"A": tensorA,
		"B": tensorB,
	}, nil)
	if err == nil {
		t.Fatal("expected error for closed input tensor")
	}
}

// addModelInputs returns fresh inputs for add_f32.onnx and a cleanup func.
func addModelInputs(t *testing.T) (map[string]*Tensor, func()) {
	t.Helper()

	a, err := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	b, err := CreateTensor[float32]([]int64{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		_ = a.Close()
		t.Fatal(err)
	}
	return map[string]*Tensor{"A": a, "B": b}, func() {
		_ = a.Close()
		_ = b.Close()
	}
}

func closeTensorMap(tensors map[string]*Tensor) {
	for _, tensor := range tensors {
		_ = tensor.Close()
	}
}

func TestRunWithOptionsCancelledBeforeCall(t *testing.T) {
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

	inputs, cleanup := addModelInputs(t)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = sess.RunWithOptions(ctx, opts, inputs, []string{"C"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	results, err := sess.RunWithOptions(context.Background(), opts, inputs, []string{"C"})
	if err != nil {
		t.Fatalf("reusing run options after a cancelled run: %v", err)
	}
	closeTensorMap(results)
}

func TestRunWithOptionsExpiredDeadline(t *testing.T) {
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

	inputs, cleanup := addModelInputs(t)
	defer cleanup()

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	_, err = sess.RunWithOptions(ctx, opts, inputs, []string{"C"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
}

// countingContext reports whether the run subscribed to cancellation.
type countingContext struct {
	context.Context
	doneCalls atomic.Int64
}

func (c *countingContext) Done() <-chan struct{} {
	c.doneCalls.Add(1)
	return c.Context.Done()
}

// TestRunWithOptionsWatchesContext pins that caller-supplied run options do not
// disable cancellation: the run must watch ctx.Done() so it can terminate an
// in-flight run, exactly as it does for the options it creates itself.
func TestRunWithOptionsWatchesContext(t *testing.T) {
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

	inputs, cleanup := addModelInputs(t)
	defer cleanup()

	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &countingContext{Context: base}

	results, err := sess.RunWithOptions(ctx, opts, inputs, []string{"C"})
	if err != nil {
		t.Fatal(err)
	}
	closeTensorMap(results)

	if ctx.doneCalls.Load() == 0 {
		t.Fatal("run never consulted ctx.Done(): cancellation is ignored when the caller supplies run options")
	}
}

// TestRunWithOptionsCancelledDuringRunKeepsOptionsUsable pins the terminate
// restore: the watcher sets the terminate flag on the caller's run options, and
// that flag is sticky, so leaving it set would abort the caller's next run.
// runInner is called directly because RunWithOptions rejects an already
// cancelled context before the watcher would ever start.
func TestRunWithOptionsCancelledDuringRunKeepsOptionsUsable(t *testing.T) {
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

	const iterations = 20
	terminated := 0
	for i := range iterations {
		inputs, cleanup := addModelInputs(t)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		results, err := sess.runInner(ctx, inputs, []string{"C"}, opts.handle)
		switch {
		case err == nil:
			// The run reached ORT's terminate check before the watcher fired; these
			// single-node models cannot be aborted once the kernel is running.
			closeTensorMap(results)
		case errors.Is(err, context.Canceled):
			terminated++
		default:
			cleanup()
			t.Fatalf("iteration %d: expected context.Canceled, got %v", i, err)
		}

		results, err = sess.RunWithOptions(context.Background(), opts, inputs, []string{"C"})
		if err != nil {
			cleanup()
			t.Fatalf("iteration %d: run options left terminated by the cancelled run: %v", i, err)
		}
		closeTensorMap(results)
		cleanup()
	}
	t.Logf("%d/%d cancellations reached ORT before the run finished", terminated, iterations)
}

// TestRunWithOptionsPreservesCallerTerminate pins that only a terminate flag the
// watcher set is cleared: one the caller set is theirs to keep.
func TestRunWithOptionsPreservesCallerTerminate(t *testing.T) {
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

	inputs, cleanup := addModelInputs(t)
	defer cleanup()

	if err := opts.SetTerminate(); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if _, err := sess.RunWithOptions(ctx, opts, inputs, []string{"C"}); err == nil {
		t.Fatal("expected error from terminated run options")
	}
	if _, err := sess.RunWithOptions(context.Background(), opts, inputs, []string{"C"}); err == nil {
		t.Fatal("terminate flag set by the caller was cleared by the run")
	}
}

// TestRunWithOptionsReuseAfterCancelledRun cancels concurrently with the run, so
// the watcher fires before, during, and after the call, and requires the
// caller's run options to stay usable in every case.
func TestRunWithOptionsReuseAfterCancelledRun(t *testing.T) {
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

	inputs, cleanup := addModelInputs(t)
	defer cleanup()

	for i := range 50 {
		ctx, cancel := context.WithCancel(context.Background())
		go cancel()

		results, err := sess.RunWithOptions(ctx, opts, inputs, []string{"C"})
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("iteration %d: expected context.Canceled or success, got %v", i, err)
		}
		closeTensorMap(results)

		results, err = sess.RunWithOptions(context.Background(), opts, inputs, []string{"C"})
		if err != nil {
			t.Fatalf("iteration %d: run options left terminated by the cancelled run: %v", i, err)
		}
		closeTensorMap(results)
	}
}

// TestRunCancelRace shakes the run options Run creates for itself: cancelling
// around the moment the run finishes leaves the watcher parked on options that
// must not be released before it is joined.
func TestRunCancelRace(t *testing.T) {
	sess, err := NewSession(testdataPath("add_f32.onnx"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	const workers = 8
	const iterations = 40

	var wg sync.WaitGroup
	errs := make(chan error, workers*iterations)
	for range workers {
		inputs, cleanup := addModelInputs(t)
		defer cleanup()

		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range iterations {
				// Timeouts straddle the run duration, so cancellation lands before,
				// during, and just after the call into ORT.
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(i%20*10)*time.Microsecond)
				results, err := sess.Run(ctx, inputs, []string{"C"})
				cancel()

				if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
					errs <- err
				}
				closeTensorMap(results)
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestNewSessionUnicodePath(t *testing.T) {
	data, err := os.ReadFile(testdataPath("add_f32.onnx"))
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "模型-mödel-🎉.onnx")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	sess, err := NewSession(path, nil)
	if err != nil {
		t.Fatalf("load model from non-ASCII path: %v", err)
	}
	defer sess.Close()

	if len(sess.Inputs()) != 2 {
		t.Errorf("expected 2 inputs, got %d", len(sess.Inputs()))
	}
}

func TestNewSessionRejectsNULInPath(t *testing.T) {
	// Without the NUL check, the path is truncated at the NUL and the model at
	// the truncated path loads instead — silently ignoring what the caller asked for.
	sess, err := NewSession(testdataPath("add_f32.onnx")+"\x00ignored", nil)
	if err == nil {
		sess.Close()
		t.Fatal("expected error for path containing NUL byte, but a model loaded")
	}
	if !strings.Contains(err.Error(), "NUL") {
		t.Errorf("expected a NUL-byte error, got: %v", err)
	}
}
