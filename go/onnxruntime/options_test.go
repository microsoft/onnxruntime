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

func TestSessionOptionsGetExecutionMode(t *testing.T) {
	if APIVersion() < 27 {
		t.Skipf("GetExecutionMode requires ORT >= 1.27 (have API version %d)", APIVersion())
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	opts.SetExecutionMode(ExecutionModeParallel)
	mode, err := opts.GetExecutionMode()
	if err != nil {
		t.Fatal(err)
	}
	if mode != ExecutionModeParallel {
		t.Errorf("expected Parallel, got %d", mode)
	}

	opts.SetExecutionMode(ExecutionModeSequential)
	mode, err = opts.GetExecutionMode()
	if err != nil {
		t.Fatal(err)
	}
	if mode != ExecutionModeSequential {
		t.Errorf("expected Sequential, got %d", mode)
	}
}

func TestSessionOptionsIsMemPatternEnabled(t *testing.T) {
	if APIVersion() < 27 {
		t.Skipf("IsMemPatternEnabled requires ORT >= 1.27 (have API version %d)", APIVersion())
	}
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	enabled, err := opts.IsMemPatternEnabled()
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		t.Error("expected mem pattern enabled by default")
	}

	opts.DisableMemPattern()
	enabled, err = opts.IsMemPatternEnabled()
	if err != nil {
		t.Fatal(err)
	}
	if enabled {
		t.Error("expected mem pattern disabled after DisableMemPattern")
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

func TestAppendExecutionProviderUnknown(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	err = opts.AppendExecutionProvider("NoSuchProvider_XYZ_999", nil)
	if err == nil {
		t.Fatal("expected error for unknown execution provider")
	}
}

func TestSessionConfigEntry(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	opts.AddConfigEntry("test.key", "test.value")

	has, err := opts.HasSessionConfigEntry("test.key")
	if err != nil {
		t.Fatal(err)
	}
	if !has {
		t.Error("expected config entry to exist")
	}

	val, err := opts.GetSessionConfigEntry("test.key")
	if err != nil {
		t.Fatal(err)
	}
	if val != "test.value" {
		t.Errorf("expected 'test.value', got %q", val)
	}

	has, err = opts.HasSessionConfigEntry("nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	if has {
		t.Error("expected config entry to not exist")
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

// newFloatTensor returns a live 2-element float32 tensor, closed by the test.
func newFloatTensor(t *testing.T) *Tensor {
	t.Helper()
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { tensor.Close() })
	return tensor
}

func TestNewSequenceNilElement(t *testing.T) {
	if _, err := NewSequence([]*Tensor{newFloatTensor(t), nil}); err == nil {
		t.Fatal("expected error for nil sequence element")
	}
}

func TestNewSequenceClosedElement(t *testing.T) {
	closed := newFloatTensor(t)
	closed.Close()

	if _, err := NewSequence([]*Tensor{newFloatTensor(t), closed}); err == nil {
		t.Fatal("expected error for closed sequence element")
	}
}

func TestNewMapNilTensors(t *testing.T) {
	if _, err := NewMap(nil, newFloatTensor(t)); err == nil {
		t.Error("expected error for nil keys")
	}
	if _, err := NewMap(newFloatTensor(t), nil); err == nil {
		t.Error("expected error for nil values")
	}
}

func TestNewMapClosedKeys(t *testing.T) {
	keys, err := CreateTensor[int64]([]int64{2}, []int64{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	keys.Close()

	if _, err := NewMap(keys, newFloatTensor(t)); err == nil {
		t.Fatal("expected error for closed map keys")
	}
}

func TestNewMapClosedValues(t *testing.T) {
	keys, err := CreateTensor[int64]([]int64{2}, []int64{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer keys.Close()

	values := newFloatTensor(t)
	values.Close()

	if _, err := NewMap(keys, values); err == nil {
		t.Fatal("expected error for closed map values")
	}
}

func TestAddInitializerNilTensor(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.AddInitializer("w", nil); err == nil {
		t.Fatal("expected error for nil initializer tensor")
	}
}

func TestAddInitializerClosedTensor(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	tensor := newFloatTensor(t)
	tensor.Close()

	if err := opts.AddInitializer("w", tensor); err == nil {
		t.Fatal("expected error for closed initializer tensor")
	}
}

func TestAddInitializerClosedOptions(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	opts.Close()

	if err := opts.AddInitializer("w", newFloatTensor(t)); err == nil {
		t.Fatal("expected error for closed session options")
	}
}

func TestAddInitializerLiveTensor(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.AddInitializer("w", newFloatTensor(t)); err != nil {
		t.Fatalf("expected live tensor to be accepted: %v", err)
	}
}
