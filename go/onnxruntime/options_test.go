package onnxruntime

import (
	"fmt"
	"math"
	"strconv"
	"sync"
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

func TestSessionOptionsMethodsAfterClose(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	if err := opts.Close(); err != nil {
		t.Fatal(err)
	}

	type testCase struct {
		name string
		call func() error
	}
	tests := []testCase{
		{"SetIntraOpNumThreads", func() error { return opts.SetIntraOpNumThreads(1) }},
		{"SetInterOpNumThreads", func() error { return opts.SetInterOpNumThreads(1) }},
		{"SetGraphOptimizationLevel", func() error {
			return opts.SetGraphOptimizationLevel(GraphOptimizationLevelBasic)
		}},
		{"AddConfigEntry", func() error { return opts.AddConfigEntry("key", "value") }},
		{"AppendExecutionProvider", func() error { return opts.AppendExecutionProvider("CPUExecutionProvider", nil) }},
		{"Clone", func() error {
			clone, err := opts.Clone()
			if clone != nil {
				_ = clone.Close()
			}
			return err
		}},
		{"DisableMemPattern", opts.DisableMemPattern},
		{"EnableMemPattern", opts.EnableMemPattern},
		{"EnableCpuMemArena", opts.EnableCpuMemArena},
		{"DisableCpuMemArena", opts.DisableCpuMemArena},
		{"EnableProfiling", func() error { return opts.EnableProfiling("profile") }},
		{"DisableProfiling", opts.DisableProfiling},
		{"AddFreeDimensionOverride", func() error { return opts.AddFreeDimensionOverride("batch", 1) }},
		{"AddFreeDimensionOverrideByName", func() error { return opts.AddFreeDimensionOverrideByName("batch", 1) }},
		{"SetExecutionMode", func() error { return opts.SetExecutionMode(ExecutionModeSequential) }},
		{"SetOptimizedModelFilePath", func() error { return opts.SetOptimizedModelFilePath("model.onnx") }},
		{"RegisterCustomOpsLibrary", func() error { return opts.RegisterCustomOpsLibrary("custom-ops.so") }},
		{"HasSessionConfigEntry", func() error {
			_, err := opts.HasSessionConfigEntry("key")
			return err
		}},
		{"GetSessionConfigEntry", func() error {
			_, err := opts.GetSessionConfigEntry("key")
			return err
		}},
	}
	if APIVersion() >= 27 {
		tests = append(tests,
			testCase{"GetExecutionMode", func() error {
				_, err := opts.GetExecutionMode()
				return err
			}},
			testCase{"IsMemPatternEnabled", func() error {
				_, err := opts.IsMemPatternEnabled()
				return err
			}},
		)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err == nil {
				t.Fatal("expected error after Close")
			}
		})
	}
}

func TestNilSessionOptions(t *testing.T) {
	var opts *SessionOptions
	if err := opts.Close(); err != nil {
		t.Fatal(err)
	}
	if err := opts.EnableMemPattern(); err == nil {
		t.Fatal("expected error using nil session options")
	}
}

func TestSessionOptionsConcurrentClose(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for range 100 {
			_, _ = opts.HasSessionConfigEntry("key")
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		_ = opts.Close()
	}()
	close(start)
	wg.Wait()

	if err := opts.EnableMemPattern(); err == nil {
		t.Fatal("expected error after concurrent Close")
	}
}

func TestSessionOptionsRejectThreadCountOutsideCInt(t *testing.T) {
	if strconv.IntSize == 32 {
		t.Skip("Go int has the same width as C int")
	}

	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetIntraOpNumThreads(int(int64(math.MaxInt32) + 1)); err == nil {
		t.Fatal("expected error for overflowing intra-op thread count")
	}
	if err := opts.SetInterOpNumThreads(int(int64(math.MinInt32) - 1)); err == nil {
		t.Fatal("expected error for overflowing inter-op thread count")
	}
}

func TestSessionOptionsThreadCountBoundaries(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	setters := []struct {
		name string
		set  func(int) error
	}{
		{"intra-op", opts.SetIntraOpNumThreads},
		{"inter-op", opts.SetInterOpNumThreads},
	}
	values := []struct {
		n       int
		wantErr bool
	}{
		{-1, true},
		{0, false},
		{1, false},
	}

	for _, setter := range setters {
		for _, value := range values {
			t.Run(fmt.Sprintf("%s/%d", setter.name, value.n), func(t *testing.T) {
				err := setter.set(value.n)
				if (err != nil) != value.wantErr {
					t.Fatalf("set thread count %d: error = %v, want error = %v", value.n, err, value.wantErr)
				}
			})
		}
	}
}

func TestSessionOptionsRejectNULStrings(t *testing.T) {
	opts, err := NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()
	tensor := newFloatTensor(t)

	tests := []struct {
		name string
		call func() error
	}{
		{"config key", func() error { return opts.AddConfigEntry("key\x00ignored", "value") }},
		{"config value", func() error { return opts.AddConfigEntry("key", "value\x00ignored") }},
		{"provider name", func() error { return opts.AppendExecutionProvider("CPU\x00ignored", nil) }},
		{"provider option key", func() error {
			return opts.AppendExecutionProvider("CPUExecutionProvider", map[string]string{"key\x00ignored": "value"})
		}},
		{"provider option value", func() error {
			return opts.AppendExecutionProvider("CPUExecutionProvider", map[string]string{"key": "value\x00ignored"})
		}},
		{"dimension denotation", func() error { return opts.AddFreeDimensionOverride("batch\x00ignored", 1) }},
		{"dimension name", func() error { return opts.AddFreeDimensionOverrideByName("batch\x00ignored", 1) }},
		{"initializer name", func() error { return opts.AddInitializer("weight\x00ignored", tensor) }},
		{"config lookup", func() error {
			_, err := opts.HasSessionConfigEntry("key\x00ignored")
			return err
		}},
		{"config get", func() error {
			_, err := opts.GetSessionConfigEntry("key\x00ignored")
			return err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err == nil {
				t.Fatal("expected NUL error")
			}
		})
	}
}
