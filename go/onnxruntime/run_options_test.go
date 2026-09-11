package onnxruntime

import (
	"context"
	"math"
	"strconv"
	"sync"
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

func TestRunOptionsMethodsAfterClose(t *testing.T) {
	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	if err := opts.Close(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		call func() error
	}{
		{"SetLogVerbosityLevel", func() error { return opts.SetLogVerbosityLevel(1) }},
		{"SetLogSeverityLevel", func() error { return opts.SetLogSeverityLevel(1) }},
		{"SetTag", func() error { return opts.SetTag("tag") }},
		{"SetTerminate", opts.SetTerminate},
		{"UnsetTerminate", opts.UnsetTerminate},
		{"AddConfigEntry", func() error { return opts.AddConfigEntry("key", "value") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err == nil {
				t.Fatal("expected error after Close")
			}
		})
	}
}

func TestRunOptionsConcurrentClose(t *testing.T) {
	opts, err := NewRunOptions()
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
			_ = opts.SetTag("tag")
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		_ = opts.Close()
	}()
	close(start)
	wg.Wait()

	if err := opts.SetTag("tag"); err == nil {
		t.Fatal("expected error after concurrent Close")
	}
}

func TestNilRunOptions(t *testing.T) {
	var opts *RunOptions
	if err := opts.Close(); err != nil {
		t.Fatal(err)
	}
	if err := opts.SetTag("tag"); err == nil {
		t.Fatal("expected error using nil run options")
	}
}

func TestRunOptionsRejectLogLevelOutsideCInt(t *testing.T) {
	if strconv.IntSize == 32 {
		t.Skip("Go int has the same width as C int")
	}
	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetLogVerbosityLevel(int(int64(math.MaxInt32) + 1)); err == nil {
		t.Fatal("expected error for overflowing verbosity level")
	}
	if err := opts.SetLogSeverityLevel(int(int64(math.MinInt32) - 1)); err == nil {
		t.Fatal("expected error for overflowing severity level")
	}
}

func TestRunOptionsRejectNUL(t *testing.T) {
	opts, err := NewRunOptions()
	if err != nil {
		t.Fatal(err)
	}
	defer opts.Close()

	if err := opts.SetTag("tag\x00ignored"); err == nil {
		t.Fatal("expected error for NUL in tag")
	}
	if err := opts.AddConfigEntry("key\x00ignored", "value"); err == nil {
		t.Fatal("expected error for NUL in config key")
	}
	if err := opts.AddConfigEntry("key", "value\x00ignored"); err == nil {
		t.Fatal("expected error for NUL in config value")
	}
}
