package onnxruntime

import (
	"runtime"
	"testing"
)

func TestSequenceRetainsGoBackedElementData(t *testing.T) {
	source := newFloatTensor(t)
	sequence, err := NewSequence([]*Tensor{source})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sequence.Close() }()

	// The sequence must keep the Go-backed buffer pinned after the source closes.
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	runtime.GC()

	element, err := sequence.SequenceAt(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = element.Close() }()
	got, err := TensorData[float32](element)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("element data was invalidated: got %v, want [1 2]", got)
	}
}
