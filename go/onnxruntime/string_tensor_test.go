package onnxruntime

import (
	"testing"
)

func TestCreateStringTensor(t *testing.T) {
	data := []string{"hello", "world", "foo"}
	tensor, err := CreateStringTensor([]int64{3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.DataType() != TensorElementDataTypeString {
		t.Errorf("expected String, got %s", tensor.DataType())
	}
	if tensor.ElementCount() != 3 {
		t.Errorf("expected 3 elements, got %d", tensor.ElementCount())
	}

	got, err := tensor.StringData()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != data[i] {
			t.Errorf("[%d]: expected %q, got %q", i, data[i], v)
		}
	}
}

func TestCreateStringTensorEmpty(t *testing.T) {
	tensor, err := CreateStringTensor([]int64{0}, []string{})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	got, err := tensor.StringData()
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Errorf("expected nil for empty string tensor, got %v", got)
	}
}

func TestCreateStringTensor2D(t *testing.T) {
	data := []string{"a", "bb", "ccc", "dddd", "eeeee", "ffffff"}
	tensor, err := CreateStringTensor([]int64{2, 3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	shape := tensor.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("expected shape [2,3], got %v", shape)
	}

	got, err := tensor.StringData()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != data[i] {
			t.Errorf("[%d]: expected %q, got %q", i, data[i], v)
		}
	}
}

func TestStringDataOnNonStringTensor(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	_, err = tensor.StringData()
	if err == nil {
		t.Fatal("expected error calling StringData on float tensor")
	}
}
