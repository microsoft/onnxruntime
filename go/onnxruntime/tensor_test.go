package onnxruntime

import (
	"testing"
)

func TestCreateTensorFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor, err := CreateTensor[float32]([]int64{2, 3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.DataType() != TensorElementDataTypeFloat32 {
		t.Errorf("expected Float32, got %s", tensor.DataType())
	}
	if tensor.ElementCount() != 6 {
		t.Errorf("expected 6 elements, got %d", tensor.ElementCount())
	}

	got, err := TensorData[float32](tensor)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != data[i] {
			t.Errorf("[%d]: expected %f, got %f", i, data[i], v)
		}
	}
}

func TestCreateTensorInt64(t *testing.T) {
	data := []int64{10, 20, 30}
	tensor, err := CreateTensor[int64]([]int64{3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.DataType() != TensorElementDataTypeInt64 {
		t.Errorf("expected Int64, got %s", tensor.DataType())
	}

	got, err := TensorData[int64](tensor)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != data[i] {
			t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
		}
	}
}

func TestCreateTensorBool(t *testing.T) {
	data := []bool{true, false, true}
	tensor, err := CreateTensor[bool]([]int64{3}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.DataType() != TensorElementDataTypeBool {
		t.Errorf("expected Bool, got %s", tensor.DataType())
	}
}

func TestCreateTensorZeroLength(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{1, 2, 0, 4}, []float32{})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.ElementCount() != 0 {
		t.Errorf("expected 0 elements, got %d", tensor.ElementCount())
	}

	shape := tensor.Shape()
	if len(shape) != 4 || shape[2] != 0 {
		t.Errorf("expected shape [1,2,0,4], got %v", shape)
	}
}

func TestTensorDataTypeMismatch(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	_, err = TensorData[int64](tensor)
	if err == nil {
		t.Fatal("expected type mismatch error")
	}
}

func TestTensorShapeMismatch(t *testing.T) {
	_, err := CreateTensor[float32]([]int64{2, 3}, []float32{1, 2, 3})
	if err == nil {
		t.Fatal("expected shape/data length mismatch error")
	}
}

func TestTensorNegativeDim(t *testing.T) {
	_, err := CreateTensor[float32]([]int64{-1, 3}, []float32{1, 2, 3})
	if err == nil {
		t.Fatal("expected negative dimension error")
	}
}

func TestTensorUseAfterClose(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	tensor.Close()

	_, err = TensorData[float32](tensor)
	if err == nil {
		t.Fatal("expected error on use after close")
	}
}

func TestTensorDoubleClose(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := tensor.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tensor.Close(); err != nil {
		t.Fatal("double close should not error")
	}
}

func TestNewTensorFromBytes(t *testing.T) {
	data := []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0x40}
	tensor, err := NewTensorFromBytes(TensorElementDataTypeFloat32, []int64{2}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	got, err := TensorData[float32](tensor)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 1.0 || got[1] != 2.0 {
		t.Errorf("expected [1.0, 2.0], got %v", got)
	}
}

func TestTensorBytes(t *testing.T) {
	data := []float32{1.0, 2.0}
	tensor, err := CreateTensor[float32]([]int64{2}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	b, err := tensor.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 8 {
		t.Errorf("expected 8 bytes, got %d", len(b))
	}
}

func TestScalarTensor(t *testing.T) {
	tensor, err := CreateTensor[float32]([]int64{}, []float32{42.0})
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.ElementCount() != 1 {
		t.Errorf("expected 1 element, got %d", tensor.ElementCount())
	}
}
