package onnxruntime

import (
	"math"
	"strings"
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

func TestCreateTensorAllTypes(t *testing.T) {
	t.Run("uint8", func(t *testing.T) {
		data := []uint8{1, 2, 3}
		tensor, err := CreateTensor[uint8]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeUint8 {
			t.Errorf("expected Uint8, got %s", tensor.DataType())
		}
		got, err := TensorData[uint8](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("int8", func(t *testing.T) {
		data := []int8{-1, 0, 1}
		tensor, err := CreateTensor[int8]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeInt8 {
			t.Errorf("expected Int8, got %s", tensor.DataType())
		}
		got, err := TensorData[int8](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("int16", func(t *testing.T) {
		data := []int16{-100, 0, 100}
		tensor, err := CreateTensor[int16]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeInt16 {
			t.Errorf("expected Int16, got %s", tensor.DataType())
		}
		got, err := TensorData[int16](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("uint16", func(t *testing.T) {
		data := []uint16{0, 1000, 65535}
		tensor, err := CreateTensor[uint16]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeUint16 {
			t.Errorf("expected Uint16, got %s", tensor.DataType())
		}
		got, err := TensorData[uint16](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("int32", func(t *testing.T) {
		data := []int32{-1, 0, 2147483647}
		tensor, err := CreateTensor[int32]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeInt32 {
			t.Errorf("expected Int32, got %s", tensor.DataType())
		}
		got, err := TensorData[int32](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("uint32", func(t *testing.T) {
		data := []uint32{0, 42, 4294967295}
		tensor, err := CreateTensor[uint32]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeUint32 {
			t.Errorf("expected Uint32, got %s", tensor.DataType())
		}
		got, err := TensorData[uint32](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("uint64", func(t *testing.T) {
		data := []uint64{0, 42, 18446744073709551615}
		tensor, err := CreateTensor[uint64]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeUint64 {
			t.Errorf("expected Uint64, got %s", tensor.DataType())
		}
		got, err := TensorData[uint64](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %d, got %d", i, data[i], v)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		data := []float64{1.5, -2.5, 3.14}
		tensor, err := CreateTensor[float64]([]int64{3}, data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()
		if tensor.DataType() != TensorElementDataTypeFloat64 {
			t.Errorf("expected Float64, got %s", tensor.DataType())
		}
		got, err := TensorData[float64](tensor)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != data[i] {
				t.Errorf("[%d]: expected %f, got %f", i, data[i], v)
			}
		}
	})
}

func TestNewTensorFromBytesUnsupportedDtype(t *testing.T) {
	_, err := NewTensorFromBytes(TensorElementDataTypeString, []int64{2}, []byte{0, 0, 0, 0})
	if err == nil {
		t.Fatal("expected error for String dtype in NewTensorFromBytes")
	}
}

func TestNewTensorFromBytesFloat16(t *testing.T) {
	// IEEE 754 half-precision: 1.0 = 0x3C00, 2.0 = 0x4000 (little-endian)
	data := []byte{0x00, 0x3C, 0x00, 0x40}
	tensor, err := NewTensorFromBytes(TensorElementDataTypeFloat16, []int64{2}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	if tensor.DataType() != TensorElementDataTypeFloat16 {
		t.Errorf("expected Float16, got %s", tensor.DataType())
	}

	shape := tensor.Shape()
	if len(shape) != 1 || shape[0] != 2 {
		t.Errorf("expected shape [2], got %v", shape)
	}

	b, err := tensor.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != len(data) {
		t.Fatalf("expected %d bytes, got %d", len(data), len(b))
	}
	for i, v := range b {
		if v != data[i] {
			t.Errorf("byte[%d]: expected 0x%02X, got 0x%02X", i, data[i], v)
		}
	}
}

func TestNewSequence(t *testing.T) {
	t1, _ := CreateTensor[float32]([]int64{3}, []float32{1, 2, 3})
	t2, _ := CreateTensor[float32]([]int64{3}, []float32{4, 5, 6})
	defer t1.Close()
	defer t2.Close()

	seq, err := NewSequence([]*Tensor{t1, t2})
	if err != nil {
		t.Fatal(err)
	}
	defer seq.Close()

	if !seq.IsSequence() {
		t.Error("expected IsSequence() = true")
	}

	n, err := seq.SequenceLen()
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("expected 2 elements, got %d", n)
	}

	elem, err := seq.SequenceAt(0)
	if err != nil {
		t.Fatal(err)
	}
	defer elem.Close()

	data, err := TensorData[float32](elem)
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != 1 || data[1] != 2 || data[2] != 3 {
		t.Errorf("expected [1,2,3], got %v", data)
	}
}

func TestNewMap(t *testing.T) {
	keys, _ := CreateTensor[int64]([]int64{2}, []int64{10, 20})
	vals, _ := CreateTensor[float32]([]int64{2}, []float32{1.5, 2.5})
	defer keys.Close()
	defer vals.Close()

	m, err := NewMap(keys, vals)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if !m.IsMap() {
		t.Error("expected IsMap() = true")
	}
}

func TestNewMapFromGoMap(t *testing.T) {
	m, err := NewMapFromGoMap[int64, float32](map[int64]float32{1: 10.5, 2: 20.5})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if !m.IsMap() {
		t.Error("expected IsMap() = true")
	}
}

func TestBytesStringTensorError(t *testing.T) {
	st, err := CreateStringTensor([]int64{2}, []string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	_, err = st.Bytes()
	if err == nil {
		t.Fatal("expected error calling Bytes() on a string tensor")
	}
}

func TestShapeElementCount(t *testing.T) {
	tests := []struct {
		name  string
		shape []int64
		want  int64
	}{
		{"scalar", []int64{}, 1},
		{"normal", []int64{2, 3, 4}, 24},
		{"zero dim", []int64{2, 0, 4}, 0},
		{"empty kv cache", []int64{1, 8, 0, 128}, 0},
		{"max dim", []int64{math.MaxInt64}, math.MaxInt64},
		{"negative dim", []int64{2, -1, 4}, -1},
		{"product overflows int64", []int64{1 << 32, 1 << 32, 2}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shapeElementCount(tt.shape); got != tt.want {
				t.Errorf("shapeElementCount(%v) = %d, want %d", tt.shape, got, tt.want)
			}
		})
	}
}

// TestTensorInvalidShapeRejected covers the count < 0 bail in both entry points.
func TestTensorInvalidShapeRejected(t *testing.T) {
	shape := []int64{1 << 32, 1 << 32, 2}

	t.Run("CreateTensor", func(t *testing.T) {
		_, err := CreateTensor[float32](shape, []float32{})
		if err == nil {
			t.Fatal("expected error for overflowing shape element count")
		}
	})

	t.Run("NewTensorFromBytes", func(t *testing.T) {
		_, err := NewTensorFromBytes(TensorElementDataTypeFloat32, shape, []byte{})
		if err == nil {
			t.Fatal("expected error for overflowing shape element count")
		}
	})
}

// TestNewTensorFromBytesByteSizeOverflow reaches the byte-size guard: the
// element count fits in int64, but count*elemSize does not fit in int. Without
// the guard the expected byte size wraps and the length check misbehaves.
func TestNewTensorFromBytesByteSizeOverflow(t *testing.T) {
	shape := []int64{1 << 62} // 2^62 elements, 2^64 bytes as float32

	if got := shapeElementCount(shape); got != 1<<62 {
		t.Fatalf("shapeElementCount(%v) = %d, want %d; test no longer reaches the byte-size guard", shape, got, int64(1)<<62)
	}

	_, err := NewTensorFromBytes(TensorElementDataTypeFloat32, shape, []byte{})
	if err == nil {
		t.Fatal("expected error for shape whose byte size overflows int")
	}
	if !strings.Contains(err.Error(), "overflows byte size") {
		t.Fatalf("expected byte-size overflow error, got: %v", err)
	}
}
