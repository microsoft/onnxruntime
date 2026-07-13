package onnxruntime

/*
#include "cshim.h"
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// Tensor wraps an OrtValue containing tensor data.
type Tensor struct {
	value  *C.OrtValue
	dtype  TensorElementDataType
	shape  []int64
	pinner runtime.Pinner
	keep   any
	owned  bool
	closed bool
}

// CreateTensor creates a tensor backed by the provided data slice.
// The data must not be freed or resized until the tensor is closed.
// The shape must have non-negative dimensions whose product equals len(data).
func CreateTensor[T TensorElement](shape []int64, data []T) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	dtype := dtypeOf[T]()
	count := shapeElementCount(shape)
	if count < 0 {
		return nil, fmt.Errorf("ort: create tensor: invalid shape (negative or overflowing element count)")
	}
	if int64(len(data)) != count {
		return nil, fmt.Errorf("ort: create tensor: data length %d does not match shape element count %d", len(data), count)
	}

	if count == 0 {
		return createEmptyTensor(dtype, shape)
	}

	t := &Tensor{dtype: dtype, shape: copyShape(shape), owned: false}
	t.pinner.Pin(&data[0])
	t.keep = data

	var cShape *C.int64_t
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	dataSize := C.size_t(len(data)) * C.size_t(elemSize(dtype))
	dataPtr := unsafe.Pointer(&data[0])

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorWithDataAsOrtValue(
		cpuMemInfo, dataPtr, dataSize, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		t.pinner.Unpin()
		return nil, wrapErr("create tensor", err)
	}

	t.value = value
	return t, nil
}

// NewTensorFromBytes creates a tensor from raw bytes for types without a
// native Go mapping (Float16, BFloat16) or for fully dynamic pipelines.
// len(data) must equal the product of shape multiplied by the element size of dtype.
func NewTensorFromBytes(dtype TensorElementDataType, shape []int64, data []byte) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	count := shapeElementCount(shape)
	if count < 0 {
		return nil, fmt.Errorf("ort: create tensor: invalid shape (negative or overflowing element count)")
	}

	es := elemSize(dtype)
	if es == 0 {
		return nil, fmt.Errorf("ort: create tensor: unsupported dtype %s", dtype)
	}

	if count > math.MaxInt/int64(es) {
		return nil, fmt.Errorf("ort: create tensor: shape element count %d overflows byte size for dtype %s", count, dtype)
	}
	expected := int(count) * es
	if len(data) != expected {
		return nil, fmt.Errorf("ort: create tensor: data length %d does not match expected %d bytes", len(data), expected)
	}

	if count == 0 {
		return createEmptyTensor(dtype, shape)
	}

	t := &Tensor{dtype: dtype, shape: copyShape(shape), owned: false}
	t.pinner.Pin(&data[0])
	t.keep = data

	var cShape *C.int64_t
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	dataSize := C.size_t(len(data))
	dataPtr := unsafe.Pointer(&data[0])

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorWithDataAsOrtValue(
		cpuMemInfo, dataPtr, dataSize, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		t.pinner.Unpin()
		return nil, wrapErr("create tensor", err)
	}

	t.value = value
	return t, nil
}

// TensorData returns a typed view of the tensor's data buffer. The returned
// slice is valid until the tensor is closed. Returns an error if T's dtype
// does not match the tensor's element type.
func TensorData[T TensorElement](t *Tensor) ([]T, error) {
	if err := t.checkUsable("tensor data"); err != nil {
		return nil, err
	}

	expected := dtypeOf[T]()
	if t.dtype != expected {
		return nil, fmt.Errorf("ort: tensor data: type mismatch: tensor is %s, requested %s", t.dtype, expected)
	}

	count := shapeElementCount(t.shape)
	if count == 0 {
		return nil, nil
	}
	if count > math.MaxInt/int64(elemSize(expected)) {
		return nil, fmt.Errorf("ort: tensor data: element count %d exceeds addressable range", count)
	}

	var dataPtr unsafe.Pointer
	if err := checkStatus(C.ort_GetTensorMutableData(t.value, &dataPtr)); err != nil {
		return nil, wrapErr("tensor data", err)
	}

	return unsafe.Slice((*T)(dataPtr), count), nil
}

// Bytes returns a raw byte view of the tensor's data buffer.
func (t *Tensor) Bytes() ([]byte, error) {
	if err := t.checkUsable("tensor bytes"); err != nil {
		return nil, err
	}

	es := elemSize(t.dtype)
	if es == 0 {
		return nil, fmt.Errorf("ort: tensor bytes: unsupported element type %s (use StringData for string tensors)", t.dtype)
	}

	count := shapeElementCount(t.shape)
	if count == 0 {
		return nil, nil
	}
	if count > math.MaxInt/int64(es) {
		return nil, fmt.Errorf("ort: tensor bytes: byte size overflows addressable range")
	}

	var dataPtr unsafe.Pointer
	if err := checkStatus(C.ort_GetTensorMutableData(t.value, &dataPtr)); err != nil {
		return nil, wrapErr("tensor bytes", err)
	}

	nbytes := int(count) * es
	return unsafe.Slice((*byte)(dataPtr), nbytes), nil
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() []int64 {
	return copyShape(t.shape)
}

// DataType returns the tensor's element data type.
func (t *Tensor) DataType() TensorElementDataType {
	return t.dtype
}

// ElementCount returns the total number of elements.
func (t *Tensor) ElementCount() int64 {
	return shapeElementCount(t.shape)
}

// ValueType returns the ONNX type of the underlying OrtValue.
func (t *Tensor) ValueType() (int, error) {
	if err := t.checkUsable("value type"); err != nil {
		return 0, err
	}
	var onnxType C.enum_ONNXType
	if err := checkStatus(C.ort_GetValueType(t.value, &onnxType)); err != nil {
		return 0, wrapErr("get value type", err)
	}
	return int(onnxType), nil
}

// IsTensor reports whether this value is a tensor (as opposed to sequence/map).
func (t *Tensor) IsTensor() bool {
	vt, err := t.ValueType()
	return err == nil && vt == int(C.ONNX_TYPE_TENSOR)
}

// IsSequence reports whether this value is a sequence.
func (t *Tensor) IsSequence() bool {
	vt, err := t.ValueType()
	return err == nil && vt == int(C.ONNX_TYPE_SEQUENCE)
}

// IsMap reports whether this value is a map.
func (t *Tensor) IsMap() bool {
	vt, err := t.ValueType()
	return err == nil && vt == int(C.ONNX_TYPE_MAP)
}

// SequenceLen returns the number of elements in a sequence value.
func (t *Tensor) SequenceLen() (int, error) {
	if err := t.checkUsable("sequence len"); err != nil {
		return 0, err
	}
	var count C.size_t
	if err := checkStatus(C.ort_GetValueCount(t.value, &count)); err != nil {
		return 0, wrapErr("get value count", err)
	}
	return int(count), nil
}

// SequenceAt returns the element at the given index from a sequence value.
// The returned tensor must be closed by the caller.
func (t *Tensor) SequenceAt(index int) (*Tensor, error) {
	if err := t.checkUsable("sequence at"); err != nil {
		return nil, err
	}
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}
	var value *C.OrtValue
	if err := checkStatus(C.ort_GetValue(t.value, C.int(index), allocator, &value)); err != nil {
		return nil, wrapErr("get value", err)
	}
	return wrapOutputTensor(value)
}

// NewSequence creates a sequence value from the given tensors.
// All elements must be the same ONNX type. The returned value must be closed.
func NewSequence(elements []*Tensor) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	if len(elements) == 0 {
		return nil, fmt.Errorf("ort: create sequence: at least one element required")
	}

	values := make([]*C.OrtValue, len(elements))
	for i, e := range elements {
		if err := e.checkUsable(fmt.Sprintf("create sequence: element %d", i)); err != nil {
			return nil, err
		}
		values[i] = e.value
	}

	var out *C.OrtValue
	if err := checkStatus(C.ort_CreateValue(&values[0], C.size_t(len(values)),
		C.ONNX_TYPE_SEQUENCE, &out)); err != nil {
		return nil, wrapErr("create sequence", err)
	}

	return &Tensor{value: out, dtype: TensorElementDataTypeUndefined, owned: true}, nil
}

// NewMap creates a map value from key and value tensors.
// Keys must be a 1-D tensor of int64 or string type. Values must be a 1-D tensor.
func NewMap(keys, values *Tensor) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}
	if err := keys.checkUsable("create map: keys"); err != nil {
		return nil, err
	}
	if err := values.checkUsable("create map: values"); err != nil {
		return nil, err
	}

	ins := []*C.OrtValue{keys.value, values.value}
	var out *C.OrtValue
	if err := checkStatus(C.ort_CreateValue(&ins[0], 2,
		C.ONNX_TYPE_MAP, &out)); err != nil {
		return nil, wrapErr("create map", err)
	}

	return &Tensor{value: out, dtype: TensorElementDataTypeUndefined, owned: true}, nil
}

// NewMapFromGoMap creates a map OrtValue from a Go map.
// Key type must be int64. Value type must be a TensorElement type.
func NewMapFromGoMap[K int64, V TensorElement](m map[K]V) (*Tensor, error) {
	if err := checkInit(); err != nil {
		return nil, err
	}

	n := len(m)
	keys := make([]K, 0, n)
	vals := make([]V, 0, n)
	for k, v := range m {
		keys = append(keys, k)
		vals = append(vals, v)
	}

	keyTensor, err := CreateTensor[K]([]int64{int64(n)}, keys)
	if err != nil {
		return nil, wrapErr("create map keys", err)
	}
	defer func() { _ = keyTensor.Close() }()

	valTensor, err := CreateTensor[V]([]int64{int64(n)}, vals)
	if err != nil {
		return nil, wrapErr("create map values", err)
	}
	defer func() { _ = valTensor.Close() }()

	return NewMap(keyTensor, valTensor)
}

// checkUsable reports an error if the tensor cannot be passed to the C API.
// op names the calling operation, e.g. `run: input "ids"`.
func (t *Tensor) checkUsable(op string) error {
	if t == nil || t.closed || t.value == nil {
		return fmt.Errorf("ort: %s: tensor is nil or closed", op)
	}
	return nil
}

// Close releases the tensor's resources. It is idempotent.
func (t *Tensor) Close() error {
	if t.closed {
		return nil
	}
	t.closed = true
	if t.value != nil {
		C.ort_ReleaseValue(t.value)
		t.value = nil
	}
	if !t.owned {
		t.pinner.Unpin()
	}
	t.keep = nil
	return nil
}

func createEmptyTensor(dtype TensorElementDataType, shape []int64) (*Tensor, error) {
	var allocator *C.OrtAllocator
	if err := checkStatus(C.ort_GetAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, wrapErr("get allocator", err)
	}

	cShape := (*C.int64_t)(nil)
	if len(shape) > 0 {
		cShape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var value *C.OrtValue
	if err := checkStatus(C.ort_CreateTensorAsOrtValue(
		allocator, cShape, C.size_t(len(shape)),
		C.ONNXTensorElementDataType(dtype), &value)); err != nil {
		return nil, wrapErr("create empty tensor", err)
	}

	return &Tensor{
		value: value,
		dtype: dtype,
		shape: copyShape(shape),
		owned: true,
	}, nil
}

func wrapOutputTensor(value *C.OrtValue) (*Tensor, error) {
	var info *C.OrtTensorTypeAndShapeInfo
	if err := checkStatus(C.ort_GetTensorTypeAndShape(value, &info)); err != nil {
		return nil, wrapErr("get output tensor type", err)
	}
	defer C.ort_ReleaseTensorTypeAndShapeInfo(info)

	var cDtype C.enum_ONNXTensorElementDataType
	if err := checkStatus(C.ort_GetTensorElementType(info, &cDtype)); err != nil {
		return nil, wrapErr("get output element type", err)
	}

	var ndims C.size_t
	if err := checkStatus(C.ort_GetDimensionsCount(info, &ndims)); err != nil {
		return nil, wrapErr("get output dims count", err)
	}

	shape := make([]int64, int(ndims))
	if ndims > 0 {
		if err := checkStatus(C.ort_GetDimensions(info, (*C.int64_t)(unsafe.Pointer(&shape[0])), ndims)); err != nil {
			return nil, wrapErr("get output dims", err)
		}
	}

	return &Tensor{
		value: value,
		dtype: TensorElementDataType(cDtype),
		shape: shape,
		owned: true,
	}, nil
}

func shapeElementCount(shape []int64) int64 {
	if len(shape) == 0 {
		return 1
	}
	count := int64(1)
	for _, d := range shape {
		if d < 0 {
			return -1
		}
		if d != 0 && count > math.MaxInt64/d {
			return -1
		}
		count *= d
	}
	return count
}

func copyShape(s []int64) []int64 {
	out := make([]int64, len(s))
	copy(out, s)
	return out
}
