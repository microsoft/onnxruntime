package onnxruntime

// TensorElementDataType represents the data type of tensor elements.
type TensorElementDataType int

const (
	TensorElementDataTypeUndefined TensorElementDataType = 0
	TensorElementDataTypeFloat32   TensorElementDataType = 1
	TensorElementDataTypeUint8     TensorElementDataType = 2
	TensorElementDataTypeInt8      TensorElementDataType = 3
	TensorElementDataTypeUint16    TensorElementDataType = 4
	TensorElementDataTypeInt16     TensorElementDataType = 5
	TensorElementDataTypeInt32     TensorElementDataType = 6
	TensorElementDataTypeInt64     TensorElementDataType = 7
	TensorElementDataTypeString    TensorElementDataType = 8
	TensorElementDataTypeBool      TensorElementDataType = 9
	TensorElementDataTypeFloat16   TensorElementDataType = 10
	TensorElementDataTypeFloat64   TensorElementDataType = 11
	TensorElementDataTypeUint32    TensorElementDataType = 12
	TensorElementDataTypeUint64    TensorElementDataType = 13
	TensorElementDataTypeBFloat16  TensorElementDataType = 16
)

func (t TensorElementDataType) String() string {
	switch t {
	case TensorElementDataTypeUndefined:
		return "Undefined"
	case TensorElementDataTypeFloat32:
		return "Float32"
	case TensorElementDataTypeUint8:
		return "Uint8"
	case TensorElementDataTypeInt8:
		return "Int8"
	case TensorElementDataTypeUint16:
		return "Uint16"
	case TensorElementDataTypeInt16:
		return "Int16"
	case TensorElementDataTypeInt32:
		return "Int32"
	case TensorElementDataTypeInt64:
		return "Int64"
	case TensorElementDataTypeString:
		return "String"
	case TensorElementDataTypeBool:
		return "Bool"
	case TensorElementDataTypeFloat16:
		return "Float16"
	case TensorElementDataTypeFloat64:
		return "Float64"
	case TensorElementDataTypeUint32:
		return "Uint32"
	case TensorElementDataTypeUint64:
		return "Uint64"
	case TensorElementDataTypeBFloat16:
		return "BFloat16"
	default:
		return "Unknown"
	}
}

// GraphOptimizationLevel controls the level of graph optimizations applied.
type GraphOptimizationLevel int

const (
	GraphOptimizationLevelDisableAll GraphOptimizationLevel = 0
	GraphOptimizationLevelBasic      GraphOptimizationLevel = 1
	GraphOptimizationLevelExtended   GraphOptimizationLevel = 2
	GraphOptimizationLevelAll        GraphOptimizationLevel = 99
)

// ExecutionMode controls sequential or parallel execution of independent ops.
type ExecutionMode int

const (
	ExecutionModeSequential ExecutionMode = 0
	ExecutionModeParallel   ExecutionMode = 1
)

// TensorElement enumerates Go types with a direct ONNX tensor element mapping.
type TensorElement interface {
	bool | int8 | uint8 | int16 | uint16 | int32 | uint32 |
		int64 | uint64 | float32 | float64
}

func dtypeOf[T TensorElement]() TensorElementDataType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return TensorElementDataTypeFloat32
	case uint8:
		return TensorElementDataTypeUint8
	case int8:
		return TensorElementDataTypeInt8
	case uint16:
		return TensorElementDataTypeUint16
	case int16:
		return TensorElementDataTypeInt16
	case int32:
		return TensorElementDataTypeInt32
	case int64:
		return TensorElementDataTypeInt64
	case bool:
		return TensorElementDataTypeBool
	case float64:
		return TensorElementDataTypeFloat64
	case uint32:
		return TensorElementDataTypeUint32
	case uint64:
		return TensorElementDataTypeUint64
	default:
		return TensorElementDataTypeUndefined
	}
}

func elemSize(dt TensorElementDataType) int {
	switch dt {
	case TensorElementDataTypeBool, TensorElementDataTypeInt8, TensorElementDataTypeUint8:
		return 1
	case TensorElementDataTypeInt16, TensorElementDataTypeUint16,
		TensorElementDataTypeFloat16, TensorElementDataTypeBFloat16:
		return 2
	case TensorElementDataTypeInt32, TensorElementDataTypeUint32, TensorElementDataTypeFloat32:
		return 4
	case TensorElementDataTypeInt64, TensorElementDataTypeUint64, TensorElementDataTypeFloat64:
		return 8
	default:
		return 0
	}
}
