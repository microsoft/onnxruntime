package onnxruntime

// LoggingLevel controls the verbosity of ONNX Runtime logging.
type LoggingLevel int32

const (
	LogVerbose LoggingLevel = 0
	LogInfo    LoggingLevel = 1
	LogWarning LoggingLevel = 2
	LogError   LoggingLevel = 3
	LogFatal   LoggingLevel = 4
)

// GraphOptimizationLevel controls the level of graph optimizations.
type GraphOptimizationLevel int32

const (
	GraphOptNone   GraphOptimizationLevel = 0
	GraphOptBasic  GraphOptimizationLevel = 1
	GraphOptExtend GraphOptimizationLevel = 2
	GraphOptLayout GraphOptimizationLevel = 3
	GraphOptAll    GraphOptimizationLevel = 99
)

// ExecutionMode controls whether operators run sequentially or in parallel.
type ExecutionMode int32

const (
	ExecSequential ExecutionMode = 0
	ExecParallel   ExecutionMode = 1
)

// TensorElementDataType maps to ONNXTensorElementDataType.
type TensorElementDataType int32

const (
	TensorUndefined      TensorElementDataType = 0
	TensorFloat32        TensorElementDataType = 1
	TensorUint8          TensorElementDataType = 2
	TensorInt8           TensorElementDataType = 3
	TensorUint16         TensorElementDataType = 4
	TensorInt16          TensorElementDataType = 5
	TensorInt32          TensorElementDataType = 6
	TensorInt64          TensorElementDataType = 7
	TensorString         TensorElementDataType = 8
	TensorBool           TensorElementDataType = 9
	TensorFloat16        TensorElementDataType = 10
	TensorFloat64        TensorElementDataType = 11
	TensorUint32         TensorElementDataType = 12
	TensorUint64         TensorElementDataType = 13
	TensorComplex64      TensorElementDataType = 14
	TensorComplex128     TensorElementDataType = 15
	TensorBFloat16       TensorElementDataType = 16
	TensorFloat8E4M3FN   TensorElementDataType = 17
	TensorFloat8E4M3FNUZ TensorElementDataType = 18
	TensorFloat8E5M2     TensorElementDataType = 19
	TensorFloat8E5M2FNUZ TensorElementDataType = 20
	TensorUint4          TensorElementDataType = 21
	TensorInt4           TensorElementDataType = 22
	TensorFloat4E2M1FN   TensorElementDataType = 23
	TensorUint2          TensorElementDataType = 24
	TensorInt2           TensorElementDataType = 25
	TensorFloat8E8M0FNU  TensorElementDataType = 26
)

// AllocatorType specifies the type of memory allocator.
type AllocatorType int32

const (
	AllocInvalid AllocatorType = -1
	AllocDevice  AllocatorType = 0
	AllocArena   AllocatorType = 1
)

// MemType specifies the type of memory (CPU input/output, temporary, etc.).
type MemType int32

const (
	MemCPUInput  MemType = 0
	MemCPUOutput MemType = 1
	MemCPU       MemType = 2
	MemDefault   MemType = 3
)

// ErrorCode maps to OrtErrorCode from the C API.
type ErrorCode int32

const (
	ErrorCodeOK               ErrorCode = 0
	ErrorCodeFail             ErrorCode = 1
	ErrorCodeInvalidArgument  ErrorCode = 2
	ErrorCodeNoSuchFile       ErrorCode = 3
	ErrorCodeNoModel          ErrorCode = 4
	ErrorCodeEngineError      ErrorCode = 5
	ErrorCodeRuntimeException ErrorCode = 6
	ErrorCodeInvalidProtobuf  ErrorCode = 7
	ErrorCodeModelLoaded      ErrorCode = 8
	ErrorCodeNotImplemented   ErrorCode = 9
	ErrorCodeInvalidGraph     ErrorCode = 10
	ErrorCodeEpFail           ErrorCode = 11
)
