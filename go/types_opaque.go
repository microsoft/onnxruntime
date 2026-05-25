package onnxruntime

// OrtStatus is an opaque handle to an ONNX Runtime status (error) object.
// nil / 0 means success; non-nil means an error occurred.
type OrtStatus uintptr

// OrtEnv is an opaque handle to the ONNX Runtime environment.
type OrtEnv uintptr

// OrtSession is an opaque handle to a loaded ONNX model session.
type OrtSession uintptr

// OrtSessionOptions configures session creation parameters.
type OrtSessionOptions uintptr

// OrtRunOptions configures per-run execution parameters.
type OrtRunOptions uintptr

// OrtValue is an opaque handle to a tensor or other value.
type OrtValue uintptr

// OrtMemoryInfo describes a memory region (device, type, etc.).
type OrtMemoryInfo uintptr

// OrtAllocator is an opaque handle to a memory allocator.
type OrtAllocator uintptr

// OrtTensorTypeAndShapeInfo describes tensor element type and dimensions.
type OrtTensorTypeAndShapeInfo uintptr

// OrtTypeInfo describes the type of an OrtValue.
type OrtTypeInfo uintptr

// OrtModelMetadata describes model-level metadata.
type OrtModelMetadata uintptr

// OrtStatusPtr is a pointer-to-pointer for out parameters returning OrtStatus.
type OrtStatusPtr = *OrtStatus

// OrtEnvPtr is a pointer-to-pointer for out parameters returning OrtEnv.
type OrtEnvPtr = *OrtEnv

// OrtSessionPtr is a pointer-to-pointer for out parameters returning OrtSession.
type OrtSessionPtr = *OrtSession

// OrtSessionOptionsPtr is a pointer-to-pointer for out parameters returning OrtSessionOptions.
type OrtSessionOptionsPtr = *OrtSessionOptions

// OrtRunOptionsPtr is a pointer-to-pointer for out parameters returning OrtRunOptions.
type OrtRunOptionsPtr = *OrtRunOptions

// OrtValuePtr is a pointer-to-pointer for out parameters returning OrtValue.
type OrtValuePtr = *OrtValue

// OrtMemoryInfoPtr is a pointer-to-pointer for out parameters returning OrtMemoryInfo.
type OrtMemoryInfoPtr = *OrtMemoryInfo

// OrtAllocatorPtr is a pointer-to-pointer for out parameters returning OrtAllocator.
type OrtAllocatorPtr = *OrtAllocator

// OrtTensorTypeAndShapeInfoPtr is a pointer-to-pointer for out parameters returning OrtTensorTypeAndShapeInfo.
type OrtTensorTypeAndShapeInfoPtr = *OrtTensorTypeAndShapeInfo

// ORTAPIVersion is the API version to request from the runtime.
// This should match the runtime's supported version (major version number).
const ORTAPIVersion uint32 = 27
