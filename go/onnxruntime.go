package onnxruntime

import (
	"sync"
	"unsafe"

	"github.com/microsoft/onnxruntime/go/internal/cstrings"
)

// Function indices in the OrtApi vtable (zero-based).
const (
	fnGetErrorCode                                  = 1
	fnGetErrorMessage                               = 2
	fnCreateEnv                                     = 3
	fnCreateSession                                 = 7
	fnCreateSessionFromArray                        = 8
	fnRun                                           = 9
	fnSessionGetInputCount                          = 30
	fnSessionGetOutputCount                         = 31
	fnSessionGetInputName                           = 36
	fnSessionGetOutputName                          = 37
	fnCreateSessionOptions                          = 10
	fnSetSessionExecutionMode                       = 13
	fnEnableMemPattern                              = 16
	fnDisableMemPattern                             = 17
	fnEnableCpuMemArena                             = 18
	fnDisableCpuMemArena                            = 19
	fnSetSessionGraphOptimizationLevel              = 23
	fnSetIntraOpNumThreads                          = 24
	fnSetInterOpNumThreads                          = 25
	fnAddSessionConfigEntry                         = 130
	fnSessionOptionsAppendExecutionProviderCUDA     = 152
	fnSessionOptionsAppendExecutionProviderTensorRT = 159
	fnCreateRunOptions                              = 39
	fnRunOptionsSetRunLogVerbosityLevel             = 40
	fnRunOptionsSetRunTag                           = 42
	fnCreateTensorAsOrtValue                        = 48
	fnCreateTensorWithDataAsOrtValue                = 49
	fnIsTensor                                      = 50
	fnGetTensorMutableData                          = 51
	fnGetTensorTypeAndShape                         = 65
	fnGetTensorElementType                          = 60
	fnGetDimensionsCount                            = 61
	fnGetDimensions                                 = 62
	fnGetTensorShapeElementCount                    = 64
	fnCreateMemoryInfo                              = 68
	fnCreateCpuMemoryInfo                           = 69
	fnGetAllocatorWithDefaultOptions                = 78
	fnAllocatorAlloc                                = 75
	fnAllocatorFree                                 = 76

	fnReleaseEnv                    = 92
	fnReleaseStatus                 = 93
	fnReleaseMemoryInfo             = 94
	fnReleaseSession                = 95
	fnReleaseValue                  = 96
	fnReleaseRunOptions             = 97
	fnReleaseTensorTypeAndShapeInfo = 99
	fnReleaseSessionOptions         = 100
	fnReleaseAllocator              = 132
)

var (
	loadOnce sync.Once
	loadErr  error

	apiBase unsafe.Pointer
)

// Initialize loads the ONNX Runtime shared library and populates the API
// function table. libPath is the path to the shared library; if empty, the
// platform default ("libonnxruntime.so" or "onnxruntime.dll") is used.
// Initialize is idempotent — subsequent calls are no-ops.
func Initialize(libPath string) error {
	loadOnce.Do(func() {
		loadErr = doInitialize(libPath)
	})
	return loadErr
}

func getFuncPtr(index int) uintptr {
	if apiBase == nil {
		return 0
	}
	return *(*uintptr)(unsafe.Add(apiBase, index*int(unsafe.Sizeof(uintptr(0)))))
}

func statusToError(status OrtStatus) error {
	codeFn := getFuncPtr(fnGetErrorCode)
	msgFn := getFuncPtr(fnGetErrorMessage)
	releaseFn := getFuncPtr(fnReleaseStatus)

	errCode := ortDispatch(codeFn, uintptr(status))
	msgPtr := ortDispatch(msgFn, uintptr(status))
	msg := cstrings.CStringToString((*byte)(unsafe.Add(unsafe.Pointer(nil), msgPtr)))

	ortDispatch(releaseFn, uintptr(status))

	return &Error{
		Code:    ErrorCode(errCode),
		Message: msg,
	}
}
