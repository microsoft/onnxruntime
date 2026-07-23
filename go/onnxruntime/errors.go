package onnxruntime

/*
#include "cshim.h"
*/
import "C"
import "fmt"

// ErrorCode mirrors OrtErrorCode from the C API.
type ErrorCode int

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
	ErrorCodeEPFail           ErrorCode = 11
)

func (c ErrorCode) String() string {
	switch c {
	case ErrorCodeOK:
		return "OK"
	case ErrorCodeFail:
		return "Fail"
	case ErrorCodeInvalidArgument:
		return "InvalidArgument"
	case ErrorCodeNoSuchFile:
		return "NoSuchFile"
	case ErrorCodeNoModel:
		return "NoModel"
	case ErrorCodeEngineError:
		return "EngineError"
	case ErrorCodeRuntimeException:
		return "RuntimeException"
	case ErrorCodeInvalidProtobuf:
		return "InvalidProtobuf"
	case ErrorCodeModelLoaded:
		return "ModelLoaded"
	case ErrorCodeNotImplemented:
		return "NotImplemented"
	case ErrorCodeInvalidGraph:
		return "InvalidGraph"
	case ErrorCodeEPFail:
		return "EPFail"
	default:
		return fmt.Sprintf("ErrorCode(%d)", int(c))
	}
}

// OrtError represents an error from the ONNX Runtime C API.
type OrtError struct {
	Code ErrorCode
	Msg  string
}

func (e *OrtError) Error() string {
	return fmt.Sprintf("onnxruntime: %s: %s", e.Code, e.Msg)
}

func checkStatus(status *C.OrtStatus) error {
	if status == nil {
		return nil
	}
	code := ErrorCode(C.ort_GetErrorCode(status))
	msg := C.GoString(C.ort_GetErrorMessage(status))
	C.ort_ReleaseStatus(status)
	return &OrtError{Code: code, Msg: msg}
}

func wrapErr(context string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("ort: %s: %w", context, err)
}

var errNotInitialized = fmt.Errorf("ort: not initialized; call Init() first")
var errShutdown = fmt.Errorf("ort: already shut down")
