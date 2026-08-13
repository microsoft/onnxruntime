package onnxruntime

import "fmt"

// Error represents an ONNX Runtime API error, wrapping an OrtStatus.
type Error struct {
	Code    ErrorCode
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("onnxruntime error [%d]: %s", e.Code, e.Message)
}

// IsError reports whether the given error is an ONNX Runtime Error.
func IsError(err error) bool {
	_, ok := err.(*Error)
	return ok
}

// GetErrorCode extracts the OrtErrorCode from an error if it is an ONNX Runtime Error.
func GetErrorCode(err error) (ErrorCode, bool) {
	if ortErr, ok := err.(*Error); ok {
		return ortErr.Code, true
	}
	return 0, false
}
