package onnxruntime

import (
	"errors"
	"testing"
)

func TestError(t *testing.T) {
	e := &Error{Code: ErrorCodeFail, Message: "something went wrong"}
	if e.Error() != "onnxruntime error [1]: something went wrong" {
		t.Errorf("unexpected Error(): %s", e.Error())
	}
}

func TestIsError(t *testing.T) {
	ortErr := &Error{Code: ErrorCodeOK, Message: "ok"}
	if !IsError(ortErr) {
		t.Error("IsError should return true for *Error")
	}
	if IsError(errors.New("plain")) {
		t.Error("IsError should return false for plain error")
	}
}

func TestGetErrorCode(t *testing.T) {
	ortErr := &Error{Code: ErrorCodeInvalidArgument, Message: "bad arg"}
	code, ok := GetErrorCode(ortErr)
	if !ok || code != ErrorCodeInvalidArgument {
		t.Errorf("GetErrorCode: ok=%v code=%d", ok, code)
	}
	_, ok = GetErrorCode(errors.New("plain"))
	if ok {
		t.Error("GetErrorCode should return false for plain error")
	}
}

func TestStatusToGoErr(t *testing.T) {
	if err := statusToGoErr(0); err != nil {
		t.Errorf("nil status should return nil error, got %v", err)
	}
}
