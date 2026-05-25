package onnxruntime

import "testing"

func TestOpaqueTypes(t *testing.T) {
	var env OrtEnv
	var sess OrtSession
	var val OrtValue
	if env != 0 || sess != 0 || val != 0 {
		t.Error("zero-initialized opaque types should be 0")
	}
}

func TestORTAPIVersion(t *testing.T) {
	if ORTAPIVersion != 27 {
		t.Errorf("API version should be 27, got %d", ORTAPIVersion)
	}
}
