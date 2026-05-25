package onnxruntime

import "testing"

func TestFunctionIndices(t *testing.T) {
	tests := []struct {
		name     string
		idx      int
		expected int
	}{
		{"GetErrorCode", fnGetErrorCode, 1},
		{"GetErrorMessage", fnGetErrorMessage, 2},
		{"CreateEnv", fnCreateEnv, 3},
		{"CreateSession", fnCreateSession, 7},
		{"CreateSessionFromArray", fnCreateSessionFromArray, 8},
		{"Run", fnRun, 9},
		{"SessionGetInputCount", fnSessionGetInputCount, 30},
		{"CreateSessionOptions", fnCreateSessionOptions, 10},
		{"SetSessionGraphOptimizationLevel", fnSetSessionGraphOptimizationLevel, 23},
		{"CreateRunOptions", fnCreateRunOptions, 39},
		{"CreateTensorAsOrtValue", fnCreateTensorAsOrtValue, 48},
		{"GetTensorMutableData", fnGetTensorMutableData, 51},
		{"GetTensorElementType", fnGetTensorElementType, 60},
		{"CreateCpuMemoryInfo", fnCreateCpuMemoryInfo, 69},
		{"GetAllocatorWithDefaultOptions", fnGetAllocatorWithDefaultOptions, 78},
		{"AllocatorAlloc", fnAllocatorAlloc, 75},
		{"ReleaseEnv", fnReleaseEnv, 92},
		{"ReleaseStatus", fnReleaseStatus, 93},
		{"ReleaseSession", fnReleaseSession, 95},
		{"ReleaseValue", fnReleaseValue, 96},
	}
	for _, tt := range tests {
		if tt.idx != tt.expected {
			t.Errorf("%s: got %d, want %d", tt.name, tt.idx, tt.expected)
		}
	}
}

func TestGetFuncPtrReturnsZeroWhenNotInitialized(t *testing.T) {
	if ptr := getFuncPtr(0); ptr != 0 {
		t.Errorf("getFuncPtr should return 0, got %d", ptr)
	}
}
