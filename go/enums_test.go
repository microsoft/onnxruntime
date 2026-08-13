package onnxruntime

import "testing"

func TestLoggingLevels(t *testing.T) {
	if LogVerbose != 0 {
		t.Error("Verbose should be 0")
	}
	if LogFatal != 4 {
		t.Error("Fatal should be 4")
	}
}

func TestGraphOptimization(t *testing.T) {
	if GraphOptNone != 0 {
		t.Error("None should be 0")
	}
	if GraphOptAll != 99 {
		t.Error("All should be 99")
	}
}

func TestExecutionMode(t *testing.T) {
	if ExecSequential != 0 {
		t.Error("Sequential should be 0")
	}
	if ExecParallel != 1 {
		t.Error("Parallel should be 1")
	}
}

func TestTensorElementTypes(t *testing.T) {
	if TensorFloat32 != 1 {
		t.Error("Float32 should be 1")
	}
	if TensorInt64 != 7 {
		t.Error("Int64 should be 7")
	}
	if TensorBool != 9 {
		t.Error("Bool should be 9")
	}
}

func TestAllocatorTypes(t *testing.T) {
	if AllocDevice != 0 {
		t.Error("DeviceAllocator should be 0")
	}
	if AllocArena != 1 {
		t.Error("ArenaAllocator should be 1")
	}
}

func TestMemTypes(t *testing.T) {
	if MemCPUInput != 0 {
		t.Error("CPUInput should be 0")
	}
	if MemDefault != 3 {
		t.Error("Default should be 3")
	}
}

func TestErrorCodes(t *testing.T) {
	if ErrorCodeOK != 0 {
		t.Error("OK should be 0")
	}
	if ErrorCodeFail != 1 {
		t.Error("Fail should be 1")
	}
}
