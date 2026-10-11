// Package onnxruntime provides Go bindings for the ONNX Runtime C API.
//
// Call SetSharedLibraryPath and Init before using any other function.
// Call Shutdown when done to release resources.
//
// The binding vendors API 32. For runtime version 1.N.x, Init first requests
// API N capped at the binding's maximum. If the version cannot be parsed,
// Init starts at the maximum. Rejected requests fall back down to API 17.
// The loader floor is an intended contract, not proof of full runtime support.
// GetBuildInfo requires API 15. SessionOptions.GetExecutionMode and
// SessionOptions.IsMemPatternEnabled require API 27. Unsupported operations
// return an OrtError with ErrorCodeNotImplemented.
// Linux CPU race tests cover ORT 1.27.0/API 27, 1.28.3/API 28,
// 1.29.1/API 29, 1.30.0/API 30, and 1.31.0/API 31.
// Capped builds on a newer runtime do not verify an older runtime's ABI.
//
// ErrorCodeDeviceReset requires releasing the old session and creating another.
// The binding does not retry inference automatically.
//
// A Session wraps an ORT inference session. It is safe for concurrent
// use: multiple goroutines may call Run on the same Session simultaneously.
// Each Tensor, SessionOptions, and other handle must not be used concurrently
// unless documented otherwise.
package onnxruntime
