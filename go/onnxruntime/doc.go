// Package onnxruntime provides Go bindings for the ONNX Runtime C API.
//
// Call SetSharedLibraryPath and Init before using any other function.
// Call Shutdown when done to release resources.
//
// A Session wraps an ORT inference session. It is safe for concurrent
// use: multiple goroutines may call Run on the same Session simultaneously.
// Each Tensor, SessionOptions, and other handle must not be used concurrently
// unless documented otherwise.
package onnxruntime
