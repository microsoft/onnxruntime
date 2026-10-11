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
// Plugin registration, discovery, V2 selection and device policies require API 22.
// Deterministic compute requires API 17; I/O binding synchronization requires API 10.
// Weightless source model buffers require API 29. The binding copies source bytes
// into native storage and retains them through clones and sessions until Close.
// Buffer sessions require successful explicit V2 selection. Automatic policy
// selection with a buffer is rejected on all accepted runtimes because native
// policy selection omits the buffer. Fresh options reset selection state.
// Public clones and private constructor snapshots restore the native buffer.
// EPContext data readers require API 31. Each payload copy is bounded by maxBytes;
// readers must bound their own I/O and allocations and handle concurrent calls.
// Reader callbacks run without package or public options locks. They may call
// options getters, Clone, setters, Clear and Close, or wait for those calls in
// another goroutine. They must not recursively construct sessions, call Shutdown,
// mutate the registry or wait for the current construction to finish. Logical
// names are UTF-8 identifiers; the binding supplies no filesystem fallback.
// Linux CPU race tests cover ORT 1.27.0/API 27, 1.28.3/API 28,
// 1.29.1/API 29, 1.30.0/API 30, and 1.31.0/API 31.
// Opt-in plugin tests require a selected-provider compute event in the native
// profile. WebGPU 0.4.0 with ORT 1.31.0 covers standard Mul only. The test
// proves execution on the plugin EP, not which adapter the plugin selected.
// The example plugin EPContext test verifies loading through a reader, not
// weightless initializer restoration or EPContext inference.
// Capped builds on a newer runtime do not verify an older runtime's ABI.
//
// ErrorCodeDeviceReset requires releasing the old session and creating another.
// The binding does not retry inference automatically.
//
// A Session wraps an ORT inference session. It is safe for concurrent
// use: multiple goroutines may call Run on the same Session simultaneously.
// Each Tensor, SessionOptions, and other handle must not be used concurrently
// unless documented otherwise.
//
// Register and unregister plugin libraries during setup or teardown without
// concurrent inference. Close outputs, inputs, I/O bindings, sessions and
// configured options before unregistering. Unregister and Shutdown reject live
// sessions, protected options and session constructors. These guards do not
// track independently owned tensors and do not authorize unload with live outputs.
// Registry mutation attempts invalidate all device selection tokens, even on
// native failure. Device attributes remain readable and map accessors return copies.
//
// V2 append and policy attempts protect options until Close, even on native
// failure. Clones retain that protection. Constructors take a private options
// snapshot under the source lock, then release the lock before native creation.
// Later setters affect later constructors. Registry mutations and Shutdown stay
// blocked until creation and introspection finish or unwind.
package onnxruntime
