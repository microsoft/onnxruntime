# ONNX Runtime Objective-C API

ONNX Runtime provides an Objective-C API.

It can be used from Objective-C/C++ or Swift with a bridging header.

Starting in 1.30, `ORTSessionOptions` can register an application-side callback that supplies external EPContext data
during session initialization. The Objective-C API does not currently expose model compilation, so it exposes the
EPContext data read callback but not the write callback.
