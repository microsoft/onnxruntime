# Go release bundle

This platform-specific ZIP is a local Go module source distribution, not a Go
module proxy or registry artifact. It contains the first-party module under
`source/` and the matching CPU ONNX Runtime shared libraries built from this
source tree under `native/lib/`.

An ordinary consumer can use the unpacked module with a local replacement:

```text
require github.com/microsoft/onnxruntime/go v0.0.0
replace github.com/microsoft/onnxruntime/go => ../onnxruntime-go/source
```

Import `github.com/microsoft/onnxruntime/go/onnxruntime`, call
`onnxruntime.SetSharedLibraryPath` with the absolute `native/lib` directory,
then call `onnxruntime.Init`. The Go package dynamically loads ONNX Runtime; the
native directory is a runtime dependency and is not embedded into the Go
module.

The archive is intentionally CPU-only and single-platform. It does not imply
module-proxy publication, signing, or compatibility with a different ONNX
Runtime binary.

The portable `//:go_module_proxy_artifacts` target separately produces the
canonical file-proxy layout for `github.com/microsoft/onnxruntime/go`:
`go_proxy/github.com/microsoft/onnxruntime/go/@v/v${VERSION_NUMBER}.{mod,info,zip}`.
The action reads `VERSION_NUMBER`; no module version is duplicated in Bazel.
The ZIP root is `github.com/microsoft/onnxruntime/go@v${VERSION_NUMBER}/`,
includes source and legal files, and contains no native runtime.

These are genuine proxy-format files, not proof of publication to a public Go
proxy. Consumers still provide and select an ONNX Runtime shared library with
`SetSharedLibraryPath`; the module does not download a prebuilt runtime or
invoke a native build wrapper.
