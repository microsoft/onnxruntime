`onnxruntime_runtime_path_test_shared_library` is a minimal shared library that exposes a public function which makes a
call to `onnxruntime::Env::GetRuntimePath()` and returns the result.

We want to ensure that `Env::GetRuntimePath()` returns the directory of the shared library it is called from, e.g.,
the onnxruntime shared library. It is used to get paths to other files which may be co-located with the shared library.

Directly calling `Env::GetRuntimePath()` from a unit test program where it is statically linked in is a slightly
different setup. We use this minimal shared library to test the `Env::GetRuntimePath()` functionality in a setup that
is closer to the real-world usage.
