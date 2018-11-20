# ONNXRuntime ABI

We release ONNXRuntime as both static library and shared library on Windows, Linux and Mac OS X. [ABI (Application Binary Interface)](https://en.wikipedia.org/wiki/Application_binary_interface) is only for the shared library. It allows you upgrade ONNXRuntime to a newer version without recompiling.

The ABI contains:

1. A set of C functions for creating an inference session and load/run a model. No global variables would be directly exported. All these functions can be directly used in .net and .net core through PInvoke. All these public symbols are C symbols, no C++.
2. (TODO) A C++ API for authoring custom ops and put them into a separated dll.

# Functionality
[C API](C_API.md)

# Integration
Q: Should I statically link to ONNXRuntime or dynamically?
A: On Windows, Any custom op DLL must dynamically link to ONNXRuntime.
Dynamical linking also helps on solving diamond dependency problem. For example, if part of your program depends on ONNX 1.2 but ONNXRuntime depends on ONNX 1.3, then dynamically linking to them would be better.

Q: Any requirement on CUDA version? My program depends on CUDA 9.0, but the ONNXRuntime binary was built with CUDA 9.1. Is it ok to put them together?
A: Yes. Because ONNXRuntime statically linked to CUDA.

# Dev Notes

## Global Variables
Global variables may get constructed or destructed inside "DllMain". There are significant limits on what you can safely do in a DLL entry point. See ['DLL General Best Practices'](https://docs.microsoft.com/en-us/windows/desktop/dlls/dynamic-link-library-best-practices). For example, you can't put a ONNXRuntime InferenceSession into a global variable.

## Component Object Model (COM)
ONNXRuntime doesn't contain a COM interface, whether it's on Windows or Linux. Because .Net Core doesn't support COM on Linux and we need to make ONNXRuntime available to .Net Core.

## Undefined symbols in a shared library
On Windows, you can't build a DLL with undefined symbols. Every symbol must be get resolved at link time. On Linux, you can.

In this project, we setup a rule: when building a shared library, every symbol must get resolved at link time, unless it's a custom op.

For custom op, on Linux, don't pass any libraries(except libc, pthreads) to linker. So that, even the application is statically linked to ONNXRuntime, they can still use the same custom op binary.


## Default visibility
On POSIX systems, please always specify "-fvisibility=hidden" and "-fPIC" when compiling any code in ONNXRuntime shared library.

 See [pybind11 FAQ](https://github.com/pybind/pybind11/blob/master/docs/faq.rst#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclassmember--wattributes)


## RTLD_LOCAL vs RTLD_GLOBAL
RTLD_LOCAL and RTLD_GLOBAL are two flags of [dlopen(3)](http://pubs.opengroup.org/onlinepubs/9699919799/functions/dlopen.html) function on Linux. By default, it's RTLD_LOCAL. And basically you can say, there no corresponding things like RTLD_GLOBAL on Windows.

If your application is a shared library, which statically linked to ONNXRuntime, and your application needs to dynamically load a custom op, then your application must be loaded with RTLD_GLOBAL. In all other cases, you should use RTLD_LOCAL. ONNXRuntime python binding is a good example of why sometimes RTLD_GLOBAL is needed.
