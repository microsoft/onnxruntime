## Global Variables
Global variables may get constructed or destructed inside "DllMain". There are significant limits on what you can safely do in a DLL entry point. See ['DLL General Best Practices'](https://docs.microsoft.com/en-us/windows/desktop/dlls/dynamic-link-library-best-practices). For example, you can't put a ONNX Runtime InferenceSession into a global variable because it has a thread pool inside.

## Thread Local variables
Onnxruntime must support explicit linking, where the operating system loads the DLL on demand at runtime, instead of process startup time. This is required by our language bindings like C#/Java.

However, there are some special restrictions on this, If a thread local variable need non-trivial construction, for the threads already exist before onnxruntime.dll is loaded, the variable won't get initialized correctly. So it's better to only access such variables from onnxruntime internal threads, or make these variables function local (Like the magic statics).


## No undefined symbols
On Windows, you can't build a DLL with undefined symbols. Every symbol must be get resolved at link time. On Linux, you can.
In order to simplify things, we require every symbol must get resolved at link time. The same rule applies for all the platforms. And this is easier for us to control symbol visibility.


## Default visibility and how to export a symbol
On Linux, by default, at linker's view, every symbol is global. It's easy to use but it's also much easier to cause conflicts and core dumps. We have encountered too many such problems in ONNX python binding. Indeed, if you have a well design, for each shared lib, you only need to export **one** function. ONNX Runtime python binding is a good example. See [pybind11 FAQ](https://github.com/pybind/pybind11/blob/master/docs/faq.rst#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclassmember--wattributes) for more info.

For controlling the visibility, we use linker version scripts on Linux and def files on Windows. They work similar. That:
1. Only C functions can be exported.
2. All the function names must be explicitly listed in a text file.
3. Don't export any C++ class/struct, or global variable.

Also, on Linux and Mac operating systems, all the code must be compiled with "-fPIC".
On Windows, we don't use dllexport but we still need dllimport.

Therefore, our DLLEXPORT macro is like:
```
#ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
#ifdef ORT_DLL_IMPORT
#define ORT_EXPORT __declspec(dllimport)
#else
#define ORT_EXPORT
#endif
#else
#define ORT_EXPORT
#endif
```

## static initialization order problem
It's well known C++ has [static initialization order problem](https://isocpp.org/wiki/faq/ctors#static-init-order). Dynamic linking can ensure that onnxruntime's static variables are already initialized before any onnxruntime's C API get called. The same thing applies to their destructors. It's good. But on the other side, static linking may have more usage restrictions on some of the APIs.
