## Global Variables
Global variables may get constructed or destructed inside "DllMain". There are significant limits on what you can safely do in a DLL entry point. See ['DLL General Best Practices'](https://docs.microsoft.com/en-us/windows/desktop/dlls/dynamic-link-library-best-practices). For example, you can't put a ONNX Runtime InferenceSession into a global variable.

## Thread Local variables
Thread Local variables must be function local, that on Windows they will be initialized as the first time of use. Otherwise, it may not work.
Also, you must destroy these thread Local variables before onnxruntime.dll is unloaded, if the variable has a non-trivial destructor. That means, only onnxruntime internal threads can access these variables. It is, the thread must be created by onnxruntime and destroyed by onnxruntime. 

## No undefined symbols
On Windows, you can't build a DLL with undefined symbols. Every symbol must be get resolved at link time. On Linux, you can.
In order to simplify things, we require every symbol must get resolved at link time. The same rule applies for all the platforms. And this is easier for us to control symbol visibility. 


## Default visibility and how to export a symbol
On Linux, by default, at linker's view, every symbol is global. It's easy to use but it's also much easier to cause conflicts and core dumps. We have encountered too many such problems in ONNX python binding. Indeed, if you have a well design, for each shared lib, you only need to export **one** function. ONNX Runtime python binding is a good example. See [pybind11 FAQ](https://github.com/pybind/pybind11/blob/master/docs/faq.rst#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclassmember--wattributes) for more info.

For controling the visibility, we use linkder version scripts on Linux and def files on Windows. They work similar. That:
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

## RTLD_LOCAL vs RTLD_GLOBAL
RTLD_LOCAL and RTLD_GLOBAL are two flags of [dlopen(3)](http://pubs.opengroup.org/onlinepubs/9699919799/functions/dlopen.html) function on POSIX systems. By default, it's RTLD_LOCAL. And basically you can say, there no corresponding things like RTLD_GLOBAL on Windows.

There is one case you need to use RTLD_GLOBAL on POSIX systems:
1. There is a shared lib which is dynamically loaded by some application(like python or dotnet)
2. The shared lib is statically linked to ONNX Runtime
3. The shared lib needs to dynamically load a custom op

Then the shared lib should be loaded with RTLD_GLOBAL, not RTLD_LOCAL. Otherwise in the custom op library, it can not find ONNX Runtime symbols.

