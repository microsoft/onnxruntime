### Use "webgpu" other than "wgpu" in this folder

This is referring to the naming convention of variables, classes and namespace.

ORT C API is using "wgpu".

Let's keep it "webgpu" for this folder for now. I have a very good reason to do so:

- search for "webgpu" in the code base shows the WebGPU EP related code and search for "wgpu" shows the WebGPU API related code. This helps me easier to find the code I want to look at.

And anyway, it's not hard to change it back to "wgpu" if we want to. (but it's harder to change it from "wgpu" to "webgpu")

### Use `OStringStream` defined in string_utils.h and macros defined in string_macros.h

Type `onnxruntime::webgpu::OStringStream` is a type alias of Abseil's OStringStream. It's a lightweight implementation
of `std::ostream`. It's recommended to use `OStringStream` instead of `std::ostringstream` in the code base.

The macros defined in `string_macros.h` are used to make coding easier:

```cpp
std::string MyFunction() {
  SS(code /* name of the string stream */, 2048 /* initial capacity */);

  code << "var my_var = ";

  // function call style string append. equivalent to:
  //
  // code << "vec4(" << type << ">(" << value1 << ", " << value2 << ", " << value3 << ", " << value4 << ")";
  //
  SS_APPEND(code, "vec4(", type, ">(", value1, ", ", value2, ", ", value3, ", ", value4, ")");

  return SS_GET(code); // return the string
}
```

### Use the subfolder for kernel implementation

Operator implementation source code need to be put under a subfolder like "math"/"nn"/"tensor".

See folder structure under onnxruntime/core/providers/cpu/ or onnxruntime/core/providers/cuda/ for examples.
