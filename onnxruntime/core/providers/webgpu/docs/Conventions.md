### Use "webgpu" other than "wgpu" in this folder

This is referring to the naming convention of variables, classes and namespace.

ORT C API is using "wgpu".

Let's keep it "webgpu" for this folder for now. I have a very good reason to do so:

- search for "webgpu" in the code base shows the WebGPU EP related code and search for "wgpu" shows the WebGPU API related code. This helps me easier to find the code I want to look at.

And anyway, it's not hard to change it back to "wgpu" if we want to. (but it's harder to change it from "wgpu" to "webgpu")

### Use macros defined in shader_macros.h

Take `SS` as example. It's a macro defined in `shader_macros.h` and it's used to concatenate strings. It's just make the `std::ostream::operator<<` to be used in a function call style.

I prefer to use the macro because I feel like it's easier to read. Check the following code:

```cpp
ss << "vec4(" << type << ">(" << value1 << ", " << value2 << ", " << value3 << ", " << value4 << ")";
```

vs.

```cpp
SS("vec4<", type, ">(", value1, ", ", value2, ", ", value3, ", ", value4, ")");
```

### Use the subfolder for kernel implementation

Operator implementation source code need to be put under a subfolder like "math"/"nn"/"tensor".

See folder structure under onnxruntime/core/providers/cpu/ or onnxruntime/core/providers/cuda/ for examples.
