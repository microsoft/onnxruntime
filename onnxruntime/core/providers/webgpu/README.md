# WebGPU Execution Provider

This folder is for the WebGPU execution provider(WebGPU EP). Currently, WebGPU EP is working in progress.

## Build WebGPU EP

Just append `--use_webgpu --skip_tests` to the `build.bat`/`build.sh` command line.

NOTE: `--skip_tests` is required for now. All existing tests are for CPU EP anyway so no need to run them.

For linux, a few dependencies need to be installed:
```sh
apt-get install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libx11-dev libx11-xcb-dev
```

## Troubleshooting

TODO: add solutions to common problems.

## Development Guide

See [How to write WebGPU EP kernel](./How_to_Write_WebGPU_EP_Kernel.md) for more information.

## Convention

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

## Best Practices

### Always use std::ostringstream to generate shader code if possible

This helps to the performance of code generation.

For example:

```cpp
ss << "var " << name << " = " << value << ";\n";
```

is better than

```cpp
ss << ("var " + name + " = " + value + ";\n");
```

### Avoid creating template class for kernel using data type as template parameter.

This basically means that we should define class like this:

```cpp
class Abs : public WebGpuKernel {
    ...
};
```

instead of

```cpp

template <typename T>  // T is tensor element type
class Abs : public WebGpuKernel {
    ...
};
```

This is because we don't really read and use `Tensor::Data<T>()`. Tensor stores a handle to a WebGPU buffer but not a pointer to the data. Using template for data type only increases the binary size with no real benefit.

## TODO items

The following items are not yet implemented:

- [ ] Validation Switch (allows to change the behavior of whether perform specific validation checks)
- [ ] pushErrorScope/popErrorScope
- [ ] Graph Capture
- [ ] Profiling supported by WebGPU Query Buffer
- [ ] WebGPU resources tracking (mainly for buffers)
- [ ] Global hanlders( unhandled exceptions and device lost )
