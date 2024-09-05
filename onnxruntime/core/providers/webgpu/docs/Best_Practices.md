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
