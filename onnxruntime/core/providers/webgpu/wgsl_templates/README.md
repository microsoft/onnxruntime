# WebGPU WGSL Templates

This directory contains the infrastructure and documentation for the WGSL
template system used by the ONNX Runtime WebGPU Execution Provider (EP). The
template system generates optimized WGSL shaders at build time, with
parameterization and reusability across different operators.

The template engine is implemented in Python and lives at
[`tools/python/wgsl_gen.py`](../../../../../tools/python/wgsl_gen.py) (with the
supporting package at
[`tools/python/wgsl_template/`](../../../../../tools/python/wgsl_template/)).
It requires only a Python 3.10+ interpreter.

## Overview

The WGSL template system provides a flexible framework for generating WebGPU
shaders for ONNX Runtime operators. Instead of writing static shader code,
developers can create parameterized templates that adapt to different input
configurations, data types, and optimization requirements.

**Key Benefits:**

- **Code Reusability**: Share common shader patterns across multiple operators
- **Type Safety**: Template parameters are validated at build time
- **Performance**: Generated shaders are optimized for specific use cases
- **Maintainability**: Centralized shader logic with clear parameterization

## Terms

- **WGSL**: The **W**eb**G**PU **S**hader **L**anguage - the shading language for WebGPU
- **WGSL Template File**: A file with the `.wgsl.template` extension containing WGSL shader code with template syntax and utilities
- **Template Parameters**: Configuration objects that control shader generation (data types, dimensions, etc.)

## How It Works

Templates are processed at **build time** to generate C++ header files embedded
in the WebGPU EP binary.

**Advantages:**

- Zero runtime overhead for template processing
- Smaller binary size (no JavaScript engine required)
- Type-safe template parameters validated at compile time
- Optimal for production deployments

CMake invokes the Python tool, which walks the `.wgsl.template` files and emits
`index.h` / `index_impl.h` (plus per-template headers) into the build directory.
The EP includes these generated headers via [`wgsl_gen.h`](wgsl_gen.h) /
[`wgsl_gen.cc`](wgsl_gen.cc).

## Development

This section describes how to use the template system during development.

1. Create WGSL template files with the `.wgsl.template` extension.

   - [Reference: Template Syntax](https://github.com/fs-eire/wgsl-template?tab=readme-ov-file#template-syntax)
   - [Reference: Built-in Utilities](https://github.com/fs-eire/wgsl-template?tab=readme-ov-file#Utilities)
   - [Example: Pad](../tensor/pad.wgsl.template)

2. In the implementation of `YourProgram::GenerateShaderCode()`, load and use the generated template files.

   - [Example: Pad](../tensor/pad.cc)

3. Build.

   The static code generator is always enabled when WebGPU is built:

   ```sh
   ./build.sh --use_webgpu
   ```

   A rebuild is needed when any C/C++ source file or WGSL template file is updated.

## Python tool reference

The build invokes [`tools/python/wgsl_gen.py`](../../../../../tools/python/wgsl_gen.py)
directly from CMake; you should not normally need to run it by hand. The CLI
surface is:

```
python tools/python/wgsl_gen.py \
    -i <source-dir> [-i <source-dir> ...] \
    --output <out-dir> \
    --generator {static-cpp|static-cpp-literal} \
    [-I <include-prefix>] \
    [--ext .wgsl.template] \
    [--preserve-code-ref] \
    [--clean] \
    [--verbose]
```

* `static-cpp` (Release): emits short `__str_N` identifiers backed by a `string_table.h` for shader-string deduplication.
* `static-cpp-literal` (Debug): inlines string literals; easier to read while debugging.

### Running the test suite

The Python tool ships with a unit + fixture test suite. CMake adds
`wgsl_template_python_tests` to `ctest`, so any standard ORT build with WGSL
templates enabled will run them:

```
ctest -R wgsl_template_python_tests
```

You can also run the suite directly from the source tree:

```
python tools/python/wgsl_template/test/run_tests.py
```

Tests cover the loader, parser, generator, build orchestrator, and a smoke test
against the in-tree templates (Pad, Transpose, im2col-matmul). The fixtures live
under
[`tools/python/wgsl_template/test/testcases/`](../../../../../tools/python/wgsl_template/test/testcases).
