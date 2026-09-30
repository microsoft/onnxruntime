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

### Storage buffer access

Use `ShaderVariableHelper` for scalar/vector storage reads and writes, including
reads from writable outputs. The helpers select the backing storage, add a
buffer view's offset, and route accesses across storage segments. This applies
to both C++ shader generators and WGSL templates. Local variables and workgroup
arrays may be indexed directly.

`AddInput` and `AddOutput` register logical names; physical storage bindings use
internal `storage_<logical_name>` names. Raw expressions such as `input[i]`,
`output[i] = value`, or `&input` therefore fail WGSL compilation. Use the helpers
instead of constructing or referring to internal names. Uniforms may still be
accessed directly.

Existing atomic and subgroup-matrix paths temporarily use the internal names
directly, retaining their existing pointer calculations and view limitations.
The source-policy test records the exact files, names, and occurrence counts for
these exceptions. New direct references or operator storage declarations fail
that check. Pointer-aware helpers replace these exceptions in a separate change.

Buffer-access tests verify that raw logical reads, writes, and pointers fail
shader compilation while helper access and local/workgroup indexing still work.

```wgsl
#use .getByOffset .setByOffset

let value = input.getByOffset(index);
output.setByOffset(index, value);
```

Pass `true` as the final argument when the operation must preserve the packed
storage representation, such as both words of an int64 value. Otherwise the
helpers perform their normal logical-value conversion.

Retain the helpers returned by `AddInput` and `AddOutput` and pass them to the
template explicitly. Use `WGSL_TEMPLATE_VARIABLE` for required bindings and
`WGSL_TEMPLATE_OPTIONAL_VARIABLE` for nullable pointers to conditional bindings.
A corresponding template condition must guard every use of an optional variable.

### Generator CLI

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

The Python tool ships with a unit + fixture test suite.

You can run the suite directly from the source tree:

```
python tools/python/wgsl_template/test/run_tests.py
```

Tests cover the loader, parser, generator, build orchestrator, and a smoke test
against the in-tree templates (Pad, Transpose, im2col-matmul). The fixtures live
under
[`tools/python/wgsl_template/test/testcases/`](../../../../../tools/python/wgsl_template/test/testcases).
