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

### Template loops

Enable the `loop` block utility to repeat template content during C++ shader
generation. It can repeat kernel blocks or generate repeated declarations,
assignments, and helper calls. Kernel loop unrolling is another usage:

```wgsl
#use loop
#param TILE_K

$MAIN {
  var acc = 0.0;
  loop(k, 0, TILE_K) {
    acc += values[k];
  }
}
```

`loop(index, begin, end[, step]) {` uses an exclusive end and defaults to a
step of one. With `TILE_K = 4`, the shader contains four assignments using
literal indices `0`, `1`, `2`, and `3`. The generated C++ contains a loop;
the emitted WGSL contains the expanded body.

For example, repeat a kernel block to process multiple elements per invocation
(assuming the kernel defines the input/output bindings and size uniform):

```wgsl
#use loop
#param ELEMENTS_PER_INVOCATION

$MAIN {
  loop(block, 0, ELEMENTS_PER_INVOCATION) {
    {
      let index = global_idx * u32(ELEMENTS_PER_INVOCATION) + u32(block);
      if (index < uniforms.output_size) {
        let value = input[index];
        output[index] = max(value, 0.0);
      }
    }
  }
}
```

With `ELEMENTS_PER_INVOCATION = 4`, generation repeats the entire block four
times with distinct literal block indices. The explicit inner block gives
each copy its own scope, so
`index` and `value` can be declared in every copy. Configure dispatch for
`ceil(output_size / ELEMENTS_PER_INVOCATION)` logical invocations.

Nested template loops can repeat a two-dimensional kernel block. For example,
this fragment copies a tile using kernel-defined `input_base`, `output_base`,
and stride uniforms (assuming the entire tile is within the buffers):

```wgsl
#use loop
#param TILE_ROWS TILE_COLS

loop(row, 0, TILE_ROWS) {
  loop(col, 0, TILE_COLS) {
    {
      let src = input_base + u32(row) * uniforms.input_stride + u32(col);
      let dst = output_base + u32(row) * uniforms.output_stride + u32(col);
      output[dst] = input[src];
    }
  }
}
```

Place this fragment inside `$MAIN`. With `TILE_ROWS = 2` and `TILE_COLS = 3`,
generation emits six scoped copies of the inner block, using `(row, col)`
indices `(0, 0)` through `(1, 2)`. Inner loop bounds can also reference an
outer loop index, such as `loop(col, row, TILE_COLS)`.

- Put the header on its own line, with the opening brace on that line.
- Bounds accept unsuffixed decimal integers, `#param` values, outer loop
  indices, and enabled integer metadata properties such as `output.rank`.
  Parentheses, addition, and subtraction are supported. Shader uniforms
  cannot supply bounds because their values are unavailable during shader generation.
- Bounds must fit in signed int32 and the step must be positive. Invalid
  values return an error from `ApplyTemplate`. Empty ranges emit no body.
- Indices work in shader expressions and helper arguments, including
  `output.indicesGet(indices, k)`. Nested loops and `#if` inside loops are supported.
- The braces belonging to `loop(...)` are template syntax and are not emitted
  into WGSL. Add an explicit block inside the body when each copy needs its
  own scope, for example when repeating local declarations. Without that
  block, repeated statements share the surrounding shader scope.
- Do not redeclare an active index
  or reuse a parameter or shader variable name as an index. `break` and
  `continue` anywhere inside template loops are unsupported in this version.
- Keep each loop body balanced, including conditional directives. Ordinary
  WGSL `for (...)` and `loop { ... }` retain their shader-side behavior.

Include parameters that affect expansion in the program's shader cache key.
When using template loops to unroll kernel loops, measure GPU performance:
larger shader bodies can increase compilation time and register pressure.

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
