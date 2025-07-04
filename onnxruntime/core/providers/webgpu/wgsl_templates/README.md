# WebGPU WGSL Templates

This directory contains the infrastructure, scripts, and documentation for the WGSL template system used by the ONNX Runtime WebGPU Execution Provider (EP). The template system enables the generation of optimized WGSL shaders at build time or runtime, with parameterization and reusability across different operators.

For detailed information about the underlying template engine, see [wgsl-template](https://github.com/fs-eire/wgsl-template).

## Overview

The WGSL template system provides a flexible framework for generating WebGPU shaders for ONNX Runtime operators. Instead of writing static shader code, developers can create parameterized templates that adapt to different input configurations, data types, and optimization requirements.

**Key Benefits:**

- **Code Reusability**: Share common shader patterns across multiple operators
- **Type Safety**: Template parameters are validated at build time (static mode)
- **Performance**: Generated shaders are optimized for specific use cases
- **Maintainability**: Centralized shader logic with clear parameterization

## Terms

- **WGSL**: The **W**eb**G**PU **S**hader **L**anguage - the shading language for WebGPU
- **WGSL Template File**: A file with `.wgsl.template` extension containing WGSL shader code with template syntax and utilities
- **WGSL-Template**: A JavaScript library developed specifically for generating WGSL code from template files
- **Template Parameters**: Configuration objects that control shader generation (data types, dimensions, etc.)

## How It Works

The system supports two generation modes to balance performance and flexibility:

### Static Generation Mode

Templates are processed at **build time** to generate C++ header files embedded in the WebGPU EP binary.

**Advantages:**

- Zero runtime overhead for template processing
- Smaller binary size (no JavaScript engine required)
- Type-safe template parameters validated at compile time
- Optimal for production deployments

**Use Cases:**

- Production builds

### Dynamic Generation Mode

Templates are processed at **runtime** using an embedded JavaScript engine to generate shaders on-demand.

**Advantages:**

- Faster development iteration (no rebuild required)
- Support for runtime-dependent parameters
- Easier debugging and experimentation
- Flexible for research and development

**Use Cases:**

- Development and debugging workflows
- Research environments with frequently changing parameters

## Development

This section includes instructions for how to use the template system in the development.

1. Create WGSL template files in `.wgsl.template` extension.

   - [Reference: Template Syntax](https://github.com/fs-eire/wgsl-template?tab=readme-ov-file#template-syntax)
   - [Reference: Built-in Utilities](#Utilities)
   - [Example: Pad](../tensor/pad.wgsl.template)

2. In the implementation of `YourProgram::GenerateShaderCode()`, load and use the generated template files.

   - [Example: Pad](../tensor/pad.cc)

3. Build.

   - Using static code generator

     Static code generator is enabled by default:

     ```sh
     ./build.sh --use_webgpu
     ```

     Rebuild is needed when any C/C++ source file or WGSL template file is updated.

   - Using dynamic code generator

     Append `--wgsl_template dynamic` to the ORT build script:

     ```sh
     ./build.sh --use_webgpu --wgsl_template dynamic
     ```

     Rebuild is not needed when only WGSL template files are updated. In this case, you just need to compile the template files. See the next section below.

4. Compile the template files. (for dynamic code generator only)

   When you are using dynamic code generator, you don't need to rebuild ONNX Runtime when you made changes to the WGSL template files.

   > Suppose `<BUILD_DIR>` is the build directory. It's usually something like `<ORT_REPO_ROOT>/build/Linux/Debug` if not explicitly specified.

   There are 2 ways to compile the template files:

   1. Use a NPM script in the current folder:

      ```sh
      npm run gen -- -i ../ --preserve-code-ref --verbose --generator dynamic --output <BUILD_DIR>/wgsl_generated/dynamic
      ```

      This script will generate the file `<BUILD_DIR>/wgsl_generated/dynamic/templates.js` and exit.

   2. Use the same script but in watch mode (recommended):

      Use the same script as above but append `--watch` flag. This will launch a service monitoring the file system change and automatically compile the templates if any change occurred.

      The typical development workflow is:
      1. Build ORT once with dynamic template mode
      2. Launch wgsl-gen in watch mode
      3. Run ORT to debug/validate the shader
      4. Make changes to the template files, and repeat step (3)
