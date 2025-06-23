## WebGPU WGSL template

The current folder is used in ONNX Runtime WebGPU EP to build the WGSL template files into a set of generated C++ header files.

For more information, check [wgsl-template](https://github.com/fs-eire/wgsl-template).

### Development

1. Create WGSL template files in `.wgsl.template` extension.

   - [Reference: Template Syntax](https://github.com/fs-eire/wgsl-template?tab=readme-ov-file#template-syntax)
   - [Reference: Built-in Utilities](#Utilities)
   - [Example: Pad](../tensor/pad.wgsl.template)

2. In the implementation of `YourProgram::GenerateShaderCode()`, load and use the generated template files.

   - [Example: Pad](../tensor/pad.cc)
   - Detailed usage TBD

3. Build and test.

   Just append `--use_wgsl_template` to the ORT build script:

   ```sh
   ./build.sh --use_webgpu --use_wgsl_template
   ```

   The following build steps are only for when you just want to build/debug the template generation.

   (in current folder)

   1. run `npm ci`
   2. run `npm run gen -- -i ../ -o {output_dir} --generator static-cpp --clean --debug`

4. Debugging

   - The generated C++ source files are in `wgsl_generated/wgsl_template_gen` inside the build directory.
