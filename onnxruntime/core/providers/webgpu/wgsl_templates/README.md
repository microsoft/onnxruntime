## WebGPU WGSL template

The current folder is used in ONNX Runtime WebGPU EP to build the WGSL template files into a set of generated C++ header files.

For more information, check [wgsl-template](https://github.com/fs-eire/wgsl-template).

### Build steps

Just append `--use_wgsl_template` to the ORT build script:

```sh
./build.sh --use_webgpu --use_wgsl_template
```


The following build steps are only for when you just want to build/debug the template generation.

(in current folder)

1. run `npm ci`
2. run `npm run gen -- -i ../ -o {output_dir} --generator static-cpp --clean --debug`
