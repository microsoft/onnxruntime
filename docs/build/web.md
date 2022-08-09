---
title: Build for web
parent: Build ONNX Runtime
description: Learn how to build ONNX Runtime from source to deploy on the web
nav_order: 4
redirect_from: /docs/how-to/build/web
---

# Build ONNX Runtime for Web
{: .no_toc }

There are 2 steps to build ONNX Runtime Web:

- Obtaining ONNX Runtime WebAssembly artifacts - can be done by - 
  - Building ONNX Runtime for WebAssembly 
  - Download the pre-built artifacts [instructions below](#prepare-onnx-runtime-webassembly-artifacts)
- Build onnxruntime-web (NPM package)
  - This step requires the ONNX Runtime WebAssembly artifacts

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build ONNX Runtime Webassembly artifacts

### Prerequisites

- Checkout the source tree:
  ```
  git clone --recursive https://github.com/Microsoft/onnxruntime
  cd onnxruntime
  ```
- [Install](https://cmake.org/download/) cmake-3.18 or higher.

- [Install](https://nodejs.org/) Node.js (16.0+)

  - (Optional) Use nvm ([Windows](https://github.com/coreybutler/nvm-windows) / [Mac/Linux](https://github.com/creationix/nvm)) to install Node.js

- Python (3.8+): https://www.python.org/downloads/
  - python should be added to the PATH environment variable

- Ninja: https://ninja-build.org/
  ```sh
  pip install ninja
  ```

- Prepare emsdk:
  emsdk should be automatically installed at `<ORT_ROOT>/cmake/external/emsdk/emsdk`. If the folder structure does not exist, run the following commands in `<ORT_ROOT>/` to install git submodules:
  ```sh
  git submodule sync --recursive
  git submodule update --init --recursive
  ```

  (If you are using Windows, you can skip this step) in `<ORT_ROOT>/cmake/external/emsdk/`, run the following commands to setup emsdk:
  ```sh
  ./emsdk install latest
  ./emsdk activate latest
  source ./emsdk_env.sh
  ```

### Build Instructions

ONNX Runtime WebAssembly can be built with or without multi-thread and Single Instruction Multiple Data (SIMD) support.
This support is added/removed by appending the following flags to the build command, the default build option is without.

| build flag              | usage                           |
| ----------------------- | ------------------------------- |
| `--enable_wasm_threads` | build with multi-thread support |
| `--enable_wasm_simd`    | build with SIMD support         |

A complete build for ONNX runtime WebAssembly artifacts will contain 4 ".wasm" files (ON/OFF configurations of the flags in the table above) with a few ".js" files.
The build command below should be run for each of the configurations.

in `<ORT_ROOT>/`, run one of the following commands to build WebAssembly:

```sh
# In windows, use 'build' to replace './build.sh'
# It's recommended to use '--skip_tests` in Release & Debug + 'debug info' configruations - please review FAQ for more details

# The following command build debug.
./build.sh --build_wasm

# The following command build debug with debug info.
./build.sh --build_wasm --skip_tests --enable_wasm_debug_info

# The following command build release.
./build.sh --config Release --build_wasm --skip_tests --disable_wasm_exception_catching --disable_rtti
```

A full list of required build artifacts:

| file name                   | `--enable_wasm_threads` | `--enable_wasm_simd` |
| --------------------------- | ----------------------- | -------------------- |
| ort-wasm.js                 | X                       | X                    |
| ort-wasm.wasm               | X                       | X                    |
| ort-wasm-threaded.js        | O                       | X                    |
| ort-wasm-threaded.wasm      | O                       | X                    |
| ort-wasm-threaded.worker.js | O                       | X                    |
| ort-wasm-simd.wasm          | X                       | O                    |
| ort-wasm-simd-threaded.wasm | O                       | O                    |

### Minimal Build Support

ONNX Runtime WebAssembly can be built with flag `--minimal_build`. This will generate smaller artifacts and also have a less runtime memory usage. 
In order to use this ONNX Runtime confiruation an ORT format model is required (vs. ONNX format). 
For more info please see also [ORT format Conversion](../reference/ort-format-models.md).

### FAQ

Q: unittest fails on Release build.

> A: unittest requires C++ exceptions to work properly. However, for performance concern, we disabled exception catching in emscripten. So please specify `--skip_tests` in Release build.

Q: unittest fails on Debug build with debug info.

> A: building with debug info will generate very huge artifacts (>1GB for unittest) and failed to load in Node.js. So please specify `--skip_tests` in build with debug info.

Q: I have a C++ project for web scenario, which runs a ML model using ONNX Runtime and generates WebAssembly as an output. Does ONNX Runtime Web support a static WebAssembly library, so that my application can link with it and make all pre/post processors to be compiled together into WebAssembly?

> A: With `--build_wasm`, a build script generates `.wasm` and `.js` files for web scenarios and intermediate libraries are not linked properly with other C/C++ projects. When you build ONNX Runtime Web using `--build_was_static_lib` instead of `--build_wasm`, a build script generates a static library of ONNX Runtime Web named `libonnxruntime_webassembly.a` in output directory. To run a simple inferencing like an [unit test](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/wasm/test_inference.cc), what you need is three header files as follows and `libonnxruntime_webassembly.a`.

- include/onnxruntime/core/session/onnxruntime_c_api.h
- include/onnxruntime/core/session/onnxruntime_cxx_api.h
- include/onnxruntime/core/session/onnxruntime_cxx_inline.h

> One important note is that ONNX Runtime has dependencies on lots of 3rd party libraries such as protobuf, onnx, and others. You may need copy necessary header files to your project. You also take care of cases of library version conflicts or emsdk version conflicts between ONNX Runtime and your project.

## Build onnxruntime-web - NPM package

The following sections are a step by step installation guide for onnxruntime-web NPM packages.
This is the last stage in the build process, please follow the sections in a sequential order. 

### Prerequisites

- [Install](https://nodejs.org/) Node.js (16.0+)

  - (Optional) Use nvm ([Windows](https://github.com/coreybutler/nvm-windows)/[Mac/Linux](https://github.com/creationix/nvm)) to install Node.js

- Chrome or Edge browser for running tests.


### Install NPM packages

   1. in `<ORT_ROOT>/js/`, run `npm ci`.
   2. in `<ORT_ROOT>/js/common/`, run `npm ci`.
   3. in `<ORT_ROOT>/js/web/`, run `npm ci`.


### Prepare ONNX Runtime WebAssembly artifacts

   You can either use the prebuilt artifacts or build it by yourself.

   - Setup by script.

     In `<ORT_ROOT>/js/web/`, run `npm run pull:wasm` to pull WebAssembly artifacts for latest master branch from CI pipeline.

   - Download artifacts from pipeline manually.

     you can download prebuilt WebAssembly artifacts from [Windows WebAssembly CI Pipeline](https://dev.azure.com/onnxruntime/onnxruntime/_build?definitionId=161&_a=summary). Select a build, download artifact "Release_wasm" and unzip. See instructions below to put files into destination folders.

   - Build WebAssembly artifacts.

     1. Build ONNX Runtime WebAssembly

        Follow [instructions above](#build-onnx-runtime-webassembly-artifacts) for building ONNX Runtime WebAssembly.

     2. Copy following files from build output folder to `<ORT_ROOT>/js/web/dist/` (create the folder if it does not exist):

         * ort-wasm.wasm
         * ort-wasm-threaded.wasm (build with flag '--enable_wasm_threads')
         * ort-wasm-simd.wasm (build with flag '--enable_wasm_simd')
         * ort-wasm-simd-threaded.wasm (build with flags '--enable_wasm_threads --enable_wasm_simd')

     3. Copy following files from build output folder to `<ORT_ROOT>/js/web/lib/wasm/binding/`:

         * ort-wasm.js
         * ort-wasm-threaded.js (build with flag '--enable_wasm_threads')
         * ort-wasm-threaded.worker.js (build with flag '--enable_wasm_threads')

Use following command in folder `<ORT_ROOT>/js/web` to build:
   ```
   npm run build
   ```
