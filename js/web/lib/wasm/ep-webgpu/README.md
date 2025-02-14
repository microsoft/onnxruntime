This folder "ep-webgpu" contains required TypeScript implementation for WebGPU EP support.

"ep-webgpu" here contains "ep" in the name, to distinguish it from other the WebGPU implementation in the JSEP folder:

- WebGPU EP is a C++ implementation. It uses the WebGPU C/C++ API provided by Dawn in the code. Emscripten will compile
  the C++ code to WebAssembly, and add internal JavaScript code to make it work in the browser.
- JSEP (JavaScript Execution Provider) is a hybrid implementation. It contains both JavaScript and C++ code, including
  the interop between them. It uses the WebGPU JavaScript API provided by the browser.

For WebGPU backend, when build definition `BUILD_DEFS.USE_WEBGPU_EP` is `true`, it is considered the WebGPU EP will be
used. Otherwise, the JSEP will be used.
