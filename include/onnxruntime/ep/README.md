## EP adapter

This folder contains a set of C++ header files. They are used specifically for allowing ONNX Runtime internal kernel-based EPs to use the plugin-style EP API while keep minimal changes to existing code.

### Folder Structure

There are 2 types of header files:

- General header files for plugin EP. This may include utilities, macros and shared routines that depending on ONNX Runtime public API only. There are multiple places for header files of this category (which we are going to unify them to one place. There is an ongoing discussion about unifying shared headers for plugin EPs):
  - `include/onnxruntime/ep/` (#26919)
  - `onnxruntime/test/autoep/library/plugin_ep_utils.h`
  - `include/onnxruntime/core/providers/utils/` (#25753)

- Header files specifically used for supporting WebGPU EP and CUDA EP to use EP APIs. These header files do not only depend on ONNX Runtime public API, but also depend on ONNX Runtime internal headers. They define adapter classes that replace their compatible, internal ONNX Runtime equivalents.
  - `include/onnxruntime/ep/adapter/`

### Usage

Make sure to include "ep/adapters.h" to include all adapter implementation code. This file brings the adapter classes into the EP's namespace, so it should be included before other EP code that relies on the adapter classes. Using "ep/adapters.h" as a pre-compiled header is the recommended way to include it first.

`ep/adapters.h` has conflicts with shared provider. Shared provider should be disabled when using these adapters.
