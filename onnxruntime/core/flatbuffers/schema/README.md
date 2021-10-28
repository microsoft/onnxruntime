# ORT File Format
This directory contains [the ORT file format schema](ort.fbs) and [the generated C++ header file](ort.fbs.h) for the ORT file format.

[The ORT file format schema](ort.fbs) uses the [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify [the generated C++ header file](ort.fbs.h) or [the generated Python binding files](../ort_flatbuffers_py).

The flatbuffers compiler (flatc) is built as part of an ONNX Runtime build. It is located in the external/flatbuffers subdirectory of the build output directory.

e.g.
  - Windows Debug build
    - \build\Windows\Debug\external\flatbuffers\Debug\flatc.exe
  - Linux Debug build
    - /build/Linux/external/flatbuffers/Debug/flatc

It is possible to use another flatc as well, e.g., from a separate installation. Note that ONNX Runtime uses FlatBuffers 1.12.

To update the ORT file format schema and generated files:
1. Modify [the ORT file format schema](ort.fbs).
2. Run [compile_schema.py](./compile_schema.py) to generate the C++ and Python bindings.

    ```
    python onnxruntime/core/flatbuffers/schema/compile_schema.py --flatc <path to flatc>
    ```

# ORT FB format version history
In [ort_format_version.h](../ort_format_version.h), see `IsOrtModelVersionSupported()` for version array and `kOrtModelVersion` for currently supported version.

## Version 1. History begins
Initial support for FlatBuffers that includes Model support. Graph support including Attributes, Tensors, Tensor Sequences, Maps and Sequences. Constant initializers are also supported. Constant nodes are converted to constant initializers in the ORT format.

## Version 2.
Support for sparse initializers. Sparse intializers are stored within ORT FlatBuffers format, which includes sparse initializers converted from a Constant node attribute.

## Version 3.
Support for storing `graph_doc_string` field in Model (ORT FlatBuffers format).

## Version 4.
Update kernel def hashing to not depend on ordering of type constraint types (NOT BACKWARDS COMPATIBLE).

## Next version (TODO finalize changes for next version before 1.10 release).
Support for storing runtime optimizations in graph.
