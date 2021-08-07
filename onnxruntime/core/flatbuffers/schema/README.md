# ORT File Format
This directory contains [the ORT file format schema](ort.fbs) and [the generated C++ header file](ort.fbs.h) for ORT file format.

[The ORT file format schema](ort.fbs) uses [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify [the generated header file](ort.fbs.h).

The flatbuffers compiler is built as part of an ONNX Runtime build. It is located in the external/flatbuffers subdirectory of the build output directory. 

e.g. 
  - Windows Debug build
    - \build\Windows\Debug\external\flatbuffers\Debug\flatc.exe
  - Linux Debug build
    - /build/Linux/external/flatbuffers/Debug/flatc

To update the ORT file format schema and generated files:
1. Modify the [the ORT file format schema](ort.fbs)
2. Use the FlatBuffers compiler to generate [the C++ header file](ort.fbs.h).
  - Change to the directory containing this file (onnxruntime/core/flatbuffers) and run as follows. Adjust paths depending on the build configuration used to refer to the flatc[.exe] binary. 
  e.g. 
    Windows
    `> ..\..\..\build\Windows\Debug\external\flatbuffers\Debug\flatc.exe --cpp --scoped-enums --filename-suffix .fbs ort.fbs`
    Linux
    `> ../../../build/Linux/Debug/external/flatbuffers/flatc --cpp --scoped-enums --filename-suffix .fbs ort.fbs`

  Verify that this results in ort.fbs.h being updated.

3. Run onnxruntime/core/flatbuffers/create_python_bindings.py to update the python bindings. Provide the path to flatc as input. 

# ORT FB format version history
`See onnxruntime/core/session/inference_session.cc:IsOrtModelVersionSupported()` for version array and `kOrtModelVersion` for currently supported version.

## Version 1. History begins
Initial support for FlatBuffers that includes Model support. Graph support including Attributes, Tensors, Tensor Sequences, Maps and Sequences. Constant initializers are also supported. Constant nodes are converted to constant initializers in the ORT format.

## Version 2. 
Support for sparse initializers. Sparse intializers are stored within ORT FlatBuffers format, which includes sparse initializers converted from a Constant node attribute.

## Version 3. 
Support for storing `graph_doc_string` field in Model (ORT FlatBuffers format).

## Version 4.
Update kernel def hashing to not depend on ordering of type constraint types (NOT BACKWARDS COMPATIBLE).
