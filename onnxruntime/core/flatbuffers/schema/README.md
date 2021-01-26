# ORT File Format
This directory contains [the ORT file format schema](ort.fbs) and [the generated C++ header file](ort.fbs.h) for ORT file format.

[The ORT file format schema](ort.fbs) uses [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify [the generated header file](ort.fbs.h).

To update the ORT file format
1. Modify the [the ORT file format schema](ort.fbs)
2. Use the FlatBuffers compiler to generate [the C++ header file](ort.fbs.h).

The FlatBuffers compiler is not built by default.
On Windows, it can be built using the generated FlatBuffers.sln in the external/flatbuffers build output directory (e.g. build/Windows/Debug/external/flatbuffers/FlatBuffers.sln)
Change to the directory containing this file (onnxruntime/core/flatbuffers) and run as follows. Adjust paths depending on the build configuration used to create flatc.exe. The example was from a Debug build.

`> ..\..\..\build\Windows\Debug\external\flatbuffers\Debug\flatc.exe --cpp --scoped-enums --filename-suffix .fbs ort.fbs`

This should result in ort.fbs.h being updated.

# ORT FB format version history
`See onnxruntime/core/session/inference_session.cc:IsOrtModelVersionSupported()` for version array and `kOrtModelVersion` for currently supported version.

## Version 1. History begins
Initial support for FlatBuffers that includes Model support. Graph support including Attributes, Tensors, Tensor Sequences, Maps and Sequences. Constant initializers are also supported. Constant nodes are converted to constant initializers in the ORT format.

## Version 2. 
Support for sparse initialiers. Sparse intializers are stored within ORT FlatBuffers format, which includes sparse initializers converted from Constant node attribute.

## Version 3. 
Support for storing `graph_doc_string` field in Model (ORT FlatBuffers format).