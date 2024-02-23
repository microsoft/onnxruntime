# ORT Flatbuffer Schemas
This directory contains [the ORT file format schema](ort.fbs) and [the generated C++ header file](ort.fbs.h) for the
ORT file format.
This directory also contains [the on-device training checkpoint schema](ort_training_checkpoint.fbs) and
[the corresponding auto generated C++ header file](ort_training_checkpoint.fbs.h) for the checkpoint format.

[The ORT file format schema](ort.fbs) and [the on-device training checkpoint schema](ort_training_checkpoint.fbs)
uses the [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify the generated C++ header file for [the ORT file format]((ort.fbs.h))
or for [the training checkpoint file format](ort_training_checkpoint.fbs.h),
or [the generated Python binding files](../ort_flatbuffers_py).

The flatbuffers compiler (flatc) is built as part of an ONNX Runtime build. It is located in the _deps/flatbuffers-build/
subdirectory of the build output directory.

e.g.
  - Windows Debug build
    - \build\Windows\Debug\_deps\flatbuffers-build\Debug\flatc.exe
  - Linux Debug build
    - /build/Linux/Debug/_deps/flatbuffers-build/flatc

It is possible to use another flatc as well, e.g., from a separate installation. Note that ONNX Runtime uses
FlatBuffers 1.12.

To update the flatbuffers schemas and generated files:
1. Modify [the ORT file format schema](ort.fbs) or [training checkpoint schema](ort_training_checkpoint.fbs).
2. Run [compile_schema.py](./compile_schema.py) to generate the C++ and Python bindings.

    ```
    python onnxruntime/core/flatbuffers/schema/compile_schema.py --flatc <path to flatc>
    ```
3. Update the version history and record the changes. Changes made to [the ORT file format schema](ort.fbs)
warrants not only updating the ort format version, but also the checkpoint version since the checkpoint schema
depends on the ort format schema.


# ORT FB format version history
In [ort_format_version.h](../ort_format_version.h), see `IsOrtModelVersionSupported()` for the supported versions and
`kOrtModelVersion` for the current version.

## Version 1
History begins.

Initial support for FlatBuffers that includes Model support. Graph support including Attributes, Tensors, Tensor
Sequences, Maps and Sequences. Constant initializers are also supported. Constant nodes are converted to constant
initializers in the ORT format.

## Version 2
Support for sparse initializers. Sparse intializers are stored within ORT FlatBuffers format, which includes sparse
initializers converted from a Constant node attribute.

## Version 3
Support for storing `graph_doc_string` field in Model (ORT FlatBuffers format).

## Version 4
Update kernel def hashing to not depend on ordering of type constraint types (NOT BACKWARDS COMPATIBLE).

## Version 5
Remove kernel def hashes and add KernelTypeStrResolver info to replace them (LIMITED BACKWARDS COMPATIBILITY).
The change to the ORT format itself is not backwards compatibility-breaking, but ORT provides limited backwards
compatibility for processing older models with missing KernelTypeStrResolver info.
The motivation for this update is to support additional execution providers with statically registered kernels.
The original approach of using kernel def hashes was not so extensible as it required the execution provider providing
hashes to be enabled at model conversion time.

## Version 6
Support for float 8 types. See [Float stored in 8 bits](https://onnx.ai/onnx/technical/float8.html)
for further details about their format and usage.

# Checkpoint format version history
In [checkpoint_version.h](../checkpoint_version.h), see `IsCheckpointVersionSupported()` for the supported versions and
`kCheckpointVersion` for the current version.

## Version 1
Initial support for the On-Device Training Checkpoint format.
The format includes support for the ModuleState (stores the module parameters), OptimizerGroups
(stores the optimizer states), and PropertyBag (stores custom user properties with support for int64, float and strings).
