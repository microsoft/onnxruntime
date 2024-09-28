# Lora Parameters Flatbuffer Schemas
This directory contains [ONNXRuntime Lora Parameter format schema](lora_schema.fbs) and [the generated C++ header file](lora_schema.fbs.h) for the
Lora Parameters file format. This file format is defined as a means to deliver Lora parameters so it can read by ONNXRuntime C++ code.

The format is generally designed to house a single Lora adapter with named Lora parameters.

[ONNXRuntime Lora Parameter file format schema](lora_schema.fbs) uses the [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify the generated C++ header file for [ONNXRuntime Lora Parameter file format]((lora_schema.fbs.h)).

Use flatc compiler for the purpose.

e.g.
  - Windows Debug build
    - \build\Windows\Debug\_deps\flatbuffers-build\Debug\flatc.exe
  - Linux Debug build
    - /build/Linux/Debug/_deps/flatbuffers-build/flatc

It is possible to use another flatc as well, e.g., from a separate installation.

To update the flatbuffers schemas and generated files:
1. Modify [ONNXRuntime Lora Parameter file format schema](lora_schema.fbs).
2. Run [compile_schema.py](./compile_schema.py) to generate the C++ bindings.

    ```
    python onnxruntime/lora/lora_format/compile_schema.py --flatc <path to flatc>
    ```
# Lora format version history
In [lora_format_version.h](../lora_format_version.h), see `IsLoraParameterslVersionSupported()` for the supported versions and
`kLoraParametersVersion` for the current version.

## Version 1
History begins.

Initial support for FlatBuffers that Lora Parameters support. This includes a definition of Tensor entity
so it can be saved in a tensor per file format.
