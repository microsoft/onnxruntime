# ORT File Format
This directory contains [the ORT file format schema](ort.fbs) and [the generated C++ header file](ort.fbs.h) for ORT file format.

[The ORT file format schema](ort.fbs) uses [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not modify [the generated header file](ort.fbs.h).

To update the ORT file format
1. Modify the [the ORT file format schema](ort.fbs)
2. Use the FlatBuffers compiler to generate [the C++ header file](ort.fbs.h),
    ```
    flatc --cpp --filename-suffix '.fbs' ort.fbs
    ```

For more information about FlatBuffers, please see https://github.com/google/flatbuffers


