---
title: Adapter file spec
description: Specification for the .onnx_adapter file format
has_children: false
parent: Reference
grand_parent: Generate API (Preview)
nav_order: 2
---

# Adapter file specification


## File format

The adapter file format is flatbuffers

## File extension

The file extension is ".onnx_adapter"

## Schema

Link to live [schema definition](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/lora/adapter_format/adapter_schema.fbs).

The schema definition is as follows

```
File:= 
  format_version := integer
  adapter_version := integer
  model_version := integer
  [parameter := Parameter]
```

```
Parameter:=
  name := string
  dimensions := [int64]
  data_type := TensorDataType
  [data := uint8] 
```

```
TensorDataType:= 
  UNDEFINED = 0 |
  FLOAT = 1 |
  UINT8 = 2 |
  INT8 = 3 |
  UINT16 = 4 |
  INT16 = 5 |
  INT32 = 6 |
  INT64 = 7 |
  STRING = 8 |
  BOOL = 9 | 
  FLOAT16 = 10 |
  DOUBLE = 11 |
  UINT32 = 12 |
  UINT64 = 13 |
  COMPLEX64 = 14 |
  COMPLEX128 = 15 |
  BFLOAT16 = 16 |
  FLOAT8E4M3FN = 17 |
  FLOAT8E4M3FNUZ = 18 |
  FLOAT8E5M2 = 19 |
  FLOAT8E5M2FNUZ = 20
```