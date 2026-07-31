# Compile API In-Memory External Initializers

## Goal

Support external initializer data without filesystem access through two separate capabilities:

- Allow `OrtCompileApi` to write an external initializer file to a caller-owned buffer when saving a compiled or
  optimized ONNX model.
- Allow an application to provide an external initializer file as a buffer when creating an inference session.

Together, these capabilities support models whose initializer data makes the complete model exceed the protobuf 2 GB
limit.

Compilation produces two buffers:

- the ONNX model, using the existing output-model buffer or write callback; and
- one logical external initializer file held in a caller-owned buffer.

The ONNX model records the caller-provided logical filename, offset, and length for every externalized initializer.
Multiple initializers share the same logical file and buffer.

## Compile API

Add `ModelCompilationOptions_SetOutputModelExternalInitializersBuffer` with:

- a relative UTF-8 logical filename to store in each initializer's `TensorProto`;
- the minimum initializer size to externalize;
- an `OrtAllocator`; and
- output pointers for the allocated buffer and its size.

```cpp
ModelCompilationOptions_SetOutputModelExternalInitializersBuffer(
  OrtModelCompilationOptions* options,
  const ORTCHAR_T* logical_file_name,
  size_t initializer_size_threshold,
  OrtAllocator* allocator,
  void** output_buffer,
  size_t* output_buffer_size);
```

Add the corresponding C++ wrapper and an `ExternalInitializerBufferHolder` alternative to
`epctx::ModelGenOptions::initializers_location`.

Add `ModelCompilationOptions_SetOutputModelExternalInitializersAlignment`, which accepts a power-of-two alignment and a
minimum initializer size at which to apply it. It affects both file and buffer output; an alignment of zero disables
this additional policy. This allows a buffer to be persisted later with mmap-friendly offsets.

```cpp
ModelCompilationOptions_SetOutputModelExternalInitializersAlignment(
  OrtModelCompilationOptions* options,
  size_t alignment,
  size_t minimum_size);
```

Store the alignment settings separately from the initializer destination in `ModelGenOptions`, then apply them when
constructing `ModelSavingOptions` for either file or buffer output. Setter call order does not matter.

On failure, ORT must free any temporary allocation and leave the caller's output pointer and size unchanged. On
success, the caller owns the buffer and releases it with the supplied allocator. If no initializer meets the threshold,
return a null buffer and size zero.

## Serialization

Refactor the existing external-initializer save path so its physical destination can be either a file stream or an
allocated memory buffer. Preserve the existing initializer traversal, threshold, external-data metadata, endian
conversion, subgraph handling, and prepacked-weight handling.

Use a two-pass implementation:

1. Compute each externalized initializer's offset and the total buffer size with checked arithmetic.
2. Allocate the exact size once, write each initializer to its assigned span, and emit its logical filename, offset,
   and length into the `TensorProto`.

Write externalized initializers in load-ready tensor storage and naturally align each tensor. Apply alignment configured
by `ModelCompilationOptions_SetOutputModelExternalInitializersAlignment` above its size threshold; the existing default
policy is mmap-friendly 4 KiB alignment for data larger than 1 MiB. Prepacked blobs are opaque and use only the
configured alignment policy.

The serialized ONNX protobuf must still be smaller than 2 GB. Externalizing initializer bytes keeps the protobuf small;
this feature does not support graph metadata that independently exceeds protobuf's limit.

## Loading

The existing `OrtApi::AddExternalInitializersFromFilesInMemory` accepts whole logical files, and one supplied file may
contain multiple initializers. It copies initializer data during session creation by default.

Add `kOrtSessionOptionsConfigUseExternalInitializerFileBuffersDirectly` with the config key
`session.use_external_initializer_file_buffers_directly`. Its default is `"0"`. When set to `"1"`, buffers supplied
through `AddExternalInitializersFromFilesInMemory` are borrowed and used directly for initializers. This follows the
existing ORT-format direct-buffer options and avoids adding another C/C++ API. Update the existing API documentation to
state that every session created from the options may outlive the options, and the application must keep each buffer
unchanged and alive until those sessions are released. If session creation fails, the buffers may be released after the
call returns.

During model load, recursively match each external initializer's logical filename, validate its declared and computed
size, and use checked arithmetic to validate its offset and length against the file buffer. Create an initializer
`OrtValue` over the validated slice and retain it in the graph/session instead of copying its data into the
`TensorProto`. Small values that must remain in the `TensorProto` for shape inference and values requiring endian
conversion may be copied.

Require each direct buffer base address plus initializer offset to satisfy the runtime storage type's natural
alignment. Reject a null buffer, a filename mismatch, out-of-range slices, and misalignment rather than silently falling
back to a copy. Valid overlapping slices are allowed.

## Validation

Reject an empty or absolute logical filename, null allocator or output pointers, invalid thresholds, a non-power-of-two
alignment, a misaligned base allocation, and offset or size overflow. Validate that generated offsets and lengths fit
the ONNX signed 64-bit external-data fields.

## Tests

- Compile a model to an ONNX buffer plus one external initializer buffer, with multiple initializers sharing the buffer.
- Reload both buffers through the existing API in its default copying mode and direct-buffer mode, and verify inference
  results.
- Verify internally that large direct-use initializer tensors point into the supplied file buffer, while required small
  or endian-converted values use owned storage.
- Cover threshold boundaries, subgraph initializers, natural and configured offset alignment, and misaligned direct
  buffers.
- Verify logical filename metadata, offsets, lengths, output ownership, and direct-buffer lifetime requirements.
- Verify invalid arguments, allocation failure, checked-arithmetic failure, and unchanged outputs on failure.
- Exercise aggregate external-data sizes beyond 2 GB with a counting or sparse test sink so routine CI does not require
  a 2 GB allocation.
