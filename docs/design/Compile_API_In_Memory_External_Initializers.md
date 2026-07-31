# Compile API In-Memory External Initializers

## Goal

Allow `OrtCompileApi` to save and reload compiled or optimized ONNX models whose initializer data makes the complete
model exceed the protobuf 2 GB limit, without filesystem access.

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

Add the corresponding C++ wrapper and an `ExternalInitializerBufferHolder` alternative to
`epctx::ModelGenOptions::initializers_location`.

On failure, ORT must free any temporary allocation and leave the caller's output pointer and size unchanged. On
success, the caller owns the buffer and releases it with the supplied allocator.

## Serialization

Refactor the existing external-initializer save path so its physical destination can be either a file stream or an
allocated memory buffer. Preserve the existing initializer traversal, threshold, external-data metadata, endian
conversion, subgraph handling, and prepacked-weight handling.

Use a two-pass implementation:

1. Compute each externalized initializer's offset and the total buffer size with checked arithmetic.
2. Allocate the exact size once, write each initializer to its assigned span, and emit its logical filename, offset,
   and length into the `TensorProto`.

Initializers may be tightly packed. ONNX external data has no data-type alignment requirement because loading uses byte
offsets and lengths. If the existing optional mmap alignment policy is enabled, preserve its padding and 4 KiB offset
alignment; do not introduce a new data-type-based policy.

The serialized ONNX protobuf must still be smaller than 2 GB. Externalizing initializer bytes keeps the protobuf small;
this feature does not support graph metadata that independently exceeds protobuf's limit.

## Loading

The existing `OrtApi::AddExternalInitializersFromFilesInMemory` API already accepts an array of logical files, not an
array of initializers. Pass one entry whose filename matches the logical filename recorded in the model and whose buffer
contains all externalized initializers. ORT locates each initializer within that buffer using its `TensorProto` offset
and length.

That API copies initializer data during session creation so the application can immediately release the file buffer.
Keep it as the compatible loading path.

Add a separate borrowed-buffer variant for memory-constrained callers. It uses the same filename-to-buffer mapping but
creates tensors over validated buffer slices without copying. Its contract requires the application to keep every
buffer unchanged and alive until the session is released. Small initializers needed by shape inference and data that
requires endian conversion may still be copied.

## Validation

Reject an empty or absolute logical filename, null allocator or output pointers, invalid thresholds, and offset or size
overflow. Validate that generated offsets and lengths fit the ONNX signed 64-bit external-data fields.

## Tests

- Compile a model to an ONNX buffer plus one external initializer buffer, with multiple initializers sharing the buffer.
- Reload both buffers through both the copying and borrowed-buffer APIs and verify inference results.
- Cover threshold boundaries, subgraph initializers, and existing optional alignment behavior.
- Verify logical filename metadata, offsets, lengths, output ownership, and borrowed-buffer lifetime requirements.
- Verify invalid arguments, allocation failure, checked-arithmetic failure, and unchanged outputs on failure.
- Exercise aggregate external-data sizes beyond 2 GB with a counting or sparse test sink so routine CI does not require
  a 2 GB allocation.
