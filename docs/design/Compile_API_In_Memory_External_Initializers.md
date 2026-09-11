# Compile API In-Memory External Initializers

## Goal

Support external initializer data without filesystem access through two separate capabilities:

- Allow `OrtCompileApi` to write an external initializer file to a caller-owned buffer when saving a compiled or
  optimized ONNX model.
- Allow an application to provide an external initializer file as a buffer when creating an inference session.

Together, these capabilities support models whose initializer data makes the complete model exceed the protobuf 2 GB
limit. Compiling execution providers hit the same limit through EPContext node data; that path is covered separately
below.

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

The external-initializer file destination (`ModelCompilationOptions_SetOutputModelExternalInitializersFile`) and the
buffer destination (`ModelCompilationOptions_SetOutputModelExternalInitializersBuffer`) are mutually exclusive: the
caller provides one or the other, never both. Because both map to the single `initializers_location` variant, the
last setter called wins and silently replaces any prior external-initializer destination. The output model destination
is independent: it may be a file, buffer, or write callback. For file output, the caller is responsible for persisting
or otherwise supplying the returned initializer buffer under the logical filename recorded in the model.

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

1. Run the existing external-initializer serializer against a counting stream. This computes offsets and the total
  size with exactly the same traversal, alignment, endian conversion, and prepacked-blob handling as the write pass.
2. Allocate the exact size once and run the serializer again against a fixed-size stream over that allocation. Emit
  the logical filename, offset, and length into each externalized `TensorProto`.

Write externalized initializers in load-ready tensor storage and align each tensor's offset to its natural alignment;
the writer controls the layout, so this alignment is guaranteed for ORT-produced buffers. Additionally apply the
alignment configured by `ModelCompilationOptions_SetOutputModelExternalInitializersAlignment` above its size threshold;
the existing default policy is mmap-friendly 4 KiB alignment for data larger than 1 MiB. Prepacked blobs are opaque and
use only the configured alignment policy.

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

During model load, recursively match each external initializer's logical filename (the existing copy path only visits
the top-level graph's initializers, so extend the traversal into subgraphs for both the copy and direct-use paths),
validate its declared and computed size, and use checked arithmetic to validate its offset and length against the file
buffer. When the slice is naturally aligned for the runtime storage type, create an initializer `OrtValue` over the
validated slice with non-owning storage and retain it in the graph/session instead of copying. The borrowed `OrtValue`
must be non-owning (wrap the slice in a `Tensor` with a plain CPU `OrtMemoryInfo` and no deleter, as the existing
native-endian inject branch already does) so the `SessionOptions` can be released while the underlying buffer persists.
Small values that must remain in the `TensorProto` for shape inference and values requiring endian conversion may be
copied.

Natural alignment is preferable but not required in direct-use mode. Even for an ORT-produced buffer whose tensor
offsets are naturally aligned, the supplied buffer's base address is caller/allocator-controlled, and the buffer may not
have been produced by ORT at all, so `base + offset` can fail to meet the runtime storage type's natural alignment. When
that happens, fall back to copying that individual initializer into owned storage instead of borrowing; the rest of the
initializers in the same buffer still borrow directly. Reject a null buffer, a filename mismatch, and out-of-range
slices. Valid overlapping slices are allowed.

## Compiling EPs and EPContext node data

Compiling execution providers wrap each fused subgraph in an `EPContext` node whose provider binary is carried in the
node's `ep_cache_context` attribute. This data is a node attribute, not an initializer, so the initializer-buffer path
above does not cover it, and the same 2 GB limit applies independently.

Embed mode cannot represent data at or above 2 GB. The blob lives inside the node's `AttributeProto` within the
`ModelProto`, and `OrtApi::CreateOpAttr` takes an `int` length, so a plugin EP cannot even create such an attribute. Whether to embed remains the caller's choice. Today the
EP creates the `EPContext` node itself (plugin EPs through `OrtModelEditorApi::CreateNode`, in-tree EPs by building the
`Node` directly) and chooses both whether to embed and, when not embedding, where the data goes; the existing write
callback (`OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc` retrieved via `OrtEpContextConfig`) is
opt-in, so ORT cannot force an arbitrary EP to route through it.

Promote the WRITE callback transport to the stable API. The compile API accepts an `OrtWriteNamedBufferFunc`; plugin
EPs snapshot it from session options through an owned `OrtEpContextConfig` and retrieve the callback through
`OrtEpApi`. Add `OrtEpContextDataSupportFlags_WRITE` and `OrtEp::GetEpContextDataSupport` so ORT rejects a configured
WRITE callback before `Compile()` when a plugin EP cannot honor it. READ callback support remains experimental.

`CreateEpContextNode` is intentionally deferred. EPs continue to create their own EPContext nodes and decide how to
encode `ep_cache_context`. This change provides stable callback transport and capability negotiation only; it does not
centralize node creation or alter the generic `CreateNode` behavior.

## Validation

Reject an empty or absolute logical filename, null allocator or output pointers, a non-power-of-two alignment, and
offset or size overflow. Validate that generated offsets and lengths fit the ONNX signed 64-bit external-data fields.
A misaligned supplied buffer is valid; direct-use loading copies only slices that do not meet natural alignment.

## Tests

- Compile a model to an ONNX buffer plus one external initializer buffer, with multiple initializers sharing the buffer.
- Reload both buffers through the existing API in its default copying mode and direct-buffer mode, and verify inference
  results.
- Verify internally that large direct-use initializer tensors point into the supplied file buffer, while required small
  or endian-converted values use owned storage.
- Cover threshold boundaries, subgraph initializers (compiled and reloaded through both copy and direct-use paths),
  natural and configured offset alignment, and misaligned direct buffers that fall back to per-initializer copies while
  neighboring aligned initializers still borrow.
- Verify logical filename metadata, offsets, lengths, output ownership, and direct-buffer lifetime requirements.
- Verify that the file and buffer external-initializer destinations are mutually exclusive (last setter wins), and
  that a buffer destination works with file, buffer, and callback output-model destinations.
- Verify invalid arguments, allocation failure, checked-arithmetic failure, and unchanged outputs on failure.
- Exercise aggregate external-data sizes beyond 2 GB with a counting or sparse test sink so routine CI does not require
  a 2 GB allocation.
- Compile with the example plugin EP and verify that its non-embedded EPContext data reaches the configured stable
  WRITE callback.
- Verify callback snapshot and clearing behavior, and reject a configured WRITE callback when a plugin EP does not
  advertise `OrtEpContextDataSupportFlags_WRITE`.
