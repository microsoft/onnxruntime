# Native WebGPU queue uploads

ONNX Runtime automatically uses WebGPU `Queue.WriteBuffer` for eligible
CPU-to-GPU uploads up to and including **1 MiB (1,048,576 bytes)**. No
environment-variable toggle or provider option is required.

The upload API is byte-oriented and writes at **destination offset zero**.
An upload may cover a prefix of a larger buffer. Queue uploads require a byte
count divisible by four and no larger than 1 MiB, an unmapped destination with
`CopyDst` usage, and ordinary (non-capture, non-replay) command recording. No new
tensor element-type restriction is introduced, and the host source pointer
need not be aligned.
Zero-byte uploads copy no data; mapped destinations are still unmapped.

Mapped-at-creation destinations retain their direct host-copy/unmap path.
Larger transfers, unaligned byte counts, and graph capture/replay retain the
existing staging path. That path rounds the GPU copy size up to four bytes;
padding bytes are not part of the logical tensor and remain unspecified.
Queue uploads do not add support for nonzero destination offsets or unaligned
GPU copy offsets.

## Ordering and lifetime

Before a queue upload, ORT submits any work already recorded on the caller's
command timeline. This is necessary when a previously recorded dispatch or
copy still reads the destination: recording a command does not yet order it
before `WriteBuffer`. Subsequent commands submitted to the same queue observe
the upload in queue order.

`WriteBuffer` copies the host source data before returning. The caller may
immediately modify or release the host allocation. Dawn owns any staging
allocation and retains resources until its transfer completes. The
destination still follows the existing ORT allocation and stream ownership
rules; do not destroy it or race writes to the same tensor.
Each session's recording timeline still requires serialization. Independent
sessions sharing tensors still require caller-provided dependency ordering;
the flush and queue write are not a device-wide atomic transaction.

Submitting prior commands is **not a CPU wait for GPU completion**. Queue
uploads remove no readback, completion, error-scope, device-loss, or session
synchronization checks. Queue-write validation failures use the existing
WebGPU error-reporting path; they are not silently retried as staging copies.

## Why this can help

The old unmapped-buffer path creates a fresh mapped staging buffer, copies
host bytes, unmaps it, records `CopyBufferToBuffer`, and finishes/submits an
encoder for every upload. `WriteBuffer` lets Dawn use its dynamic uploader,
reuse staging allocations, select backend-specific host-visible fast paths,
and batch pending transfers with later queue work. It may still submit work
internally, so fewer ORT encoders do not imply an identical reduction in
backend submissions.

The 1 MiB limit is conservative: A100/Vulkan measurements consistently
improved both host and completion-fenced latency through that size, with and
without preceding GPU reads. At 4 MiB + 4 bytes, completion-fenced transfers
regressed. Sizes between 1 MiB and that regression were not validated, so they
remain on staging rather than extrapolating the small-upload improvement.
The limit is also below the pinned Dawn dynamic uploader's 4 MiB ring size;
larger uploads can require dedicated staging rather than allocation reuse.

`WriteBuffer` is a core WebGPU API, not a Vulkan-only extension. Its queue and
host-source lifetime guarantees apply to Dawn's Vulkan, D3D12, and Metal
backends. A100/Vulkan correctness has been exercised locally; macOS/Metal and
Windows/D3D12 native CI must still validate default routing, ordering,
threshold boundaries, and concurrency. Backend performance, memory pressure,
and external Dawn hosts require platform-specific validation before raising
the size limit.

Correctness coverage is part of `onnxruntime_provider_test`, including the
`WebGpuQueueUploadTest` cases, typed byte/lifetime tests,
deferred-dispatch and copy ordering, and independent concurrent transfers.
These run with the native WebGPU test suites in Windows and macOS CI.
Performance benchmarks and model-generation comparisons are not CI tests.
