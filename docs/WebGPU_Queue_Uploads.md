# Opt-in native WebGPU queue uploads

Set `ORT_WEBGPU_ENABLE_QUEUE_UPLOADS=1` to use WebGPU `Queue.WriteBuffer` for
eligible CPU-to-GPU uploads. The default is off; unset, empty, and `0` retain
the staging-buffer path. Other values are rejected. Set the variable before
using WebGPU and do not change process environment variables concurrently with
WebGPU operations.

The upload API is byte-oriented and writes at **destination offset zero**.
An upload may cover a prefix of a larger buffer. Queue uploads require a byte
count divisible by four, an unmapped destination with `CopyDst` usage, and
ordinary (non-capture, non-replay) command recording. No new tensor element-type
restriction is introduced, and the host source pointer need not be aligned.
Zero-byte uploads copy no data; mapped destinations are still unmapped.

Mapped-at-creation destinations retain their direct host-copy/unmap path.
Unaligned byte counts and graph capture/replay retain the existing staging
path. That path rounds the GPU copy size up to four bytes; padding bytes are
not part of the logical tensor and remain unspecified. This option does not
add support for nonzero destination offsets or unaligned GPU copy offsets.

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

Submitting prior commands is **not a CPU wait for GPU completion**. This
option removes no readback, completion, error-scope, device-loss, or session
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

`WriteBuffer` is a core WebGPU API, not a Vulkan-only extension. The queue and
host-source lifetime guarantees apply to Dawn's Vulkan, D3D12, and Metal
backends. Backend performance, memory pressure during large weight uploads,
and external Dawn hosts still require platform-specific validation. Large
transfers can still require dedicated staging and may be slower, particularly
when overwriting a buffer recently read by the GPU. Keeping
this option off by default avoids treating one backend's measured improvement
as a universal performance guarantee.

Correctness coverage is part of `onnxruntime_provider_test`, including the
`WebGpuQueueUploadTest` parameterized tests, typed byte/lifetime tests,
deferred-dispatch and copy ordering, and independent concurrent transfers.
These run with the native WebGPU test suites in Windows and macOS CI.
Performance benchmarks and model-generation comparisons are not CI tests.
