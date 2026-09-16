# CUDA host-pageable `GatherBlockQuantized`

The CUDA Execution Provider option `enable_host_pageable_gather` enables direct access to CPU-resident FP8
`com.microsoft::GatherBlockQuantized` input data. It is disabled by default.

Direct access requires a CUDA device that reports both `cudaDevAttrPageableMemoryAccess` and
`cudaDevAttrPageableMemoryAccessUsesHostPageTables`. If either device capability is unavailable, ONNX Runtime emits a
warning and makes one persistent CUDA copy of a constant input instead.

The option does not create or manage a file mapping. The model initializer must already be supplied as CPU memory,
such as a file-backed mapping, and that mapping remains live for the session. The direct path does not register,
prefetch, hash, scan, or copy the complete initializer, and the initializer is not accounted as CUDA-resident memory.
Fallback copies are allocated by the CUDA initializer allocator; they are not currently included in capacity-aware
partitioning estimates.

CUDA Graph capture is supported for direct host-pageable access. The initializer mapping must remain alive at the same
virtual address until the graph executable is destroyed, and indices, scales, and outputs must retain their normal
CUDA Graph-stable addresses. The direct path performs no mapping, registration, allocation, copy, capability query, or
synchronization during capture. Persistent-copy fallbacks are prepared during prepacking or an uncaptured warmup run;
if lazy fallback initialization is still required when capture starts, the run fails instead of allocating or copying
during capture.

Because the CPU memory contract is part of the static CUDA kernel registration, non-constant input data is staged to
CUDA on every run, including for non-FP8 types. Such runtime input is not supported during CUDA Graph capture. Multiple
nodes that use the same initializer can also create separate fallback copies; direct host access does not duplicate the
initializer.

This mode primarily reduces GPU memory capacity requirements. Performance depends on storage latency and operating
system page-cache state, so cold prefill can be slower and less predictable than using resident GPU memory.
