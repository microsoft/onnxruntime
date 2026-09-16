# CUDA host-pageable `GatherBlockQuantized`

The CUDA Execution Provider option `enable_host_pageable_gather` enables direct access to CPU-resident FP8
`com.microsoft::GatherBlockQuantized` input data. It is disabled by default.

Direct access requires a CUDA device that reports both `cudaDevAttrPageableMemoryAccess` and
`cudaDevAttrPageableMemoryAccessUsesHostPageTables`. CUDA Graph capture is not currently supported with this mode.
If either device capability is unavailable or CUDA Graphs are enabled, ONNX Runtime emits a warning and makes one
persistent CUDA copy of a constant input instead.

The option does not create or manage a file mapping. The model initializer must already be supplied as CPU memory,
such as a file-backed mapping, and that mapping remains live for the session. The direct path does not register,
prefetch, hash, scan, or copy the complete initializer, and the initializer is not accounted as CUDA-resident memory.
Fallback copies are allocated by the CUDA initializer allocator; they are not currently included in capacity-aware
partitioning estimates.

This mode primarily reduces GPU memory capacity requirements. Performance depends on storage latency and operating
system page-cache state, so cold prefill can be slower and less predictable than using resident GPU memory.
