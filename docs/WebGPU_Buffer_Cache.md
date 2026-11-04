# WebGPU buffer caching

The native WebGPU provider's default storage cache mode, `bucket`, manages spare
buffers automatically. No bucket table configuration is required.

- Sizes use eight intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. Above 128 bytes, rounding adds at most 12.5%. For example,
  a 4,195,328-byte KV tensor uses a 4.5 MiB buffer rather than an 8 MiB buffer.
- Spare counts follow the buffers returned by the workload, without fixed
  per-size count limits. Each bucket-mode cache retains at most 256 MiB in total.
  A working set larger than this budget can still require repeated allocations.
- When space is needed, buffers from the least recently requested size are
  released first. Cache misses reclaim spare space before creating a new buffer.
- A size not requested for 64 cache refreshes loses half its spares, rounded up.
  This repeats until the bucket is empty. Refreshes occur at submission boundaries,
  including empty flushes; they do not correspond one-to-one with inference runs.
  There is no background timer while the context is idle.

The budget covers spare buffers only. Live tensors, buffers awaiting their
recording's submission, other cache modes, and driver allocations are additional
memory. Releasing a cache reference does not force the GPU driver to reclaim
memory immediately. Buffers awaiting submission retain their existing lifetime
protection. Graph capture uses its separate retention policy. The deprecated JSEP
backend is also separate.
