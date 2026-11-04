# WebGPU buffer caching

The native WebGPU provider's default storage cache mode, `bucket`, manages spare
buffers automatically. No bucket table configuration is required.

- Sizes use eight intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. Above 128 bytes, rounding adds at most 12.5%. The rule is
  the same across sizes and models; 4 MiB has no special treatment. For example,
  requests of 3,000 bytes, 70 KiB and 33 MiB use 3,072 bytes, 72 KiB and 36 MiB.
  The reported 4,195,328-byte KV request uses 4.5 MiB rather than 8 MiB.
- Spare counts follow the buffers returned by the workload, without fixed
  per-size count limits. Each bucket-mode cache retains at most 256 MiB in total.
  A working set larger than this budget can still require repeated allocations.
- Newly returned sizes enter a probationary queue. Cache hits or requests recurring
  across refreshes earn protection for that size. One-off bursts within a refresh
  do not earn protection just by allocating many buffers.
- Eviction removes the oldest probationary buffers first. Protected buffers target
  three quarters of the byte budget; older protected buffers move back to probation
  to leave room for new demand. A single large protected buffer may exceed this
  target while remaining within the total budget. Cache misses reclaim probationary
  space before allocation, preserving protected buffers for expected reuse.
- Every 64 refreshes, the cache releases individual spares idle for at least 64
  refreshes. Reusing one buffer does not keep an entire size bucket alive. Refreshes
  occur at submission boundaries, including empty flushes; they do not correspond
  one-to-one with inference runs. There is no background timer while the context
  is idle.

The provider cannot query remaining GPU memory through portable WebGPU APIs.
The spare-byte budget is an internal cache-pressure threshold, not a measurement
of operating-system or device memory pressure. Reuse and request history guide
which buffers are worth retaining; the limits are shared policy constants, not
model-specific settings or guarantees against device allocation failure.

The budget covers spare buffers only. Live tensors, buffers awaiting their
recording's submission, other cache modes, and driver allocations are additional
memory. Releasing a cache reference does not force the GPU driver to reclaim
memory immediately. Buffers awaiting submission retain their existing lifetime
protection. Graph capture uses its separate retention policy. The deprecated JSEP
backend is also separate.
