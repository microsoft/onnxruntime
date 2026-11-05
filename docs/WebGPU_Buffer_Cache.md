# WebGPU buffer caching

The WebGPU provider's default storage cache mode, `bucket`, manages spare buffers
automatically. No bucket table configuration is required.

- Sizes use two intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. For requests of at least 64 bytes, rounding adds at most
  50%. For request `n >= 64`, let `p` be the largest power of two no larger
  than `n`, and round up to a multiple of `max(16, p / 2)`. Requests of 3,000 bytes,
  70 KiB and 33 MiB use 3,072 bytes, 96 KiB and 48 MiB. The reported
  4,195,328-byte KV request uses 6 MiB rather than 8 MiB. Coarser classes retain
  reuse over a wider range of growing requests; finer classes can retain more
  abandoned capacities and increase physical allocations.
- Spare counts follow actual demand. There is no fixed per-size count limit or
  total spare-byte cap that would force an otherwise reusable working set to be
  recreated. Different sessions can reuse compatible spares from the shared
  context; concurrently live buffers still require separate allocations.
- Reuse takes the most recently returned buffer of the matching capacity. Every
  spare records its own return time, so reusing one buffer does not keep an
  entire size class's historical peak alive.
- Reclamation runs synchronously when a submission boundary (including an empty
  flush) or a cache miss triggers a maintenance check. A check skips scanning if
  the previous scan was less than one second ago; otherwise it releases spares
  unused for at least one second. These time thresholds only control eligibility
  and throttle event-triggered work. They never schedule a scan. Returning a
  buffer records its return time without triggering a scan.
  Elapsed time, rather than a submission count, prevents a busy session from
  advancing other sessions' aging clocks. The thresholds are policy heuristics,
  not measurements of available GPU memory.

There is no background thread or timer for cache maintenance. With no submission
or cache-miss event, no scan occurs, regardless of how much time passes. Context
destruction releases its cache. A compatible spare can still be reused before the
next maintenance pass even after its idle period. Workloads whose reuse intervals
exceed the idle period may recreate buffers.

Portable WebGPU APIs do not expose remaining GPU memory. This policy does not
bound total device memory or guarantee allocation success. Live tensors,
unsubmitted recordings, weights, other cache modes and driver allocations have
separate lifetimes. Releasing a cache reference does not force the GPU driver to
reclaim memory immediately. No memory-budget option or allocation-source hints
are introduced.

Buffers awaiting submission retain their existing lifetime protection. Graph
capture uses the same storage size classes, while retaining buffers under its
separate lifetime policy without idle eviction. No predefined size table is used.
The deprecated JSEP backend is separate.
