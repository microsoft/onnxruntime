# WebGPU buffer caching

The WebGPU provider's default storage cache mode, `bucket`, manages spare buffers
automatically. No bucket table configuration is required.

- Sizes use eight intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. For requests of at least 128 bytes, rounding adds at most
  12.5%. The rule is the same across sizes and models; 4 MiB has no special
  treatment. Requests of 3,000 bytes, 70 KiB and 33 MiB use 3,072 bytes, 72 KiB and
  36 MiB. The reported 4,195,328-byte KV request uses 4.5 MiB rather than 8 MiB.
- Spare counts follow actual demand. There is no fixed per-size count limit or
  total spare-byte cap that would force an otherwise reusable working set to be
  recreated. Different sessions can reuse compatible spares from the shared
  context; concurrently live buffers still require separate allocations.
- Reuse takes the most recently returned buffer of the matching capacity. Every
  spare records its own return time, so reusing one buffer does not keep an
  entire size class's historical peak alive.
- At submission boundaries, including empty flushes, and before cache-miss
  allocations, the cache checks for idle spares at most once per second and
  releases entries unused for one second.
  Elapsed time, rather than a submission count, prevents a busy session from
  advancing other sessions' aging clocks. These intervals favor reuse while
  eventually retiring abandoned capacities and surplus buffers. They are policy
  heuristics, not measurements of available GPU memory.

There is no background timer while the context is idle. Maintenance resumes on
the next flush, and context destruction releases its cache. A compatible spare
can still be reused before the next maintenance pass even after its idle period.
Workloads whose reuse intervals exceed the idle period may recreate buffers.

Portable WebGPU APIs do not expose remaining GPU memory. This policy does not
bound total device memory or guarantee allocation success. Live tensors,
unsubmitted recordings, weights, other cache modes and driver allocations have
separate lifetimes. Releasing a cache reference does not force the GPU driver to
reclaim memory immediately. No memory-budget option or allocation-source hints
are introduced.

Buffers awaiting submission retain their existing lifetime protection. Graph
capture keeps its separate retention policy. The deprecated JSEP backend is also
separate.
