# WebGPU buffer caching

The WebGPU provider's default storage cache mode, `bucket`, manages spare buffers
automatically. No bucket table configuration is required.

- Sizes use four intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. For requests of at least 64 bytes, rounding adds at most
  25%. For request `n >= 64`, let `p` be the largest power of two no larger
  than `n`, and round up to a multiple of `max(16, p / 4)`. Requests of 3,000 bytes,
  70 KiB and 33 MiB use 3,072 bytes, 80 KiB and 40 MiB. The reported
  4,195,328-byte KV request uses 5 MiB rather than 8 MiB. Coarser classes retain
  reuse over a wider range of growing requests; finer classes can retain more
  abandoned capacities and increase physical allocations.
- Spare counts follow actual demand. There is no fixed per-size count limit or
  total spare-byte cap that would force an otherwise reusable working set to be
  recreated. Different sessions can reuse compatible spares from the shared
  context; concurrently live buffers still require separate allocations.
- Reuse takes the most recently returned buffer of the matching capacity. Each
  session records peak simultaneous demand per capacity over its 16 most recent
  completed runs, including live buffers carried into a run and buffers still
  awaiting submission. A smaller working set eventually replaces an earlier peak,
  allowing surplus buffers within a still-active size class to be released.
- Shared spare demand combines the sessions' recent peaks after subtracting their
  outstanding buffers. Each session advances only its own history. Partial
  submissions and another session's runs do not age its demand. Graph replay counts
  as a run; the replay performed during initial capture does not count twice.
- After a session has no active run for 60 seconds, its retained-demand history
  can expire at a later maintenance event. A slow active run does not expire.
  Session destruction removes its contribution immediately. Allocations outside a
  tracked session run use a 60-second spare-return grace period instead.
  If all sessions are idle and grace periods have expired, all ordinary bucket
  spares can be released; another session's recent demand can still retain spares.
- The same 16-run window controls demand history and ordinary reclamation.
  Cleanup is batched every 16 completed runs of each session, while demand history
  advances each run. An old peak can remain cached until the next cleanup after
  it leaves that history. Late buffer returns from a cleanup run check only that
  buffer's size class; returns between batches wait for a later cleanup.
  Unsuccessful runs, idle expiry and session release trigger maintenance
  immediately, without advancing another session's run count.
  A shared scan can reclaim any surplus whose demand has already aged out.
  Run starts, submissions (including empty flushes) and allocation misses can
  trigger idle maintenance once the next idle deadline is reached. The normal
  run-demand check inspects surplus entries rather than scanning retained buffers.

There is no background thread or timer for cache maintenance. With no subsequent
maintenance event, buffers remain cached even after 60 seconds. Only free spares
can be evicted; live buffers and unsubmitted recordings remain protected. The
history window and idle timeout are reuse heuristics, not measurements of available
GPU memory. A later request can recreate buffers whose previous demand aged out.

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
