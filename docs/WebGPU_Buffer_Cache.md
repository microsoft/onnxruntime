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
  abandoned capacities and increase physical allocations. If rounding exceeds the
  device's buffer-size limit, reduce padding to fit while covering the requested
  size; the resulting capacity need not be a multiple of 16 bytes.
- Spare counts follow actual demand. There is no fixed per-size count limit or
  total spare-byte cap that would force an otherwise reusable working set to be
  recreated. Different sessions can reuse compatible spares from the shared
  context; concurrently live buffers still require separate allocations.
- Reuse takes the most recently returned buffer of the matching capacity. A shared
  window ends after 16 successful runs across the sessions using this context's
  pool. Each session records its peak simultaneous demand per capacity within
  that window, including live buffers carried into a run and buffers awaiting
  submission. Graph replay counts as a run; initial capture does not count twice.
- At a window boundary, a session with completed runs and no active run replaces
  its retained demand with its window peaks. Unused capacities can then lose their
  spare reservation. Sessions without completed runs preserve their previous
  demand. Active sessions defer replacement until their full run ends and carry
  their window peaks forward: a slow run may need more buffers later, even after
  an earlier small run completed. This update does not add a pool scan; surplus
  waits for ordinary maintenance. A session stores retained and window peaks
  rather than a history array per run.
- Shared spare demand sums each session's retained, window and active-run maximum
  after subtracting its outstanding buffers. Acquisition follows the current
  borrower, regardless of the recording used to release the buffer. Free buffers
  have no session owner and can serve any compatible request.
- Pauses between runs do not age retained demand, so a chat can reuse its working
  set after a long gap. Session destruction removes its contribution immediately.
  Allocations outside tracked runs share the spare pool without reserving extra
  demand; surplus can be reclaimed at the next run or session-release maintenance.
- The same global 16-run window controls demand statistics and ordinary
  reclamation. A smaller observed working set replaces an earlier peak at a
  window boundary or when a deferred run finishes. Late returns from maintenance check
  only that buffer's size class; starting another run ends that return opportunity.
  Unsuccessful runs and session release trigger maintenance immediately, without
  advancing or resetting the successful-run counter. Failures can raise observed
  demand but do not justify reducing it. Partial submissions do not count as runs.
  Maintenance considers all sessions and releases only surplus spares.
  Run starts, cache refreshes and allocation misses do not scan for eviction.
  Submissions still retire pending buffers and apply the late-return rule above.
  The run-demand check releases surplus entries without scanning retained buffers.

There is no background thread, timer or wall-clock expiry for cache maintenance.
Only free spares can be evicted; live buffers and unsubmitted recordings remain
protected. The run window is a reuse heuristic, not a measurement of available
GPU memory. One small completed run in a global window can replace a larger
previous peak at its boundary, so a later large request may recreate buffers.

Portable WebGPU APIs do not expose remaining GPU memory. This policy does not
bound total device memory or guarantee allocation success. Live tensors,
unsubmitted recordings, weights, other cache modes and driver allocations have
separate lifetimes. Releasing a cache reference does not force the GPU driver to
reclaim memory immediately. No memory-budget option or allocation-source hints
are introduced.

Buffers awaiting submission retain their existing lifetime protection. Graph
capture uses the same storage size classes, while retaining buffers under its
separate lifetime policy. No predefined size table is used.
The deprecated JSEP backend is separate.
