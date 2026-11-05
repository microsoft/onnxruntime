# WebGPU buffer caching

The WebGPU provider's default storage cache mode, `bucket`, manages spare buffers
automatically. No bucket table configuration is required.

- Sizes use two intervals per doubling, with a minimum allocation of 64 bytes
  and 16-byte alignment. For requests of at least 64 bytes, rounding adds at most
  50%. The rule is the same across sizes and models; 4 MiB has no special
  treatment. For request `n >= 64`, let `p` be the largest power of two no larger
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
capture keeps its separate retention policy. The deprecated JSEP backend is also
separate.

## Native evaluation

The following measurements compare the upstream cache at `527971bcec` with this
policy on Windows/D3D12, an RTX 5080 (driver 616.64), and the same executable,
kernels and default Basic validation. The issue workload uses
[LFM3-1.7B-Web q4f16](https://huggingface.co/daveycodez/LFM3-1.7B-Web).
Phi uses the Phi-4-mini-instruct WebGPU export from
[webai-community/ai-models](https://huggingface.co/webai-community/ai-models),
with 4-bit weights, block size 32 and `multiRotaryCacheConcatOffset=4096`.
Past outputs are fed into the next input; graph capture is disabled in this
cache comparison.

| Workload (prompt + decode tokens) | Peak capacity, MB (old → new) | Decode-end capacity, MB | Large decode creations | Median decode, ms | Mean decode, ms |
| --------------------------------- | ----------------------------- | ----------------------- | ---------------------- | ----------------- | --------------- |
| LFM3, 3731 + 450                  | 1978.2 → 1800.3               | 1978.1 → 1363.0         | 1781 → 62              | 10.538 → 9.558    | 11.941 → 9.529  |
| Phi-4-mini, 128 + 192             | 3720.5 → 3720.5               | 3288.0 → 3274.4         | 989 → 0                | 18.041 → 17.937   | 17.625 → 17.911 |
| Phi-4-mini, 3731 + 450            | 5239.5 → 5792.7               | 5239.1 → 4717.7         | 24324 → 190            | 50.084 → 17.175   | 52.454 → 17.316 |

Capacity counts buffer bytes with outstanding API references, including weights,
pending work and cached spares; it is not driver residency or browser page memory.
MB is decimal. One tracked process per policy records full peaks and decode-end
capacity. Large creations count storage buffers of at least 1 MiB during decode, excluding
prefill. Timing uses separate untracked processes in three old/new/new/old rounds,
with prefill and the first 32 decode steps excluded. Reported latencies are the
medians of six process medians and six process means per policy: 2508 warm samples
for each long workload and 960 for the short workload. All compared output logit
hashes and token IDs match across the 42 processes.

The issue workload reduces peak capacity by 9.0% and decode-end capacity by 31.1%.
Long-context Phi trades a 10.6% higher transient peak for fewer creations, lower
decode-end capacity and faster decoding. Short-context Phi's mean latency rises
1.6% while its median is nearly unchanged. Retaining a useful working set can
increase peaks, and timings vary across runs; these results do not establish
universal memory or speed improvements. Browser/Safari page limits and other
GPUs have not been measured.
