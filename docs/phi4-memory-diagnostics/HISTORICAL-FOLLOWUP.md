# Historical arena, initializer and last-row follow-up

Prepared October 2, 2026 from completed, saved diagnostic captures. These are
**single-run, debugger-instrumented historical results**, not current-upstream
benchmarks, clean latency measurements or a matched llama.cpp rerun. No inference
was run to prepare this documentation update. The original [median tables](README.md),
five historical evidence CSV/JSON files, reference artifacts and historical
patches are unchanged.

The [curated evidence](evidence/historical-followup.json) records source-artifact
hashes, runtime/model/configuration identities, acceptance outcomes, phase
coverage and allocator accounting. Its source labels identify excluded artifacts,
not public download locations. See [independent review requirements](#independent-review).

## Workload and identities

All four conditions used one assigned NVIDIA A100-SXM4-80GB (driver 580.105.08),
28,747 identical saved input tokens, maximum length 28,811, exactly 64 generated
IDs, batch/beam 1, greedy decoding, no chunking and shared cache.
Both allocator domains and the actual C API wrapper-to-GenAI-global binding were
verified before inference. The owned binding-return breakpoint was confirmed
deleted before GO; live library identities, a 300-second child deadline,
five-second termination grace and owned-process-only cleanup were retained.

- ORT base: `94e76459dc9a759888707aa430d51a95d878da8f`.
- GenAI base: `ed5f4e87147731e5b07810f9f5c90103b3603cdf`, with the historical
  redundant full-prompt FP32 destination allocation removed in **every** condition.
- The arena-off provider variant bypasses CUDA-device BFC wrapping; it is not
  a CUDA-pool substitution. Its binary hash differs from the arena-on provider.
- Original unpruned graph SHA256:
  `f8ff300b85719f79f1c220c88575626c62cc772a228244a6b96031d1a8d25250`.
- Last-row candidate graph SHA256:
  `cf8a7b6279ff66a7a9a3b3d064eae47b38fbe5879991b382284a4e5b72172e11`.
- Both graphs' external weight contents have SHA256
  `b32f38f9211040b01c1483759cef7a6cce28df9f83b6b8bb91a474575d559d0f`.

All four reused the same long-input diagnostic application and accepted
lifecycle-repaired observer. These identities describe the historical artifacts,
not production changes proposed by this PR or the separate rotary work.

## Overall sampled peaks and correctness

Memory is **MiB (2^20 bytes)**. Process NVML and whole-device NVML are separate
measurements. These are observed maxima over the recorded interval, not medians
and not proven continuously covered lifetime peaks.

| Condition | Process peak | Whole-device peak | First peak phase | Complete first-token logits / 64 IDs |
| --- | ---: | ---: | --- | --- |
| Arena on, setting 0, original graph | 27592 | 28368.125 | Decode | Exact / exact |
| Arena off, setting 0, original graph | 21004 | 21780.125 | Prefill | Exact / exact |
| Arena on, setting 1, original graph | 27392 | 28168.125 | Decode | Exact / exact |
| Arena off, last-hidden-row candidate, setting 0 | 10036 | 10812.125 | Prefill | **Failed** / exact |

The arena-on/off pair matched the saved long reference and each other.
Setting 1 matched both that reference and the accepted arena-on baseline.
The candidate was compared against both the reference and arena-off baseline;
matching tokens did **not** satisfy its original byte-exact logits contract.

### Arena bypass

Bypassing the arena lowered both observed peaks by **6,588 MiB (6.434 GiB)**.
This establishes the measured bypass effect for this workload. It does not
establish how much arena tuning could recover, or that the entire difference
is exclusively retained slack.

### Completed initializer-setting experiment

This result supersedes the earlier proposal to try setting 1. With BFCArena
**on** and the original graph, the only effective configuration change was:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "session.use_device_allocator_for_initializers": "1"
      }
    }
  }
}
```

The final dotted string is one literal key; its original value was the string
`"0"`. The full configuration was checked for exactly that delta, including
types, with no extra changes. The separate GenAI-global allocator-only session
was unchanged. Baseline configuration SHA256 is
`7c31a5416f68bb25c38ed30bf5a604b46d3b516c746d0ac8912a94aabebcb194`;
experimental configuration SHA256 is
`60271eb04195d1e00c0c890e101773b0bb0545a487d49d1db141e4b1540039da`.
The application's applied-config record matches the supplied overlay and search
readback. **Applied-config is an overlay record, not an ORT option getter.**

Initialization and setup sampled peaks fell **1,224 MiB**, but overall peak
and post-generator-cleanup usage fell only **200 MiB**. This did not recover
the 6,588 MiB bypass benefit. The older pip-runtime initializer matrices are
not equivalent to this logits-fixed experiment and are not substituted for it.

Pinned ORT [session initialization](https://github.com/microsoft/onnxruntime/blob/94e76459dc9a759888707aa430d51a95d878da8f/onnxruntime/core/framework/session_state_utils.cc#L211)
selects the Reserve route for applicable initializer allocations when the
option is `"1"`; the [helper calls Reserve](https://github.com/microsoft/onnxruntime/blob/94e76459dc9a759888707aa430d51a95d878da8f/onnxruntime/core/framework/session_state_utils.cc#L40).
Preallocated/shared initializer paths can differ. GenAI passes decoder config
entries through AddConfigEntry, whereas its
[global allocator session](https://github.com/microsoft/onnxruntime-genai/blob/ed5f4e87147731e5b07810f9f5c90103b3603cdf/src/models/model.cpp#L445)
creates separate options and a trivial constant model, not the decoder weights.
Ordinary GenAI output tensors and graph workspaces are not initializer
allocations, so this option does not directly reroute them.

### Rejected last-hidden-row candidate

The isolated graph added Gather (axis 1, scalar index -1), then Unsqueeze
(axes `[1]`), selecting the last normalized hidden row **before** the unchanged
quantized vocabulary projection. For this batch-1 workload, the projection
input became FP16 `[1,1,3072]` and logits FP16 `[1,1,200064]`.
This is not slicing after full vocabulary projection or shrinking an
incompatible destination. All 518 original initializers, weight contents,
external-data descriptors, transformer inputs and KV outputs were unchanged.

Standalone ONNX checking failed on the existing ORT-specific
SimplifiedLayerNormalization schema in **both** graphs. The authorized
alternative validated the exact graph delta and exact added standard nodes
in a host-only model; it did not certify the full graph with standalone ONNX.
The candidate then loaded within its sole authorized runtime attempt.

Its process peak was **10,968 MiB lower** than the arena-off original graph.
However, **52,128 of 200,064** captured FP32 logits differed, with maximum
absolute difference **0.046875**. All 64 IDs matched. It remains **rejected**,
not a validated accuracy-preserving optimization. Attribute memory reduction
to the transformation as a whole, including changed execution/allocation
behavior, not exclusively the removed full-prompt FP16 output payload.

Changing projection row count from 28,747 to 1 can select different
MatMulNBits arithmetic. This is a plausible explanation, not proven causality:
runtime kernel execution and equality of the entering hidden vectors were
not captured. No tolerance was relaxed and no candidate retry was performed.

## Phase-specific memory and coverage

Each cell is **process / whole-device MiB**, with maxima requiring the entire
memory query to fall inside the phase. Setup is the interval after model
initialization and before prefill, including generator/cache preparation and
diagnostic handshakes. Decode excludes the separately marked first-token
transition. Cleanup keeps the model alive; it is not process teardown.

| Phase | On, setting 0 | Off, setting 0 | On, setting 1 | Rejected candidate |
| --- | ---: | ---: | ---: | ---: |
| Initialization | 5048 / 5824.125 | 3822 / 4598.125 | 3824 / 4600.125 | 3824 / 4600.125 |
| Setup | 9152 / 9928.125 | 7542 / 8318.125 | 7928 / 8704.125 | 7544 / 8320.125 |
| Prefill | 27590 / 28366.125 | 21004 / 21780.125 | 27390 / 28166.125 | 10036 / 10812.125 |
| Decode | 27592 / 28368.125 | 7552 / 8328.125 | 27392 / 28168.125 | 7552 / 8328.125 |
| Post-generator cleanup | 27592 / 28368.125 | 3830 / 4606.125 | 27392 / 28168.125 | 3832 / 4608.125 |

| Coverage | On, setting 0 | Off, setting 0 | On, setting 1 | Rejected candidate |
| --- | ---: | ---: | ---: | ---: |
| Whole-device samples | 2384 | 2439 | 2372 | 2255 |
| Missing process readings | 359 | 348 | 349 | 328 |
| Maximum sampling interval, ms | 34.440036 | 31.807867 | 34.396509 | 30.752744 |
| Intervals over 15 ms | 4 | 3 | 4 | 4 |

Sampling requested 10 ms; median intervals were approximately 10 ms.
Initialization has process-attribution gaps. Logits capture has no complete
samples in any condition. First-token/cleanup transitions have zero complete
samples in the on conditions and only one/two in the off conditions.
The JSON retains these sparse/null rows and complete per-phase counts.
Unavailable readings are **not zero memory**. Boundary-crossing queries are
not assigned to a phase; short transient peaks may be missed.

All recorded pre-run whole-device medians were 767.25 MiB. No foreign-process
or process-query-error samples were recorded; sampling cannot exclude
interference between observations. Debugger stops, synchronization, snapshots,
hashing and handshakes perturb execution. One run per condition cannot
establish repeatability, statistical significance or clean latency.

## Reserve accounting without double counting

The pinned [BFC implementation](https://github.com/microsoft/onnxruntime/blob/94e76459dc9a759888707aa430d51a95d878da8f/onnxruntime/core/framework/bfc_arena.cc#L36)
defines:

- Total backing = BFC region bytes + outstanding separately tracked reserves.
- Live bytes = rounded BFC live chunks + outstanding reserves.
- Requested live bytes also **include** those reserves.
- BFC live excluding reserves = live bytes minus reserve bytes.
- Slack = total backing minus live bytes, not all retained process memory.
- `num_reserves` is cumulative calls, not the number of outstanding allocations.

Setting 1 recorded **392 cumulative Reserve calls** and **3,255.076187 MiB**
outstanding reserves at initialization, post-generation and post-generator
cleanup. Setting 0 recorded zero. The global allocator recorded zero in both.
Configuration evidence, source-predicted routing and observed counters agree;
there is no per-initializer call/pointer trace.

| Checkpoint / allocator | Setting | BFC regions MiB | BFC live excluding reserves MiB | Outstanding reserves MiB |
| --- | ---: | ---: | ---: | ---: |
| Initialization / decoder | 0 | 4608 | 3282.234375 | 0 |
| Initialization / decoder | 1 | 1 | 0.031250 | 3255.076187 |
| Post-generation / decoder | 0 | 6656 | 3282.234375 | 0 |
| Post-generation / decoder | 1 | 3073 | 0.031250 | 3255.076187 |
| Post-generation / global | Both | 20489 | 3700.901367 | 0 |
| Post-generator cleanup / global | Both | 20489 | 0 | 0 |

Decoder counters are unchanged between post-generation and generator cleanup;
its model weights remain alive. Global counters are zero at initialization.
The JSON retains requested/live byte accounting for all three checkpoints.
Domain labels use observed allocator creation order and pinned registry order:
raw checkpoint lines do not include allocator pointers. This is source/order
correlation, not directly pointer-tagged domain sampling.

After generation/cleanup, total device-allocator backing was **27,145 MiB**
for setting 0 versus **23,562 + 3,255.076187 = 26,817.076187 MiB** for setting 1.
Thus the backing difference is **327.923813 MiB**, not the 3,583 MiB difference
between BFC regions alone, and not the 200 MiB sampled GPU difference.
Do not add reserves again to total/live counters, add overlapping allocator
quantities to NVML usage, sum non-simultaneous per-arena high-water marks into
a GPU peak, or treat cleanup slack as peak savings.
Device BFC metrics for both off conditions are **not applicable**, not zero.

## Recovered llama.cpp behavior and remaining questions

The preserved Python comparison harness used llama-cpp-python **0.3.35**,
not llama-cli. The cached source archive identifies binding commit
`3691546f1c9e0c1bf93323dff02230bd959cf562` and vendored llama.cpp commit
`4df29be4f4c3673f428170fda944a5b19f743bb8`. Archive/member hashes are in the
curated evidence. Installed bindings match the archive, and the installed
native library matches a cached wheel. Old inventory size/mtime agree.

In that pinned Phi graph,
[requested rows are gathered](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/models/phi3.cpp#L130)
before the final FFN and
[vocabulary projection](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/src/models/phi3.cpp#L182).
The [Python batch builder](https://github.com/abetlen/llama-cpp-python/blob/3691546f1c9e0c1bf93323dff02230bd959cf562/llama_cpp/_internals.py#L507)
requests the last token of **each prompt batch** when `logits_all=False`.
The historical invocation used default logical/physical batch sizes 512/512,
context 32,768, all-layer offload with no splitting, temperature 0 and a
64-token completion cap. Ordinary decode uses one-token batches.
It did not compute full-prompt vocabulary logits and only then slice them.

**Provenance limit:** native source/build identity was recovered afterward.
Contemporaneous process maps, native library hashes and a literal shell-command
record were not retained for the early matrix. Size/mtime continuity and the
cached wheel are weaker than run-time binary attestation. This is not a
substitution of current-main behavior. The old long llama.cpp rows reported
45 completion tokens, counted by retokenizing streamed text, not an exact
64-ID comparison; its quantizer also differed.

No matched llama.cpp rerun or larger-model scaling experiment was completed.
There is **no validated approximately 0.8 GiB residual gap** and no evidence
that the remaining gap is constant or proportional to model size.
Local inspection did not establish a ready, comparably identified larger
ORT/GGUF pair. An MoE model is not intrinsically disqualified: a future
comparison must document its architecture, total/active parameters,
quantization, context/cache and matched workload rather than assume dense
and MoE memory scaling are interchangeable.

## Independent review

During curation, the saved audits were reconciled with complete logits/IDs,
input/configuration records, raw NVML samples and phase markers, BFC logs,
allocator/binding/deletion gates and runtime/model identities. No original
controller was rerun. Saved cleanup/preservation records report successful
owned-process cleanup and unchanged protected artifacts/PR worktrees.

Only the narrative and curated JSON are published. Prompt contents, raw
logits/IDs, weights, binaries, machine paths/addresses, internal links,
large inventories and logs remain excluded. Independent raw verification
needs authorized access to those original captures, configurations, sample
logs, phase markers, gate records, model assets and exact runtime variants.
Fresh reproduction additionally needs the isolated controllers, accepted
observer and matching application; these follow-up runners are not added
to this package. Hashes identify required artifacts but cannot reconstruct
them, independently prove execution or grant access.

The existing offline reviewer still validates the **original five evidence
files and median tables**. Its integrity check covers the new files' bytes,
but it does **not** validate this new JSON schema or replay these follow-ups.
Likewise its `--raw-captures` option applies only to the original historical
Phase A layout, not the new initializer delta or rejected candidate.
See [review/reproduction guidance](REPRODUCING.md#october-2-follow-up-evidence).
