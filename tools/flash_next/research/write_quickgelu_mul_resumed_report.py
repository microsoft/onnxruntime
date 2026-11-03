import json
from pathlib import Path

root = Path(__file__).parent
overlay = Path("/home/sayanshaw/repos/qwen38-int4-quickgelu-next")
qualification = json.loads((overlay / "qualification-resumed.json").read_text())
performance = qualification["performance"]
rows = "\n".join(f"| {r['block']} | {r['mode']} | {r['median']:.6f} |"
                 for r in qualification["ABBA"]["rows"])
sources = "\n".join(f"- `{name}`: `{digest}`"
                    for name, digest in qualification["source_sha256"].items())
metrics = "\n".join(
    f"| {name} | {values['control']:.6f} | {values['candidate']:.6f} | {values['gain_percent']:+.3f}% |"
    for name, values in performance.items())
text = f"""# Shared-expert QuickGelu + Mul continuation qualification

## Result

{qualification['decision']}.

| Metric (2K context, 256 tokens, three measured repetitions per block) | Control | Candidate | Change |
|---|---:|---:|---:|
{metrics}

Same provider, same GPU1, same options, process-block ABBA order. Each block also
contains one excluded warmup. Six measured requests per graph, not six process
replicates. Both exact-dense selectors remain enabled. This is measured throughput,
not a dispatch-count prediction; shared-host variation is not eliminated.
The per-block ranges overlap: the second candidate block is slower than the final
control block. Thus the positive pooled result is modest and does not establish a
stable or statistically significant causal speedup.

| Block | Graph | Median decode tokens/s |
|---:|---|---:|
{rows}

## Correctness and actual execution

- Reused the cancelled worker's existing compiled overlay; did not overwrite its
  source, build, profiling evidence, or interrupted benchmark files.
- The interrupted control has one persisted warmup, stale running metadata, and no
  live process/container. Its decoder state could not be resumed; separate resumed
  measurement files were created instead. No partial result entered the table.
- 26 native fixture pairs, three runs each: **430,926 finite bitwise-equal FP16
  elements**, no tolerance changes. This includes inherited 16 pairs plus 10 new
  pairs covering scalar shape, M=2/16, width7/2560, tails and saturation up to
  ±65504; inherited coverage includes M=0/1 and additional ranks/widths.
- Full WebGPU validation and CPU fallback disabled in native comparisons.
  Profiles prove the actual registered fused kernel ran; zero-size fixtures issue
  no shader. Two new runtime guards reject broadcasting and unequal ranks.
- Fixed scalar alpha=1 by construction. The reference is production-optimized
  `QuickGelu(alpha=1)` then `Mul`, preserving the existing negative sigmoid branch
  and FP16 activation rounding boundary. This is not arbitrary-alpha support or
  bitwise equivalence to a separately executed, unoptimized raw Sigmoid chain.
- Existing graph-rewrite tests: **27 passed**, using the already-installed ONNX
  package; no installation required.
- Full-model profiles (full validation) prove **1581 actual dispatches**, versus
  the qualified shared-post stack's **1629**: 48 `QuickGeluMulF16` invocations,
  no remaining QuickGelu invocations. This is a net **48-dispatch reduction**.
- Every one of **16 full 256-token sequences** (all four prompts including warmup
  in all four blocks) matches the original frozen control's prompt SHA and token
  IDs. No major page faults in any recorded prefill/decode phase.
- Weights/config remain the original same-inode files; initializer descriptors
  unchanged. No model conversion semantics or main working-tree source changed.

## Provenance

- Baseline provider SHA256: `{qualification['baseline_provider_sha256']}`
- Candidate provider SHA256: `{qualification['provider_sha256']}`
- Candidate graph SHA256: `{qualification['graph_sha256']}`
- Baseline relink was byte-identical in the inherited retained build manifest;
  qualification rechecks both live provider hashes and every retained source hash.

{sources}

## Reproduction and persistent evidence

Base evidence directory: `{root}`.

- Overlay source, provider and exact compile/link commands:
  `{overlay / 'build'}`.
- Machine-checkable qualification: `{overlay / 'qualification-resumed.json'}`.
- Resumed driver/summary: `resume_quickgelu_mul_abba.py`,
  `quickgelu-mul-resumed-abba-summary.json`; each block has JSON, log and guard
  metadata containing its complete command, GPU UUID and provider/graph hashes.
- Final all-eight-GPU/disk/container cleanup evidence:
  `quickgelu-mul-resumed-resources.json`.
- Extended fixtures: `extend_quickgelu_mul_fixtures.py`,
  `quickgelu-mul-resumed-fixtures/`; native/negative commands are preserved in
  `quickgelu-mul-resumed-native.guard.json` and
  `quickgelu-mul-resumed-negative-{{broadcast,rank}}.guard.json`.
- Revalidate without launching a GPU run:

```bash
E={root}
PYTHONPATH={root.parent / 'h200-native-combined-venv/lib/python3.12/site-packages'} \\
  python3 "$E/qualify_quickgelu_mul_resumed.py"
```

Fixture generation and measurement drivers deliberately reject existing output
paths; choose a fresh evidence suffix before replaying, rather than overwriting.

## Safety, surfaced errors and limits

Initial all-eight-GPU inspection found GPU0/1 idle, GPU5 reserved, GPU6 owned by
another workload; GPU6/7 were untouched. Native additions used GPU0; model runs
used GPU1 only. The new runner uses network=none, user664503:664503, all capabilities
dropped and no-new-privileges. NVML is read-only, with the original 130000MiB idle
guard and owned-process/container cleanup. No clocks/power changes, weight copies,
large installs or main-source integration.

The private Docker socket was initially absent. Only its existing isolated
daemon/data/exec roots were restarted; the system daemon was untouched. The
parallel GPU5 worker was notified and used it successfully. The private daemon
remains available to its parent owner; experiment containers and GPU allocations
are released after the final check.

Preparation errors are retained in `shared-activation-build-20261027*.log`:
an initial redundant prototype missed the pre-existing alternate-named work,
hit an output-basename relink gate and a C++ ambiguous-context error, then built.
On discovering the existing qualified overlay, that unused prototype's source
and build directory were removed. None of its binaries/results enters this
qualification. The earlier stale interrupted control is preserved verbatim.

No whole-model quality/logit equivalence, 100TPS, native FP8, 8K/32K update,
randomized causal speedup, publication, commit or integration claim. Host/page
cache/power variation remains; the table must not be mixed with other providers'
historical long-context numbers. `quality_qualified=false`, `headline_eligible=false`.
"""
(overlay / "report-resumed.md").write_text(text)
print(overlay / "report-resumed.md")
