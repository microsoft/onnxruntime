# Small-M MatMulNBits selector: correctness investigation

## Conclusion

The reproduced width-3 divergence is **a numerical argmax change, not evidence of an
accept/reject, cache-state, or verification-row bug**. With byte-identical target inputs and
saved state, changing only the selector reproduces the different argmax outside DFlash2.
An ordinary standalone four-token target forward reproduces the Engine's logits bit-for-bit
under each selector.

This does **not** establish lossless equivalence to single-token greedy decoding. At the
first divergent position, baseline greedy has an exact FP16 tie. Both the old and new
width-7 runs already disagree with that baseline at the same position, despite matching
each other. The candidate can remain an explicitly experimental investigation artifact;
it should not be promoted as a production-default, greedy-output-preserving optimization.

No production selection policy, shader, or speculative-decoding implementation was changed
in this investigation. All diagnostic builds, model metadata, scripts, snapshots, and
results are outside tracked production code.

## Reproduction and exact disagreement

The experiment reused the previous model exports, plugin variants, tokenizer, and benchmark
request. Each of six configurations was run twice, with a new process/Engine per configuration,
new request state per repetition, no prefix caching, and graph capture disabled. Each pair
of repetitions produced identical output IDs.

The prompt was the same 86-token rendered rainbow prompt. Generation used 128 output tokens,
greedy selection, seed 0, minimum generated tokens 0, repetition penalty 1, no-repeat-ngram
size 0, no stop strings, and the same DFlash2 overlays as the performance experiment.

The first width-3 disagreement is at **zero-based output index 67**, the 68th generated token.
The preceding context contains 153 tokens: 86 prompt tokens and 67 generated tokens.

| Mode | Token at index 67 | First difference from baseline | Verification rounds / accepted drafts |
| --- | --- | --- | ---: |
| Baseline greedy, old selector | 1599, `" many"` | None in 128 tokens | 0 / 0 |
| Baseline greedy, candidate | 1599, `" many"` | None in 128 tokens | 0 / 0 |
| Width 3, old selector | 1599, `" many"` | None in 128 tokens | 42 / 84 |
| Width 3, candidate | 11346, `" detailed"` | 67 | 38 / 89 |
| Width 7, old selector | 11346, `" detailed"` | 67 | 32 / 95 |
| Width 7, candidate | 11346, `" detailed"` | 67 | 33 / 94 |

The two width-7 sequences match each other for all 128 tokens. They are not baseline-equivalent.
They also are not identical to candidate width 3: those sequences first differ at index 83.

For width 3, the first 20 target invocations have identical token IDs, position IDs, past
lengths, cumulative lengths, block tables, capture counts, and activity flags under both
selectors. In particular, **different acceptance decisions do not precede the first split**.
There are already numerical differences in the computed state, but the committed token
contexts and invocation geometries are the same.

The divergent invocation is step 20, with past length 150 and four input token IDs
`[11782, 13, 14235, 11346]`. Row 2 predicts output index 67; its input is token 14235
(`" Need"`) at absolute position 152. The old selector accepts the first two proposed tokens
and corrects the third to 1599. The candidate accepts all three proposed tokens, including
11346, then emits the bonus token. These decisions follow the recorded target argmaxes.

## Logits and exact-state counterfactuals

These are unmodified-graph Engine logits at the common prefix, before any sampling penalties.
At this position the two largest logits belong to token IDs 1599 and 11346.

| Target evaluation | Logit 1599 | Logit 11346 | Argmax | Top-2 margin |
| --- | ---: | ---: | ---: | ---: |
| Baseline greedy, either selector | 24.296875 | 24.296875 | 1599, first-index tie break | 0 |
| Old width-3 verification | 24.343750 | 24.218750 | 1599 | 0.125000 |
| Candidate width-3 verification | 24.234375 | 24.359375 | 11346 | 0.125000 |

The maximum absolute old/candidate logit difference across the entire vocabulary at this row
is **0.28515625**. This is also the maximum across this four-row invocation; row 3 of the old
invocation follows a rejected draft and is not a common committed-context comparison.

Saved feeds were replayed directly through `onnxruntime.InferenceSession` with the original
target ONNX graph, without a drafter, verifier, scheduler, or state-pool commit. Both original
selector/state combinations reproduced every Engine logit bit-for-bit. The baseline
single-token snapshot also reproduced bit-for-bit under both selectors.

Crossing the two selectors with the two saved pre-step states gives:

| Saved state, same four input IDs | Evaluation selector | Logit 1599 | Logit 11346 | Argmax / margin |
| --- | --- | ---: | ---: | --- |
| Old width-3 state | Old | 24.343750 | 24.218750 | 1599 / 0.125000 |
| Old width-3 state | Candidate | 24.265625 | 24.312500 | 11346 / 0.046875 |
| Candidate width-3 state | Old | 24.281250 | 24.281250 | 1599 / 0 |
| Candidate width-3 state | Candidate | 24.234375 | 24.359375 | 11346 / 0.125000 |

Thus the selector change alone can flip the decision on the **same state**, before any
different acceptance or commit can occur. The maximum divergent-row cross-selector
logit differences are 0.29296875 on the old snapshot and 0.306640625 on the candidate snapshot.

As a separate control, forcing the same four input tokens through four sequential M=1
target forwards from each saved initial state produces identical results under old and
candidate selectors. At the relevant third row, the old initial state gives 1599 with
margin 0.046875; the candidate initial state gives a tie, resolved to 1599.
This control has the same initial state and forced context but a different evaluation shape:
its intermediate states are recomputed by M=1 kernels. It is not a claim that M=1 and M=4
have bitwise-identical internal recurrence. In particular, both selectors retain subgroup
for the M=4 output head but use regular MatMulNBits at M=1.

## Intermediate outputs and identical-activation isolation

Readback-only diagnostic plugins captured MatMulNBits inputs and outputs **without adding
graph outputs or changing kernel selection**. Their complete logits matched the original
Engine snapshots bit-for-bit. Eleven MatMul sites across layers 0, 31, 63, and the head were
captured. At divergent row 2:

| Original-graph intermediate, old versus candidate | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: |
| Layer-0 linear-attention QKV input | 0 | 0 |
| Layer-0 linear-attention QKV output | 0.187500 | 0.3040% |
| Layer-31 attention QKV input | 0.022461 | 0.4334% |
| Layer-31 attention QKV output | 0.070313 | 0.4522% |
| Layer-63 MLP down output | 6.125000 | 2.5077% |
| Final normalized hidden state, input to LM head | 0.914063 | 1.0954% |
| LM-head output | 0.285156 | 2.5991% |

The first selected projection already differs on **identical input activations**, with no
preceding state-dependent operation needed to explain the difference. Later layers combine
local reduction differences with propagated activation/state differences; their deltas are
not isolated errors attributable to one kernel.

Six representative operators were then extracted with their exact real weights and run
through regular and subgroup kernels on each of the two captured activation arrays:
24 kernel evaluations, or 12 identical-activation comparisons. GPU traces explicitly showed
regular `MatMulNBits` versus `SubgroupMatrixMatMulLayout` plus `SubgroupMatrixMatMulNBits`;
CPU fallback was disabled. All 12 selected original-graph outputs were reproduced bit-for-bit
by the matching isolated operator.

The independent reference dequantizes INT4/block32 weights and performs FP64 matmul. The
following relative L2 errors use the old captured activations and all four rows:

| Operator | Subgroup versus FP64 | Regular versus FP64 |
| --- | ---: | ---: |
| Layer-0 linear QKV | 0.3237% | 0.0755% |
| Layer-0 linear output | 0.1820% | 0.0760% |
| Layer-0 MLP down | 0.4879% | 0.0814% |
| Layer-31 attention QKV | 0.3072% | 0.0753% |
| Layer-63 MLP down | 2.1718% | 0.0679% |
| LM head | 0.3634% | 0.0928% |

Regular was closer to the reference in all 12 comparisons, including the candidate activation
arrays. The large layer-63 down error is an existing subgroup FP16 accumulation limitation,
not evidence that the candidate's regular kernel introduced an inaccurate implementation.
Nevertheless, smaller operator errors do not guarantee identical greedy decisions.

The **unchanged output head** provides a particularly direct explanation. On the candidate's
exact captured final hidden state:

| LM-head computation, identical input and weights | Logit 1599 | Logit 11346 | Argmax / margin |
| --- | ---: | ---: | --- |
| Existing subgroup, used by candidate M=4 | 24.234375 | 24.359375 | 11346 / 0.125000 |
| Existing regular, diagnostic only | 24.296875 | 24.296875 | 1599 / 0 |
| FP64 dequantized reference | 24.317016 | 24.290651 | 1599 / 0.026365 |

The maximum subgroup/regular difference over this row's vocabulary is 0.1171875.
On the old hidden state, the corresponding maximum is 0.140625; FP64 favors 1599 by
0.059487 and both FP16 head kernels also favor 1599. The candidate changes the hidden
activations presented to an unchanged, differently rounded head; it does not change the
head's selected shader. A precise whole-model FP64 or source-model quality conclusion does
not follow from these single-operator references.

## State and verification checks

Every emitted token in the captured 80-token width-3 and baseline runs was checked against
the appropriate target argmax row and the actual visible context. The accepted input prefix
matched the committed token history at each step. No shifted row, extra rejected token, or
missing committed token was found.

Around the split, steps 17-20 were checked against the next step's saved fixed-state inputs:
768 tensor-transition checks across the two selectors, covering 48 convolution and 48
recurrent states per transition. Full acceptance matched the full target state outputs
bit-for-bit. Partial acceptance convolution states matched exact prefix replay bit-for-bit.
Partial recurrent states matched the existing compact CPU recurrence to maximum absolute
error 7.62939453125e-6 and maximum per-tensor relative L2 error 8.234e-8, consistent with
FP32 device arithmetic. Accounting for contraction as `fma(state, decay, key * delta)`
reproduced **all 144 partial recurrent-state tensors bit-for-bit**. Thus all 768 transitions
matched exactly using either the full output, exact convolution replay, or an FMA-equivalent
compact recurrence. This check used the captured decay/key/delta capsule and actual kept
count, not independently regenerated activations.

At step 20 specifically, old selection keeps three input rows (anchor plus two accepted
drafts), while the candidate keeps four. All 32 committed attention K/V prefixes matched
the corresponding standalone target outputs bit-for-bit under each selector. The old next
past length is 153 and the candidate next past length is 154, exactly as their different
accepted prefixes require; neither is a context-length mismatch.

Five existing GenAI tests passed, with three real WebGPU tests and no skips:
`WebGpuFixedStatePoolTest.*`, `FixedStatePoolTest.CompactPartialAcceptanceReplaysConvAndGdn`,
and `FixedStatePoolTest.DirectBindingsMixPartialAndFullAcceptance`.
The WebGPU suite includes adversarial replay and failure-before-publication checks.

These checks exclude a state/verification bug as the explanation for the reproduced first
split; they are not a proof covering every speculative workload, adapter, or failure path.

## Reproducibility and production prerequisites

Evidence and runnable scripts are retained under `build/nbits-small-m-selector/results/`.
Key entries are `correctness-mode-comparison.json`, `correctness-row-analysis.json`,
`correctness-replay/`, `correctness-single-op/`, `correctness-state-commits.json`,
`correctness-asset-hashes.json`, and `correctness-replay-tests.log`. Raw target snapshots
under `correctness/` are losslessly compressed; manifests carry uncompressed SHA-256 hashes.

The pinned GenAI runtime is commit `f3be4a517f6b0349add2c47a9086e00bfb7d0af2`; the compatible
ORT core reports `dc1ae7a129`. The WebGPU plugin variants are the same current-kernel A100/Vulkan
builds used in the selector experiment. Target and draft graph SHA-256 hashes are respectively
`ca4d44eb82d011fde4d3180a15165597cb7983def141bf203514c31174b4b544` and
`e3dbed272590346eb2adc55ba64e293ca3af55f88c4bde0b99c5e7e575e3ce58`.
Hashes of the external weights, tokenizer, config, workload, and loaded libraries are also
preserved. The diagnostic GenAI object was compiled against the pinned commit's headers,
not the newer checkout's incompatible class layouts.

The existing reproduction command, run for A/implemented and widths 0/3/7, was:

```bash
source /workspace/venv/bin/activate
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R=/workspace/onnxruntime/build/nbits-small-m-selector/results
python "$R/dflash_benchmark.py" --variant A --width 3 --tokens 128 \
  --warmups 0 --repetitions 2 --diagnostic-tokens 0 --label=-correctness-repro
```

All six two-repetition configurations completed successfully and deterministically. Exact-state
and same-activation commands included:

```bash
python "$R/replay_correctness.py" --variant A \
  --captures A-w3-pinned/step-20 implemented-w3-pinned/step-20 A-w0-pinned/step-68
python "$R/replay_correctness.py" --variant implemented \
  --captures A-w3-pinned/step-20 implemented-w3-pinned/step-20 A-w0-pinned/step-68
python "$R/isolate_actual_matmuls.py" --variant A
python "$R/isolate_actual_matmuls.py" --variant R
python "$R/check_state_commits.py"
```

These commands passed. Sequential M=1 controls and graph-preserving readback variants passed
as described above. An exploratory graph with extra debug outputs changed optimization/fusion
and shifted logits, so its values were discarded for causal attribution; the reported
intermediates instead use readback-only plugins whose logits were verified unchanged.

The focused existing test command passed five tests:

```bash
export ORT_LIB_PATH=/workspace/onnxruntime-genai/build/qwen38-current-attribution-20261019/ort-core/Release/onnxruntime/capi/libonnxruntime.so.1.32.0
cd /workspace/onnxruntime-genai/build/qwen38-current-attribution-20261019/genai-build/Release
./engine_unit_tests --ep_dir "$R/variants/implemented" \
  --gtest_filter='WebGpuFixedStatePoolTest.*:FixedStatePoolTest.CompactPartialAcceptanceReplaysConvAndGdn:FixedStatePoolTest.PrefixCheckpointAfterPartialAcceptanceSeesReplayedState:FixedStatePoolTest.DiscardedPartialAcceptanceLeavesStateUnchanged:FixedStatePoolTest.DirectBindingsMixPartialAndFullAcceptance'
```

Two named CPU cases in that filter are absent from this pinned binary; the confirmed count
is five executed tests, not seven.

**Before a production PR:** define whether the contract requires exact greedy equivalence or
permits documented FP16 near-tie changes. If exact/lossless equivalence is required, the present
candidate does not satisfy it, and the old width-7 path already needs the same numerical
consistency investigation. Assess existing precision controls and shape-dependent reduction
consistency, particularly the output head, in a separately scoped change rather than patching
acceptance decisions. Add a captured-activation near-tie regression and same-state grouped
versus sequential target checks, then broaden prompt/model-quality and adapter coverage.
Do not infer model-quality preservation from lower single-op L2 error or from one matching
128-token sequence, and do not describe the measured end-to-end ratios as identical-work speedups.
