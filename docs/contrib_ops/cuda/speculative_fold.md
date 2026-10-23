# CUDA speculative fold operators

`com.microsoft::FoldMid2` and `FoldTail` schema version 1 implement batch-one
speculative acceptance and result publication without a custom-op library.
This replaces the earlier two-row schema and supports two through four head
rows, variable rotary width, configurable cache capacity, optional hidden-state
padding, and deferred publication after downstream validation.

`FoldMid2` takes verifier token predictions, input token IDs, status words,
control state, committed context, prefix contexts, result-ring metadata,
rotary caches, and optional hidden rows. Its outputs are head token IDs,
committed length, slot IDs, sequence length, gathered cosine/sine rows, accepted
prefix length, optional padded hidden rows, updated control, and updated context.
The last two outputs **alias** inputs 3 and 4. The caller must serialize access
to those device buffers. For four head rows, control has at least 16 int64 words
and prefix contexts contain four pairs of int64 words.

`FoldTail` takes a token, status words, the mid-stage control output, and
result-ring metadata. It returns a dependency token and updated control.
The latter aliases input 2. `ctl_slot` selects word 2, 8, or 9. The tail writes
the ring's sequence word only after a system fence, so host polling cannot
observe a partial record.

The ring metadata is `[device_accessible_address, positive_record_capacity]`.
Each record has eight int64 words. Its address must be aligned and its allocation
must hold the declared number of records; allocation ownership remains with
the caller. Invalid length, rotary position, or ring metadata sets error bits
4, 8, or 16, mutes control, and avoids unsafe ring access. Muted prefill uses
zero positions/cache length and does not commit contexts or access ring metadata.
Muted prefill may contain more verifier rows than `head_rows`; non-muted
verification is limited to `head_rows`. The tail unions its downstream status
with the verifier status stored by the mid stage. The kernels do not allocate
or synchronize the stream and support CUDA graph replay.

## Migrating legacy packages

Use GenAI's `tools/qwen38_graph/convert_fold_to_contrib.py` on the target ONNX
graph. It adds explicit state outputs and connects the tail to the mid control
output, preserving external-weight paths. Remove `custom_ops_library` and
`custom_ops_libraries` from model session options only after all graph operations
are available in the rebuilt ORT stack. Keep the required GenAI data-plane
library: it is a runtime helper, not a custom-op registration library.

Build ORT with CUDA and contrib operators enabled. The plugin build registers
the schemas in the core and the kernels in the CUDA provider. Rebuild both,
not just the provider. Ordinary upstream wheels do not contain these additions.

`FoldContribTest.*` in `onnxruntime_provider_test` covers all S1-S4 prefix
lengths, hidden-row padding, deferred/wrapped publication, muted prefill, invalid
metadata, verifier/downstream status union, tail slots, and captured-graph replay.
`SpeculativeFoldTest.InferenceSessionLoadsVersionOneAndAcceptsMutedPrefillRows`
loads and runs the version-1 schemas through `InferenceSession`, including
muted prefill batches larger than `head_rows`.
