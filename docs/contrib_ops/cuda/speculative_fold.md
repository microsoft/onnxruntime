# CUDA speculative fold operators (version 2)

`com.microsoft::FoldMid2` and `FoldTail` version 2 implement batch-one
speculative acceptance and result publication without a custom-op library.
They extend the two-row version-1 design for two through four head rows,
variable rotary width, configurable cache capacity, optional hidden-state
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
observe a partial record. This differs from version 1, which published in mid.

The ring metadata is `[device_accessible_address, positive_record_capacity]`.
Each record has eight int64 words. Its address must be aligned and its allocation
must hold the declared number of records; allocation ownership remains with
the caller. Invalid length, rotary position, or ring metadata sets error bits
4, 8, or 16, mutes control, and avoids unsafe ring access. Muted prefill uses
zero positions/cache length and does not commit contexts or access ring metadata.
The kernels do not allocate or synchronize the stream and support CUDA graph replay.

## Migrating legacy packages

Use GenAI's `tools/qwen38_graph/convert_fold_to_contrib.py` on the target ONNX
graph. It adds explicit state outputs and connects the tail to the mid control
output, preserving external-weight paths. Remove `custom_ops_library` and
`custom_ops_libraries` from model session options only after all graph operations
are available in the rebuilt ORT stack. Use the matching GenAI integration:
its data-plane kernels are compiled into the normal GenAI CUDA add-on, not a
separate model-local helper library.

Build ORT with CUDA and contrib operators enabled. The plugin build registers
both schemas in the core and both kernels in the CUDA provider. Rebuild both,
not just the provider. Ordinary upstream wheels do not contain these additions.

`FoldContribV2Test.*` in `onnxruntime_provider_test` covers all S1-S4 prefix
lengths, hidden-row padding, deferred/wrapped publication, muted prefill, invalid
metadata, status aggregation, tail slots, and captured-graph replay.
