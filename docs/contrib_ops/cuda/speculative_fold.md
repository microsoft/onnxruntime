# CUDA SpeculativePrefixCommit operator

`com.microsoft::SpeculativePrefixCommit` (schema version 1, CUDA EP only) is the commit step of a speculative
decoding verification pass. For every request in the batch it accepts the longest prefix of draft tokens that the
verifier confirmed, advances the persistent control state, selects the committed context for the accepted length, and
publishes a record to a host-visible result ring that the runtime polls. It runs entirely on the device and can be
captured in a CUDA graph (no allocation, no synchronization).

## Interface

| Kind | Name | Type / shape | Meaning |
|---|---|---|---|
| attribute | `head_rows` | int, 3..8, default 8 | Number of verifier rows. Non-muted requests must provide exactly this many rows. |
| input 0 | `next_tokens` | int64 `[batch, rows]` or `[rows]` (batch one) | Verifier predictions. Any other rank is rejected. |
| input 1 | `input_ids` | int64, `batch * stride` elements, `stride >= rows` | Verifier input; the draft tokens are `input_ids[b, 1:rows]`. |
| input 2 | `status` | int32, any size | Verifier status words, OR-ed together and shared by all batch items. |
| input 3 | `ctl` | int64, 16 words per item | Control state (see below). |
| input 4 | `ctx_commit` | int64, 2 words per item | Committed context pair. |
| input 5 | `prefixes` | int64, `>= 2 * head_rows` words per item | Context pair `k` (words `2k`, `2k+1`) is committed when `k` drafts are accepted. |
| input 6 | `ptrs` | int64 `[2]` | `{result ring address, records per batch item}`. |
| output 0 | `head_c` | int32 `[batch]` | Cache length before the commit; zero for muted or invalid requests. |
| output 1 | `updated_ctl` | int64, shape of `ctl` | Updated control state. **Required.** |
| output 2 | `updated_ctx_commit` | int64, shape of `ctx_commit` | Updated committed context. **Required.** |

`ctl` words used by the operator: `[0]` pending token, `[1]` length, `[3]` muted flag, `[4]` ring sequence number,
`[5]` error flags, `[6]` accepted count, `[7]` cache length. All other words are copied unchanged.

## State handling

The operator is stateful but never mutates its inputs through hidden aliasing. It reads `ctl` / `ctx_commit` and
always writes the new state to `updated_ctl` / `updated_ctx_commit`. Each output may be the same device buffer as its
input (in-place, which the kernel declares via `MayInplace`) or a disjoint buffer; partially overlapping buffers are
rejected. The kernel copies the unchanged words when the buffers differ, and each batch item is handled by a single
thread, so aliasing is safe.

A caller that keeps the state in persistent device buffers must bind the two outputs to those buffers (for example with
`IOBinding`, passing the same device pointer as the input). An output that ORT allocates itself is a separate buffer,
so the update would be lost unless the caller feeds the returned tensors into the next run. Callers must serialize
access to the state buffers.

## Result ring

`ptrs[0]` is the device-accessible address of an 8-byte aligned allocation that holds `batch * ptrs[1]` records of
eight int64 words; batch item `b` uses records `[b * ptrs[1], (b + 1) * ptrs[1])` and writes sequence `s` to record
`(s - 1) % ptrs[1]`. Allocation ownership stays with the caller. Words (`low | high << 32` where two values share a
word):

| Word | Contents |
|---|---|
| 0 | sequence number (`ctl[4]` after the commit), written last after `__threadfence_system()` |
| 1 | `tokens[0]` \| `tokens[4]` |
| 2 | `tokens[1]` \| `tokens[5]` |
| 3 | accepted count \| `tokens[6]` |
| 4 | OR of the status words |
| 5 | pending token \| `tokens[7]` |
| 6 | new length |
| 7 | `tokens[2]` \| `tokens[3]` |

Tokens beyond `rows` are stored as zero. A host poller that reads word 0 first never observes a partial record.

## Muted requests and errors

A request with `ctl[3] != 0` (for example during prefill) does not commit state, does not access the ring, reports
`head_c = 0`, and may use any number of `next_tokens` rows. Invalid device metadata mutes the request and sets error
flags in `ctl[5]`: `4` for a negative length or cache length, a non-muted row count different from `head_rows`, and `16`
for an invalid ring (null or misaligned address, non-positive or overflowing record count, bad sequence number). A
non-zero OR of the status words sets flag `2`.

## Using it in a graph

Replace the previous fold node as follows: connect the seven inputs above, drop any rotary-cache inputs and the
duplicate `prefix2` input, and expose `updated_ctl` and `updated_ctx_commit` as graph outputs (or otherwise bind them)
so the runtime can bind them to its persistent control and context buffers. Only `head_c` is typically consumed as a
regular graph value.

Build ORT with CUDA and contrib operators enabled. The schema lives in the core library and the kernel in the CUDA
provider, so both must be rebuilt; ordinary upstream wheels do not contain this operator.

## Tests

- `FoldContribTest.*` (`onnxruntime_provider_test`, kernel level) covers every accepted-prefix length for 3, 4 and 8
  rows, in-place and separate state outputs, batched per-item state and ring, muted prefill, invalid metadata, the
  status union, and captured-graph replay.
- `SpeculativeFoldTest.*` (session level) covers batched S8 commits through `InferenceSession`, carrying the returned
  state into later runs (the sequence counter advances), muted prefill with more rows than `head_rows`, shape inference
  for known and unknown `next_tokens` shapes, and rejection of ranks other than 1 and 2.
