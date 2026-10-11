# Experimental runtime-owned KV cache

This branch is a coordinated **experimental ONNX fork**, not an upstream ONNX
release. It uses the dedicated `TypeProto.kv_cache_type`, IR 15 and standard-domain
opset 29. It does not encode caches as opaque values or serialize C++ objects.
CPU float32 is the only supported execution/payload type.

## Reproducible Windows build

Use a separate environment and build directory. No sibling checkout is modified.
The revision in `cmake/kvcache_onnx_revision.txt` is mandatory: configuration
rejects stock ONNX, a different commit, or tracked changes to the ONNX source.
The wrapper makes its own clone of the read-only sibling repository at that pin.
Do not substitute a stock ONNX wheel when generating these models.
The coordinated source revision is
`4443045c53d13a8d3157fcbd2cd5e52450a0784b`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install cmake ninja
python tools\kvcache\build.py --onnx-source <path-to-ONNX-fork-repository>
```

The wrapper source-builds Protobuf and ONNX in the same Release configuration,
avoiding installed-package debug/release runtime mismatches. The initial local
prerequisite configure discovered a debug-only shared Protobuf package; that
package is deliberately not used for the Release build.
Contrib CPU kernels remain enabled because Mobius exports
`com.microsoft::SkipSimplifiedLayerNormalization`; only unused ONNX-ML ops are
disabled.

The wrapper builds `onnxruntime_kvcache_test` and the Python extension, then a
Release wheel under `build\kvcache-native\Release\Release\dist`. It uses two compile
workers. Install that wheel and the matching sibling ONNX wheel into the
environment used by the driver. The native test executable is under
`build\kvcache-native\Release\Release`; run it from that directory. Run
the following with the installed wheels:

```powershell
$env:ALLOW_RELEASED_ONNX_OPSET_ONLY = "0"
python tools\kvcache\smoke_test.py
```

This explicit development opt-in is required before session creation. The
existing Python path/bytes loading paths do not honor the session-local
`session.allow_released_opsets_only` option; the process environment controls
their experimental-opset acceptance.

### Verified local artifacts

The Release build was exercised on Windows x64 with CPython 3.13.15,
VS 2022/MSVC 14.44 and the pinned ONNX wheel.
`onnxruntime_kvcache_test.exe --gtest_filter="KVCacheTest.*"` passed 12 tests.
The Python smoke test passed using the installed wheel, covering identical
model bytes across eight backend/window combinations. Native coverage includes
window-512 chunks `500,24,3,520,1`, logical masks across wrap, fully masked rows,
GQA, immutable ownership, later-layer failure, and rejected preallocated fetches.
The smoke graph also preserves the Qwen INT64-to-BOOL padding-mask conversion,
exercising the explicit CPU Cast-28 registration with existing supported dtypes.
An additional numerical test exercises the direct float32 Swish-24 kernel under
opset 29, including zero and negative alpha. Baseline graphs previously relied
on function expansion for Swish; lifted graphs require the explicit kernel.
Real pretrained model parity is performed by the coordinated Mobius driver;
these focused tests alone do not establish model-level parity.

The coordinated driver completed real pretrained CPU float32 comparisons:

| Model | Prefill / decode | Runtime-selected storage | Maximum absolute logit error |
| --- | --- | --- | --- |
| Qwen3-0.6B | 8 tokens / 3 single-token steps | contiguous, paged | 9.6321106e-5 |
| Gemma-3-270m | chunks 500,24,3 / 3 single-token steps | contiguous, paged, mixed ring/paged | 5.531311e-5 |

All steps passed `atol=rtol=1e-4` with matching greedy tokens. Qwen used model
revision `c1899de289a04d12100db370d81485cdf75e47ca`; Gemma used
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Gemma has 15 window-512
sliding layers and 3 full layers. Its computed ends were
`500,524,527,528,529,530`, sliding retained begins were `0,12,15,16,17,18`,
and full-layer retained begins remained zero. Thus the 24-token chunk crosses
the physical ring boundary without changing logical attention. Qwen's real
run does not cover multi-chunk prefill; Gemma and native tests provide that
coverage. The model owner persists per-backend machine-readable run records,
including model/wheel hashes, exact token IDs, and per-step cache states.

The wheel is `onnxruntime-1.32.0-cp313-cp313-win_amd64.whl`, SHA256
`ab1fc5ef214e69ce17ee566fec58ffe1a4a8bf4d0965cc97513cc1c9cd0205d4`.
The matching ONNX wheel is `onnx-1.24.0-cp312-abi3-win_amd64.whl`, SHA256
`03029ba7a488ee71bc1ad2f4647a695097e1eca95fb56000e52c89218482f744`.
Wheel hashes identify these local artifacts, not deterministic build outputs.

## Python feed/fetch API

```python
import os
os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"

cache = ort.OrtValue.kv_cache(
    batch_size=1, num_kv_heads=8, key_head_size=128, value_head_size=128,
    backend="paged", page_size=16, sliding_window=0,
)
options = ort.SessionOptions()
options.add_session_config_entry("session.allow_released_opsets_only", "0")
session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
feeds = {"past_cache": cache, "q": ort.OrtValue.ortvalue_from_numpy(q),
         "k": ort.OrtValue.ortvalue_from_numpy(k),
         "v": ort.OrtValue.ortvalue_from_numpy(v)}
y, next_cache = session.run_with_ort_values(["y", "present_cache"], feeds)
```

Graph cache input/output names are model-defined. Feed `next_cache` on the next
successful invocation. Do not convert cache objects to NumPy. Ordinary `run`
is not the object-valued interface. `kv_cache_describe()` returns
`computed_tokens`, `retained_begin`, and `backend`.

Choose `contiguous`, `paged`, or `ring` without changing model bytes. Ring
requires a positive `sliding_window` matching the model; full layers may remain
paged or contiguous while sliding layers use ring storage. For models with
`KVCacheCreate` nodes, the session config key `session.kv_cache.backend` selects
their backend (default contiguous).

In-tree native callers use `CreateKVCacheOrtValue(KVCacheOptions, OrtValue&)`
and `InferenceSession::Run`. `OrtValue` owns a `KVCacheValue` holding an immutable
shared cache version. C API type inspection identifies `ONNX_TYPE_KV_CACHE`,
not a tensor or opaque object. This prototype does not add a portable plugin EP
ABI or public C cache-construction function.

## Semantics and limits

Q/K/V use BNSD, with equal positive append/query counts and uniform batch ends.
GQA maps query head `h` to KV head `h / (Nq / Nkv)`. K/V contain only new tokens;
position inputs, if present, are int64 `[B,S]` and must exactly match the next
contiguous absolute positions. RoPE is performed by the model, not the cache.

Masks are boolean visibility or float32 additive bias broadcasting to
`[B,Nq,S,total_computed_keys]`; the final axis addresses absolute logical
positions, including evicted positions, never page/slot indices. Scale defaults
to `1/sqrt(Dk)` only when omitted; explicit zero is valid. Softcap precedes mask
bias. Fully masked rows produce zeros. Nonfinite Q/K/V, NaN or positive-infinity
mask bias, gaps, ragged batches, cross-attention, noncausal windows, quantization,
other payload/softmax precision, and GPU execution fail explicitly.

Window W includes the query token: visible keys satisfy
`q-W+1 <= p <= q`. Committed caches retain `[max(0,end-W),end)`, but a whole
chunk attends over the union of the previous retained history and all its new
tokens **before** eviction. A committed ring owns W physical slots per head;
absolute positions are not relabeled after wrapping.

All storage versions are functional. Contiguous appends allocate private
payload; paged versions share immutable old pages and allocate new pages;
ring appends stage a private union and only seal their final bank after attention.
No input object is mutated or destructively aliased, even when a later layer
fails. The caller must discard all fetches from a failed invocation and advance
its per-layer cache set only after the **entire** run succeeds. Computed end
counts tokens actually passed through the model, not sampled tokens.

The attention kernel streams blocks with a numerically stable running maximum,
normalization sum, and weighted-value accumulator. Paged attention does not
gather dense KV history or allocate a full score matrix. `KVCacheExport` is a
separate diagnostic materialization and must not be inserted into the production
attention path. `KVCacheDescribe` returns int64 `[B]` ends and `[B,2] retained
ranges. Continuous batching, prefix sharing, speculative acceptance, quantized
storage, offload and GPU optimization are outside this demonstration.

Cache fetches must be unallocated placeholders. Preallocating a cache output
(particularly with a caller input object) is rejected before execution; it could
otherwise overwrite a shared OrtValue wrapper. The Python object-valued `run`
interface creates fresh fetch placeholders automatically.
