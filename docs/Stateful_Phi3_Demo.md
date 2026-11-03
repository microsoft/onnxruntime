# Local Phi-3 stateful text generation

This demonstration uses the public
[`onnxruntime.stateful` API](Stateful_Python.md), not a Python generation engine:

1. A sampler-free SessionLoop processes the full prompt and exits on EndOfInputs.
2. A decode SessionLoop samples its first token from prefill logits without
   invoking the decoder, then performs native greedy decode.
3. A SessionChain shares64 managed key/value caches between those stages.

The [CLI and reusable helpers](../samples/python/phi3_demo.py) produce separate
portable CPU/CUDA packages. Each package references one ONNX file and external
weight file for both stages; weights are not duplicated for prefill/decode.

## Requirements and approved assets

- An M13-or-later local ONNX Runtime Python build/wheel. Native GPU cache/greedy
  requires the **CUDA plugin** build, a matching CUDA/cuDNN installation and a
  supported NVIDIA GPU. Legacy CUDA is not a native-cache/sampler substitute.
- Optional [demo dependencies](../samples/python/requirements-phi3-demo.txt),
  installed in an isolated environment. This file intentionally does not install
  a published ORT wheel or add tokenizer dependencies to the runtime.
- Approximately10.4GB for the selected models and tokenizer metadata, plus
  working space. Do not put these files in Git or a space-constrained build tree.

The explicit setup operation fetches only approved assets from
[microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx),
revision `5f5f794c1c23c9d5ee142af85df02a6cc52d6945`:

| Target | Variant | Logits/cache dtype |
|---|---|---|
| CPU | `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` | FP32 |
| CUDA | `cuda/cuda-fp16` | FP16 |

Both graphs have INT64 token/mask inputs,32 layers,32 KV heads, head size96,
vocabulary32064 and a4096-token context bound. The actual graph is inspected;
weight quantization is not used to guess cache dtype.

Graphs/weights use published LFS SHA-256 and size checks. Supporting assets use
their pinned Git blob identity and recorded SHA-256. Invalid existing files fail
rather than being silently overwritten. New downloads are validated before
atomic publication. Downloaded Python model/configuration implementations are
never fetched or executed. The MIT license is retained in each variant folder.

## Prepare the packages

Use the local CPU wheel for CPU commands and the local CUDA-plugin wheel for
CUDA commands. If using isolated `--target` wheel installations, set `PYTHONPATH`
to the relevant installation. Verify the actual imported extension, not just
the package version.

Run outside the repository root to avoid source-package shadowing:

```bash
ROOT=/path/to/onnxruntime
source "$ROOT/.venv/bin/activate"
python -m pip install -r "$ROOT/samples/python/requirements-phi3-demo.txt"
cd "$HOME"
ASSETS="$HOME/onnxruntime-stateful/phi3-mini-4k-instruct"
DEMO="$ROOT/samples/python/phi3_demo.py"

# Explicit network-enabled setup, pinned to the revision above:
python "$DEMO" --assets "$ASSETS" --target cpu acquire
python "$DEMO" --assets "$ASSETS" --target cuda acquire

# These commands use existing local assets only:
python "$DEMO" --assets "$ASSETS" --target cpu inspect
python "$DEMO" --assets "$ASSETS" --target cuda inspect
python "$DEMO" --assets "$ASSETS" --target cpu build --max-new-tokens 16
python "$DEMO" --assets "$ASSETS" --target cuda build --max-new-tokens 16 --device-id 0
```

The resulting package roots are `"$ASSETS/cpu"` and `"$ASSETS/cuda"`.
Each contains `manifest.json` and `decoder/<target>/` with the graph, external
weights, tokenizer/configuration files and license. The manifest records revision,
target, declared budget/device and asset hashes.

`build` constructs a hardlinked temporary portable package and initializes its
actual native chain before atomically publishing the manifest. A validation
failure preserves the previous manifest. Temporary hardlinks do not copy weights.
Building validates compatibility, not merely JSON syntax.

## Run offline

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python "$DEMO" --assets "$ASSETS" --target cpu run \
  --prompt 'Answer briefly: what is 2 + 2?'
python "$DEMO" --assets "$ASSETS" --target cuda run \
  --prompt 'Answer briefly: what is 2 + 2?'
```

For the qualified CUDA13.0/cuDNN9.23 environment, the matching loader paths were:

```bash
export LD_LIBRARY_PATH="/home/tianlei/cudnn_9.23_cuda13/lib:/home/tianlei/cuda13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use paths matching your own build; do not install/change global drivers to run
the demo. The actual plugin library is `libonnxruntime_providers_cuda.so`.

The local tokenizer's chat template and assistant prefix are applied by default.
`run --raw` bypasses the template without silently adding special tokens.
Local-only loading and `trust_remote_code=False` remain enabled in both modes.

JSON output includes generated continuation text, all generated token IDs,
termination reasons, model invocation counts, observed sampler routing and timings.
`setup_seconds` includes asset verification, tokenizer work and Session creation;
`generation_seconds` measures native Run. These are not optimized benchmark or
speedup claims.

The budget is declared at package construction, not overridden during execution.
Rebuild explicitly to change it. EOS IDs32000,32001 and32007 are declared from
the pinned generation metadata; an observed EOS remains in token-level results
while special tokens are omitted from rendered text.

Empty prompts, invalid token IDs, missing/corrupt assets, unsupported EP devices
and prompt-plus-budget context overflow fail explicitly. Nothing is silently
truncated, re-exported, downloaded during Run or replaced with another variant.

One intra-op thread is intentional for these independently owned stage Sessions.
On the qualification host, an initial eight-thread configuration exceeded a
300-second diagnostic bound; that is recorded as a failed diagnostic, not passing
evidence. The single-thread configuration completed CPU qualification.

## Python adapters and native contexts

Import the demo module from its source directory; it is application tooling,
not part of the installed ORT namespace.

```python
import phi3_demo

class Adapter:
    def encode(self, text: str, *, raw: bool = False) -> list[int]:
        ...  # Application-owned tokenization/formatting.

    def decode(self, tokens: list[int]) -> str:
        ...  # Render continuation, omitting special tokens as appropriate.

result = phi3_demo.generate(package_root, prompt, Adapter())
execution, adapter, prompt_length = phi3_demo.create_execution(
    package_root, prompt, Adapter()
)
with execution, execution.run() as native_result:
    tokens = native_result.get_outputs()["tokens"].numpy()
```

For a complete local adapter, use `LocalTokenizer(package_root / "decoder" /
target)` or omit the adapter in `generate`. The native context retains its inputs
and Sessions; callers must close it and preserve the ordinary
[stateful ownership/cancellation/reset contracts](Stateful_Python.md).

## Differential validation

```bash
python "$DEMO" --assets "$ASSETS" --target cpu validate \
  --prompt 'Answer briefly: what is 2 + 2?'
python "$DEMO" --assets "$ASSETS" --target cuda validate \
  --prompt 'Answer briefly: what is 2 + 2?'
```

Validation uses an ordinary stateless InferenceSession reference for the same
variant and tokenized input. It compares all component logits/cache outputs at
each executed step, greedy tokens, bootstrap/invocation counts, cache device/
length, unchanged retained prefill snapshots and native reset/replay.
Its full GPU readbacks are intentional validation work, not normal decoding.

CPU thresholds are `rtol=1e-4, atol=1e-4`; CUDA FP16 thresholds are
`rtol=1e-2, atol=2e-2`, with nonfinite mismatches rejected. Qualified runs measured
zero maximum absolute checkpoint error for both variants. Greedy token mismatch
fails explicitly, including near ties; tolerances are not expanded to hide it.
CPU INT4 and CUDA FP16 need not generate identical prose.

The explicit real-asset suite additionally validates budgets0/1/16 and typed
programmatic/package construction parity. Fixture tests cover first-EOS and
other boundaries that real prose cannot reliably force:

```bash
python -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py"
export ORT_STATEFUL_PHI3_ASSETS="$ASSETS"
python -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" -k test_cpu_real
# Use the CUDA-plugin wheel for the CUDA selector:
python -m pytest -q "$ROOT/onnxruntime/test/python/onnxruntime_test_python_phi3_demo_assets.py" -k test_cuda_real
```

The standard full Python build runner executes only offline fixture tests.
Real-asset tests require the explicit environment variable and never download.

## GPU evidence and scope

Qualified CUDA placement has all32 GQA and161 MatMul nodes on CUDA. One
attention-mask shape Gather/Cast pair intentionally runs on CPU.
Dedicated normal-decode tracing captured1,722 GPU kernels on ordinal0, including
three native greedy-partial and three greedy-finish kernels.

Normal prefill/bootstrap/two-decode execution reads back only three16-byte
token/status records: **48 bytes**, no full GPU logits or KV-cache payload.
There are128 intentional device-to-device cache-prefix copies across64 caches.
A separate explicit-copy range records one98,304-byte DtoD clone and one
98,304-byte DtoH cache copy.

Direct real-model memcheck/full-leak, racecheck and synccheck workers pass with
zero errors/leaks/hazards. They include actual decode, not bootstrap alone, and
post-context-close access to retained GPU values. The inherited
`--show-backtrace no` workaround avoids Compute Sanitizer2025.3 retaining Python
constructor frames; memory/leak/API/exit checks remain enabled.

No new native runtime feature or CUDA kernel is introduced. M15 diffusion,
additional model variants, batching, chunked prefill, stochastic sampling, async/
streaming APIs and model rewriting remain outside this milestone.
