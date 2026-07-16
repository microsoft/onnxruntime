---
name: ort-transformers-gpu-pytest
description: Run the ONNX Runtime transformers Python tests (onnxruntime/test/python/transformers) against a GPU wheel, and confirm real cuDNN/flash SDPA dispatch. Use when a transformers pytest fails with ModuleNotFoundError onnxruntime.capi, when torch's bundled CUDA/cuDNN libs shadow the ORT-built ones (wrong-version dispatch or load errors), or when you must PROVE a test exercised the cuDNN SDPA decode tier instead of silently skipping/falling back to MATH.
---

# Running ONNX Runtime transformers GPU pytest (dispatch-verified)

Reusable, hard-won knowledge for running the Python transformers tests under
`onnxruntime/test/python/transformers/` against a **GPU-built wheel**, on a box where
`torch` is also installed. Three gotchas below will each cost you an hour if
rediscovered. See the `ort-test` skill for the general test taxonomy and the
false-green modes; this skill is the GPU + transformers + SDPA-dispatch specialization.

## 1. NEUTRAL-CWD gotcha — never run pytest from the repo root

**Symptom:**

```
ModuleNotFoundError: No module named 'onnxruntime.capi'
# or an AttributeError deep inside onnxruntime import
```

**Cause:** the repo root contains a source directory `./onnxruntime/` (the C++/Python
source tree). When pytest runs with the repo root on `sys.path[0]`, `import onnxruntime`
resolves to that **source** package — which has no compiled `capi` extension — instead
of the **installed wheel** in your venv's `site-packages`. The source dir *shadows* the
wheel.

**Fix:** run pytest from a **neutral, private working directory** — a fresh
`mktemp -d`, not the repo root and not a bare shared `/tmp` — so the shadowing source
dir is not on the path, and point at the test file by absolute path. Put the
transformers test-helper dir on `PYTHONPATH` so shared helpers still import.

```bash
WORKDIR=$(mktemp -d); cd "$WORKDIR"       # NEUTRAL + PRIVATE cwd — NOT repo root, NOT /tmp
export PYTHONPATH=/abs/repo/onnxruntime/test/python/transformers
python -m pytest /abs/repo/onnxruntime/test/python/transformers/<file>.py -v
```

**Why `mktemp -d` and not a bare `cd /tmp`:** pytest prepends the cwd to `sys.path`, and
Python auto-imports `sitecustomize.py`/`usercustomize.py` from it at startup — so on a
shared box a co-tenant's planted `/tmp/sitecustomize.py` would execute as arbitrary code
in your test process. A fresh per-run `mktemp -d` (private, `0700`) keeps the neutral-cwd
source-shadow protection while removing that injection vector.

Do **not** `cd` into the repo and run `pytest onnxruntime/test/...` — that reintroduces
the shadowing. (This is the Python analogue of the C++ "run from the build output dir"
rule in `ort-test`.)

## 2. LD_PRELOAD lib-pinning gotcha — torch's CUDA/cuDNN shadow ORT's

**Symptom:** any of —
- the SDPA decode tier silently routes to `MATH` instead of `CUDNN_FLASH_ATTENTION`
  (wrong-version cuDNN loaded), or
- `libcudnn.so.9: cannot open shared object file` / `undefined symbol` / cuDNN version
  mismatch errors at first CUDA op, or
- ORT loads a different CUDA runtime than it was built against.

**Cause:** a pip-installed `torch` ships its **own** bundled CUDA runtime + cuDNN
(e.g. cu124 → CUDA 12.4 / cuDNN 9.1) under `site-packages/nvidia/*/lib`. If ORT was
built against a **different** CUDA/cuDNN (e.g. CUDA 12.9 / cuDNN 9.8), whichever set the
dynamic loader resolves **first** wins. With torch imported (or its libs on the path),
torch's older libs can shadow the ones ORT `dlopen`s → wrong-version dispatch or load
failure.

**Fix:** `LD_PRELOAD` the **system** CUDA runtime + cuDNN that ORT was built against so
they are loaded first, and add their dirs to `LD_LIBRARY_PATH`. Activate the venv that
has the ORT wheel. Concrete form used successfully (CUDA 12.9 + cuDNN 9.8; substitute
your absolute lib paths):

```bash
WORKDIR=$(mktemp -d); cd "$WORKDIR"       # neutral + private (see §1)
source /abs/repo/.venv/bin/activate
export LD_PRELOAD=/abs/cuda12.9/lib64/libcudart.so.12:/abs/cudnn9.8/lib/libcudnn.so.9
export LD_LIBRARY_PATH=/abs/cuda12.9/lib64:/abs/cudnn9.8/lib
export PYTHONPATH=/abs/repo/onnxruntime/test/python/transformers
python -m pytest /abs/repo/onnxruntime/test/python/transformers/<file>.py -v
```

Notes:
- `libcudart.so.12` is correct for **both** CUDA 12.4 and 12.9 (SONAME is major-only) —
  pinning the 12.9 file forces the right minor.
- `torch.cuda` still works fine under this preload — the bf16 IO-binding path that uses
  `torch` tensors + `.data_ptr()` runs correctly.
- Keep the two exports and the preload together; dropping `LD_LIBRARY_PATH` can still let
  a transitive dependency resolve against torch's copy.

## 3. Confirm REAL cuDNN SDPA dispatch (don't trust value-equality)

A numerically-correct result does **not** prove the cuDNN SDPA path ran — the kernel has
a `MATH` fallback that produces the same answer (false-green mode 4 in `ort-test`). To
prove the tier dispatched, **observe** ORT's routing rather than probing a version.

### Observe-dispatch (the correct probe)

ORT's ONNX-domain Attention kernel emits a debug line when
`ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1` is set **before** the `InferenceSession` is
created (the option is read once at session creation). `AttentionKernelDebugInfo::Print`
emits a token of the form:

```
SdpaKernel=CUDNN_FLASH_ATTENTION      # or =MATH, =FLASH_ATTENTION, =EFFICIENT_ATTENTION (non-exhaustive)
```

Capture stdout across a single `run()` and parse it. **Capture at the file-descriptor
level, not `contextlib.redirect_stdout`:** the `SdpaKernel=` line is written to native
fd-1 from C++, which Python-level stdout redirection never intercepts — you would get
`dispatched=None`, a silent false-negative. Mirror ORT's own `_CaptureStdout`
(`os.dup2` fd-1 to a temp file, run, restore, read it back):

```python
os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"   # BEFORE InferenceSession()

# FD-level capture (see onnxruntime's _CaptureStdout for the exact idiom):
saved_fd = os.dup(1)
tmp = tempfile.TemporaryFile()
os.dup2(tmp.fileno(), 1)            # redirect native fd-1
try:
    # ... create session, run once ...
finally:
    os.dup2(saved_fd, 1)           # restore fd-1
    os.close(saved_fd)
tmp.seek(0)
captured_text = tmp.read().decode()

m = re.search(r"SdpaKernel=(?P<kernel>[A-Z_]+)", captured_text)
dispatched = m.group("kernel") if m else None
assert dispatched == "CUDNN_FLASH_ATTENTION"
```

Caveat: `re.search` returns only the **first** `SdpaKernel=` token — correct for the
single-node decode probe here. For a graph with **multiple** attention nodes use
`re.findall` and check every token, or a later node's `MATH` fallback is masked by an
earlier cuDNN hit.

Prefer this over reading `torch.backends.cudnn.version()` or any library-version check:
a version probe reads **torch's** cuDNN, not the cuDNN ORT actually loaded/dispatched —
that mismatch is a real trustworthiness bug. Observe-dispatch reads ORT's own routing
decision, so it is correct across cuDNN versions with no hard-coded version table.

### Non-skippable canary — `ORT_TEST_REQUIRE_CUDNN_SDPA`

Gating decode tests by observed dispatch has a failure mode: if the tier silently
regresses (stops selecting cuDNN), the observation returns "not dispatched" and every
decode test **skips green**, hiding the regression as all-green skips.

Close the hole with an env-gated canary. On a known-good GPU CI leg the operator sets
`ORT_TEST_REQUIRE_CUDNN_SDPA=1`; when set, the dispatch assertion becomes
**non-skippable** — a `MATH` fallback / non-dispatch on the minimal known-good config
**FAILS LOUD** instead of skipping. When unset (dev boxes, unsupported cuDNN) it falls
back to the normal skip guard so it never false-alarms.

```python
def require_cudnn_sdpa():
    return os.environ.get("ORT_TEST_REQUIRE_CUDNN_SDPA") == "1"

# in the test:
enforce = require_cudnn_sdpa()
if not enforce and not cudnn_decode_supported(head_size):  # illustrative: your suite's own support predicate
    self.skipTest("cuDNN SDPA decode tier not dispatched; set ORT_TEST_REQUIRE_CUDNN_SDPA=1 to enforce")
# then assert dispatch == CUDNN_FLASH_ATTENTION unconditionally
```

Run both ways to prove it works AND bites:

```bash
python -m pytest <file>.py -v                              # normal: skips where unsupported
ORT_TEST_REQUIRE_CUDNN_SDPA=1 python -m pytest <file>.py -v # enforced: fails if not cuDNN
```

**Prove the teeth.** A canary you never watched fail is not verified. Force `MATH`-only
by setting the CUDA provider's `sdpa_kernel` **provider option** to the MATH bitmask
(`16`) — a monkeypatch of the C++ selector is not reachable from Python — under
`ORT_TEST_REQUIRE_CUDNN_SDPA=1`, and confirm it fails with, verbatim:

```
AssertionError: 'CUDNN_FLASH_ATTENTION' != 'MATH'
```

A run that never demonstrates this failure has not proven the canary has teeth (grounding
rule: negative/teeth evidence must actually be observed, not asserted).

## 4. Putting it together — one clean run block

```bash
WORKDIR=$(mktemp -d); cd "$WORKDIR"       # neutral + private (see §1)
source /abs/repo/.venv/bin/activate
export LD_PRELOAD=/abs/cuda12.9/lib64/libcudart.so.12:/abs/cudnn9.8/lib/libcudnn.so.9
export LD_LIBRARY_PATH=/abs/cuda12.9/lib64:/abs/cudnn9.8/lib
export PYTHONPATH=/abs/repo/onnxruntime/test/python/transformers
F=/abs/repo/onnxruntime/test/python/transformers/<file>.py

python -m pytest "$F" -v                               # A: normal
ORT_TEST_REQUIRE_CUDNN_SDPA=1 python -m pytest "$F" -v  # B: canary active, non-skippable
# C: teeth — force MATH under the env var, expect the AssertionError above
```

**Check the passed count, not just the exit code.** `pytest -v` exits **0 even if every
test skipped** (no CUDA, or an unmet `@skipUnless(ml_dtypes)` guard) — the saved log then
looks like passing evidence but proves nothing. Require a **non-zero passed count and
zero unexpected skips**, and note pytest **exit code 5 = "no tests collected"** (usually a
wrong path or `-k` filter, not success). RUN B's canary only converts *dispatch-related*
skips into failures — it does **not** rescue collection or environment skips, so still
read the summary line.

Redirect to a log (`... 2>&1 | tee "$WORKDIR/gpu_run.log"`) — the debug-info stdout and
pytest output are large, and a saved log is the evidence that the run happened and
dispatched to cuDNN. Write it **inside** `$WORKDIR` (the `mktemp -d` above), not a
predictable `/tmp/gpu_run.log` a co-tenant could pre-create as a symlink to clobber.

## Gotcha quick-reference

| Symptom | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: onnxruntime.capi` | repo-root `./onnxruntime/` source shadows the wheel | run pytest from a private `mktemp -d` (not repo root, not bare `/tmp`); abs path + `PYTHONPATH` |
| routes to `MATH` / cuDNN load error | torch's bundled CUDA/cuDNN shadow ORT's | `LD_PRELOAD` system `libcudart.so.12` + `libcudnn.so.9`, set `LD_LIBRARY_PATH` |
| test passes but path unproven | `MATH` fallback gives same numbers | observe `SdpaKernel=` via `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1` |
| all tests skip green, regression hidden | dispatch-gated skip | `ORT_TEST_REQUIRE_CUDNN_SDPA=1` makes assertions non-skippable |
