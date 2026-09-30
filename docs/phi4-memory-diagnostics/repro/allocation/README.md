# Allocation-fix reproduction (Phase A → Phase B)

This is a focused adaptation of the historical allocation experiment, not the
rotary benchmark, a synthetic result generator, or a copy of the old controllers.
Source trace and original SHA-256 values are in [provenance.json](provenance.json).
`production.diff` is byte-identical to the historical three-line removal:
`34723df6327bb3430ff00a05d8329d3c69660ceaf67a7efd3d2b268314124e38`.

**Status:** the correction phase passed 73 host tests and 26 integration checks
across the package. Packaged Phase A and Phase B applications compiled/linked
against both recorded historical GenAI variants using GCC 13.2, CMake 4.1.2
and CUDA 13.0.48. ORT/GenAI and native regression-test targets were not rebuilt.
Application execution, actual runtime loading and fresh GPU replay were not
demonstrated by this build-only check.
Fresh replay from this artifact alone is **BLOCKED**: saved inputs,
model, diagnostic ORT/runtime builds and tiny regression model fixtures are
external prerequisites. The complete historical native regression target/source
is included as a separate patch; no private instrumentation framework is needed. Missing
prerequisites are errors, not skipped tests or a reduced correctness gate.
No command downloads, installs, stages, publishes, or falls back to another model.

## Source and data contract

- GenAI pin: `ed5f4e87147731e5b07810f9f5c90103b3603cdf`.
- ORT pin: `94e76459dc9a759888707aa430d51a95d878da8f`.
  This public `microsoft/onnxruntime` benchmark pin already includes
  `OrtApi_DebugLogAndShrinkGpuArenas_SinceV29`; no reconstruction patch is
  required. Supply matching built headers and libraries from that pin.
  A stock pip wheel without that API cannot measure these snapshots.
- Existing evidence is read from
  [`../../evidence/model-manifest.json`](../../evidence/model-manifest.json).
  No evidence files are rewritten.
- External inputs are **headerless signed int32 little-endian (`i32le`)**, not
  JSON, text, a prompt to retokenize, or newly generated replacements:

  | Nominal context | Token count / bytes | SHA-256 of complete i32le file |
  |---|---:|---|
  | 2048 | 1801 / 7204 | `a7100ece1396cf13e81616381b11fed562406cadf539c72a7cae8c955b0877f4` |
  | 32768 | 28747 / 114988 | `fbe6fe0929666ac5a91dd152aea07bc2bf7675c765f31ad57e701604ccaa3831` |

Saved input vectors and raw capture data are **not approved for publication** and
are not included here. Keep supplied inputs and all future output directories
external to this artifact. Generated output includes those vectors and generated
tokens/logits; do not commit or publish it without separate approval.

The workload's `source_config_sha256` and embedded `effective_config` are
authoritative. The manifest also records the downloaded `genai_config.json`
artifact hash; it differs from the locally edited configuration used in the
experiment. The checker deliberately requires the workload configuration, not
the untouched downloaded configuration. Other model artifacts are fully hashed.

## Host-only commands

From this directory, with an existing Python **3.11+** interpreter:

```bash
python -B allocation.py --help
python -B allocation.py config
python -B allocation.py check
python -B -m unittest -v test_allocation
# If pytest is already available:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q -p no:cacheprovider --confcutdir=. test_allocation.py
```

`config` prints the template; it does not silently fill missing paths or digests.
`check` without a config verifies artifact/manifest contracts only, **not replay
readiness**. The unit tests use small invented test fixtures, not historical input
vectors or measurements. They remove their project-local scratch files.

Prepare a JSON config **outside** this artifact using `config.example.json`.
Every path must be absolute. Replace all path/digest/UUID placeholders; explicitly
choose resource limits. `output_dir` must not exist and must not overlap any
input/model/runtime/toolkit path. No resume, overwrite, or automatic retry exists.

```bash
python -B allocation.py check --config /absolute/external/allocation-config.json
python -B allocation.py check-fixtures --models-dir /absolute/external/genai-source/test/models
```

This remains host-only: checks files/hashes, resources, tool availability and ELF
metadata using `readelf`/`nm`. It does not initialize NVML, load a runtime, execute
an app, use `ldd`, or contact a GPU. Library source pins are operator declarations;
file digests bind supplied artifacts but do not independently prove their build
origin. Preserve external source/build provenance separately.

## External prerequisites and optional build recipes

Required: Linux `/proc`, Python 3.11+, `nvidia-ml-py` (`pynvml`, imported only by
`run`), NVML permissions for **compute and graphics** process enumeration,
CUDA toolkit 13, compatible cuDNN, and the pinned model/runtime files.
`readelf` and `nm` are required even for configured checks. Both variants use
identical ORT core/provider binaries and different GenAI core binaries.

Supply or build both test executables per variant: `logits_allocation_tests`
and `unit_tests`. The byte-identical
[`historical-tests-normalized.patch`](historical-tests-normalized.patch) includes
the entire historical `test/logits_allocation/tests.cpp` and its CMake target:
SHA-256 `1485beda53090cc53f4c1cfc0a19c3938d2f4f720a881632e7398d25eb7f038b`.
It makes **no production-source changes**. Preserve its mixed CRLF/LF bytes.
Do not substitute later review-bundle tests: selectors/counts differ.

For each variant, `genai_library_dir`, `allocation_tests.path`, and
`existing_tests.path` must refer to the same complete build directory produced
by `build_runtime.sh`: both test executables sit beside
`libonnxruntime-genai.so` and `libonnxruntime-genai-cuda.so`. The historical
allocation test links GenAI objects into its executable; the pinned loader
looks for its CUDA companion beside that executable, not via `LD_LIBRARY_PATH`.
Config validation rejects detached test executables (using resolved paths), and
host preflight verifies the adjacent companion against that variant's configured
SHA-256. Missing or mismatched companions block before GPU initialization.
Checks do not copy libraries, repair layouts, or prove that the native loader
executes successfully.

[`fixture-manifest.json`](fixture-manifest.json) records nine exact file sizes
and SHA-256 hashes under `test/models/hf-internal-testing/`: CPU
`tiny-random-gpt2-fp32`, CUDA `tiny-random-gpt2-fp16-cuda`, and CUDA
`tiny-random-gpt2-fp32-cuda`. The last fixture is also exercised inside both
selected CUDA model tests. All nine historical hashes were additionally verified
against existing local Git blobs at the public GenAI pin above; no downloads or
fixture binaries were copied into this artifact. The existing public source's
`test/model_tests.cpp`, `test/c_api_tests.cpp`, and CMake `MODEL_PATH` definition
were inspected to verify the complete selected fixture chain.

Materialize those exact files in each external pinned source checkout; do not
regenerate models. `test_models_dir` in each runtime config must name that
checkout's `test/models` directory, which CMake compiles into the executables.
Host preflight and the runtime build recipe verify fixture hashes before tests.
Supplied executable hashes bind binaries; operators must preserve their build
provenance, including this compiled fixture location.

The scripts below require already available sources/dependencies/tools, use at
most two parallel build jobs, and never obtain missing dependencies. The
parameterized application recipe was build-validated against existing pinned
runtimes; source preparation, runtime builds and regression-target rebuilds were
not performed in that check.

```bash
# Optional acquisition from an ALREADY AVAILABLE LOCAL public-GenAI seed.
# Every variable is an absolute local path; no remote clone/fetch is performed.
git clone --no-hardlinks "$GENAI_SEED" "$BASELINE_SOURCE"
git clone --no-hardlinks "$GENAI_SEED" "$PATCHED_SOURCE"
git -C "$BASELINE_SOURCE" checkout --detach ed5f4e87147731e5b07810f9f5c90103b3603cdf
git -C "$PATCHED_SOURCE" checkout --detach ed5f4e87147731e5b07810f9f5c90103b3603cdf

# Host checks and source preparation only; existing clean checkouts required.
python -B allocation.py check-fixtures --models-dir "$BASELINE_SOURCE/test/models"
python -B allocation.py check-fixtures --models-dir "$PATCHED_SOURCE/test/models"
bash prepare_sources.sh "$BASELINE_SOURCE" "$PATCHED_SOURCE"

# DEPS has googletest-src, onnxruntime_extensions-src, gsl-src,
# nlohmann_json-src, dr_libs-src, and dlib-src, already materialized.
PYTHON="$PYTHON" bash build_runtime.sh "$BASELINE_SOURCE" "$ORT_HOME" "$CUDA_TOOLKIT" "$CUDNN_LIB" "$DEPS" 80 "$BASELINE_BUILD"
PYTHON="$PYTHON" bash build_runtime.sh "$PATCHED_SOURCE" "$ORT_HOME" "$CUDA_TOOLKIT" "$CUDNN_LIB" "$DEPS" 80 "$PATCHED_BUILD"
bash build_apps.sh "$ORT_HOME" "$BASELINE_SOURCE" "$BASELINE_BUILD" "$CUDA_TOOLKIT" "$BASELINE_APPS"
bash build_apps.sh "$ORT_HOME" "$PATCHED_SOURCE" "$PATCHED_BUILD" "$CUDA_TOOLKIT" "$PATCHED_APPS"
```

`PYTHON` is an existing Python 3.11+ interpreter. All variables denote explicit
absolute external paths. The preparation script verifies both source pins and
clean trees, checks both patches before applying, applies the regression patch to
both trees, and applies `production.diff` only to patched. No private original
directory or prior preservation controller is used. The two app directories provide per-variant
`phi_phase_a`. Configure a single `phi_phase_b` (for example the baseline-linked
one) for both variants. Its RUNPATH is overridable; the child environment selects
the variant, and loaded core/CUDA companion/ORT/provider paths **and hashes** are
checked before GO and after inference. Build artifacts include hash listings and
Phase B ELF dynamic metadata. Configure their real SHA-256 digests before replay.

## Conditional GPU execution and report

Only after prerequisites are supplied and the configured check succeeds:

```bash
python -B allocation.py run --config /absolute/external/allocation-config.json
python -B allocation.py report --results /absolute/external/new-allocation-run
```

`run` is the **only** GPU command. It cannot bypass these stages:

1. Five regression jobs: baseline allocation (9 tests; exactly six expected
   `NoPromptSizedFp32/{0..5}` oversized-allocation failures), baseline semantics
   (20), patched regression (29), baseline and patched related existing tests
   (7 each). Skips, missing fixtures/tests, OOM or other failures abort.
2. Four Phase A captures: both variants at both saved input lengths. Capture all
   `[1,1,200064]` float32 **prefill** logits before token generation and exactly
   64 greedy tokens. Full logits and token bytes must agree between variants.
   Metadata alone, top-k comparisons, token-only checks and stale booleans cannot
   open the Phase B gate; actual Phase A files are revalidated for every run.
3. Twelve fresh serial Phase B processes: 2048 then 32768, repetitions 1–3,
   baseline then patched in each pair. No diagnostic logits calls, profiling,
   verbose allocation trace or shrinking. TTFT starts immediately before
   `AppendTokens` and ends after first-token completion synchronization; decode
   is **63 / (end − first-token)** with final-token completion synchronization
   included. Initialization/cleanup snapshots are outside timing. GO/ACK
   handshakes check cross-process monotonic-clock boundaries.

Each launch enforces 2-second cooldown, 50 idle NVML samples, no foreign compute
or graphics PIDs, process-query availability, explicit host/swap/disk/GPU-memory
resource limits, 300-second watchdog and 5-second termination grace. Sampling
requests 10 ms; reports disclose actual gaps. Peaks use only complete NVML
queries inside inference with available process attribution. Device and
pinned-host arenas remain separate, and paired differences remain signed.

### Output artifacts

Everything is below the new configured external output directory:

- `config.json`, `manifest.json`, `status.json`: configuration, supplied/runtime
  identities, local runner hashes, GPU identity, plan and final state.
- `test-results/*`: commands, stdout/stderr, monitoring, GTest XML/classification;
  `gate.json` exists only after all required classifications pass.
- `phase-a/{baseline,patched}-{2048,32768}/`: saved/loaded inputs, effective/applied
  config/readback, loaded maps/hashes before and after, full logits/metadata,
  64 generated tokens, state, commands, watchdog/resource/NVML capture.
  `complete.json` records byte-equal comparisons only after all four succeed.
- `measurement-results/context-*-rep-*-*/`: equivalent identity/monitoring files,
  `timing.json`, `protocol.json`, non-shrinking snapshot calls and arena stdout,
  tokens, `record.json`. No full logits are copied during Phase B.
- `measurement-results/{records.json,summary.json,runs.csv,arenas.csv}`:
  raw-derived rows, median/min/max groups and paired baseline-minus-patched
  differences. `report` revalidates real raw outputs and gates before printing
  summary JSON; it cannot replay a historical summary as a fresh result.

In each `summary.json` group, `arenas_mib` contains only byte-valued arena
fields converted to MiB (including `max_alloc_size`). `arena_counts` contains
unscaled median/min/max values for `num_allocs`, `num_reserves`,
`num_arena_extensions`, and `num_arena_shrinkages`, with the same checkpoint and
device/pinned group structure. Raw records and `arenas.csv` retain bytes and
integer counts; paired memory deltas remain signed MiB. Consumers of the former
all-fields `arenas_mib` layout must read counters from `arena_counts` instead.
`report` rejects summaries using that former layout.

## Validation and limits

Initial packaging validation used the existing Python 3.11 interpreter: **18 unittest
tests passed**, artifact check passed, `production.diff` matched the historical
file byte-for-byte, and the regression patch matched its source byte-for-byte.
Both patches were successfully applied to project-local scratch copies of the
exact pinned source files; the resulting test source and CMake hashes matched
historical build identities. All nine fixture hashes also matched pinned public
Git blobs. Shell syntax and repository Ruff checks passed. Pytest was
unavailable in that interpreter and was not installed. Editor/Pylance could not
analyze this out-of-workspace tree; executed imports/tests checked Python syntax.

Additional host-only regression fixtures cover valid/detached/wrong-variant test
layouts, missing/mismatched CUDA companion hashes, nonzero memory and counter
summaries, unchanged raw CSV units, and rejection of old or corrupted summary
counter fields. Report-schema tests mock raw capture analysis and the Phase A
gate; they do not validate a native capture or loader execution.

The correction phase passed 23 allocation host tests as part of the package's
73 tests and 26 integration checks. A subsequent build-only check compiled and
linked both packaged applications against each recorded historical runtime
variant. Static ELF inspection checked the resulting layout, library requirements
and overridable RUNPATH. No ORT/GenAI or native regression-test target rebuilds
were performed, and no application or regression test was executed.

Actual runtime loading, NVML permissions, allocation watchdog behavior with live
children, model execution, complete fresh raw-capture reporting and all fresh
GPU measurements remain **unvalidated**. Application build success in the
pinned environment does not certify a different build or hardware environment.
No machine-wide inventories, fixture binaries,
saved input vectors, raw evidence, stopped controllers or machine-specific
fallback paths are bundled.
