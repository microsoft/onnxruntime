# Reviewing and reproducing the historical evidence

## Start here: four different checks

Run commands from `docs/phi4-memory-diagnostics/`. Use an existing Python
3.11+ environment for the host-only tools. Host checks and the completed
application-build check below are separate from application execution,
actual runtime loading and fresh GPU replay, which remain unperformed.

| Level | Entry point | What it establishes |
| --- | --- | --- |
| Package integrity | `sha256sum --check SHA256SUMS` | Files agree with the supplied checksum list; not independent authenticity. |
| Compact consistency | `python -B repro/review_results.py --check-integrity` | Recomputes both published median tables, checks CSV repetition grids, arena arithmetic, settings and parity records. |
| Full raw comparison | Same reviewer with `--raw-captures` | Compares actual external historical logits/tokens and checks input/config/recorded-runtime metadata. |
| Fresh GPU reproduction | Separate comparison or allocation workflow | Requires suitable hardware, model/runtime dependencies and, for historical A/B, approved original inputs. Not run during this preparation. |

The offline reviewer uses only the Python standard library and imports no
inference, NVML or network modules. Its default JSON output goes to stdout:

```sh
python -B repro/review_results.py --check-integrity
python -B repro/review_results.py --check-integrity \
  --output /tmp/phi4-offline-summary.json
```

The output file must not already exist and must be outside this package.
It includes 12 allocation rows, 45 earlier comparison rows, four allocation
median groups and 15 earlier-runtime median groups. It explicitly labels
raw-vector comparison and fresh GPU reproduction as not run when applicable.
It checks all reported medians, including process peaks of
8,002/5,954 MiB and 59,848/27,082 MiB.

### Optional external historical captures

Obtain authorized access separately; no public download or sharing approval
is implied. An exact historical Phase A directory has four subdirectories:
`baseline-2048`, `patched-2048`, `baseline-32768`, `patched-32768`.
Each needs the original input and loaded-input binaries, effective/applied
configuration and search readback, logits binary and metadata, generated
IDs in JSON and binary, capture state, and before/after loaded-library hash
records. These files are **not** distributed in the PR.

```sh
python -B repro/review_results.py --check-integrity \
  --raw-captures /absolute/path/to/approved-phase-a \
  --output /tmp/phi4-raw-review.json
```

The checker retains the original complete comparison without tolerances:
800,256-byte FP32 logits and 256-byte signed-int32 little-endian generated
IDs at each context. It checks all 200,064 logits for finite values,
input hashes, settings and recorded runtime identities. Missing requested
captures cause a nonzero exit, not a skipped or success-shaped result.
It does not rehash excluded runtime binaries or execute them.

The original vectors are identified in the unchanged
[model manifest](evidence/model-manifest.json) and
[parity records](evidence/parity-checks.json). Hashes identify required
artifacts; they cannot reconstruct those artifacts.

## What can be checked from this contribution

This directory contains compact evidence and separately documented
reproduction tooling, not model weights, captured vectors or prebuilt
native runtimes. The offline commands do not download anything or run GPU
work. See the [comparison workflow](repro/comparison/README.md) and
[allocation workflow](repro/allocation/README.md) for their distinct scope.

From this directory:

```sh
sha256sum --check SHA256SUMS
```

This checks package integrity, not whether a model run is correct.
The package's narrowly scoped [.gitattributes](.gitattributes) preserves both
exact historical patches, including the regression patch's mixed line endings,
through Git staging and checkout. The checksums describe the documented
Linux/LF package checkout; they do not normalize files or replace the original
patch digests.
For an offline result review:

1. Group `evidence/memory-before-after.csv` by `context` and `variant`.
   Each group has repetitions 1, 2 and 3. Compute column medians, retaining
   the distinction between process NVML, whole-device NVML and BFC capacity.
2. Confirm that both context pairs in `evidence/parity-checks.json` have
   matching logits hashes, 800,256-byte FP32 vectors and 64 matching output
   IDs. Check that all 12 recorded Phase B token hashes match the appropriate
   Phase A reference. This checks report consistency, not the omitted raw
   comparisons.
3. Group the 45 earlier-runtime rows by context/runtime/allocator. They have
   three repetitions each. Keep actual input/output counts visible when
   computing medians; most llama.cpp conditions stopped before 64 tokens.
4. Use `evidence/model-manifest.json` for saved model/runtime hashes,
   input hashes and effective configurations. The older exploratory matrix
   is not assigned the later controlled experiment's configuration.

The original Phase A binary pairs were compared in full, without a
tolerance. Initial September 28 packaging rechecked logits but not raw
generated-token files. During the September 29 tooling preparation, the
new reviewer was run from an isolated package copy with explicitly supplied
external historical captures. Both complete logits and generated-token
pairs, input/configuration records and recorded library identities passed.
No captures were copied into the proposed repository package and no
inference was executed.

## Exploratory ORT GenAI / llama.cpp comparison

The [comparison README](repro/comparison/README.md) describes dependencies,
build prerequisites and the portability changes. The included entry points
are [prepare_ort_llama_benchmark.sh](repro/comparison/prepare_ort_llama_benchmark.sh)
and [run_ort_llama_benchmark.py](repro/comparison/run_ort_llama_benchmark.py).
The shell delegates artifact handling to one local
[setup helper](repro/comparison/prepare_benchmark_artifacts.py). It never
installs packages, builds native code or starts a benchmark.

Recorded exploratory versions: Python 3.11.16, GenAI CUDA 0.15.2,
ORT GPU 1.30.0, llama-cpp-python 0.3.35, pynvml 13.0.1 and psutil 7.2.2.
llama-cpp-python needs a CUDA-enabled build and a compatible local toolkit,
CMake and C++ compiler. The historical Hugging Face client pin is unknown.
Do not interpret installing today's unpinned dependencies as restoring the
historical environment. Provision dependencies separately; no installation
commands were executed for this addition.

Host-only help and a plan that performs no downloads or writes:

```sh
bash repro/comparison/prepare_ort_llama_benchmark.sh --help
python -B repro/comparison/run_ort_llama_benchmark.py --help
bash repro/comparison/prepare_ort_llama_benchmark.sh --python python3 \
  --plan --download-root /absolute/path/to/new-models \
  --output-dir /absolute/path/to/new-artifact-record
```

For an explicitly authorized future download, replace `--plan` with
`--download`. The setup selects only the exact ONNX
`gpu/gpu-int4-rtn-block-32` variant at revision
`fc04c8f93df696602fd9f300a30d1bf2e3081347` and the exact
`Phi-4-mini-instruct-Q4_K_M.gguf` file at revision
`78eb92a46fc37e6b524df991ed9aca9bc6aa7b80`. It writes
`artifact-manifest.json` to the selected output directory. It does not find
and silently choose the first model configuration.

Use already acquired artifacts to validate paths/options without loading
a model, importing the optional runtime packages or initializing a GPU:

```sh
python -B repro/comparison/run_ort_llama_benchmark.py \
  --runtime both \
  --ort-model /absolute/path/to/gpu-int4-rtn-block-32 \
  --gguf-model /absolute/path/to/Phi-4-mini-instruct-Q4_K_M.gguf \
  --gpu-index 0 --output-dir /absolute/path/to/new-comparison \
  --contexts 2048,4096,8192,16384,32768 --output-tokens 64 \
  --repetitions 3 --sampling-interval-ms 10 --ort-device-allocator \
  --validate-only
```

For a separately authorized GPU comparison, run the same command **without
`--validate-only`**. `--gpu-index` is the physical NVML index, not a remapped
CUDA ordinal. The output directory is new by default; do not use historical
evidence directories. Outputs are `run-config.json`, `results.csv` and
`report.md`.
The optional `--log-arena` flag retains the original verbose baseline arena
logging mode; it is not allocation-call-site attribution.

For a separately authorized sequential-request experiment, replace the
matrix settings with `--sequential-requests 5 --sequential-context 32768`
and choose another output directory. It produces `run-config.json`,
`sequential.csv` and `report.md`, not an additional historical measurement.

The source's prompt construction, runtime iteration/order, repetitions,
token-selection and early-EOS behavior are retained. The default matrix
has 45 runs when both runtimes and the ORT initializer allocator variant
are selected. Ordering is supplied contexts, then repetitions, then ORT
baseline, optional ORT device allocator, and llama.cpp. Sequential mode runs
ORT baseline then llama.cpp, keeping one model alive per runtime.
Whole-device NVML, not per-process GPU memory, is sampled.
The original quantizers differ and llama.cpp may generate fewer than 64
tokens. No equal-output-length replacement is labeled a replay, and the
runner does not reproduce the custom GenAI allocation-fix measurement.

Worker lifecycle and failure reporting have deliberate robustness corrections:
each worker has a finite wall-clock deadline, results are drained before
waiting for process exit, and cleanup targets only children owned by this
run. Any failed requested condition makes the overall exit nonzero while
successful rows and failure details remain in the output. These changes do
not alter successful inference timing boundaries, token selection or EOS
behavior; they do mean the entire runner is no longer identical to the
original. See the comparison README for the timeout option and cleanup bounds.

## Requirements for an exact historical allocation A/B replay

The distinct entry point is [allocation.py](repro/allocation/allocation.py).
Its [workflow README](repro/allocation/README.md) gives the native build
recipes, full configuration contract, safeguard thresholds and output layout.
The production and native-regression patches are historical reproduction
material against GenAI `ed5f4e8...`, not fixes proposed for current main.
The regression target's complete patch is included; there is no dependency
on an unshared regression-source implementation.

Host-only commands needing no external model/runtime artifacts:

```sh
python -B repro/allocation/allocation.py --help
python -B repro/allocation/allocation.py check
python -B repro/allocation/allocation.py config
```

`check` without configuration verifies only the included artifact contract;
it explicitly reports that GPU execution is unvalidated. Save the printed
template to a **new external JSON file** and replace every placeholder
before configured preflight. The template also exists as
[config.example.json](repro/allocation/config.example.json).
Configuration includes model location, ORT/toolkit/cuDNN locations,
baseline/patched GenAI libraries and hashes, native application and test
executables and hashes, per-variant fixture directories, physical GPU
UUID/index, explicit resource limits, inputs and a new output directory.

The required saved input files are signed int32 little-endian with no header:

| Context | Input IDs | Bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| 2048 | 1801 | 7204 | `a7100ece1396cf13e81616381b11fed562406cadf539c72a7cae8c955b0877f4` |
| 32768 | 28747 | 114988 | `fbe6fe0929666ac5a91dd152aea07bc2bf7675c765f31ad57e701604ccaa3831` |

They require separately approved access. They are not distributed, cannot
be recovered from hashes, and must not be replaced with synthetic inputs
while claiming an exact historical replay. The native regression fixtures
are a separate requirement: [fixture-manifest.json](repro/allocation/fixture-manifest.json)
identifies nine files in three tiny fixture directories by recorded hashes.
Fixture binaries are not included; materialize them from the pinned public
GenAI source/fixture mechanism described in the workflow README.

Once external sources/fixtures are supplied, prepare two clean, separate
GenAI trees and verify fixtures, without compiling:

```sh
bash repro/allocation/prepare_sources.sh \
  /absolute/path/to/clean-baseline-genai /absolute/path/to/clean-patched-genai
python -B repro/allocation/allocation.py check-fixtures \
  --models-dir /absolute/path/to/clean-baseline-genai/test/models
```

The source-preparation command is deliberate and modifies only those
explicitly supplied clean pinned trees: regression instrumentation goes to
both; the allocation production patch goes only to the patched tree.
Follow the linked native build recipes to produce matching applications
and test executables. Only the packaged application recipe has been
build-validated; ORT/GenAI and native regression-test targets were not rebuilt.

The build entry points take absolute paths and require new build directories.
For each variant, after materializing the pinned public dependency sources
and fixtures described in the allocation README:

```sh
PYTHON=/absolute/path/to/existing-python \
  bash repro/allocation/build_runtime.sh \
  /absolute/path/to/prepared-variant-genai \
  /absolute/path/to/diagnostic-ort-prefix \
  /absolute/path/to/cuda-toolkit /absolute/path/to/cudnn/lib \
  /absolute/path/to/pinned-dependency-sources 80 \
  /absolute/path/to/new-variant-runtime-build

bash repro/allocation/build_apps.sh \
  /absolute/path/to/diagnostic-ort-prefix \
  /absolute/path/to/prepared-variant-genai \
  /absolute/path/to/new-variant-runtime-build \
  /absolute/path/to/cuda-toolkit \
  /absolute/path/to/new-variant-app-build
```

Run those recipes separately for baseline and patched sources. They do not
acquire dependencies. Runtime outputs include the GenAI/CUDA libraries,
`logits_allocation_tests`, `unit_tests` and `runtime-hashes.sha256`.
Application outputs include `phi_phase_a`, `phi_phase_b`,
`app-hashes.sha256` and `phase-b-dynamic.txt`. Phase B selects the configured
runtime dynamically using a checked RUNPATH/loader setup; an application
pinned to baseline by RPATH is rejected rather than misreported as patched.
The configuration selects each variant's Phase A application and one shared
Phase B application. Native prerequisites include Linux, a C++20 toolchain,
CMake, CUDA/cuDNN and `readelf`/`nm`; historical versions and pinned sources
are documented in the allocation README.

Keep each variant's `logits_allocation_tests`, `unit_tests`, GenAI library
directory and CUDA companion in their corresponding complete runtime build
directory. In particular, do not relocate the allocation-test executable
alone: the historical object-linked executable loads its CUDA companion
beside itself, not through a general `LD_LIBRARY_PATH` search. Configured
preflight rejects an incompatible layout; it never copies or replaces libraries.
Host fixture tests establish the path/hash contract, not successful native
dynamic loading.

After those prerequisites and the saved inputs are available:

```sh
python -B repro/allocation/allocation.py check \
  --config /absolute/path/to/completed-allocation-config.json
```

This is a host-only prerequisite check, not inference. Missing, mismatched
or placeholder prerequisites cause a nonzero exit. A future GPU run must
be separately authorized and uses:

```sh
python -B repro/allocation/allocation.py run \
  --config /absolute/path/to/completed-allocation-config.json
```

It requires the native regression gate, then four correctness captures
with complete first-token logits and all 64 tokens matching, before any
of the 12 performance measurements. It retains baseline-then-patched
ordering, CUDA completion barriers, no diagnostic logits copy in Phase B,
loaded-runtime identity checks, interference/resource gates and watchdogs.
No shrinking, synthetic substitution or automatic measurement retries are
introduced.

For an actual completed new run, revalidate its captures without a GPU:

```sh
python -B repro/allocation/allocation.py report \
  --results /absolute/path/to/completed-new-allocation-run
```

This is distinct from compact historical review: it requires the new run's
complete raw result set and does not accept the checked-in summary CSV as
a substitute. Output artifacts include configuration/identity/status
records, native-test results, separate Phase A captures and comparisons,
and Phase B records, summary, run CSV and arena CSV. TTFT starts before
AppendTokens and ends after the first generated token's completion barrier;
decode throughput uses the remaining 63 tokens. Process and whole-device
NVML peaks remain distinct from BFC capacity/live/slack snapshots.
Allocation summaries place only byte-valued arena fields in `arenas_mib`;
allocation/reserve/extension/shrink counters remain unscaled in
`arena_counts`. Raw records and CSV values retain their original units.

**The included compact evidence alone is insufficient for exact replay.**
Saved input-token vectors, baseline/patched runtime builds, model artifacts
and raw historical reference outputs remain external. The
[allocation workflow](repro/allocation/README.md) specifies the included
source/tooling, its command interface and any remaining blockers.
Access and sharing approval for external inputs/captures must be arranged
separately; no public archive link is implied.

Use the following as the protocol, not as a claim that replay has been
performed from this documentation:

1. Pin ORT to benchmark commit
   `94e76459dc9a759888707aa430d51a95d878da8f`, GenAI to
   `ed5f4e87147731e5b07810f9f5c90103b3603cdf`, and the model to
   `fc04c8f93df696602fd9f300a30d1bf2e3081347`.
   Reproduce the historical baseline and allocation-only GenAI variant
   separately. Using GenAI main after #2607 is not the historical baseline.
2. Obtain the original saved 1,801/28,747 input-token vectors and verify
   their hashes using the original serialization. A prompt of similar
   length is not an equivalent input.
3. Match the recorded effective model/search settings. The manifest contains
   the configurations and hashes of their original serialized files.
   Reformatting JSON changes file hashes; distinguish byte identity from
   semantic configuration equality.
4. Build GenAI against the matching custom ORT headers/libraries. For the
   native application, select the intended library using `ORT_LIB_PATH`
   and a matching `LD_LIBRARY_PATH`. Verify the actual loaded GenAI,
   CUDA companion, ORT and provider library paths/hashes, rather than
   assuming the environment variable selected them.
5. Set `ORT_ARENA_DIAGNOSTICS=1` before model creation. The branch's
   `OrtApi_DebugLogAndShrinkGpuArenas` experimental API is explicitly
   called by a diagnostic native application, not automatically by every
   GenAI application. Capture after initialization, after generation and
   after per-request generator cleanup while the model is alive.
   **Use `shrink=false` for this measurement protocol.**
6. Separate correctness captures from measurements. In Phase A, capture
   the full first-token logits after AppendTokens and before generation,
   and all 64 output IDs. In Phase B, omit diagnostic logits copying and
   verbose attribution/profiler instrumentation. Take allocator snapshots
   outside timing and apply identical CUDA completion barriers to both
   variants.
7. Run fresh processes serially, with three repetitions per condition,
   baseline then patched within context/repetition, short context before
   long context. The original controlled protocol had no inference warmup.
   A different ordering or warmup policy is a new experiment and must be
   labeled as such.
8. Use an approved idle GPU and resource budget. The historical controlled
   runs used a 300-second watchdog, pre-run interference checks, process
   and whole-device NVML sampling requested at 10 ms, and recorded actual
   sampling windows. Reserve sufficient memory for the baseline, which
   peaked near 60,624 MiB whole-device on an 80 GB A100.
   Do not introduce automatic retries, shrinking or concurrent workloads
   into the historical comparison.
9. Compare complete logits and token vectors, record every failure/skip,
   and retain loaded identities plus raw sampling/allocator evidence.
   New results must be labeled with their own exact build and artifact
   identities, not appended to the historical CSV as original measurements.

The observed historical request and its small replacement should be
correlated with allocation events in an attribution run, separate from
performance measurements. Arena capacity alone does not identify an
allocation owner, exact CUDA free or kernel workspace.

## Scope of reproducibility

Model graph, external weights and tokenizer/config hashes were recomputed
during the original evidence packaging. Model revisions were recovered from
saved download metadata, not newly resolved for these measurements.
The earlier GGUF revision/hash in the manifest belongs only to the initial
ORT/llama.cpp comparison; that GGUF was not rehashed during packaging and
was not used for the GenAI allocation-fix experiment.

Do not claim a new current-main benchmark, current-main memory saving,
full chunking correctness, causal TTFT improvement, shrink safety or
concurrency validation from this historical protocol.

## Initial validation and subsequent correction checks

The initial addition's executable host checks were repeated from a clean temporary copy of
only this proposed package, without historical-directory imports. Python
was run with bytecode writing disabled; the test runner used isolated
imports and no installed package was added. The copy was removed after
retaining local validation records outside the source diff.

| Check | Result |
| --- | --- |
| Offline-review unittest cases | 15 passed |
| Comparison unittest cases | 24 passed |
| Allocation unittest cases | 18 passed |
| Compact review and complete package checksums | Passed; all published median rows reproduced |
| Python syntax; Ruff lint and formatting | Passed |
| All four shell entry points, `bash -n` | Passed |
| Help, setup plan, allocation template/artifact check | Passed; no download, inference or GPU initialization |
| Missing models/configuration/fixtures/raw results | Explicit nonzero failures verified |
| Optional full historical raw comparison | Passed with explicitly supplied external captures; not part of the distributed package |
| Native source formatting and historical patch identity/application | Checked without compilation |

The tests retain historical reporting/byte-comparison cases and add negative
checks for malformed paths/settings, missing or changed prerequisites,
duplicate repetitions, nonfinite values, wrong arena arithmetic, changed
parity records and unsafe output reuse. The existing environment lacked
pytest, so stdlib `unittest` was used; nothing was installed.

**Not performed:** model downloads, dependency installations, ORT/GenAI or
native regression-test target rebuilds,
real loader/NVML checks against newly built binaries, CUDA execution,
fresh baseline/patched captures, GPU performance measurements or cross-machine
replay. Full configured allocation preflight still requires external inputs,
model, fixtures, dependencies and builds. Host test success does not certify
those unavailable prerequisites or a new native binary.

The bounded pre-publication corrections are checked with all three affected
host suites in one invocation, from a clean package copy. From that copy's
`docs/phi4-memory-diagnostics/` directory (or its equivalent package root):

```sh
python -I -B -c '
import unittest
from pathlib import Path
root = Path.cwd() / "repro"
suite = unittest.TestSuite(
    unittest.TestLoader().discover(str(directory), pattern="test_*.py")
    for directory in (root, root / "comparison", root / "allocation")
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
'
```

These tests add bounded CPU-only worker completion/stall/exception/large-result
cases, mixed/all-success/all-failure reporting, native-loader layout checks and
independent byte/count statistics expectations. The Git regression stages only
inside disposable repositories, checks both patches' bytes with `core.autocrlf` set to
`false`, `input` and `true`, verifies staged patch export/application, and checks
full package integrity on Linux/LF checkouts. It never stages the real index.
Git is a host-test prerequisite; every Git subprocess has a 30-second timeout.

The correction phase passed **73 host tests**: 16 offline-review/Git-integrity,
34 comparison and 23 allocation tests, with no skips. All **26 integration
checks** passed, including offline medians/checksums, help/planning,
missing-prerequisite failures, Python/Ruff, shell syntax and disposable Git
staging/export/checkout integrity. The initial raw-vector comparison above was
not rerun by these correction checks. The historical evidence CSV/JSON and both
historical patch files are unchanged.

### Completed application-build check

From a clean copy of the corrected package, `build_apps.sh` compiled and linked
both `phi_phase_a` and `phi_phase_b` against each of the recorded baseline and
patched GenAI variants. The builds used GCC 13.2, CMake 4.1.2 and CUDA 13.0.48,
with `RelWithDebInfo`, C++20 and at most two build jobs; variants were built
serially. GenAI headers matched `ed5f4e87147731e5b07810f9f5c90103b3603cdf`,
ORT headers matched `94e76459dc9a759888707aa430d51a95d878da8f`, and runtime
library hashes matched the historical evidence.

The documented application layout was produced. Static ELF checks confirmed
the expected library requirements and overridable RUNPATH; Phase A imports
the logits API, while Phase B imports CUDA synchronization without the
diagnostic logits API. ORT/GenAI and native regression-test targets were not
rebuilt. No application or native regression test was executed.

This establishes application compile/link validation in that pinned environment,
not actual runtime loading, application correctness or fresh GPU replay.
No new measurements were taken; build binaries and raw logs are not distributed
in this package.
