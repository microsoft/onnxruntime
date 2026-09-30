# Portable ORT GenAI / llama.cpp comparison reproduction

These scripts are derived from the historical `phi-benchmark/scripts` harness,
not a redesigned benchmark. They run the matrix or repeated requests with one
loaded model. Copy this entire directory anywhere; no repository-relative model
lookup, private package feed, existing virtual environment, or machine path is
assumed. Paths are explicit and relative paths resolve against the caller's
working directory. Do **not** run inference on a shared/busy GPU without approval.

## Dependencies and artifact identity

Host help, validation, preparation plans, and unit tests need only Python's standard
library (3.10+; historical Python **3.11.16**) and Bash for the optional shell wrapper.
Actual inference additionally needs compatible NVIDIA hardware, driver, CUDA/cuDNN
libraries on the loader path, GPU-enabled llama.cpp, and these recorded packages:

| Distribution | Recorded version |
|---|---|
| `onnxruntime-genai-cuda` | `0.15.2` |
| `onnxruntime-gpu` | `1.30.0` |
| `llama-cpp-python` | `0.3.35` |
| `pynvml` | `13.0.1` |
| `psutil` | `7.2.2` |
| `huggingface_hub` (optional download only) | **Unknown historical pin** |

Provision dependencies separately with your approved package source. No script
installs anything or builds native code. Package version strings alone do not
reproduce locally rebuilt ORT binaries; preserve wheel/build identity and loader
configuration independently. CUDA-enabled llama.cpp is required; its version alone
does not prove GPU support. No CUDA library path is fabricated by these scripts.

Pinned setup selections (commit IDs recovered from local download metadata):

- `microsoft/Phi-4-mini-instruct-onnx` at
  `fc04c8f93df696602fd9f300a30d1bf2e3081347`, **only**
  `gpu/gpu-int4-rtn-block-32/*`.
- `unsloth/Phi-4-mini-instruct-GGUF` at
  `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80`, **only**
  `Phi-4-mini-instruct-Q4_K_M.gguf`.

These are different 4-bit quantizers, not numerically equivalent artifacts.
Already-downloaded models may be used without setup. Local edits to the graph or
GenAI config must be recorded separately; a repository revision is not proof of
unchanged local bytes. Configuration validation checks the explicit directory's
nonempty `genai_config.json`, `model.onnx`, `model.onnx.data`, `tokenizer.json`,
and `tokenizer_config.json`, and the decoder filename. It does not parse ONNX,
validate external-data offsets, or load the runtime.

## Optional setup — deliberately separate from execution

Run commands from this directory, or substitute absolute script paths.

```bash
bash prepare_ort_llama_benchmark.sh --help
bash prepare_ort_llama_benchmark.sh --python python3 --plan \
  --download-root ./models --output-dir ./artifact-record

# OPTIONAL network action: explicitly opt in; never starts inference.
bash prepare_ort_llama_benchmark.sh --python python3 --download \
  --download-root ./models --output-dir ./artifact-record
```

`--plan` performs no writes or network access and imports no optional package.
`--download` imports Hugging Face Hub, downloads only the pinned selections, and
records revisions, full file SHA256s, config, platform, and actual Hub version in
`artifact-record/artifact-manifest.json`. Hashing large model files can take time.
Explicit `--ort-revision` / `--gguf-revision` overrides require full commit SHAs
and are a different reproduction input. Existing artifact manifests are not
overwritten. Download caching/network behavior otherwise follows the installed Hub
client; use an independently prepared environment/cache if isolation is required.

## Validate, then execute explicitly

```bash
python3 run_ort_llama_benchmark.py --help
python3 run_ort_llama_benchmark.py \
  --ort-model ./models/ort/gpu/gpu-int4-rtn-block-32 \
  --gguf-model ./models/gguf/Phi-4-mini-instruct-Q4_K_M.gguf \
  --gpu-index 0 --runtime both --contexts 2048,4096,8192,16384,32768 \
  --output-tokens 64 --repetitions 3 --sampling-interval-ms 10 \
  --output-dir ./matrix-results --validate-only
```

Remove **only** `--validate-only` to run that matrix. Add
`--ort-device-allocator` for baseline ORT, device-initializer allocator ORT, then
llama.cpp, in that order for every repetition/context. Without it, baseline ORT
then llama.cpp run. There is no warm-up or reordered/ABBA scheduling.

Sequential mode loads one model per runtime and sends requests back-to-back:

```bash
python3 run_ort_llama_benchmark.py \
  --ort-model ./models/ort/gpu/gpu-int4-rtn-block-32 \
  --gguf-model ./models/gguf/Phi-4-mini-instruct-Q4_K_M.gguf \
  --gpu-index 0 --runtime both --sequential-requests 5 --sequential-context 8192 \
  --output-tokens 64 --sampling-interval-ms 10 --output-dir ./sequential-results
```

`--sequential-requests 1` means matrix mode, as historically. Sequential mode
does not use matrix contexts/repetitions or the ORT device allocator variant.
`--runtime ort` or `llama_cpp` only requires that runtime's model argument.
`--log-arena` captures ORT baseline native stderr in `ort-arena.log`.
Use a new output directory for each run. Deliberate
`--allow-existing-output` permits replacement and the historical behavior of
including an existing matrix `results.csv` in a subsequent sequential report.
It is not resume support: matrix output is rewritten, and arena logs are truncated.
`run-config.json` records explicit inputs and the resolved GPU UUID. Matrix runs
write `results.csv` and `report.md` after each row; sequential runs write
`sequential.csv` and the report after all selected runtimes finish.

### Worker deadlines and failed runs

`--worker-timeout-seconds` is a finite positive wall-clock deadline (default
**3600 seconds**) for each spawned worker, not each token or sequential request.
It includes process startup, model loading, inference, cleanup, result transfer,
and worker exit. In sequential mode the entire runtime's request sequence shares
one deadline. Choose a larger explicit value for slow hardware/long sequences;
the value is recorded in `run-config.json`.

The parent drains a one-way result pipe while the child runs, before waiting for
exit. A dedicated receiver thread keeps a partially transferred message from
blocking the parent's deadline. On expiry, the parent terminates only its owned
worker, waits at most one second, escalates to kill, and waits at most one more
second to reap it. Receiver cleanup waits at most one additional second. These
are wall-clock waits, not a hard real-time guarantee against OS scheduling,
uninterruptible kernel operations, or process-start failures; inability to reap
or stop the receiver is explicitly reported. No unrelated process is signalled.
There is no unbounded queue feeder flush or child join.

Runtime errors, OOM, deadline expiry, missing results, and abnormal exit produce
failed rows. Other requested conditions still run in the original order.
**Any requested row failure returns nonzero**, unlike the historical
any-success policy. Reports expose success/failure/total counts per condition and
failure diagnostics; medians use only successful rows, with missing conditions
identified rather than treated as zero. Raw native arena logs remain available.
Previously loaded matrix rows in an explicitly reused sequential report do not
change the current invocation's exit status.

Sequential workers deliver each completed row after its measured snapshots.
Those rows survive later exceptions/timeouts; remaining requested indices are
failed rows without invented measurements. If teardown fails after all requests,
the last row retains its measurements but is marked failed. `sequential.csv` now
includes `status` and `error`. Progress delivery adds inter-request IPC overhead
outside the measured inference windows; it does not alter token loops, order,
EOS handling, or metric calculations. Parent interruption/crash and machine
failure are not durable checkpoint/resume support.

## Preserved measurements and portability limits

- Synthetic prompt seed, repeat count `context // 9`, character truncation to
  `max(100, context * 5)`, and `"\nAnswer briefly:"` are unchanged. Context denotes
  capacity, **not** a guarantee of that many prompt tokens. Each runtime uses its
  own tokenizer and truncates to `context - output_tokens`.
- ORT search still sets **only** `max_length` and `batch_size=1`; other search
  choices, including token selection/EOS, come from the model config. llama.cpp
  still uses streaming completion at `temperature=0.0`, with its runtime defaults.
  Neither ignores EOS or forces the requested number of generated tokens.
- ORT TTFT times the first generation call, including append/prefill. llama.cpp
  TTFT is the first **nonempty text chunk**, not necessarily its first token;
  completion token count is retokenized generated text. Decode rate remains
  `(completion_tokens - 1) / (total_seconds - TTFT_seconds)`, clamped exactly as
  in the source. These are asymmetric historical metrics, not normalized speed
  comparisons. Initialization, cleanup, and the matrix's three-second idle
  sampling sleep retain their original timing boundaries.
- Matrix rows use spawned worker processes, ordered context → repetition →
  runtime. Sequential uses one spawned worker/model per runtime (ORT first).
- `--gpu-index` is a **physical NVML** index. Immediately before inference the
  runner maps it to a UUID and overrides `CUDA_VISIBLE_DEVICES` so both runtimes
  use the matching logical CUDA device 0. The sampler uses that physical NVML
  device's **whole-device used memory**, not per-process VRAM. Other processes,
  driver allocations, and sampling gaps affect observations. RAM is process RSS.
  Clamped memory deltas and inference-window maxima are unchanged.
- Linux + NVIDIA is the execution target. MIG, distributed execution, CPU-only
  benchmarks, and non-NVIDIA accelerators are not supported. Host-only validation
  cannot certify GPU availability, full layer offload, or the historical
  90%-of-ONNX-bytes residency heuristic. Inference intentionally still enforces
  that heuristic and llama.cpp's all-layer offload log check.
- Host tests use fake runtimes/artifacts only; they cannot establish actual GPU
  numerical equality, performance, binary compatibility, or successful downloads.

## Host tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -S -m unittest discover -s . -p 'test_comparison.py' -v
bash -n prepare_ort_llama_benchmark.sh
```

Tests use a self-cleaning `host-test-work-*` directory beside the tests, never a
system temporary directory. They cover paths, missing files, arguments, host-only
help/validation/plans, explicit download selection with a mocked client, GPU
selection with mocked NVML, workload ordering, and report/memory calculations.
CPU-only spawned-worker tests cover oversized results, stalled workers that
ignore termination, partial pipe transfers, exceptions, missing results,
partial sequential rows, and mixed/all-success/all-failure exit statuses.
Worker tests have an independent POSIX alarm watchdog and bounded owned-child
rescue cleanup; a deliberate join-before-drain regression tests that guard itself.
The watchdog restores the caller's signal handler/timer. These worker tests are
skipped on hosts without POSIX interval timers.

## Source provenance and intentional semantic changes

Original basenames under `phi-benchmark/scripts`, SHA256 before copying:

| Source | SHA256 |
|---|---|
| `prepare_ort_llama_benchmark.sh` | `adb6566e04fe9d773125ef21a733c87f2520534e62c3a67f8a22a7005fe3c9cb` |
| `run_ort_llama_benchmark.py` | `4bacaadcbb62465127cc98c7199275bfe26de7c983bf16c431f6fe8c1d4041f3` |

The original runner has no project-local imports; the old shell invoked it.
The portable shell now delegates only to `prepare_benchmark_artifacts.py`, which
imports the runner's standard-library validation helpers **only for download**.

Changes are limited to lazy optional imports; explicit physical GPU mapping;
host-only validation; stricter path/argument/output handling; input metadata;
bounded worker/result handling, partial sequential row retention, explicit
failure reporting and any-failure exit status;
and replacing unconditional broad downloads, private-feed installation, implicit
first-config discovery, machine-specific loader setup, and implicit execution with
optional pinned setup. Token-generation loops, token selection, EOS, metric
formulas, requested counts, ordering, and measured timing boundaries are retained.
Worker orchestration, sequential progress delivery, failure policy, and report
presentation intentionally differ; this is not a blanket AST-identity claim.
ORT MIT notices have been added; no source notices were removed.
