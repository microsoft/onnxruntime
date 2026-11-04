# Local SD1.5 stateful image generation

The [CLI and helpers](../samples/python/sd15_demo.py) construct one native
[SessionChain](Stateful_Python.md):

1. Encode positive and negative prompts with ordinary text-encoder Sessions.
2. Run the UNet through a SessionLoop with native FP32 DDIM/guidance state.
3. Decode latents with the VAE.
4. Run declared CPU ONNX safety preprocessing and the mandatory CUDA safety checker.
5. Save a PNG only after an unflagged safety verdict.

Python is application/validation tooling, not a second generation engine.
Diffusion latents are native execution state, not KV caches.

## Requirements and assets

- A local M15-or-later **CUDA-plugin** Python wheel, matching CUDA/cuDNN libraries
  and a supported NVIDIA GPU. These optimized pretrained graphs are CUDA-only;
  deterministic CPU EP fixtures do not imply real CPU model support.
- Optional [demo dependencies](../samples/python/requirements-sd15-demo.txt) in
  an isolated environment. No published ORT wheel or new runtime dependency is
  installed by this file. For validation, install CPU-only Torch first to avoid
  unrelated bundled CUDA/cuDNN libraries shadowing the local ORT toolchain.
- Approximately 2.7GB for immutable original models and a similar amount for
  derived graphs, plus working space. Do not put models or generated images in Git.

Explicit acquisition is allowlisted to
[`tlwu/stable-diffusion-v1-5-onnxruntime`](https://huggingface.co/tlwu/stable-diffusion-v1-5-onnxruntime),
revision `f22c0e90205a7a2c08bf3e8927b39dc84e4064c2`: text encoder,
UNet, VAE decoder, safety checker, tokenizer/feature/scheduler metadata and model
card. The authoritative CreativeML Open RAIL-M license is retained from
`CompVis/stable-diffusion-license`, revision
`14d42d09bffd871b1666a084fc954a50cff72ac0`.

Models use pinned LFS SHA-256/size identities; supporting files use pinned Git
blob identities and recorded SHA-256. Invalid existing files fail instead of
being silently replaced. New downloads are bounded, checked and atomically
published. VAE encoder/image-to-image and downloaded Python code are excluded.
Normal build/run/validation commands never download missing assets.

## Prepare the package

Use the local plugin installation, not an installed public wheel. Run outside
the repository to avoid source-package shadowing:

```bash
ROOT=/path/to/onnxruntime
source "$ROOT/.venv/bin/activate"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$ROOT/samples/python/requirements-sd15-demo.txt"
cd "$HOME"
ASSETS="$HOME/onnxruntime-stateful/sd15"
DEMO="$ROOT/samples/python/sd15_demo.py"

# Explicit network-enabled setup:
python "$DEMO" --assets "$ASSETS" acquire

# Local-only native construction/validation:
python "$DEMO" --assets "$ASSETS" build
```

Default policy: 512x512 RGB, 20 complete DDIM updates, guidance 7.5, batched
classifier-free guidance, device ordinal 0. `build --steps`, `--guidance`,
`--sequential` and `--device-id` explicitly construct a different policy;
`run` does not silently override it.

CUDA Sessions explicitly use `cudnn_conv_algo_search=HEURISTIC`, recorded in
package provenance. Rebuild older packages before running this qualified
configuration. Exhaustive-search diagnostics with the installed cuDNN version
leak allocations even for an ordinary unmodified VAE Session; this existing
non-exhaustive option avoids that vendor-library path without changing weights,
native DDIM arithmetic, GPU placement or global library/driver settings.

The package is `"$ASSETS/package"`: a native schema-1.1 manifest, declared
component variants, tokenizer/feature metadata and license. Every original
initializer is preserved byte-for-byte. Derived boundaries expose FP32 around
FP16 model internals, retain INT32 token IDs/BOOL safety verdicts, and apply VAE
latent scaling exactly once (`1/0.18215` before the FP16 cast).

The small CPU ONNX preprocessor clips `sample/2+0.5`, emits NHWC images and
implements the pinned byte-rounding, Pillow-13 bicubic resize to 224x224 and
CLIP rescale/normalization. It is an explicit ordinary Session stage, not an
undeclared Python callback.

Native chain construction validates actual models/bindings/EP selection.
A construction failure preserves the previous manifest. Models are derived
only from verified originals; existing linked metadata is checked as well.

## Generate offline

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python "$DEMO" --assets "$ASSETS" run \
  --prompt 'A peaceful mountain lake at sunrise, soft golden light, landscape painting' \
  --seed 42 --output "$ASSETS/qualification/sd15.png"
```

The negative prompt defaults to empty; use `--negative` to supply one.
The local CLIP tokenizer includes special tokens, pads to 77 INT32 IDs, rejects
overlength text and never loads remote Python. Positive prompts must be nonempty.
Seeds are unsigned 64-bit integers. Output paths must not already exist.

The checked writer accepts only finite FP32 NHWC `[1,512,512,3]` values in
`[0,1]` and one BOOL verdict. A flagged verdict raises an explicit error and
does not save an unfiltered image. PNG encoding/validation happens in a temporary
directory, followed by exclusive publication; failed writes are not reported as
successful images.

JSON reports image identity, seed, safety verdict, total model invocations and
setup/native-execution/output timings. These are not benchmark or speedup claims.
The default chain invokes the UNet 20 times and five ordinary stages, for 25
total model invocations; sequential guidance invokes UNet 40 times, for 45 total.

For programmatic use, import the helper from its source directory:

```python
import sd15_demo

execution = sd15_demo.create_execution(package_root, prompt, seed=42)
with execution, execution.run() as result:
    outputs = result.get_outputs()
    sd15_demo.save_image(outputs["image"].numpy(), outputs["unsafe"].numpy(), output_path)
```

A caller adapter's `encode(text)` returns INT32 `[1,77]`. Optional supplied
initial latents are finite FP32 `[1,4,64,64]`. Native cancellation, failed-work
Reset, retained immutable values and explicit close follow the existing
[stateful ownership contracts](Stateful_Python.md).

## Differential validation

The upstream metadata declares **PNDMScheduler**. This demo intentionally uses
the approved native `sd15_ddim_v1` policy; it does not claim PNDM parity.

```bash
python "$DEMO" --assets "$ASSETS" validate \
  --prompt 'A peaceful mountain lake at sunrise, soft golden light, landscape painting' \
  --output "$ASSETS/qualification/sd15-validated.png"
```

Validation uses ordinary CUDA Sessions and Diffusers **0.35.1** DDIMScheduler:
1000 training steps, scaled-linear betas 0.00085..0.012, epsilon prediction,
leading timesteps with offset 1, `set_alpha_to_one=false`, no clipping/thresholding,
and eta 0. It obtains native initial noise before the first Step and supplies
identical latents to the independent reference; Torch and native RNG seeds are
not interchangeable.

Current approved comparisons are conditioning `rtol=atol=0.002`, same-input
local DDIM updates `rtol=0.002, atol=0.02`, and matching-latent VAE
`rtol=atol=0.02`. Independent closed-loop/replay/parity latents require
RMSE<=0.01; normalized images require RMSE<=0.005. Absolute maxima are measured
and reported, not used as accumulated-error gates. Existing CUDA GroupNorm
explicitly has a nondeterministic reduction; same seeds do not promise
byte-identical encoded CUDA PNGs.
Safety preprocessing for the same decoded image and BOOL verdicts are exact.
Real FP16 image replay uses the declared image tolerance; CPU fixtures and
retained latent snapshots remain exact. Separate original-versus-wrapper
qualification checks all components and verifies actual CUDA large-op placement,
so shared wrapper defects cannot masquerade as agreement.

Full UNet predictions intentionally return to the inherited CPU FP32 scheduler:
20x131,072 bytes batched or 40x65,536 bytes sequential = 2,621,440 prediction
DtoH bytes. Conditioning/VAE/safety/final-image traffic and explicit validation
reads are separate categories, not zero-copy/GPU-resident scheduling claims.

## Tests and qualification status

The [offline suite](../onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py)
uses small deterministic CPU EP graphs and never downloads models. Optional
Pillow/reference dependencies produce explicit skips when absent; qualification
must install them and report nonzero executed counts and exact line coverage.

The [real-asset suite and workers](../onnxruntime/test/python/onnxruntime_test_python_sd15_demo_assets.py)
require `ORT_STATEFUL_SD15_ASSETS` explicitly. They check original/wrapper
equivalence, both supported UNet batch sizes, actual CUDA placement, independent
DDIM/image/safety checkpoints, typed/package parity and lifecycle behavior.

The approved plugin operator-identity repair is rebuilt in a fresh byte-verified
local wheel. Original/wrapper CUDA checks pass with unchanged weights. The
separately approved
[specification amendment](../SPEC.md#o9-cudnn-search-and-accumulated-error-amendment-approved)
and plan are implemented. Both real guidance-mode qualification tests pass in
three recorded runs with unchanged hard RMSE/local/safety checks.

| Qualification surface | Result |
|---|---|
| CPU EP fixture/regression Python | 60 tests / 195 subtests |
| Installed CUDA-plugin Python | 64 tests / 213 subtests |
| New production-line coverage | 503/503 Python + 1/1 native; no exclusions |
| CPU native/public API | 85/153 tests |
| CUDA-plugin native/public API | 157/216 tests |
| Full 20-update memcheck/full leaks | Zero errors and zero leaked allocations |
| Full-size two-update all-kernel synccheck | Zero errors |
| Approved bounded GroupNorm racecheck | Zero hazards/errors/warnings; nonzero sum-kernel instrumentation verified |
| Real 20-update batched/sequential generation | Both pass, safe 512x512 RGB PNGs |

Final independent-reference latent/image RMSE is 0.005744151/0.002029416
batched and 0.007715663/0.002610830 sequential. A separate offline CLI run
produced a safe PNG in 2.491 seconds of native execution after 12.383 seconds
of setup (one observed run, not a benchmark). CUDA nondeterminism means these
values and PNG hashes need not repeat bit-for-bit.

Actual native-denoising transfer captures confirm exactly 2,621,440 prediction
DtoH bytes per guidance mode. Sequential internal DtoD copies are recorded
separately; setup, reference and image-output traffic are outside that capture.

Race qualification has an explicitly
[approved bounded scope](../SPEC.md#o10-bounded-real-model-race-qualification-approved):
sample the actual GroupNorm scale and sum160/sum256 variants while executing
complete full-size two-update lifecycle workers. It is not an unfiltered
full-model race-freedom claim. The unfiltered two-update attempt timed out;
earlier interrupted attempts and a case-mismatched filter are not passing
evidence. Full 20-update memory/leak and all-kernel two-update synchronization
checks retain their full scopes. Exact commands, launch bounds, final proof
and publication tracking are in [TASKS](../TASKS.md).

The accepted race sample includes two sum160 launches and all 50 sum256
launches in that full worker. Explicit synchronization-event counts,
10,240 and 466,944, match captured grid/block geometry. All 302 scale launches
have no shared memory; full memcheck covers their global accesses. Low launch
caps that reached no selected kernel are discarded, not reported as passes.
