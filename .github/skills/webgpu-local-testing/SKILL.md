---
name: webgpu-local-testing
description: "Build and run ONNX Runtime WebGPU provider tests on Linux WITHOUT a real GPU, using a software Vulkan adapter (Mesa lavapipe). Use when you need to exercise WebGPU EP kernels off-Mac — the Linux webgpu CI leg is build-only, so software Vulkan is how you actually run WebGPU correctness tests locally. SCOPE - lavapipe only validates host-side enforce/shape bugs and MatMul-free kernels; any graph containing MatMul (including the expanded-Attention node tests) crashes lavapipe and runs ONLY on macOS-arm64 Metal, which is the source of truth for those. Covers install (dnf on Azure Linux), the --use_webgpu build flag, the onnxruntime_provider_test target, VK_ICD_FILENAMES, and the lavapipe MatMul crash gotcha."
---

# Running ONNX Runtime WebGPU Tests Locally on Linux (No GPU)

Reusable knowledge for exercising the **WebGPU execution provider** on a Linux box
with no physical GPU.

> **Scope**: Linux ORT WebGPU only. macOS uses the Metal backend and a real GPU;
> this skill is the off-Mac story for running WebGPU EP kernels in CI-less dev loops.

## 1. Why software Vulkan is enough

On Linux, ORT's WebGPU EP runs on **Dawn** with the **Vulkan** backend. Vulkan does
not require a hardware GPU — **Mesa lavapipe** is a software (CPU) Vulkan adapter that
Dawn enumerates like any other device. For **EP correctness tests** this is sufficient
because:

- Many host-side validation paths (shape/broadcast checks, `ORT_ENFORCE`s) fire
  **before** any shader is dispatched. E.g. the WebGPU broadcast `ORT_ENFORCE` runs
  host-side, so the failure is observable on a software adapter without ever touching
  the GPU.
- Element-wise and broadcasting kernels that do dispatch run correctly (if slowly) on
  lavapipe, so their numeric output can be validated against the CPU reference.

You are trading speed for not needing hardware. It is **not** a substitute for a real
GPU on perf-sensitive or driver-specific paths — but for kernel correctness it is the
practical local loop.

> **Scope — what lavapipe can and cannot validate.** Software Vulkan covers (a)
> **host-side** failures (shape/broadcast `ORT_ENFORCE`s that fire *before* any shader
> dispatch) and (b) **MatMul-free** kernels that dispatch. It does **NOT** cover any
> graph that contains a **MatMul** — the MatMul family crashes lavapipe's LLVM JIT (see
> §5). This explicitly includes the motivating expanded-Attention node tests
> (`test_attention_4d_softcap_neginf_mask_expanded`): they decompose to
> `softmax(Q·Kᵀ + bias)·V`, which **contains MatMuls**, so they **cannot run on
> lavapipe**. For those, **macOS-arm64 Metal is the source of truth**. Concretely, the
> #28969 WebGPU broadcast-underflow fix was validated on lavapipe via a standalone
> Add-broadcast `OpTester` proxy (a host-side enforce/shape path) — **NOT** via the
> expanded-Attention node test. **Never** run lavapipe green and conclude an
> Attention/MatMul fix is validated off-Mac.

## 2. Install the software Vulkan stack (Azure Linux — use `dnf`, NOT `apt`)

```bash
dnf install -y mesa-vulkan-drivers vulkan-loader
# optional, for sanity-checking the adapter:
dnf install -y vulkan-tools && vulkaninfo | head
```

`mesa-vulkan-drivers` provides lavapipe; `vulkan-loader` provides the ICD loader.
The lavapipe ICD manifest lands at `/usr/share/vulkan/icd.d/lvp_icd.<arch>.json` —
`lvp_icd.x86_64.json` on x86_64, `lvp_icd.aarch64.json` on arm64. The examples below
use the x86_64 name; substitute your arch, or glob it:
`VK_ICD_FILENAMES=$(echo /usr/share/vulkan/icd.d/lvp_icd.*.json)`.

## 3. Build with `--use_webgpu`

```bash
./build.sh --config Release --parallel --use_webgpu
```

See the `ort-build` skill for general build phases and flags.

## 4. Run the WebGPU provider tests

WebGPU operator/kernel tests are **provider op tests** — they build into the
`onnxruntime_provider_test` target (**NOT** `onnxruntime_test_all`; see the `ort-test`
skill for the executable taxonomy). Point the Vulkan loader at the lavapipe ICD and
select a subset with `--gtest_filter`:

```bash
cd build/Linux/Release
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  ./onnxruntime_provider_test --gtest_filter="*WebGPU*"
```

`VK_ICD_FILENAMES` forces Vulkan to load **only** lavapipe, so the run is
deterministic regardless of what else is installed.

## 5. Gotcha: lavapipe crashes on the MatMul family

`MathOpTest.MatMulFloatType` (and other MatMul-family tests) crash lavapipe with:

```
LLVM ERROR: Instruction Combining did not reach a fixpoint after 1 iterations
```

This is a **pre-existing limitation of software Vulkan (Mesa lavapipe's LLVM JIT)**,
not an ORT bug. **Exclude the MatMul family** from broad lavapipe runs:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  ./onnxruntime_provider_test --gtest_filter="*WebGPU*:-*MatMul*"
```

## 6. Why this matters: the Linux webgpu CI leg is build-only

The Linux webgpu CI leg (`py-linux-webgpu-stage.yml`) **only builds** — it does not run
WebGPU kernels. A green Linux webgpu leg therefore does **not** mean any WebGPU test
actually executed. The macOS-arm64 webgpu leg is the only CI leg that runs WebGPU
backend node tests. So a local lavapipe run is the practical way to **actually exercise
WebGPU kernels off-Mac** before you push.

But mind the §1 scope: lavapipe covers host-side enforce/shape paths and **MatMul-free**
kernels only. Any **MatMul-containing** graph — including the expanded-Attention node
tests (`test_attention_4d_softcap_neginf_mask_expanded`) — crashes lavapipe and runs
**only** on the macOS-arm64 Metal leg, which is the source of truth for those. A green
lavapipe run never validates a MatMul/Attention fix off-Mac.
