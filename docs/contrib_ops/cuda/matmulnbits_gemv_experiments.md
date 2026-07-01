# MatMulNBits M=1 GEMV `__launch_bounds__` Experiments

This file records the occupancy‑tuning experiments on the single‑row (M=1) weight‑only GEMV
kernels of `MatMulNBits`, together with the exact steps to reproduce them, the Nsight Compute
(ncu) before/after measurements, and the analysis that motivated the final code shape.

> **Note**: These are **point‑in‑time** measurements captured on the specific GPU, driver, CUDA
> toolkit, and ORT build noted below. Treat the numbers as a historical baseline for regression
> comparison, not as current performance guidance — re‑run on your own hardware before drawing
> conclusions.

Related source:

- `onnxruntime/contrib_ops/cuda/quantization/matmul_4bits.cu` — 4‑bit GEMV (`MatMulFloatInt4Kernel`, `MatMulFloat4BatchedKernel`).
- `onnxruntime/contrib_ops/cuda/quantization/matmul_8bits.cu` — 8‑bit GEMV (`MatMulFloat8bKernelM1`).
- `onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cuh` — shared header; home of the
  arch‑aware constant `kMatMulNBitsM1MinBlocksPerSM`.

---

## Background

`MatMulNBits` dispatches by the number of rows `M` of activation `A(M, K)`:

- **M = 1** → a dedicated single‑row GEMV kernel (`MatMulFloatInt4Kernel` for 4‑bit,
  `MatMulFloat8bKernelM1` for 8‑bit). Thread block is `(kWarpSize, kColsPerThreadBlock) = (32, 8) = 256` threads.
- **M ≥ 2** → the batched kernel (`MatMulFloat4BatchedKernel`, half/bf16) or a dequant + cuBLAS path.

The M=1 kernels are latency‑bound **pure weight streaming**: they read the quantized weight matrix
once and do a warp reduction. With no occupancy hint they use ~39–40 registers/thread, which on a
64‑warp/SM part caps them to 6 of 8 resident blocks (~75 % theoretical occupancy), leaving too few
resident warps to hide global‑load latency.

`__launch_bounds__(threads, minBlocksPerSM)` lets us request a register cap so more blocks stay
resident. The second argument is a **compile‑time** hint — it cannot come from a runtime device
query — so it is selected per architecture via `__CUDA_ARCH__` into
`kMatMulNBitsM1MinBlocksPerSM`:

| Resident warps / SM | Architectures | `minBlocksPerSM` |
|---|---|---|
| 64 warps (2048 threads) | sm_50/52/53, sm_60/61/62, sm_70/72, **sm_80 (A100)**, sm_90, sm_100/103 | 8 |
| 32 warps (1024 threads) | sm_75 (Turing / RTX 20xx) | 4 |
| 48 warps (1536 threads) | sm_86/87 (client Ampere / Orin), sm_89 (Ada RTX 40xx), sm_120 (RTX 50xx) | 6 |

> Ada (CC 8.9) doubled the FP32 datapaths but — like client Ampere (CC 8.6) — keeps only **48**
> resident warps / 1536 threads per SM (NOT 64). Verified with ptxas: under `__launch_bounds__(256, 8)`
> sm_89 clamps to 6 blocks (39–40 regs) while the 2048‑thread sm_80 is cut to 32. The `6` bucket is a
> safe no‑op on those parts (they already reach full occupancy at ~40 regs), and the `else` default of
> 6 is the safe under‑estimate for any unlisted/future arch.

---

## Environment

- **GPU**: NVIDIA A100‑SXM4‑80GB, `sm_80`. All 8 GPUs free; experiments pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1`.
- **Clock lock** (stabilize ncu / wall‑clock timing):
  ```bash
  sudo -n nvidia-smi -i 1 -lgc 1410   # lock GPU1 graphics clock to 1410 MHz
  # ... run experiments ...
  sudo -n nvidia-smi -i 1 -rgc        # reset when done
  ```
- **Toolchain**: CUDA 13.0 (`~/cuda13.0`, `nvcc`/`ncu` V13.0).
- **Python**: `~/git/onnxruntime/.venv/bin/python`.
- **Build dir**: `build/cu130_fp4_bench/Release` (configured for `sm_80`).
- **Branch**: `tlwu/matmul_nbits_m1_opt`.

### Benchmark / profiling script

`onnxruntime/test/python/transformers/profile_matmul_nbits.py`

Builds a single `MatMulNBits` model, binds device I/O, and prints a machine‑parseable line
`MATMUL_NBITS_RESULT {json}` with the average kernel latency in microseconds.

> The script is not tracked at the current HEAD. It was restored from
> `origin/jambayk/mnb-small-m-gemv` (commit `23a4c56226`, "Fix bf16 scales in MatMulNBits profiler"):
> ```bash
> git checkout 23a4c56226 -- onnxruntime/test/python/transformers/profile_matmul_nbits.py
> ```

Key flags: `--k --n --m --block-size --bits {4,8} --dtype {fp16,bf16} --warmup --repeat`.

### Build + stage the provider

Provider‑only incremental build, then copy the freshly linked `.so` over the wheel copy in the venv
(the installed wheel `.so` is otherwise stale for provider‑only edits):

```bash
cd ~/git/onnxruntime && source .venv/bin/activate
cmake --build build/cu130_fp4_bench/Release --target onnxruntime_providers_cuda --parallel 16
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
   .venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
```

### ncu capture

`ncu` needs elevated GPU counter permission (`sudo -n -E`, else `ERR_NVGPUCTRPERM`). Run the Python
process from `/tmp`:

```bash
cd /tmp
CUDA_VISIBLE_DEVICES=1 sudo -n -E ~/cuda13.0/bin/ncu --set full \
  -k regex:MatMulFloat8bKernelM1 -c 2 -f -o /tmp/m8_m1_opt \
  ~/git/onnxruntime/.venv/bin/python \
  ~/git/onnxruntime/onnxruntime/test/python/transformers/profile_matmul_nbits.py \
  --k 4096 --n 12288 --m 1 --block-size 32 --bits 8 --dtype fp16 --warmup 3 --repeat 5

# read back the report
~/cuda13.0/bin/ncu -i /tmp/m8_m1_opt.ncu-rep --page details
```

Kernel name filters: `regex:MatMulFloatInt4Kernel` (4‑bit M=1), `regex:MatMulFloat8bKernelM1`
(8‑bit M=1), `regex:MatMulFloat4BatchedKernel` (4‑bit M≥2).

---

## Experiment 1 — 4‑bit M=1: `__launch_bounds__(256, 8)` (ACCEPTED, committed)

Applied the arch‑aware `minBlocksPerSM` hint to `MatMulFloatInt4Kernel`. On sm_80 this lowers the
register footprint 40 → 32, raising the block‑limit from 6 to 8 (75 % → 100 % theoretical occupancy).

### ncu — after (A100 sm_80 @1410 MHz, gate_up shape K=4096 N=12288 fp16 blk=32)

| Metric | Baseline (no hint) | Optimized `(256, 8)` |
|---|---|---|
| Registers / thread | 40 | **32** |
| Block Limit (Registers) | 6 | **8** |
| Theoretical Occupancy | 75 % | **100 %** |
| Achieved Occupancy | ~75 % | **78 %** |
| Compute (SM) Throughput | — | 53 % |
| DRAM Throughput | — | 49 % |
| Dominant warp stall | — | Long Scoreboard (~52 %) |
| Kernel Duration | — | **30.85 µs** |

Result: **~13 % faster** at M=1 on A100 (recorded during the original 4‑bit work). The optimized
kernel is balanced (Compute ≈ DRAM) and memory‑latency bound — near optimal for a weight‑streaming
GEMV.

> The pre‑hint 4‑bit duration was measured in the original 4‑bit session (≈40 regs / 75 % occ,
> ~13 % slower); only the optimized state was re‑captured here. The 8‑bit experiment below has a full
> before/after captured in one sitting.

---

## Experiment 2 — 8‑bit M=1: same hint (REJECTED, negative result)

Hypothesis: the 8‑bit M=1 kernel (`MatMulFloat8bKernelM1`) is structurally analogous to the 4‑bit
one, so `__launch_bounds__(256, 8)` should give the same occupancy win.

### Steps to reproduce

1. Lock GPU1 clocks: `sudo -n nvidia-smi -i 1 -lgc 1410`.
2. **Baseline capture** (pristine kernel, no second `__launch_bounds__` arg): build + stage, then
   ```bash
   cd /tmp
   CUDA_VISIBLE_DEVICES=1 sudo -n -E ~/cuda13.0/bin/ncu --set full \
     -k regex:MatMulFloat8bKernelM1 -c 2 -f -o /tmp/m8_m1_base \
     ~/git/onnxruntime/.venv/bin/python \
     .../profile_matmul_nbits.py --k 4096 --n 12288 --m 1 --block-size 32 --bits 8 --dtype fp16 --warmup 3 --repeat 5
   ```
3. Add `kMatMulNBitsM1MinBlocksPerSM` to the kernel's `__launch_bounds__`, rebuild + stage.
4. **Optimized capture** to `/tmp/m8_m1_opt` (same command, `-o /tmp/m8_m1_opt`).
5. Clean wall‑clock A/B (no ncu serialization) for three shapes, `--warmup 25 --repeat 200`,
   toggling the hint and rebuilding between runs.
6. Reset clocks: `sudo -n nvidia-smi -i 1 -rgc`.

### ncu — before vs after (A100 sm_80 @1410 MHz, K=4096 N=12288 fp16 blk=32)

| Metric | Baseline (no hint) | `(256, 8)` |
|---|---|---|
| Registers / thread | 39 | 32 |
| Block Limit (Registers) | 6 | 8 |
| Theoretical Occupancy | 75 % | 100 % |
| Achieved Occupancy | 61.8 % | 79.2 % |
| Compute (SM) Throughput | 66.3 % | 63.7 % |
| DRAM Throughput | 47.5 % | 44.9 % |
| Memory Throughput | 968 GB/s | 915 GB/s |
| **Kernel Duration** | **59.33 µs** | **62.78 µs (WORSE)** |

Occupancy went up, **but the kernel got slower** and memory throughput dropped.

### Clean wall‑clock A/B (M=1, fp16, blk32, warmup 25 / repeat 200)

| Shape (K × N) | Baseline (no hint) | `minBlocks=8` | Δ |
|---|---|---|---|
| 4096 × 4096 | 32.12 µs | 32.71 µs | **+1.8 %** |
| 4096 × 12288 | 62.96 µs | 65.68 µs | **+4.3 %** |
| 12288 × 4096 | 68.75 µs | 76.06 µs | **+10.6 %** |

Uniformly slower; the regression is worst for the large‑K shape.

### Why 8‑bit differs from 4‑bit

8‑bit weights are **2× the bytes** of 4‑bit, so the 8‑bit GEMV is more **DRAM‑bandwidth‑bound** and
already has enough resident warps to hide load latency. Cutting registers 39 → 32 forces extra
recompute, and the extra concurrent blocks contend for DRAM/L2 — a net loss.

**Decision**: do **not** apply the hint to the 8‑bit kernel. `matmul_8bits.cu` is left pristine.

---

## Experiment 3 — 4‑bit batched (M ≥ 2): higher occupancy (REJECTED, negative result)

For completeness: the M=1 change does not touch the M ≥ 2 path (a different kernel,
`MatMulFloat4BatchedKernel`), confirmed by ncu (batched metrics unchanged). A separate attempt to
raise batched occupancy also regressed.

- Natural batched register use is 70–80 across all `CtaM` instantiations. `__launch_bounds__(256, 3)`
  caps at 85 → 80 used, no spill (near‑optimal).
- Forcing `__launch_bounds__(256, 4)` → 64 regs → 50 % theo / 43 % achieved occupancy **but slower**:
  - M=4: 50.75 → 53.06 µs (**+4.5 %**)
  - M=16: 146.6 → 154.0 µs (**+5.1 %**)
- The 37.5 % occupancy is baked into the register‑heavy tiling that buys ILP/reuse; occupancy is not
  the lever. Real M ≥ 2 gains need an algorithmic tiling redesign, not a `launch_bounds` bump.

---

## Experiment 4 — 4‑bit M=1 reproduction on consumer GPUs sm_86 & sm_120 (bucket 6, ACCEPTED no‑op)

Re‑running the 4‑bit M=1 A/B on two `minBlocksPerSM = 6` ("48 warps/SM") parts to close the first
item under **Next tasks** — confirm the hint is a **no‑op, never a regression**, on architectures
that take a different bucket than the A100 (`sm_80`, bucket 8) where the original win was measured.

### Environment (machine `TLWU-18`)

- **GPUs**:
  - GPU 0 — NVIDIA GeForce **RTX 5060 Ti**, `sm_120` (CC 12.0, consumer Blackwell), 16 GB.
  - GPU 1 — NVIDIA GeForce **RTX 3060**, `sm_86` (CC 8.6, client Ampere), 12 GB.
  - Both map to `kMatMulNBitsM1MinBlocksPerSM = 6` (48 resident warps / 1536 threads per SM).
- **Host**: WSL2 (Ubuntu) on Windows. **Toolchain**: CUDA 13.0 (`~/cuda13.0`), cuDNN 9.12
  (`~/cudnn_9.12_cuda13`). **Python**: `/home/tlwu/venv` (3.12).
- **Build**: `build/cuda130_bench/Release`, `CMAKE_CUDA_ARCHITECTURES="86-real;120-real"`,
  `--enable_lto`, built via `.env/build_cuda13_bench.sh`. **Branch**: `tlwu/matmul_nbits_m1_opt`
  (`onnxruntime_gpu 1.28.0`).
- **WSL2 limitations (methodology adapted accordingly)**:
  - `nvidia-smi -lgc` (clock lock) is **not supported** under WSL2 (`Unknown Error`), so GPU boost
    clocks could not be pinned — wall‑clock timing has large run‑to‑run variance (quantified below).
  - Nsight Compute **hardware counters are blocked** (`ERR_NVGPUCTRPERM`) even under `sudo`; this is
    a Windows‑driver‑side setting, so the ncu `--set full` occupancy/throughput capture used in
    Experiments 1–2 was **not available here**.
  - **Substitute for ncu**: register footprint and occupancy limits are static properties of the
    compiled cubin, so they were read directly and deterministically with
    `cuobjdump -res-usage` (no GPU counters needed). This is exactly the metric that governs the
    occupancy hint, so it is sufficient to prove the no‑op.

### A/B method

Toggle only the second `__launch_bounds__` argument on `MatMulFloatInt4Kernel`, provider‑only rebuild,
and stage the `.so` (as in the Experiment 2 steps):

```bash
# baseline: __launch_bounds__(kWarpSize*kColsPerThreadBlock)                          -> no hint
# optimized: __launch_bounds__(kWarpSize*kColsPerThreadBlock, kMatMulNBitsM1MinBlocksPerSM)
cmake --build build/cuda130_bench/Release --target onnxruntime_providers_cuda --parallel 4
cp build/cuda130_bench/Release/libonnxruntime_providers_cuda.so \
   /home/tlwu/venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so

# read the compiled register/occupancy footprint (no GPU counters required):
cuobjdump -res-usage build/cuda130_bench/Release/libonnxruntime_providers_cuda.so \
  | grep -A1 'MatMulFloatInt4KernelI6__halfLi32ELb1E'   # <half, block_size=32, has_zero_point>
```

### Register / occupancy — before vs after (`cuobjdump -res-usage`, `<half, blk 32, zp>`)

| Metric | `sm_86` baseline → optimized | `sm_120` baseline → optimized |
|---|---|---|
| Registers / thread | 40 → **40** | 40 → **40** |
| Local (spill) bytes | 0 → 0 | 0 → 0 |
| Max resident warps / SM | 48 | 48 |
| Block limit (warps: 1536/256) | 6 | 6 |
| Block limit (registers: 65536/(40·256)=6.4) | 6 | 6 |
| Theoretical occupancy | 100 % → **100 %** | 100 % → **100 %** |

On both parts the compiler **already used exactly 40 registers without the hint**, and 40 regs already
permits the full 6 resident blocks (= 48 warps = 100 % theoretical occupancy). Requesting
`minBlocksPerSM = 6` therefore forces **no** register cut and leaves the block/occupancy limit
unchanged — the hint is an occupancy **no‑op** on `sm_86` and `sm_120`, exactly as the header comment
predicts. (Contrast `sm_80`, where the bucket‑8 hint cut 40 → 32 regs and lifted 6 → 8 blocks.)

> The SASS is *not* byte‑identical between the two builds (ptxas re‑schedules slightly when the
> min‑blocks attribute is present: `sm_86` 1256 → 1248, `sm_120` 1624 → 1592 instructions), but the
> register allocation and occupancy limits are unchanged, so there is no occupancy effect either way.

### Wall‑clock A/B (M=1, fp16, blk 32, `--warmup 25 --repeat 200`)

Because clocks could not be locked, each configuration was run **4×** and the table reports the
**minimum** (least boost‑perturbed) with the full observed range. The profiler itself already reports
the best of 10 inner trials per run.

**GPU 0 — RTX 5060 Ti (`sm_120`)** — avg µs, lower is better:

| Shape (K × N) | Baseline min (range) | Optimized min (range) | Δ (min) |
|---|---|---|---|
| 4096 × 4096 | 53.83 (53.8–75.1) | 53.23 (53.2–60.5) | −1.1 % |
| 4096 × 12288 | 66.90 (66.9–93.8) | 69.63 (69.6–85.9) | +4.1 % |
| 12288 × 4096 | 71.81 (71.8–81.7) | 73.20 (73.2–76.0) | +1.9 % |

**GPU 1 — RTX 3060 (`sm_86`)** — avg µs, lower is better:

| Shape (K × N) | Baseline min (range) | Optimized min (range) | Δ (min) |
|---|---|---|---|
| 4096 × 4096 | 68.61 (68.6–74.9) | 67.13 (67.1–80.1) | −2.2 % |
| 4096 × 12288 | 133.02 (133.0–151.8) | 128.77 (128.8–149.6) | −3.2 % |
| 12288 × 4096 | 129.97 (130.0–159.6) | 140.74 (140.7–157.2) | +8.3 % |

The Δ(min) sign is **inconsistent** across shapes (both faster and slower) and every baseline/optimized
range **overlaps**. A noise‑floor control — the *same* baseline `.so` measured 3× — showed run‑to‑run
spread as large as the A/B deltas (e.g. GPU 0 `4096×12288`: 66.9 / 93.8 / 75.2 µs, ~40 %; GPU 1
`12288×4096`: 130.0 / 159.6 µs, ~23 %). The wall‑clock differences are therefore **within measurement
noise**, consistent with the register/occupancy evidence that the two builds are occupancy‑equivalent.

### Result

**Reproduced as expected**: on `sm_86` and `sm_120` (bucket 6) the M=1 4‑bit hint is a confirmed
**no‑op** — identical 40‑register footprint and 100 % theoretical occupancy with and without it, and
no consistent wall‑clock change (all within the unlocked‑clock noise band). No regression on either
GPU. This matches the `matmul_nbits.cuh` header rationale that bucket 6 is a *safe no‑op* on parts
that already reach full occupancy at ~40 registers; the ~13 % speed‑up remains specific to the
bucket‑8 datacenter parts (e.g. A100 `sm_80`) where the hint actually cuts registers 40 → 32.

---

## Final code shape

- `kMatMulNBitsM1MinBlocksPerSM` (arch‑aware) lives in the shared header
  `matmul_nbits.cuh` and is consumed **only** by the 4‑bit `MatMulFloatInt4Kernel`.
- `matmul_8bits.cu` is unchanged (no hint), documented in the header comment as a deliberate opt‑out
  backed by the measured regression above. The constant stays in the shared header so a future N‑bit
  M=1 kernel can opt in after its own A/B.

---

## Analysis summary

1. The `__launch_bounds__` `minBlocksPerSM` win is **not portable across bit widths**. It helps the
   4‑bit M=1 GEMV (compute‑latency bound, benefits from more warps) but hurts the more
   DRAM‑bound 8‑bit M=1 GEMV.
2. Occupancy is a means, not an end: for both the 8‑bit M=1 and the 4‑bit batched kernels, pushing
   theoretical occupancy up while cutting registers **regressed** wall‑clock latency.
3. Always validate an occupancy hint per (kernel, arch) with a clean wall‑clock A/B — ncu occupancy
   numbers alone are misleading.

## Next tasks / ideas

- [x] Re‑confirm the 4‑bit M=1 win on non‑A100 arches that take a different bucket — **done for
      bucket 6 in Experiment 4**: `sm_86` (RTX 3060) and `sm_120` (RTX 5060 Ti) are a confirmed
      no‑op (40 regs / 100 % occ unchanged, no wall‑clock regression). Still open: `sm_89 → 6`
      (Ada) and `sm_75 → 4` (Turing) to cover the remaining buckets.
- [ ] Capture a fresh 4‑bit M=1 **baseline** duration in one sitting (this doc only re‑captured the
      optimized state) so the 4‑bit before/after is self‑contained.
- [ ] Investigate an algorithmic redesign for the M ≥ 2 batched kernel (reduce live registers without
      spilling) — the only path to real M ≥ 2 speedups, since occupancy tuning does not help.
- [ ] Consider whether an 8‑bit M=1 kernel rewrite (wider vectorized loads / better L2 reuse) could
      shift it off the DRAM bound before revisiting any occupancy hint.
- [ ] Upstream the `profile_matmul_nbits.py` script (currently only on `origin/jambayk/mnb-small-m-gemv`)
      so these experiments are reproducible from a tracked path.
