# CPU Performance Feature Crew Findings

## 1. MatMulNBits

### A. DQ → MatMul Fusion Rules

#### Findings

* Only 4-bit block-quantized models are currently fused.
* There are no fusion rules for 2-bit or 8-bit models.
* FP16 models introduce Cast ops around MatMul, which block MatMulNBits fusion.

#### Recommendations

* Update fusion rules to support 2-bit and 8-bit quantized models.
* Enable MatMulNBits fusion for FP16 models.

#### Data: MNB vs Unfused QDQ Latency (Qwen2.5 LLMs, AMD + Intel avg)

The tables below compare **MNB** (native `MatMulNBits`) vs **QDQ** (unfused `DequantizeLinear → MatMul`) latency across model sizes and sequence lengths, averaged over AMD and Intel devices. QDQ models use block-quantized, asymmetric, unsigned weights with original layout. For all cases shown, `qdq_fused` scenario is used — however no actual fusion occurs because (a) 8-bit fusion rules don't exist, and (b) FP16 Cast ops block the pattern match.

##### Table 1 — FP32 8-bit: No Fusion Rules for 8-bit

| Model | MNB seq=1 (ms) | QDQ/MNB | MNB seq=128 (ms) | QDQ/MNB | MNB seq=256 (ms) | QDQ/MNB | MNB seq=512 (ms) | QDQ/MNB | MNB seq=1024 (ms) | QDQ/MNB |
|-------|----------------|---------|-------------------|---------|-------------------|---------|-------------------|---------|---------------------|---------|
| 0.5b  | 12.4           | 14.9×   | 228.9             | 1.9×    | 478.9             | 1.5×    | 979.7             | 1.3×    | 2,113.6             | 1.3×    |
| 1.5b  | 38.9           | 16.3×   | 797.5             | 1.7×    | 1,542.8           | 1.4×    | 3,206.4           | 1.2×    | 6,987.1             | 1.1×    |
| 3b    | 79.0           | 17.0×   | 1,558.2           | 1.8×    | 3,072.0           | 1.5×    | 6,147.3           | 1.3×    | 13,271.2            | 1.2×    |
| 7b    | 166.0          | 17.8×   | 3,522.0           | 1.7×    | 6,886.2           | 1.4×    | 13,734.8          | 1.3×    | 28,092.1            | 1.2×    |

At decode (seq=1), unfused 8-bit QDQ models are **15–18× slower** than MNB. The gap narrows at longer sequences as compute shifts from weight-bound to activation-bound, but remains 10–30% slower even at seq=1024.

##### Table 2 — FP16 4-bit: Cast Ops Block Fusion

| Model | MNB seq=1 (ms) | QDQ/MNB | MNB seq=128 (ms) | QDQ/MNB | MNB seq=256 (ms) | QDQ/MNB | MNB seq=512 (ms) | QDQ/MNB | MNB seq=1024 (ms) | QDQ/MNB |
|-------|----------------|---------|-------------------|---------|-------------------|---------|-------------------|---------|---------------------|---------|
| 0.5b  | 10.0           | 275×    | 293.5             | 10.2×   | 609.0             | 5.3×    | 1,387.1           | 2.7×    | 3,042.7             | 1.6×    |
| 1.5b  | 24.7           | 343×    | 865.9             | 10.6×   | 1,833.4           | 5.4×    | 3,593.5           | 3.2×    | 7,546.1             | 2.0×    |
| 3b    | 49.3           | 369×    | 1,691.8           | 11.5×   | 3,507.5           | 5.9×    | 7,034.6           | 3.4×    | 14,817.3            | 2.1×    |
| 7b    | 97.6           | 475×    | 3,795.1           | 13.4×   | 6,850.0           | 8.3×    | 13,764.8          | 5.5×    | 28,653.6            | 4.1×    |

This is the most impactful gap: 4-bit block QDQ models fuse perfectly in FP32 (achieving MNB-equivalent performance), but in FP16 they are **275–475× slower at decode** due to Cast ops blocking the pattern match. Even at seq=1024, the gap remains 1.6–4.1×.

##### Table 3 — FP16 8-bit: No Fusion Rules + Cast Ops

| Model | MNB seq=1 (ms) | QDQ/MNB | MNB seq=128 (ms) | QDQ/MNB | MNB seq=256 (ms) | QDQ/MNB | MNB seq=512 (ms) | QDQ/MNB | MNB seq=1024 (ms) | QDQ/MNB |
|-------|----------------|---------|-------------------|---------|-------------------|---------|-------------------|---------|---------------------|---------|
| 0.5b  | 14.5           | 125×    | 317.4             | 6.5×    | 638.3             | 3.5×    | 1,379.0           | 2.0×    | 2,934.6             | 1.4×    |
| 1.5b  | 39.3           | 138×    | 892.9             | 6.7×    | 1,878.4           | 3.5×    | 3,470.1           | 2.4×    | 7,430.5             | 1.6×    |
| 3b    | 75.7           | 154×    | 1,557.8           | 8.4×    | 3,255.6           | 4.4×    | 6,409.0           | 2.7×    | 13,642.2            | 1.8×    |
| 7b    | 159.0          | 431×    | 3,200.5           | 23.0×   | 6,358.3           | 12.0×   | 13,049.7          | 6.8×    | 27,104.1            | 4.1×    |

FP16 8-bit models are doubly impacted: no 8-bit fusion rules exist and Cast ops would block fusion even if rules were added. The 7b model shows particularly large ratios due to memory pressure from unfused DQ ops materializing full-precision weight tensors.

---

### B. Arm 8-bit MatMulNBits Kernel (FP16 scales)

#### Findings

* The 8-bit kernel for FP16 scales does not implement the `accuracy_level=4` variant.

#### Recommendation

* Implement this kernel if FP16 CPU models are a priority.

#### Data: ARM MNB 8-bit vs 4-bit Latency (Qwen2.5 LLMs, ARM Surface Laptop)

The tables below compare MNB 8-bit vs 4-bit latency on ARM for FP32 and FP16 models. In a well-optimized kernel, the 8b/4b ratio should be similar across precision. A significantly higher ratio for FP16 indicates a missing or slower kernel path.

##### Table 1 — FP32: MNB 8-bit vs 4-bit (8b/4b ratio)

| Model | 4b seq=1 (ms) | 8b/4b | 4b seq=128 (ms) | 8b/4b | 4b seq=256 (ms) | 8b/4b | 4b seq=512 (ms) | 8b/4b | 4b seq=1024 (ms) | 8b/4b |
|-------|----------------|-------|------------------|-------|------------------|-------|------------------|-------|-------------------|-------|
| 0.5b  | 9.7            | 4.8×  | 317.2            | 0.9×  | 763.9            | 0.9×  | 1,652.4          | 0.8×  | 3,390.1           | 0.7×  |
| 1.5b  | 87.9           | 0.6×  | 1,331.4          | 0.5×  | 2,353.8          | 0.5×  | 4,750.0          | 0.5×  | 8,388.8           | 0.6×  |
| 3b    | 116.5          | 1.7×  | 1,923.1          | 0.6×  | 3,922.6          | 0.6×  | 7,311.0          | 0.7×  | 14,182.5          | 0.7×  |

In FP32, 8-bit MNB performs **comparably or faster** than 4-bit at longer sequences (ratio ≤ 1 for 1.5b–3b), as expected since 8-bit kernels avoid nibble extraction.

##### Table 2 — FP16: MNB 8-bit vs 4-bit (8b/4b ratio)

| Model | 4b seq=1 (ms) | 8b/4b  | 4b seq=128 (ms) | 8b/4b | 4b seq=256 (ms) | 8b/4b | 4b seq=512 (ms) | 8b/4b | 4b seq=1024 (ms) | 8b/4b |
|-------|----------------|--------|------------------|-------|------------------|-------|------------------|-------|-------------------|-------|
| 0.5b  | 10.6           | 93.4×  | 403.7            | 3.1×  | 679.3            | 2.3×  | 1,355.4          | 1.6×  | 2,908.7           | 1.4×  |
| 1.5b  | 51.1           | 86.2×  | 1,308.8          | 4.0×  | 2,309.9          | 2.4×  | 4,482.5          | 1.6×  | 9,622.8           | 1.1×  |
| 3b    | 57.5           | 94.2×  | 2,067.4          | 3.3×  | 4,134.7          | 1.9×  | 7,810.8          | 1.5×  | 16,492.4          | 1.3×  |

In FP16, the 8-bit kernel is **86–94× slower at decode** (seq=1) and **1.1–4.0× slower** at longer sequences. This contrast with FP32 (where 8-bit is comparable or faster) confirms the FP16 8-bit kernel lacks an optimized code path. The 4-bit FP16 kernel benefits from the `accuracy_level=4` variant, while the 8-bit FP16 kernel does not implement it.

---

### C. Arm: KleidiAI Support for 4-bit Asymmetric Quantization

#### Findings

* KleidiAI’s optimized packed kernel (SDOT/I8MM, tiled) is disabled for asymmetric models.
* Asymmetric models fall back to generic NEON CompInt8, with a 30 to 46 percent penalty for models ≥1.5B parameters.
* Supporting asymmetric quantization requires adding per-block zero-points to the `qsi4c32p` packing format. It currently hardcodes `rhs_zero_point = 8`.

#### Recommendation

* Asymmetric quantization generally leads to better accuracy than symmetric models. We recommend using asymmetric quantization where possible, so optimizing this code path is important.

#### Data: ARM MNB 4-bit Asymmetric vs Symmetric Latency (Qwen2.5 LLMs, ARM Surface Laptop)

The tables below compare MNB 4-bit asymmetric vs symmetric latency on ARM. KleidiAI's optimized packed kernel is only used for symmetric models; asymmetric models fall back to generic NEON CompInt8. An asym/sym ratio > 1 indicates the asymmetric penalty from the fallback.

##### Table 1 — FP32: Asymmetric vs Symmetric (asym/sym ratio)

| Model | sym seq=1 (ms) | asym/sym | sym seq=128 (ms) | asym/sym | sym seq=256 (ms) | asym/sym | sym seq=512 (ms) | asym/sym | sym seq=1024 (ms) | asym/sym |
|-------|----------------|----------|-------------------|----------|-------------------|----------|-------------------|----------|---------------------|----------|
| 0.5b  | 29.1           | 0.33×    | 539.2             | 0.59×    | 1,013.1           | 0.75×    | 1,943.2           | 0.85×    | 3,035.0             | 1.12×    |
| 1.5b  | 64.3           | 1.37×    | 841.4             | 1.58×    | 1,565.4           | 1.50×    | 3,063.2           | 1.55×    | 6,081.3             | 1.38×    |
| 3b    | 66.9           | 1.74×    | 1,442.2           | 1.33×    | 2,612.1           | 1.50×    | 5,078.3           | 1.44×    | 10,086.8            | 1.41×    |

For models ≥1.5B parameters, asymmetric is **33–58% slower** than symmetric in FP32, consistent with the KleidiAI fallback to generic NEON. The 0.5b model is too small for the penalty to manifest (compute is not weight-bound enough).

*FP16 data not included: KleidiAI's optimized kernel (`accuracy_level=4`) is only available for FP32 scales. FP16 models use a generic kernel for both symmetric and asymmetric, so no asym/sym difference is observed.*

---

## 2. DequantizeLinear (Per-channel, Blockwise)

### A. Common

#### Findings

* 8-bit is generally faster than 4-bit.
* Separate kernels exist for 8-bit and sub-byte data types.
* Sub-byte kernels require nibble extraction, which adds overhead.
* Multithreading and SIMD are not implemented.

  * Per-channel has 8-bit multithreading support, but it is not enabled by default.
* The DQ kernel performs intermediate computation in FP32, casting to and from FP16.

#### Suggestions

* Implement multithreading and SIMD support.

  * MLAS includes blockwise dequantization kernels that may be leveraged.
* Evaluate whether FP16 casts can be avoided. CPU lacks native FP16 arithmetic, so implicit casts may still occur. Verification is required.

---

### B. Blockwise Quantization

#### Findings

* 4-bit FP32 DQ → MatMul patterns are fused into MatMulNBits.
* Fusion could be extended to 2-bit and 8-bit models.
* The priority of standalone DQ improvements depends on whether expanded fusion is sufficient.

#### Suggestions

* Revisit weight layout.

  * MatMulNBits uses NxK layout.
  * DQ uses KxN layout.
  * For standalone blockwise dequantization, KxN is likely preferable.

---

### C. Per-channel Quantization

#### Findings

* Per-channel is slower than blockwise, although it could be faster in principle.
* Incorrect graph transformations are applied for signed symmetric models.

#### Suggestions

* Reevaluate weight layout.

  * DQ uses KxN, but NxK may be better for per-channel quantization.
* Fix the incorrect graph transformation for signed symmetric models.

---

## 3. FP16 Models

### Findings

* FP16 support exists for DQ, Q, and contrib operators such as MatMulNBits and GQA.
* Core operators such as MatMul, Mul, and Add do not support FP16.
* This introduces Cast ops and blocks some fusion patterns.

### Suggestions

* Consider extending FP16 support to core compute operators.
* Investigate the cost of Cast ops. Even with FP16 support, implicit FP32 casts may still occur due to lack of native FP16 compute.

### Data

*Qwen2.5 models, AMD + Intel average. FP16/FP32 = ratio of FP16 model latency to FP32 model latency (>1 means FP16 is slower).*

**Table 1 — MNB Asym 4-bit: FP16 / FP32**

| model | seq=1 fp32 (ms) | seq=1 fp16/fp32 | seq=128 fp32 (ms) | seq=128 fp16/fp32 | seq=256 fp32 (ms) | seq=256 fp16/fp32 | seq=512 fp32 (ms) | seq=512 fp16/fp32 | seq=1024 fp32 (ms) | seq=1024 fp16/fp32 |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5b | 8.0 | 1.2 | 265.9 | 1.1 | 466.1 | 1.3 | 912.1 | 1.5 | 2184.5 | 1.4 |
| 1.5b | 21.0 | 1.2 | 690.4 | 1.3 | 1354.0 | 1.4 | 2784.6 | 1.3 | 5963.5 | 1.3 |
| 3b | 44.2 | 1.1 | 1641.6 | 1.0 | 3207.7 | 1.1 | 6477.9 | 1.1 | 13474.5 | 1.1 |

MNB 4-bit FP16 models are 1.0–1.5× the latency of FP32 — a modest overhead, indicating the MatMulNBits kernel handles FP16 scales reasonably well.

**Table 2 — MNB Asym 8-bit: FP16 / FP32**

| model | seq=1 fp32 (ms) | seq=1 fp16/fp32 | seq=128 fp32 (ms) | seq=128 fp16/fp32 | seq=256 fp32 (ms) | seq=256 fp16/fp32 | seq=512 fp32 (ms) | seq=512 fp16/fp32 | seq=1024 fp32 (ms) | seq=1024 fp16/fp32 |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5b | 12.4 | 1.2 | 228.9 | 1.4 | 478.9 | 1.3 | 979.7 | 1.4 | 2113.6 | 1.4 |
| 1.5b | 38.9 | 1.0 | 797.5 | 1.1 | 1542.8 | 1.2 | 3206.4 | 1.1 | 6987.1 | 1.1 |
| 3b | 79.0 | 1.0 | 1558.2 | 1.0 | 3072.0 | 1.1 | 6147.3 | 1.0 | 13271.2 | 1.0 |

MNB 8-bit FP16 is nearly on par with FP32 (1.0–1.4×), showing minimal overhead from FP16 scales.

**Table 3 — QDQ Unfused Asym Unsigned 4-bit: FP16 / FP32**

| model | seq=1 fp32 (ms) | seq=1 fp16/fp32 | seq=128 fp32 (ms) | seq=128 fp16/fp32 | seq=256 fp32 (ms) | seq=256 fp16/fp32 | seq=512 fp32 (ms) | seq=512 fp16/fp32 | seq=1024 fp32 (ms) | seq=1024 fp16/fp32 |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5b | 686.0 | 4.1 | 944.6 | 3.2 | 1213.6 | 2.6 | 1782.5 | 2.1 | 3172.6 | 1.7 |
| 1.5b | 2322.9 | 3.7 | 3176.8 | 2.9 | 3890.2 | 2.6 | 5772.6 | 2.0 | 9488.0 | 1.6 |
| 3b | 4566.8 | 3.9 | 6060.4 | 3.3 | 7718.3 | 2.8 | 11490.4 | 2.1 | 19918.6 | 1.6 |

QDQ 4-bit FP16 is **1.6–4.1× slower** than FP32. The overhead is highest at decode (seq=1) and decreases at longer sequences, consistent with the FP16 Cast overhead being a fixed per-layer cost.

**Table 4 — QDQ Unfused Asym Unsigned 8-bit: FP16 / FP32**

| model | seq=1 fp32 (ms) | seq=1 fp16/fp32 | seq=128 fp32 (ms) | seq=128 fp16/fp32 | seq=256 fp32 (ms) | seq=256 fp16/fp32 | seq=512 fp32 (ms) | seq=512 fp16/fp32 | seq=1024 fp32 (ms) | seq=1024 fp16/fp32 |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5b | 185.0 | 9.8 | 445.4 | 4.6 | 715.7 | 3.2 | 1298.1 | 2.2 | 2732.3 | 1.5 |
| 1.5b | 634.1 | 8.5 | 1337.4 | 4.5 | 2104.1 | 3.2 | 3835.0 | 2.2 | 7649.0 | 1.5 |
| 3b | 1340.9 | 8.7 | 2847.0 | 4.6 | 4598.2 | 3.1 | 8263.6 | 2.1 | 16510.9 | 1.5 |

QDQ 8-bit FP16 is 1.5–10× slower than FP32. The overhead is smaller than 4-bit but still substantial, confirming that the unfused DQ + Cast path adds significant cost for FP16 models.

---

## 4. Full QDQ Models (Per-tensor, Activation + Weight)

### A. uint16 Activation, uint8 Weight

#### Findings

* ONNX does not define QLinear operators for this combination. It only supports uint8/int8 activation and weight.
* This configuration is common in non-LLM models such as Stable Diffusion and ASG perception models.
* Without QLinear operators, DQ and Q nodes remain unfused and add compute overhead to float operators.
* ASG is proposing new operators to support this combination.

#### Suggestions

* Collaborate with ASG to implement QLinear operators for 16-bit activations and 8-bit weights.
* Update the ONNX spec if possible. Otherwise, use contrib operators.

---

### B. CPU EP Handling of QDQ Models

#### Findings

* Fully QDQ models are used as shared artifacts across EPs.
* These models target EPs such as QNN EP, which fuse DQ → Op → Q patterns into integer kernels.
* CPU EP does not have a broad set of efficient integer kernels.
* QDQ nodes block fusion patterns such as MatMulNBits and LayerNorm.
* Unfused QDQ nodes add computational overhead.
* ORT has a session option `session.enable_quant_qdq_cleanup` that can remove redundant QDQ pairs.
  * It does not however handle cases where Q and DQ ops have shape related operators in between, such as Q → Reshape → DQ.
  * It does not fold DQ nodes on weights into float initializers.

#### Suggestions

* Consider specialized handling of QDQ models in CPU EP:

  * Retain DQ nodes for weights. There are different scenarios here:
    * DQ (blockwise) -> MatMul: Keep DQ to enable MatMulNBits fusion.
    * DQ (per-tensor) on scalar: No benefit to keep DQ. Fold into float initializer.
    * DQ with same bit-width as float compute (e.g. fp32 compute with int32 weights): No benefit to keep DQ. Fold into float initializer.
    * DQ on non-scalar weights: There are potential memory savings to keep DQ, but it adds overhead. This requires further investigation to determine the best approach.
  * Retain activation QDQ nodes only if they can be fused.
    * Extend qdq cleanup optimizer to handle patterns like Q → Reshape → DQ.
* There are open questions about output mismatches:

  * Output differences are expected.
  * Overall quality may improve unless the model was trained with activation quantization awareness. For standard static activation quantization, this is not a concern.

#### Data: Stable Diffusion Component Benchmarks (Intel CPU, ORT 1.24.1)

The following tables use per-component benchmarks of Stable Diffusion v1.5 (text_encoder, unet, vae_decoder, vae_encoder) to quantify the impact of QDQ on CPU EP. Three model variants are compared: **float** (baseline), **qdq** (full QDQ quantization), and **qdq_cleanup** (with `session.enable_quant_qdq_cleanup` enabled).

##### Table 1 — Latency Comparison

| Component     | Float (ms) | QDQ (ms)  | QDQ Cleanup (ms) | QDQ / Float | Cleanup / Float | Cleanup Improvement |
|---------------|-----------|-----------|-------------------|-------------|-----------------|---------------------|
| text_encoder  | 57.1      | 94.2      | 93.5              | 1.65×       | 1.64×           | 0.7%                |
| unet          | 1,977.9   | 4,428.6   | 3,819.6           | 2.24×       | 1.93×           | 13.8%               |
| vae_decoder   | 5,702.6   | 10,349.9  | 7,485.4           | 1.81×       | 1.31×           | 27.7%               |
| vae_encoder   | 3,024.1   | 4,401.3   | 3,855.9           | 1.46×       | 1.28×           | 12.4%               |

QDQ models are **1.5–2.2× slower** than float on CPU EP. Cleanup reduces the gap (1.3–1.9×) but models remain slower than float. Cleanup is most effective for Conv-heavy components (vae_decoder: 27.7% improvement) and least effective when Q/DQ ops pass through shape operators (text_encoder: 0.7%).

##### Table 2 — Op Count & Fusion Analysis

| Component     | Variant     | Total Ops | Q    | DQ    | FusedMatMul | QuickGelu |
|---------------|-------------|-----------|------|-------|-------------|-----------|
| text_encoder  | float       | 565       | 0    | 0     | 12          | 12        |
| text_encoder  | qdq         | 1,097     | 206  | 487   | 0           | 12        |
| text_encoder  | qdq_cleanup | 902       | 121  | 378   | 0           | 12        |
| unet          | float       | 2,605     | 0    | 0     | 0           | 47        |
| unet          | qdq         | 4,507     | 930  | 2,050 | 0           | 0         |
| unet          | qdq_cleanup | 3,199     | 402  | 1,355 | 0           | 47        |
| vae_decoder   | float       | 402       | 0    | 0     | 3           | 29        |
| vae_decoder   | qdq         | 958       | 212  | 460   | 0           | 0         |
| vae_decoder   | qdq_cleanup | 594       | 66   | 285   | 0           | 29        |
| vae_encoder   | float       | 333       | 0    | 0     | 3           | 21        |
| vae_encoder   | qdq         | 739       | 163  | 354   | 0           | 0         |
| vae_encoder   | qdq_cleanup | 463       | 51   | 221   | 0           | 21        |

Key observations:
* QDQ nearly **doubles** total op count (e.g. unet: 2,605 → 4,507).
* QuickGelu is **lost** in QDQ models (unet: 47 → 0). Cleanup restores it.
* FusedMatMul is **blocked** in QDQ text_encoder (12 → 0). Not restored by cleanup because DQ nodes on scalar weights remain. Folding scalar weight DQ into float initializers (as suggested below) would unblock this fusion.

##### Table 3 — Q/DQ Breakdown: Activation vs Weight & Shape-Op Barriers

| Component     | Variant     | Q (act) | DQ (act) | DQ (wt) | Q → shape op | DQ ← shape op | Q shape % |
|---------------|-------------|---------|----------|---------|--------------|----------------|-----------|
| text_encoder  | qdq         | 206     | 256      | 231     | 121          | 145            | 59%       |
| text_encoder  | qdq_cleanup | 121     | 147      | 231     | 121          | 145            | 100%      |
| unet          | qdq         | 930     | 1,177    | 873     | 401          | 401            | 43%       |
| unet          | qdq_cleanup | 402     | 482      | 873     | 401          | 401            | 100%      |
| vae_decoder   | qdq         | 212     | 258      | 202     | 66           | 68             | 31%       |
| vae_decoder   | qdq_cleanup | 66      | 83       | 202     | 66           | 68             | 100%      |
| vae_encoder   | qdq         | 163     | 198      | 156     | 50           | 52             | 31%       |
| vae_encoder   | qdq_cleanup | 51      | 65       | 156     | 50           | 52             | 100%      |

Key observations:
* **Weight DQ ops are never removed** by cleanup (e.g. unet: DQ(wt) stays at 873).
* After cleanup, **all remaining activation Q ops are shape-blocked** (Q shape % = 100%), confirming that cleanup successfully removes all non-shape-blocked pairs.
* A large fraction of original Q ops flow into shape operators (unet: 43%, text_encoder: 59%). Extending cleanup to handle Q → Reshape → DQ patterns would remove these.

---

## 5. Vision Models (Float)

### Findings

* ARM64 could benefit from improvements to the NCHWc Conv kernel suite.
* Conv-heavy models are well suited for NCHWc acceleration but require optimized vectorized kernels, which ARM64 currently lacks.
* On some platforms, certain Conv parameter configurations result in slowdowns under NCHWc compared to default NCHW.

  * NCHWc is enabled by default through graph optimization.
  * Users should be able to opt out via a session option.
  * OLIVE recipes should reflect the optimal configuration for the target platform.
* The NCHWc layout transformer does not fully handle elementwise ops.

  * It inserts Reorder nodes around layout-agnostic ops.
  * This introduces unnecessary transpose overhead, particularly in lightweight Conv models.

### Suggestions

* Evaluate alternative Conv algorithms such as Winograd for standard 3×3 Conv.

  * Reference: [https://github.com/microsoft/onnxruntime/issues/12834](https://github.com/microsoft/onnxruntime/issues/12834)
* Explore additional activation fusions in MLAS Conv micro-kernels.

  * Currently only ReLU is fused.
  * Adding Gelu, SiLU, etc., requires significant development effort due to register pressure constraints.
* When customers report CPU EP being 2× slower than OpenVINO EP, confirm configuration parity.

  * OpenVINO may use mixed precision via performance hints.
  * CPU EP runs the model as provided.
* Investigate the threading model in the NCHWc Conv kernel suite, particularly for unbatched, ungrouped Conv (B=1, G=1).
