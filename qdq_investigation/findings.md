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

---

### B. Arm 8-bit MatMulNBits Kernel (FP16 scales)

#### Findings

* The 8-bit kernel for FP16 scales does not implement the `accuracy_level=4` variant.

#### Recommendation

* Implement this kernel if FP16 CPU models are a priority.

---

### C. Arm: KleidiAI Support for 4-bit Asymmetric Quantization

#### Findings

* KleidiAI’s optimized packed kernel (SDOT/I8MM, tiled) is disabled for asymmetric models.
* Asymmetric models fall back to generic NEON CompInt8, with a 30 to 46 percent penalty for models ≥1.5B parameters.
* Supporting asymmetric quantization requires adding per-block zero-points to the `qsi4c32p` packing format. It currently hardcodes `rhs_zero_point = 8`.

#### Recommendation

* Not a top priority. Further investigation is needed to estimate effort and performance gains.

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

#### Suggestions

* Consider specialized handling of QDQ models in CPU EP:

  * Retain DQ nodes for weights.
  * Retain activation QDQ nodes only if they can be fused.
* There are open questions about output mismatches:

  * Output differences are expected.
  * Overall quality may improve unless the model was trained with activation quantization awareness. For standard static activation quantization, this is not a concern.

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
