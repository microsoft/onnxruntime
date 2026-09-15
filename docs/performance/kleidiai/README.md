<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->

# Arm® KleidiAI™ software compatibility in ONNX Runtime

This page outlines support details for the [Arm KleidiAI software](https://www.arm.com/markets/artificial-intelligence/software/kleidi) micro-kernel library. Micro-kernels are integrated into ONNX Runtime via Microsoft Linear Algebra Subprograms (MLAS) to support CPU-Based inference acceleration on Arm-Based CPUs.

Arm, Kleidi, KleidiAI, KleidiCV and Kleidi Libraries are registered trademarks or trademarks of Arm Limited (or its subsidiaries or affiliates) in the US and/or elsewhere.

- KleidiAI tagged release: [`v1.31.0`](https://github.com/ARM-software/kleidiai/tree/v1.31.0)
- ONNX Runtime release: [`v1.30.0`](https://github.com/microsoft/onnxruntime/tree/v1.30.0)
- ONNX Runtime KleidiAI pin: [`v1.20.0`](https://github.com/ARM-software/kleidiai/tree/v1.20.0)
- Last updated date: `2026-09-15`

## Summary

| Classification | Micro-kernels |
| --- | ---: |
| ✅ Integrated | 29 |
| 🔗 Referenced only | 1 |
| 🟣 PR available | 10 |
| — Not integrated | 76 |
| Unavailable in ONNX Runtime's `v1.20.0` pin | 52 |
| **Total SVE/SME-family micro-kernels** | **116** |

> Integration and pin availability are separate. A kernel can be absent from the current ONNX Runtime pin even when a newer integration pull request exists.

## Kernel compatibility

Sections are grouped by user-facing operation. Kernel symbols are hidden behind compact source links; hover a link to see its full symbol.

### Depthwise RHS pack (2)

<details>
<summary>Show 2 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.c "kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme") | `X16P · X16 · X16` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.c "kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme") | `X32P · X32 · X32` | SME | ✅ | 🟣 PR available<br>[PR #29441](https://github.com/microsoft/onnxruntime/pull/29441)<br>[PR #26654](https://github.com/microsoft/onnxruntime/pull/26654) |

</details>

### Depthwise convolution + clamp (2)

<details>
<summary>Show 2 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.c "kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla") | `F16 · F16 · F16P` | SME2 · `4x4` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.c "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla") | `F32 · F32 · F32P` | SME2 | ✅ | 🟣 PR available<br>[PR #29441](https://github.com/microsoft/onnxruntime/pull/29441)<br>[PR #26654](https://github.com/microsoft/onnxruntime/pull/26654) |

</details>

### GEMM (1)

<details>
<summary>Show 1 micro-kernel</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_i32_u8p_u8p/kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa.c "kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa") | `I32 · U8P · U8P` | SME2 · `8vsx8vs` | ❌ | 🟣 PR available<br>[PR #28745](https://github.com/microsoft/onnxruntime/pull/28745) |

</details>

### GEMM + clamp (32)

<details>
<summary>Show 32 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot.c "kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot") | `F16 · F16 · F16P` | SVE2P1 · `6x16vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa") | `F16 · F16P · F16P` | SME2 · `2vlx2vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.c "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa") | `F16 · F16P · F16P` | SME · `2vlx2vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa.c "kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa") | `F16 · F16P · F16P` | SME2 · `8vsx8vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f16_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa") | `F16 · QAI8DXP · QSI8CXP` | SME2 · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa") | `F16 · QAI8DXP · QSI4CXP` | SME2 · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa") | `F16 · QSI8D32P · QAI4C32P` | SME2 · `1vlx4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.c "kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa") | `F32 · BF16P · BF16P` | SME2 · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:306](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L306 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:306") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa.c "kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme_mopa") | `F32 · BF16P · BF16P` | SME · `2vlx2vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa") | `F32 · F16P · QSI4C32P` | SME2 · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f16p_qai4c32p/kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa.c "kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa") | `F32 · F16P · QAI4C32P` | SME2 · `4vsx16vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla.c "kai_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla") | `F32 · F32 · F32P` | SVE · `6x4vl` | ✅ | 🟣 PR available<br>[PR #31143](https://github.com/microsoft/onnxruntime/pull/31143)<br>[PR #27643](https://github.com/microsoft/onnxruntime/pull/27643) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.c "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa") | `F32 · F32P · F32P` | SME · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:290](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L290 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:290") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa") | `F32 · F32P · F32P` | SME2 | ✅ | ✅ Integrated<br>[code:314](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L314 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:314") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa.c "kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa") | `F32 · F32P · F32P` | SME2 · `8vsx8vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa") | `F32 · QAI8DXP · QSI4C32P` | SME2 · `1vlx4vl` | ✅ | ✅ Integrated<br>[code:257](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L257 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:257")<br>[code:258](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L258 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:258") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa") | `F32 · QAI8DXP · QSI4CXP` | SME · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa") | `F32 · QAI8DXP · QSI8CXP` | SME2 · `1vlx4vl` | ✅ | ✅ Integrated<br>[code:320](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L320 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:320") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa") | `F32 · QAI8DXP · QSI8CXP` | SME · `1vlx4vl` | ✅ | ✅ Integrated<br>[code:317](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L317 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:317") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa") | `F32 · QAI8DXP · QSU2CXP` | SME2 · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa") | `F32 · QAI8DXP · QSI4CXP` | SME2 · `1vlx4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa") | `F32 · QSI8D32P · QAI4C32P` | SME2 · `1vlx4vl` | ✅ | ✅ Integrated<br>[code:281](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L281 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:281")<br>[code:282](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L282 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:282") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa") | `F32 · QSI8D32P · QSI4C32P` | SME2 · `1vlx4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa") | `F32 · QSI8D32P · QSI4C32P` | SME · `1vlx4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm") | `F32 · QSI8D32P · QSI4C32P` | SVE · `16x8` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_u8p_u8p/kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa.c "kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa") | `F32 · U8P · U8P` | SME2 · `8vsx8vs` | ❌ | 🟣 PR available<br>[PR #28745](https://github.com/microsoft/onnxruntime/pull/28745) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.c "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME · `2vlx2vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.c "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME2 · `2vlx2vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa.c "kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME2 · `8vsx8vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa.c "kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME2P1 · `8vsx8vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi4cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa.c "kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa") | `QAI8 · QAI8P · QSI4CXP` | SME2 · `8vsx8vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsu2cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa.c "kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa") | `QAI8 · QAI8P · QSU2CXP` | SME2 · `8vsx16vs` | ❌ | — Not integrated |

</details>

### GEMV + clamp (26)

<details>
<summary>Show 26 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.c "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot") | `F16 · F16 · F16P` | SME2 · `1x16vl` | ✅ | ✅ Integrated<br>[code:326](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L326 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:326") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.c "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla") | `F16 · F16 · F16P` | SME · `1x8vl` | ✅ | ✅ Integrated<br>[code:323](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L323 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:323") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot.c "kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot") | `F16 · F16 · F16P` | SME2 · `1x32vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.c "kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot") | `F16 · QAI8DXP · QSI4CXP` | SME2 · `1x4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot") | `F16 · QAI8DXP · QSI8CXP` | SME2 · `1x4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot") | `F16 · QSI8D32P · QAI4C32P` | SME2 · `1x4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.c "kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla") | `F32 · F32 · F32P` | SME2 · `1x16vl` | ✅ | 🔗 Referenced only<br>[code:48](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L48 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:48") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.c "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla") | `F32 · F32 · F32P` | SME2 · `1x16vl` | ✅ | ✅ Integrated<br>[code:357](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L357 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:357")<br>[code:358](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L358 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:358") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.c "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla") | `F32 · F32 · F32P` | SME · `1x8vl` | ✅ | ✅ Integrated<br>[code:342](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L342 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:342")<br>[code:343](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L343 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:343") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla.c "kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla") | `F32 · F32 · F32P` | SME2 · `1x32vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot") | `F32 · QAI8DXP · QSI4C32P` | SME2 · `1x4vl` | ✅ | ✅ Integrated<br>[code:261](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L261 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:261")<br>[code:262](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L262 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:262") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot") | `F32 · QAI8DXP · QSI4CXP` | SME2 · `1x4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot") | `F32 · QAI8DXP · QSI4CXP` | SME · `1x4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot") | `F32 · QAI8DXP · QSI8CXP` | SME2 · `1x4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot") | `F32 · QAI8DXP · QSI8CXP` | SME · `1x4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot") | `F32 · QAI8DXP · QSU2CXP` | SME2 · `1x4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot.c "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot") | `F32 · QSI8D32P · QAI4C32P` | SME2 · `1x16vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.c "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot") | `F32 · QSI8D32P · QAI4C32P` | SME2 · `1x4vl` | ✅ | ✅ Integrated<br>[code:285](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L285 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:285")<br>[code:286](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L286 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:286") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot") | `F32 · QSI8D32P · QSI4C32P` | SME2 · `1x4vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot.c "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot") | `F32 · QSI8D32P · QSI4C32P` | SME · `1x4vl` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.c "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod") | `F32 · QSI8D32P · QSI4C32P` | SVE · `1x8` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod") | `F32 · QSI8D32P · QSI4C32P` | SVE · `1x8` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi4cxp/kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot.c "kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot") | `QAI8 · QAI8 · QSI4CXP` | SME2 · `1x64vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.c "kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot") | `QAI8 · QAI8 · QSI8CXP` | SME2 · `1x16vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot.c "kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot") | `QAI8 · QAI8 · QSI8CXP` | SME2 · `1x32vs` | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsu2cxp/kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot.c "kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot") | `QAI8 · QAI8 · QSU2CXP` | SME2 · `1x64vs` | ❌ | — Not integrated |

</details>

### Indirect GEMM + clamp (7)

<details>
<summary>Show 7 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.c "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa") | `F16 · F16P · F16P` | SME2 · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:303](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L303 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:303") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.c "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa") | `F16 · F16P · F16P` | SME · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:300](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L300 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:300") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_f32_f32_f32p/kai_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla.c "kai_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla") | `F32 · F32 · F32P` | SVE · `6x4vl` | ❌ | 🟣 PR available<br>[PR #27643](https://github.com/microsoft/onnxruntime/pull/27643) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.c "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa") | `F32 · F32P · F32P` | SME2 · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:297](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L297 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:297") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.c "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa") | `F32 · F32P · F32P` | SME · `2vlx2vl` | ✅ | ✅ Integrated<br>[code:294](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp#L294 "onnxruntime/core/mlas/lib/kleidiai/kai_ukernel_interface.cpp:294") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.c "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME · `2vlx2vl` | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.c "kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa") | `QAI8 · QAI8P · QSI8CXP` | SME2 · `2vlx2vl` | ✅ | — Not integrated |

</details>

### LHS pack (10)

<details>
<summary>Show 10 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.c "kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme") | `X16P · X16` | SME | ✅ | ✅ Integrated<br>[code:676](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp#L676 "onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp:676")<br>[code:693](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp#L693 "onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp:693") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.c "kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme") | `X32P · X32` | SME | ✅ | ✅ Integrated<br>[code:157](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp#L157 "onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp:157")<br>[code:522](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp#L522 "onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp:522") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.c "kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme") | `X8P · X8` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c "kai_lhs_pack_bf16p2vlx2_f32_sme") | `BF16P · F32` | SME | ✅ | ✅ Integrated<br>[code:279](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp#L279 "onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp:279")<br>[code:300](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp#L300 "onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp:300") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme2.c "kai_lhs_pack_bf16p2vlx2_f32_sme2") | `BF16P · F32` | SME2 | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c "kai_lhs_pack_f32p2vlx1_f32_sme") | `F32P · F32` | SME | ✅ | ✅ Integrated<br>[code:542](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L542 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:542")<br>[code:563](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L563 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:563") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.c "kai_lhs_pack_x16p2vlx2_x16_sme") | `X16P · X16` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.c "kai_lhs_pack_x8p2vlx4_x8_sme") | `X8P · X8` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c "kai_lhs_quant_pack_qai8dxp_f32") | `QAI8DXP · F32` | Scalar / generic | ✅ | ✅ Integrated<br>[code:328](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp#L328 "onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp:328")<br>[code:427](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp#L427 "onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp:427") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c "kai_lhs_quant_pack_qsi8d32p_f32") | `QSI8D32P · F32` | Scalar / generic | ✅ | — Not integrated |

</details>

### Packing (11)

<details>
<summary>Show 11 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme.c "kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme") | `MATMUL · PACK · LHS · MXK · X16P4VSX2 · X16` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme.c "kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme") | `MATMUL · PACK · LHS · MXK · X32P4VSX1 · X32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme.c "kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme") | `MATMUL · PACK · LHS · MXK · X8P4VSX4 · X8` | SME | ❌ | 🟣 PR available<br>[PR #28745](https://github.com/microsoft/onnxruntime/pull/28745) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme.c "kai_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme") | `MATMUL · PACK · RHS · KXN · QSU2CXP16VSX4SF32BI32 · QSU2CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme.c "kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme") | `MATMUL · PACK · RHS · KXN · X16P4VSX2BX16 · X16 · X16` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme.c "kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme") | `MATMUL · PACK · RHS · KXN · X32P4VSX1BX32 · X32 · X32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme.c "kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme") | `MATMUL · PACK · RHS · KXN · X8P4VSX4 · X8` | SME | ❌ | 🟣 PR available<br>[PR #28745](https://github.com/microsoft/onnxruntime/pull/28745) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme.c "kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme") | `MATMUL · PACK · RHS · NXK · QAI4C32P16VSX4S1S0SF16 · QAI4C32K256SF16S32S0` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme.c "kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme") | `MATMUL · PACK · RHS · NXK · QSU2CXP16VSX4SF32BI32 · QSU2CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme.c "kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme") | `MATMUL · PACK · RHS · NXK · X32P4VSX1BX32 · X32 · X32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme.c "kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme") | `MATMUL · PACK · RHS · NXK · X8P4VSX4 · X8` | SME | ❌ | — Not integrated |

</details>

### RHS pack K×N (17)

<details>
<summary>Show 17 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme.c "kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme") | `QSI8CXP · QSI8 · I32 · F32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c "kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme") | `QSI8CXP · QSI8CX · F32 · I32` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.c "kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme") | `X16P · X16 · X16` | SME | ✅ | ✅ Integrated<br>[code:272](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp#L272 "onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp:272")<br>[code:513](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp#L513 "onnxruntime/core/mlas/lib/kleidiai/halfconv_kleidiai.cpp:513") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.c "kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme") | `X32P · X32 · X32` | SME | ✅ | ✅ Integrated<br>[code:308](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp#L308 "onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp:308")<br>[code:351](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp#L351 "onnxruntime/core/mlas/lib/kleidiai/convolve_kleidiai.cpp:351") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve.c "kai_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve") | `X32P · X32 · X32` | SVE | ❌ | 🟣 PR available<br>[PR #27643](https://github.com/microsoft/onnxruntime/pull/27643) |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c "kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme") | `BF16P · F32 · X32` | SME | ✅ | ✅ Integrated<br>[code:146](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp#L146 "onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp:146")<br>[code:206](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp#L206 "onnxruntime/core/mlas/lib/kleidiai/sbgemm_kleidiai.cpp:206") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme.c "kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme") | `F32P · F32 · F32` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c "kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme") | `F32P · F32 · F32` | SME | ✅ | ✅ Integrated<br>[code:351](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L351 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:351")<br>[code:422](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L422 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:422") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.c "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0") | `QSI4C32P · QSU4C32` | Scalar / generic | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme.c "kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme") | `QSI4CXP · QSI4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsu4cx_f32_i32_sme.c "kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsu4cx_f32_i32_sme") | `QSI4CXP · QSU4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme.c "kai_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme") | `QSI4CXP · QSX4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.c "kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0") | `QSI4CXP · QSI4CX` | Scalar / generic | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c "kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme") | `QSI8CXP · QSI8CX · F32 · I32` | SME | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve.c "kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve") | `X16P · X16 · X16` | SVE | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.c "kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme") | `X16P · X16 · X16` | SME | ✅ | ✅ Integrated<br>[code:77](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfgemm_kleidiai.cpp#L77 "onnxruntime/core/mlas/lib/kleidiai/halfgemm_kleidiai.cpp:77")<br>[code:112](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/halfgemm_kleidiai.cpp#L112 "onnxruntime/core/mlas/lib/kleidiai/halfgemm_kleidiai.cpp:112") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve.c "kai_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve") | `X32P · X32 · X32` | SVE | ✅ | 🟣 PR available<br>[PR #31143](https://github.com/microsoft/onnxruntime/pull/31143)<br>[PR #27643](https://github.com/microsoft/onnxruntime/pull/27643) |

</details>

### RHS pack N×K (8)

<details>
<summary>Show 8 micro-kernels</summary>

| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |
| :---: | --- | --- | :---: | --- |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.c "kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme") | `F32P · F32 · F32` | SME | ✅ | ✅ Integrated<br>[code:354](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L354 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:354")<br>[code:427](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp#L427 "onnxruntime/core/mlas/lib/kleidiai/sgemm_kleidiai.cpp:427") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0") | `QSI4C32P · QSU4C32` | Scalar / generic | ✅ | ✅ Integrated<br>[code:110](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp#L110 "onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp:110")<br>[code:206](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp#L206 "onnxruntime/core/mlas/lib/kleidiai/qnbitgemm_kleidiai.cpp:206") |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0") | `QSI4C32P · QSU4C32` | Scalar / generic | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme.c "kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme") | `QSI4CXP · QSI4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsu4cx_f32_i32_sme.c "kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsu4cx_f32_i32_sme") | `QSI4CXP · QSU4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme.c "kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme") | `QSI4CXP · QSX4CX · F32 · I32` | SME | ❌ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0") | `QSI4CXP · QSI4CX` | Scalar / generic | ✅ | — Not integrated |
| [source](https://github.com/ARM-software/kleidiai/blob/v1.31.0/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.c "kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme") | `X16P · X16 · X16` | SME | ✅ | — Not integrated |

</details>
