/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    sbgemm_kernel_neon.cpp

Abstract:

    This module implements bfloat16 precision GEMM kernel for neon.

--*/

#if defined(__aarch64__) && defined(__linux__)

#include "arm_neon.h"
#include "mlasi.h"
#include "sbgemm.h"

struct MLAS_SBGEMM_KERNEL_NEON {
    static constexpr bool PackNeeded = true;
    static constexpr size_t KernelMaxM = 8;  // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 4;
    static constexpr size_t PackedN = MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    static constexpr MLAS_SBGEMM_STRIDES Strides{128, 128, 256};  // M:N:K
};

bool MLASCALL
MlasBf16AccelerationSupported()
{
#if defined(MLAS_TARGET_ARM64)
    return MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_BF16();
#else
    return false;
#endif
}

/*
    This routine converts fp32 to bf16 and copies elements from the source
     matrix to the destination packed buffer.

    4x2 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SBGEMM kernels. The remaining
    rows and columns are padded to 4 and 2 alignment.
*/
MLAS_FORCEINLINE
void
MlasSBGemmConvertCopyPackB(bfloat16_t* D, const float* B, size_t ldb, size_t CountN, size_t CountK)
{
    //
    // Copy data from matrix B into the destination buffer 4x2 blocks at a
    // time.
    //
    //
    while (CountN >= 8) {
        const float* b = B;
        int y = static_cast<int>(CountK);

        while (y > 0) {
            MLAS_FLOAT32X4 t0_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t0_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t1_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t1_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t2_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t2_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t3_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t3_h = MlasZeroFloat32x4();

            if (y >= 4) {
                t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                t2_l = MlasLoadFloat32x4(&b[ldb * 2]);
                t2_h = MlasLoadFloat32x4(&b[ldb * 2 + 4]);
                t3_l = MlasLoadFloat32x4(&b[ldb * 3]);
                t3_h = MlasLoadFloat32x4(&b[ldb * 3 + 4]);
            } else {
                switch (y) {
                    case 3:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                        t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                        t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                        t2_l = MlasLoadFloat32x4(&b[ldb * 2]);
                        t2_h = MlasLoadFloat32x4(&b[ldb * 2 + 4]);
                        break;
                    case 2:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                        t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                        t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                        break;
                    case 1:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                        break;
                }
            }

            float32x4x2_t z0_l = vzipq_f32(t0_l, t2_l);
            float32x4x2_t z1_l = vzipq_f32(t1_l, t3_l);
            float32x4x2_t o0_l = vzipq_f32(z0_l.val[0], z1_l.val[0]);
            float32x4x2_t o1_l = vzipq_f32(z0_l.val[1], z1_l.val[1]);
            t0_l = o0_l.val[0];
            t1_l = o0_l.val[1];
            t2_l = o1_l.val[0];
            t3_l = o1_l.val[1];

            bfloat16x8_t t0t1_l_4h = vcvtq_low_bf16_f32(t0_l);
            bfloat16x8_t t0t1_l_8h = vcvtq_high_bf16_f32(t0t1_l_4h, t1_l);

            bfloat16x8_t t2t3_l_4h = vcvtq_low_bf16_f32(t2_l);
            bfloat16x8_t t2t3_l_8h = vcvtq_high_bf16_f32(t2t3_l_4h, t3_l);

            vst1q_bf16(&D[0], t0t1_l_8h);
            vst1q_bf16(&D[8], t2t3_l_8h);

            float32x4x2_t z0_h = vzipq_f32(t0_h, t2_h);
            float32x4x2_t z1_h = vzipq_f32(t1_h, t3_h);
            float32x4x2_t o0_h = vzipq_f32(z0_h.val[0], z1_h.val[0]);
            float32x4x2_t o1_h = vzipq_f32(z0_h.val[1], z1_h.val[1]);
            t0_h = o0_h.val[0];
            t1_h = o0_h.val[1];
            t2_h = o1_h.val[0];
            t3_h = o1_h.val[1];

            bfloat16x8_t t0t1_h_4h = vcvtq_low_bf16_f32(t0_h);
            bfloat16x8_t t0t1_h_8h = vcvtq_high_bf16_f32(t0t1_h_4h, t1_h);

            bfloat16x8_t t2t3_h_4h = vcvtq_low_bf16_f32(t2_h);
            bfloat16x8_t t2t3_h_8h = vcvtq_high_bf16_f32(t2t3_h_4h, t3_h);

            vst1q_bf16(&D[16], t0t1_h_8h);
            vst1q_bf16(&D[24], t2t3_h_8h);

            D += 32;
            b += ldb * 4;
            y -= 4;
        };
        B += 8;
        CountN -= 8;
    }

    //
    // Special case the handling of the remaining columns less than 8 elements
    // wide.
    //
    if (CountN > 0) {
        int y = static_cast<int>(CountK);
        while (y > 0) {
            const float* b = B;
            size_t b_inc = 0;
            if ((CountN & 4) != 0) {
                MLAS_FLOAT32X4 t0 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t1 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t2 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t3 = MlasZeroFloat32x4();
                if (y >= 4) {
                    t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                    t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                    t2 = MlasLoadFloat32x4(&b[ldb * 2]);
                    t3 = MlasLoadFloat32x4(&b[ldb * 3]);
                } else {
                    switch (y) {
                        case 3:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                            t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                            t2 = MlasLoadFloat32x4(&b[ldb * 2]);
                            break;
                        case 2:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                            t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                            break;
                        case 1:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                            break;
                    }
                }

                float32x4x2_t z0 = vzipq_f32(t0, t2);
                float32x4x2_t z1 = vzipq_f32(t1, t3);
                float32x4x2_t o0 = vzipq_f32(z0.val[0], z1.val[0]);
                float32x4x2_t o1 = vzipq_f32(z0.val[1], z1.val[1]);

                t0 = o0.val[0];
                t1 = o0.val[1];
                t2 = o1.val[0];
                t3 = o1.val[1];

                bfloat16x8_t t0t1_4h = vcvtq_low_bf16_f32(t0);
                bfloat16x8_t t0t1_8h = vcvtq_high_bf16_f32(t0t1_4h, t1);

                bfloat16x8_t t2t3_4h = vcvtq_low_bf16_f32(t2);
                bfloat16x8_t t2t3_8h = vcvtq_high_bf16_f32(t2t3_4h, t3);

                vst1q_bf16(&D[0], t0t1_8h);
                vst1q_bf16(&D[8], t2t3_8h);

                D += 16;
                b += 4;
                b_inc += 4;
            }

            if ((CountN & 2) != 0) {
                float32x2_t t0 = {0x0, 0x0};
                float32x2_t t1 = {0x0, 0x0};
                float32x2_t t2 = {0x0, 0x0};
                float32x2_t t3 = {0x0, 0x0};

                if (y >= 4) {
                    t0 = vld1_f32(&b[ldb * 0]);
                    t1 = vld1_f32(&b[ldb * 1]);
                    t2 = vld1_f32(&b[ldb * 2]);
                    t3 = vld1_f32(&b[ldb * 3]);
                } else {
                    switch (y) {
                        case 3:
                            t0 = vld1_f32(&b[ldb * 0]);
                            t1 = vld1_f32(&b[ldb * 1]);
                            t2 = vld1_f32(&b[ldb * 2]);
                            break;
                        case 2:
                            t0 = vld1_f32(&b[ldb * 0]);
                            t1 = vld1_f32(&b[ldb * 1]);
                            break;
                        case 1:
                            t0 = vld1_f32(&b[ldb * 0]);
                            break;
                    }
                }

                float32x2x2_t z0 = vzip_f32(t0, t2);
                float32x2x2_t z1 = vzip_f32(t1, t3);
                float32x2x2_t o0 = vzip_f32(z0.val[0], z1.val[0]);
                float32x2x2_t o1 = vzip_f32(z0.val[1], z1.val[1]);

                float32x4_t tt0 = vcombine_f32(o0.val[0], o0.val[1]);
                float32x4_t tt1 = vcombine_f32(o1.val[0], o1.val[1]);

                bfloat16x8_t t_4h = vcvtq_low_bf16_f32(tt0);
                bfloat16x8_t t_8h = vcvtq_high_bf16_f32(t_4h, tt1);

                vst1q_bf16(&D[0], t_8h);

                D += 8;
                b += 2;
                b_inc += 2;
            }
            if ((CountN & 1) != 0) {
                float a = 0.0f;
                float b = 0.0f;
                float c = 0.0f;
                float d = 0.0f;

                if (y >= 4) {
                    a = *(float*)(&B[ldb * 0 + b_inc]);
                    b = *(float*)(&B[ldb * 1 + b_inc]);
                    c = *(float*)(&B[ldb * 2 + b_inc]);
                    d = *(float*)(&B[ldb * 3 + b_inc]);
                } else {
                    switch (y) {
                        case 3:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                            b = *(float*)(&B[ldb * 1 + b_inc]);
                            c = *(float*)(&B[ldb * 2 + b_inc]);
                            break;
                        case 2:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                            b = *(float*)(&B[ldb * 1 + b_inc]);
                            break;
                        case 1:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                            break;
                    }
                }

                float32x2_t t0 = {a, 0x0};
                float32x2_t t1 = {b, 0x0};
                float32x2_t t2 = {c, 0x0};
                float32x2_t t3 = {d, 0x0};

                float32x2x2_t z0 = vzip_f32(t0, t2);
                float32x2x2_t z1 = vzip_f32(t1, t3);
                float32x2x2_t o0 = vzip_f32(z0.val[0], z1.val[0]);
                float32x2x2_t o1 = vzip_f32(z0.val[1], z1.val[1]);

                float32x4_t tt0 = vcombine_f32(o0.val[0], o0.val[1]);
                float32x4_t tt1 = vcombine_f32(o1.val[0], o1.val[1]);

                bfloat16x8_t t_4h = vcvtq_low_bf16_f32(tt0);
                bfloat16x8_t t_8h = vcvtq_high_bf16_f32(t_4h, tt1);

                vst1q_bf16(&D[0], t_8h);

                D += 8;
                b += 1;
                b_inc += 1;
            }
            B += 4 * ldb;
            y -= 4;
        }
    }
}

template <typename KernelType>
void
MlasSBGemmConvertPackB(
    bfloat16_t* PackedB, const float* B, size_t ldb, size_t CountN, size_t CountK)
{
    const auto* dispatch = MlasSBGemmGetDispatch();
    if (dispatch == nullptr) return;

    const auto PackedN = dispatch->PackedN;

    const size_t AlignedN = (CountN + PackedN - 1) & ~(PackedN - 1);

    //
    // Step through each slice of matrix B along the K dimension.
    //
    size_t K_block_size;
    constexpr MLAS_SBGEMM_STRIDES Strides = KernelType::Strides;

    for (size_t k = 0; k < CountK; k += K_block_size) {
        K_block_size = std::min(CountK - k, Strides.K);

        MlasSBGemmConvertCopyPackB((bfloat16_t*)PackedB, B + k * ldb, ldb, CountN, K_block_size);
        PackedB = (bfloat16_t*)PackedB + AlignedN * K_block_size;
    }
}

template <>
MLAS_FORCEINLINE void
MlasSBGemmKernel<MLAS_SBGEMM_KERNEL_NEON>(size_t CountM,
                                          size_t CountN,
                                          size_t CountK,
                                          const float* A,
                                          size_t lda,
                                          const bfloat16_t* B,
                                          float* C,
                                          size_t ldc,
                                          const float* Bias,
                                          const bool ZeroMode)
{
    while (CountM > 0) {
        size_t RowsHandled;
        if (ZeroMode) {
            RowsHandled = MlasSbgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, Bias);
        } else {
            RowsHandled = MlasSbgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, Bias);
        }
        C += ldc * RowsHandled;
        A += lda * RowsHandled;
        CountM -= RowsHandled;
    }
}

const MLAS_SBGEMM_DISPATCH MlasSBGemmDispatchNeon = {
    MlasSBGemmOperation<MLAS_SBGEMM_KERNEL_NEON>,
    MlasSBGemmConvertPackB<MLAS_SBGEMM_KERNEL_NEON>,
    MLAS_SBGEMM_KERNEL_NEON::PackedK,
    MLAS_SBGEMM_KERNEL_NEON::PackedN,
    MLAS_SBGEMM_KERNEL_NEON::KernelMaxM,
    32  // kernel may read beyond buffer end by 32 bytes
};
#endif //defined(__aarch64__) && defined(__linux__)
