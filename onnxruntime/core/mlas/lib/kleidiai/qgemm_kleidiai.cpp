//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"

namespace ArmKleidiAI {
namespace {

// Thread-local reusable buffers to reduce allocation overhead across tiles.
struct KaiTlsBuffersQgemm {
    std::vector<std::byte> lhs_packed;
    std::vector<const std::byte*> lhs_base_table;
};
static thread_local KaiTlsBuffersQgemm g_kai_tls_qgemm;

const KaiDynamicQGemmKernel qgemm_gemm = GetKleidiAIQGemmUKernel();
constexpr int32_t kPackedBAccColBiasInvalidZeroPointA = -1;

constexpr size_t kStaticQgemmMTileByteLimit = 128 * 1024;
constexpr size_t kDynamicQgemmNTileByteLimit = 128 * 1024;

static size_t
CapStepByPackedBytes(size_t step, size_t base_step, size_t base_packed_bytes, size_t byte_limit) {
    if (byte_limit == 0 || base_step == 0 || base_packed_bytes == 0 || step <= base_step) {
        return step;
    }

    const size_t max_blocks = std::max<size_t>(size_t{1}, byte_limit / base_packed_bytes);

    size_t capped_step = 0;
    if (mul_overflow_size_t_builtin(base_step, max_blocks, &capped_step)) {
        return step;
    }

    return std::min(step, capped_step);
}

static size_t
CapDynamicQgemmNStepByPackedRhs(size_t n_step, size_t base_n_step, size_t k, size_t byte_limit) {
    return CapStepByPackedBytes(
        n_step, base_n_step, qgemm_gemm.ukernel.get_rhs_packed_offset(base_n_step, k), byte_limit);
}

struct KaiTlsBuffersQgemmInteger {
    std::vector<uint8_t> lhs_unsigned;
    std::vector<uint8_t> rhs_unsigned;
    std::vector<std::byte> lhs_packed;
    std::vector<std::byte> rhs_packed;
    std::vector<int32_t> acc_row_bias;
    std::vector<int32_t> acc_col_bias;
    std::vector<int32_t> zero_acc_row_bias;
    std::vector<int32_t> zero_acc_col_bias;
    std::vector<float> zero_col_bias;
};
static thread_local KaiTlsBuffersQgemmInteger g_kai_tls_qgemm_int;

const kai_matmul_pack_lhs_uker_api qgemm_int_lhs_pack = kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
const kai_matmul_pack_rhs_uker_api qgemm_int_rhs_pack = kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
const kai_matmul_uker_api qgemm_int_i32 = kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa();
const kai_matmul_uker_api qgemm_int_f32 = kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa();

static int32_t
AdjustedZeroPoint(uint8_t zero_point, bool is_signed) {
    return is_signed ? static_cast<int32_t>(zero_point ^ 0x80) : static_cast<int32_t>(zero_point);
}

template <typename T>
static void
EnsureVectorSize(std::vector<T>& buffer, size_t size) {
    if (buffer.size() < size) {
        buffer.resize(size);
    }
}

template <typename T>
static T*
EnsureZeroVectorSize(std::vector<T>& buffer, size_t size) {
    if (buffer.size() < size) {
        buffer.resize(size, T{});
    }
    return buffer.data();
}

static bool
CheckedAddSize(size_t a, size_t b, size_t* out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool
CheckedAlignUp(size_t value, size_t alignment, size_t* out) {
    if (alignment == 0) {
        *out = value;
        return true;
    }

    size_t rounded = 0;
    if (!CheckedAddSize(value, alignment - 1, &rounded)) {
        return false;
    }

    *out = rounded & ~(alignment - 1);
    return true;
}

static bool
PackedBColumnSumsSize(size_t n, size_t* size) {
    size_t bytes = 0;
    if (mul_overflow_size_t_builtin(n, sizeof(int32_t), &bytes)) {
        return false;
    }
    return CheckedAlignUp(bytes, MlasGetPreferredBufferAlignment(), size);
}

static bool
PackedBAccColBiasSize(size_t n, size_t* size) {
    size_t elements = 0;
    if (!CheckedAddSize(n, 1, &elements)) {
        return false;
    }

    size_t bytes = 0;
    if (mul_overflow_size_t_builtin(elements, sizeof(int32_t), &bytes)) {
        return false;
    }

    return CheckedAlignUp(bytes, MlasGetPreferredBufferAlignment(), size);
}

static int32_t*
PackedBColumnSums(void* packed_b) {
    return reinterpret_cast<int32_t*>(packed_b);
}

static const int32_t*
PackedBColumnSums(const void* packed_b) {
    return reinterpret_cast<const int32_t*>(packed_b);
}

static int32_t*
PackedBAccColBias(void* packed_b, size_t n) {
    size_t column_sums_size = 0;
    if (!PackedBColumnSumsSize(n, &column_sums_size)) {
        return nullptr;
    }

    return reinterpret_cast<int32_t*>(static_cast<std::byte*>(packed_b) + column_sums_size);
}

static const int32_t*
PackedBAccColBias(const void* packed_b, size_t n) {
    size_t column_sums_size = 0;
    if (!PackedBColumnSumsSize(n, &column_sums_size)) {
        return nullptr;
    }

    return reinterpret_cast<const int32_t*>(static_cast<const std::byte*>(packed_b) + column_sums_size);
}

static const int32_t*
PackedBAccColBiasData(const void* packed_b, size_t n, int32_t zero_point_a) {
    const int32_t* acc_col_bias = PackedBAccColBias(packed_b, n);
    if (acc_col_bias == nullptr || acc_col_bias[0] != zero_point_a) {
        return nullptr;
    }

    return acc_col_bias + 1;
}

static void*
PackedBData(void* packed_b, size_t n) {
    size_t column_sums_size = 0;
    if (!PackedBColumnSumsSize(n, &column_sums_size)) {
        return nullptr;
    }

    size_t acc_col_bias_size = 0;
    if (!PackedBAccColBiasSize(n, &acc_col_bias_size)) {
        return nullptr;
    }

    return static_cast<std::byte*>(packed_b) + column_sums_size + acc_col_bias_size;
}

static const void*
PackedBData(const void* packed_b, size_t n) {
    size_t column_sums_size = 0;
    if (!PackedBColumnSumsSize(n, &column_sums_size)) {
        return nullptr;
    }

    size_t acc_col_bias_size = 0;
    if (!PackedBAccColBiasSize(n, &acc_col_bias_size)) {
        return nullptr;
    }

    return static_cast<const std::byte*>(packed_b) + column_sums_size + acc_col_bias_size;
}

template <bool LhsIsSigned, bool RhsIsSigned, bool OutputFloat>
static bool
GemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    const size_t m = Shape.M;
    const size_t n = Shape.N;
    const size_t k = Shape.K;

    if (!UseSME2) {
        return false;
    }
    if (DataParams == nullptr) {
        return false;
    }
    if (m == 0 || n == 0 || k == 0 || BatchN == 0) {
        return true;
    }

    for (size_t i = 0; i < BatchN; ++i) {
        const auto& p = DataParams[i];

        if (p.PerColumnZeroPoints) {
            return false;
        }
        if (p.A == nullptr || p.B == nullptr) {
            return false;
        }

        const MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR* scale_bias_processor = nullptr;
        if constexpr (OutputFloat) {
            if (p.OutputProcessor == nullptr) {
                return false;
            }

            scale_bias_processor = p.OutputProcessor->AsScaleBiasOutputProcessor();
            if (scale_bias_processor == nullptr) {
                return false;
            }
            if (scale_bias_processor->OutputMode() != MLAS_QGEMM_OUTPUT_MODE::ZeroMode) {
                return false;
            }
            if (scale_bias_processor->QuantGranularity() != MLAS_QUANTIZATION_GRANULARITY::PerMatrix) {
                return false;
            }
            if (scale_bias_processor->Scale() == nullptr) {
                return false;
            }
            if (scale_bias_processor->Output() == nullptr) {
                return false;
            }
        } else {
            if (p.C == nullptr) {
                return false;
            }
            if (p.OutputProcessor != nullptr) {
                return false;
            }
        }

        const void* lhs_qdata = static_cast<const void*>(p.A);
        size_t lhs_stride = p.lda * sizeof(uint8_t);
        int32_t zpA = AdjustedZeroPoint(p.ZeroPointA, LhsIsSigned);
        if constexpr (LhsIsSigned) {
            size_t lhs_unsigned_size = 0;
            if (mul_overflow_size_t_builtin(m, k, &lhs_unsigned_size)) {
                return false;
            }
            EnsureVectorSize(g_kai_tls_qgemm_int.lhs_unsigned, lhs_unsigned_size);

            const uint8_t* src = p.A;
            uint8_t* dst = g_kai_tls_qgemm_int.lhs_unsigned.data();
            for (size_t row = 0; row < m; ++row) {
                for (size_t col = 0; col < k; ++col) {
                    dst[col] = src[col] ^ 0x80;
                }
                src += p.lda;
                dst += k;
            }

            lhs_qdata = static_cast<const void*>(g_kai_tls_qgemm_int.lhs_unsigned.data());
            lhs_stride = k * sizeof(uint8_t);
        }

        const bool rhs_is_packed = p.BIsPacked;
        const void* rhs_qdata = static_cast<const void*>(p.B);
        size_t rhs_stride = p.ldb * sizeof(uint8_t);
        const uint8_t zero_point_b = p.ZeroPointB ? p.ZeroPointB[0] : uint8_t{0};
        const int32_t zpB = AdjustedZeroPoint(zero_point_b, RhsIsSigned);
        if constexpr (RhsIsSigned) {
            if (!rhs_is_packed) {
                size_t rhs_unsigned_size = 0;
                if (mul_overflow_size_t_builtin(k, n, &rhs_unsigned_size)) {
                    return false;
                }
                EnsureVectorSize(g_kai_tls_qgemm_int.rhs_unsigned, rhs_unsigned_size);

                const uint8_t* src = static_cast<const uint8_t*>(p.B);
                uint8_t* dst = g_kai_tls_qgemm_int.rhs_unsigned.data();
                for (size_t row = 0; row < k; ++row) {
                    for (size_t col = 0; col < n; ++col) {
                        dst[col] = src[col] ^ 0x80;
                    }
                    src += p.ldb;
                    dst += n;
                }

                rhs_qdata = static_cast<const void*>(g_kai_tls_qgemm_int.rhs_unsigned.data());
                rhs_stride = n * sizeof(uint8_t);
            }
        }

        const void* acc_row_bias = nullptr;
        if (zpB == 0) {
            acc_row_bias = EnsureZeroVectorSize(g_kai_tls_qgemm_int.zero_acc_row_bias, m);
        } else {
            EnsureVectorSize(g_kai_tls_qgemm_int.acc_row_bias, m);
            for (size_t row = 0; row < m; ++row) {
                const uint8_t* lhs_row = static_cast<const uint8_t*>(lhs_qdata) + row * lhs_stride;
                int64_t row_sum = 0;
                for (size_t col = 0; col < k; ++col) {
                    row_sum += lhs_row[col];
                }
                g_kai_tls_qgemm_int.acc_row_bias[row] =
                    static_cast<int32_t>(-static_cast<int64_t>(zpB) * (row_sum - static_cast<int64_t>(k) * zpA));
            }
            acc_row_bias = g_kai_tls_qgemm_int.acc_row_bias.data();
        }

        const void* acc_col_bias = nullptr;
        if (zpA == 0) {
            acc_col_bias = EnsureZeroVectorSize(g_kai_tls_qgemm_int.zero_acc_col_bias, n);
        } else if (rhs_is_packed) {
            const int32_t* packed_acc_col_bias = PackedBAccColBiasData(p.B, n, zpA);
            if (packed_acc_col_bias != nullptr) {
                acc_col_bias = packed_acc_col_bias;
            } else {
                EnsureVectorSize(g_kai_tls_qgemm_int.acc_col_bias, n);
                const int32_t* rhs_col_sums = PackedBColumnSums(p.B);
                for (size_t col = 0; col < n; ++col) {
                    g_kai_tls_qgemm_int.acc_col_bias[col] = static_cast<int32_t>(-static_cast<int64_t>(zpA) * rhs_col_sums[col]);
                }
                acc_col_bias = g_kai_tls_qgemm_int.acc_col_bias.data();
            }
        } else {
            EnsureVectorSize(g_kai_tls_qgemm_int.acc_col_bias, n);
            for (size_t col = 0; col < n; ++col) {
                const uint8_t* rhs_col = static_cast<const uint8_t*>(rhs_qdata) + col;
                int64_t col_sum = 0;
                for (size_t row = 0; row < k; ++row) {
                    col_sum += rhs_col[row * rhs_stride];
                }
                g_kai_tls_qgemm_int.acc_col_bias[col] = static_cast<int32_t>(-static_cast<int64_t>(zpA) * col_sum);
            }
            acc_col_bias = g_kai_tls_qgemm_int.acc_col_bias.data();
        }

        const kai_matmul_pack_lhs_uker_config lhs_pack_config{};
        const auto lhs_pack_step = qgemm_int_lhs_pack.get_step(&lhs_pack_config);
        const size_t lhs_base_m_step = lhs_pack_step.m == 0 ? m : lhs_pack_step.m;
        const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape{lhs_base_m_step, k};
        const auto lhs_packed_stride =
            qgemm_int_lhs_pack.get_lhs_packed_stride(&lhs_pack_config, &lhs_packed_shape);
        const size_t m_tile_step =
            CapStepByPackedBytes(m, lhs_base_m_step, lhs_packed_stride.m, kStaticQgemmMTileByteLimit);

        const kai_matmul_pack_rhs_uker_config rhs_pack_config{};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape{n, k};
        const auto rhs_packed_stride =
            qgemm_int_rhs_pack.get_rhs_packed_stride(&rhs_pack_config, &rhs_packed_shape);
        const void* rhs_packed_data = nullptr;

        if (rhs_is_packed) {
            rhs_packed_data = PackedBData(p.B, n);
            if (rhs_packed_data == nullptr) {
                return false;
            }
        } else {
            const size_t rhs_packed_size =
                qgemm_int_rhs_pack.get_rhs_packed_size(&rhs_pack_config, &rhs_packed_shape, &rhs_packed_stride);
            EnsureVectorSize(g_kai_tls_qgemm_int.rhs_packed, rhs_packed_size);

            kai_matmul_pack_rhs_uker_args rhs_pack_args{};
            rhs_pack_args.shape.n = n;
            rhs_pack_args.shape.k = k;
            rhs_pack_args.operand.rhs.ptr = rhs_qdata;
            rhs_pack_args.operand.rhs.stride.n = sizeof(uint8_t);
            rhs_pack_args.operand.rhs.stride.k = rhs_stride;
            rhs_pack_args.operand.rhs_packed.ptr = g_kai_tls_qgemm_int.rhs_packed.data();
            rhs_pack_args.operand.rhs_packed.stride = rhs_packed_stride;
            rhs_pack_args.operand.bias_n.ptr = nullptr;
            KLEIDIAI_KERNEL_LOG("kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme"
                                << " Batch=" << i << " N=" << n << " K=" << k);
            qgemm_int_rhs_pack.run(&rhs_pack_config, &rhs_pack_args);
            rhs_packed_data = g_kai_tls_qgemm_int.rhs_packed.data();
        }

        const void* scale = nullptr;
        const void* scale_bias_n = nullptr;
        void* dst = nullptr;
        size_t dst_stride = 0;
        float clamp_min = std::numeric_limits<float>::lowest();
        float clamp_max = std::numeric_limits<float>::max();
        if constexpr (OutputFloat) {
            scale = scale_bias_processor->Scale();
            if (scale_bias_processor->Bias() != nullptr) {
                scale_bias_n = scale_bias_processor->Bias();
            } else {
                scale_bias_n = EnsureZeroVectorSize(g_kai_tls_qgemm_int.zero_col_bias, n);
            }
            dst = static_cast<void*>(scale_bias_processor->Output());
            if (mul_overflow_size_t_builtin(scale_bias_processor->LeadingDimensionOutput(), sizeof(float), &dst_stride)) {
                return false;
            }
        } else {
            dst = static_cast<void*>(p.C);
            if (mul_overflow_size_t_builtin(p.ldc, sizeof(int32_t), &dst_stride)) {
                return false;
            }
        }

        const kai_matmul_uker_config matmul_config{};
        const kai_matmul_uker_api& matmul = OutputFloat ? qgemm_int_f32 : qgemm_int_i32;

        for (size_t m_start = 0; m_start < m; m_start += m_tile_step) {
            const size_t m_tile = std::min(m - m_start, m_tile_step);
            const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_tile_packed_shape{m_tile, k};
            const size_t lhs_packed_size =
                qgemm_int_lhs_pack.get_lhs_packed_size(&lhs_pack_config, &lhs_tile_packed_shape, &lhs_packed_stride);
            EnsureVectorSize(g_kai_tls_qgemm_int.lhs_packed, lhs_packed_size);

            kai_matmul_pack_lhs_uker_args lhs_pack_args{};
            lhs_pack_args.shape.m = m_tile;
            lhs_pack_args.shape.k = k;
            lhs_pack_args.operand.lhs.ptr = static_cast<const uint8_t*>(lhs_qdata) + m_start * lhs_stride;
            lhs_pack_args.operand.lhs.stride.m = lhs_stride;
            lhs_pack_args.operand.lhs_packed.ptr = g_kai_tls_qgemm_int.lhs_packed.data();
            lhs_pack_args.operand.lhs_packed.stride = lhs_packed_stride;
            KLEIDIAI_KERNEL_LOG("kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme"
                                << " Batch=" << i << " M=" << m_tile << " K=" << k);
            qgemm_int_lhs_pack.run(&lhs_pack_config, &lhs_pack_args);

            kai_matmul_uker_args matmul_args{};
            matmul_args.flags = OutputFloat ? KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP : 0;
            matmul_args.shape.m = m_tile;
            matmul_args.shape.n = n;
            matmul_args.shape.k = k;
            matmul_args.operand.lhs.ptr = g_kai_tls_qgemm_int.lhs_packed.data();
            matmul_args.operand.lhs.stride.m = lhs_packed_stride.m;
            matmul_args.operand.rhs.ptr = rhs_packed_data;
            matmul_args.operand.rhs.stride.n = rhs_packed_stride.n;
            matmul_args.operand.dst.ptr = static_cast<std::byte*>(dst) + m_start * dst_stride;
            matmul_args.operand.dst.stride.m = dst_stride;
            matmul_args.operand.bias.acc_bias_m.ptr =
                static_cast<const int32_t*>(acc_row_bias) + m_start;
            matmul_args.operand.bias.acc_bias_n.ptr = acc_col_bias;
            if constexpr (OutputFloat) {
                matmul_args.operand.scale.acc_scale_global.ptr = scale;
                matmul_args.operand.bias.scale_bias_n.ptr = scale_bias_n;
                matmul_args.activation.clamp.min_ptr = &clamp_min;
                matmul_args.activation.clamp.max_ptr = &clamp_max;
            }
            if constexpr (OutputFloat) {
                KLEIDIAI_KERNEL_LOG("kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa"
                                    << " Batch=" << i << " M=" << m_tile << " N=" << n << " K=" << k);
            } else {
                KLEIDIAI_KERNEL_LOG("kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa"
                                    << " Batch=" << i << " M=" << m_tile << " N=" << n << " K=" << k);
            }
            matmul.run(&matmul_config, &matmul_args);
        }
    }

    return true;
}

static bool
GemmBatchU8U8ToI32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<false, false, false>(Shape, DataParams, BatchN);
}

static bool
GemmBatchU8I8ToI32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<false, true, false>(Shape, DataParams, BatchN);
}

static bool
GemmBatchI8U8ToI32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<true, false, false>(Shape, DataParams, BatchN);
}

static bool
GemmBatchI8I8ToI32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<true, true, false>(Shape, DataParams, BatchN);
}

static bool
GemmBatchU8U8ToF32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<false, false, true>(Shape, DataParams, BatchN);
}

static bool
GemmBatchU8I8ToF32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<false, true, true>(Shape, DataParams, BatchN);
}

static bool
GemmBatchI8U8ToF32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<true, false, true>(Shape, DataParams, BatchN);
}

static bool
GemmBatchI8I8ToF32(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    size_t BatchN
) {
    return GemmBatch<true, true, true>(Shape, DataParams, BatchN);
}

} // namespace
} // namespace ArmKleidiAI

bool
MLASCALL
ArmKleidiAI::MlasGemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* /*ThreadPool*/
) {
    // Implement all unpacked {u8, i8} x {u8, i8} -> i32 variants here.
    if (Shape.IsAccumulateMode) {
        return false;
    }
    if (BatchN > 0 && DataParams[0].OutputProcessor != nullptr) {
        if (Shape.AIsSigned) {
            return Shape.BIsSigned
                ? GemmBatchI8I8ToF32(Shape, DataParams, BatchN)
                : GemmBatchI8U8ToF32(Shape, DataParams, BatchN);
        } else {
            return Shape.BIsSigned
                ? GemmBatchU8I8ToF32(Shape, DataParams, BatchN)
                : GemmBatchU8U8ToF32(Shape, DataParams, BatchN);
        }
    }
    if (Shape.AIsSigned) {
        return Shape.BIsSigned
            ? GemmBatchI8I8ToI32(Shape, DataParams, BatchN)
            : GemmBatchI8U8ToI32(Shape, DataParams, BatchN);
    } else {
        return Shape.BIsSigned
            ? GemmBatchU8I8ToI32(Shape, DataParams, BatchN)
            : GemmBatchU8U8ToI32(Shape, DataParams, BatchN);
    }
}

size_t
MLASCALL
ArmKleidiAI::MlasQGemmPackBSize(
    size_t N,
    size_t K,
    bool AIsSigned,
    bool BIsSigned
) {
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    if (!UseSME2 || N == 0 || K == 0) {
        return 0;
    }

    size_t column_sums_size = 0;
    if (!PackedBColumnSumsSize(N, &column_sums_size)) {
        return 0;
    }

    size_t acc_col_bias_size = 0;
    if (!PackedBAccColBiasSize(N, &acc_col_bias_size)) {
        return 0;
    }

    size_t rhs_elements = 0;
    if (mul_overflow_size_t_builtin(N, K, &rhs_elements)) {
        return 0;
    }

    const kai_matmul_pack_rhs_uker_config rhs_pack_config{};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape{N, K};
    const auto rhs_packed_stride =
        qgemm_int_rhs_pack.get_rhs_packed_stride(&rhs_pack_config, &rhs_packed_shape);
    const size_t rhs_packed_size =
        qgemm_int_rhs_pack.get_rhs_packed_size(&rhs_pack_config, &rhs_packed_shape, &rhs_packed_stride);

    size_t total_size = 0;
    if (!CheckedAddSize(column_sums_size, rhs_packed_size, &total_size)) {
        return 0;
    }
    if (!CheckedAddSize(total_size, acc_col_bias_size, &total_size)) {
        return 0;
    }

    return total_size;
}

bool
MLASCALL
ArmKleidiAI::MlasQGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool AIsSigned,
    bool BIsSigned,
    void* PackedB,
    const uint8_t* ZeroPointA
) {
    if (ArmKleidiAI::MlasQGemmPackBSize(N, K, AIsSigned, BIsSigned) == 0 || B == nullptr || PackedB == nullptr) {
        return false;
    }

    int32_t* rhs_col_sums = PackedBColumnSums(PackedB);
    std::fill_n(rhs_col_sums, N, int32_t{0});
    int32_t* cached_acc_col_bias = PackedBAccColBias(PackedB, N);
    if (cached_acc_col_bias == nullptr) {
        return false;
    }
    cached_acc_col_bias[0] = kPackedBAccColBiasInvalidZeroPointA;

    const void* rhs_qdata = static_cast<const void*>(B);
    size_t rhs_stride = ldb * sizeof(uint8_t);
    std::vector<uint8_t> rhs_unsigned;

    if (BIsSigned) {
        size_t rhs_unsigned_size = 0;
        if (mul_overflow_size_t_builtin(N, K, &rhs_unsigned_size)) {
            return false;
        }
        rhs_unsigned.resize(rhs_unsigned_size);

        const uint8_t* src = B;
        uint8_t* dst = rhs_unsigned.data();
        for (size_t row = 0; row < K; ++row) {
            for (size_t col = 0; col < N; ++col) {
                const uint8_t value = src[col] ^ 0x80;
                dst[col] = value;
                rhs_col_sums[col] += value;
            }
            src += ldb;
            dst += N;
        }

        rhs_qdata = static_cast<const void*>(rhs_unsigned.data());
        rhs_stride = N * sizeof(uint8_t);
    } else {
        const uint8_t* src = B;
        for (size_t row = 0; row < K; ++row) {
            for (size_t col = 0; col < N; ++col) {
                rhs_col_sums[col] += src[col];
            }
            src += ldb;
        }
    }

    if (ZeroPointA != nullptr) {
        const int32_t zpA = AdjustedZeroPoint(*ZeroPointA, AIsSigned);
        cached_acc_col_bias[0] = zpA;
        for (size_t col = 0; col < N; ++col) {
            cached_acc_col_bias[col + 1] = static_cast<int32_t>(-static_cast<int64_t>(zpA) * rhs_col_sums[col]);
        }
    }

    const kai_matmul_pack_rhs_uker_config rhs_pack_config{};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape{N, K};
    const auto rhs_packed_stride =
        qgemm_int_rhs_pack.get_rhs_packed_stride(&rhs_pack_config, &rhs_packed_shape);

    kai_matmul_pack_rhs_uker_args rhs_pack_args{};
    rhs_pack_args.shape.n = N;
    rhs_pack_args.shape.k = K;
    rhs_pack_args.operand.rhs.ptr = rhs_qdata;
    rhs_pack_args.operand.rhs.stride.n = sizeof(uint8_t);
    rhs_pack_args.operand.rhs.stride.k = rhs_stride;
    rhs_pack_args.operand.rhs_packed.ptr = PackedBData(PackedB, N);
    rhs_pack_args.operand.rhs_packed.stride = rhs_packed_stride;
    rhs_pack_args.operand.bias_n.ptr = nullptr;

    KLEIDIAI_KERNEL_LOG("kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme"
                        << " N=" << N << " K=" << K << " packed=1");
    qgemm_int_rhs_pack.run(&rhs_pack_config, &rhs_pack_args);

    return true;
}

size_t
MLASCALL
ArmKleidiAI::MlasDynamicQGemmPackBSize(
    size_t N,
    size_t K
) {
    // Degenerate shapes: there is nothing to pack.
    if (N == 0 || K == 0) {
        return 0;
    }

    auto nr = qgemm_gemm.ukernel.get_nr();
    auto kr = qgemm_gemm.ukernel.get_kr();
    auto sr = qgemm_gemm.ukernel.get_sr();

    // Regardless of kernel variant, use the NEON packing variant.
    KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon Groups=1"
                        << " N="<< N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr);
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQGemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
) {
    // Degenerate shapes: nothing to pack. Avoid calling into packers that may not tolerate K==0.
    if (N == 0 || K == 0) {
        return;
    }

    auto nr = qgemm_gemm.ukernel.get_nr();
    auto kr = qgemm_gemm.ukernel.get_kr();
    auto sr = qgemm_gemm.ukernel.get_sr();

    // y - float output
    // scale_factor_lhs - lhs scaling factor
    // scale_factor_rhs - rhs scaling factor
    // lhs_q - lhs quantized (asymmetric, so has zero point)
    // rhs_q - rhs quantized (symmetric so no zero point)
    // lhs_zp - lhs zero point
    // y = (1/(scale_factor_lhs * scale_factor_rhs) * sum( (lhs_q + lhs_zp)*rhs_q )) + bias

    // RHS packing requires lhs_zp because it will perform lhs_zp*rhs_q during RHS packing.
    // Because LHS quantization is hidden from us by LHS quant packing, we don't have a value for lhs_zp.
    // LHS uses dynamic quantization.

    kai_rhs_pack_qsi8cx_params params{
        1,  // lhs_zp - set to 1 so it becomes sum((lhs_q + 1)*rhs_q )),
            // the actual lhs_zp is applied during the matmul
        1.f  // it is not used
    };

    // Regardless of kernel variant, use the NEON packing variant.
    KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon Groups=1"
                        << " N=" << N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr);
    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1, N, K, nr, kr, sr, B,
                                             // N bias values
                                             Bias,
                                             // N scale values
                                             Scales, PackedB, 0, &params);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
) {

    const size_t mr = qgemm_gemm.ukernel.get_mr();
    const size_t kr = qgemm_gemm.ukernel.get_kr();
    const size_t sr = qgemm_gemm.ukernel.get_sr();

    size_t m_step = qgemm_gemm.ukernel.get_m_step();
    size_t n_step = qgemm_gemm.ukernel.get_n_step();
    const size_t base_n_step = n_step;

    if (BatchSize == 0 || Shape.M == 0 || Shape.N == 0 || Shape.K == 0) {
        return;
    }

    // We are required to fail fast when we reach this stage as we will not be able
    // to reverse the packing decision that was made for RHS.

    if (DataParams == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires valid DataParams.");
    }

    for (size_t batch_idx = 0; batch_idx < BatchSize; ++batch_idx) {
        const auto& params = DataParams[batch_idx];

        if (params.A == nullptr) {
            MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires non-null A pointer.");
        }
        if (params.C == nullptr) {
            MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires non-null C pointer.");
        }
        if (params.PackedB == nullptr) {
            MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires non-null PackedB pointer.");
        }

        const size_t lda = params.lda != 0 ? params.lda : Shape.K;
        const size_t ldc = params.ldc != 0 ? params.ldc : Shape.N;

        if (lda < Shape.K) {
            MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires lda >= K.");
        }
        if (ldc < Shape.N) {
            MLAS_THROW_EX(std::runtime_error, "Dynamic QGEMM requires ldc >= N.");
        }
    }

    // Dynamic-quantize A (LHS).
    const size_t LhsPackedStride = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr);
    std::byte* LhsPackedData = nullptr;

    EnsureVectorSize(g_kai_tls_qgemm.lhs_packed, LhsPackedStride * BatchSize);
    LhsPackedData = g_kai_tls_qgemm.lhs_packed.data();

    // Per-batch table of LHS base pointers.
    EnsureVectorSize(g_kai_tls_qgemm.lhs_base_table, BatchSize);
    // Capture the shared batch table pointer so worker threads use the same backing storage.
    const std::byte** tls_lhs_base = g_kai_tls_qgemm.lhs_base_table.data();

    // B batches require no packing.
    // We have already decided the matmul variant we are using before having values for M, N, and K.
    MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {

        std::byte* lhs = nullptr;
        if (DataParams[batch_idx].Workspace && DataParams[batch_idx].WorkspaceSize >= LhsPackedStride) {
            lhs = static_cast<std::byte*>(DataParams[batch_idx].Workspace);
        } else {
            lhs = &(LhsPackedData[LhsPackedStride * batch_idx]);
        }
        KLEIDIAI_KERNEL_LOG("kai_run_lhs_quant_pack_qai8dxp_f32"
                            << " M=" << Shape.M << " K=" << Shape.K << " mr=" << mr << " kr=" << kr << " sr=" << sr << " m_idx_start=0");
        kai_run_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr, 0, DataParams[batch_idx].A, DataParams[batch_idx].lda * sizeof(float), lhs);
        tls_lhs_base[batch_idx] = lhs;
    });

    // Tile iteration dimensions.
    std::array<size_t, 3> dim;
    dim[0] = BatchSize;                  // B
    dim[1] = MlasDivRoundup(Shape.M, m_step);  // M
    dim[2] = MlasDivRoundup(Shape.N, n_step);  // N

    // Minimize the kernel call count for the number of available threads.
    auto RequiredTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0] * dim[1] * dim[2]);

    // Scale required tiles over available tile processors.
    dim[1] = MlasDivRoundup(RequiredTiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(RequiredTiles * dim[2], dim[1] * dim[2]);

    // Compute new step sizes.
    m_step *= MlasDivRoundup(MlasDivRoundup(Shape.M, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(Shape.N, dim[2]), n_step);

    n_step = CapDynamicQgemmNStepByPackedRhs(n_step, base_n_step, Shape.K, kDynamicQgemmNTileByteLimit);

    // Update tile iterations.
    dim[1] = MlasDivRoundup(Shape.M, m_step);
    dim[2] = MlasDivRoundup(Shape.N, n_step);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [=](ptrdiff_t tid) {

        // Compute B, M, N indices from the iteration index.
        ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
        ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
        ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

        // Get rhs tile, B
        const size_t rhs_packed_offset = qgemm_gemm.ukernel.get_rhs_packed_offset(NIdx * n_step, Shape.K);

        const std::byte* B_base = reinterpret_cast<const std::byte*>(DataParams[BIdx].PackedB);
        auto BTile = reinterpret_cast<const void*>(B_base + rhs_packed_offset);

        // Get lhs tile, A
        const size_t lhs_packed_offset = qgemm_gemm.ukernel.get_lhs_packed_offset(MIdx * m_step, Shape.K);

        const std::byte* A_base = tls_lhs_base[BIdx]; // LhsPackedData + LhsPackedStride * BIdx; OR DataParams[batch_idx].Workspace;
        auto ATile = reinterpret_cast<const std::byte*>(A_base + lhs_packed_offset);

        auto TileSizeM = (MIdx + 1) * m_step > Shape.M ? (Shape.M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > Shape.N ? (Shape.N - NIdx * n_step) : n_step;

        float* dst_tile = reinterpret_cast<float*>(
        reinterpret_cast<std::byte*>(DataParams[BIdx].C) +
        MIdx * m_step * DataParams[BIdx].ldc * sizeof(float) +
        NIdx * n_step * sizeof(float)
        );
        
        KLEIDIAI_KERNEL_LOG(qgemm_gemm.name
                            << " M=" << TileSizeM << " N=" << TileSizeN << " K=" << Shape.K);
        qgemm_gemm.ukernel.run_matmul(
                TileSizeM, TileSizeN, Shape.K, ATile, BTile,
                dst_tile,
                DataParams[BIdx].ldc * sizeof(float),
                sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
                );
    });
}
