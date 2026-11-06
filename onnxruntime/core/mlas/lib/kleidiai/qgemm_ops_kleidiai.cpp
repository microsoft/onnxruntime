//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common_internal/utils.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "mlasi_kleidiai.h"

namespace
{

constexpr uint64_t kPackedMagic = UINT64_C(0x38514f54524f494b);
constexpr uint32_t kPackedVersion = 1;
constexpr size_t kAlignment = 64;

struct PackedHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t vector_length;
    size_t n;
    size_t k;
    size_t payload_offset;
    size_t payload_size;
    unsigned int inner_block_size;
    unsigned int outer_block_size;
    uint32_t weight_format;
    char kernel_name[96];
};

using Gemm = kai::ops::GemmCommon<int8_t, int8_t, int32_t>;

const CPUInfo&
GetCPUInfo()
{
    static const CPUInfo cpu_info = []() {
        CPUInfo info;
        // PackB size and layout must not change when the caller migrates between core models.
        info.midr_override(0);
        return info;
    }();
    return cpu_info;
}

uint32_t
VectorLength(std::string_view kernel)
{
    if (kernel.compare(0, 4, "sme_") == 0 || kernel.compare(0, 5, "sme2_") == 0) {
        return GetCPUInfo().has_sme() ? static_cast<uint32_t>(kai::ops::utils::sme::get_vector_length<uint8_t>()) : 0;
    }
    if (kernel.compare(0, 4, "sve_") == 0) {
        return GetCPUInfo().has_sve() ? static_cast<uint32_t>(kai::ops::utils::get_vector_length<uint8_t>()) : 0;
    }
    return 0;
}

bool
FitsOpsDimensions(size_t m, size_t n, size_t k)
{
    constexpr size_t int_limit = static_cast<size_t>(std::numeric_limits<int>::max());
    if (m == 0 || n == 0 || k == 0 || m > int_limit || n > int_limit || k > int_limit) {
        return false;
    }
    // Ops rounds and multiplies packing/window dimensions in unsigned arithmetic.
    const uint64_t padded_m = static_cast<uint64_t>(m) + 256;
    const uint64_t padded_n = static_cast<uint64_t>(n) + 1024;
    const uint64_t padded_k = static_cast<uint64_t>(k) + 64;
    constexpr uint64_t limit = std::numeric_limits<unsigned int>::max();
    return padded_n * padded_k <= limit && padded_m * padded_n <= limit && padded_m * padded_k <= limit;
}

bool
FitsMatrixSpan(size_t rows, size_t columns, size_t stride)
{
    size_t span = 0;
    return rows != 0 && stride >= columns && stride <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           !MlasMultiplyOverflowsSizeT(rows - 1, stride, &span) &&
           !MlasAddOverflowsSizeT(span, columns, &span) &&
           span <= static_cast<size_t>(std::numeric_limits<int>::max());
}

std::unique_ptr<Gemm>
SelectGemm(size_t m, size_t n, size_t k, const kai::ops::GemmConfig* config = nullptr)
{
    if (!FitsOpsDimensions(m, n, k)) {
        return nullptr;
    }
    const kai::ops::GemmArgs args(
        &GetCPUInfo(), static_cast<unsigned int>(m), static_cast<unsigned int>(n), static_cast<unsigned int>(k),
        1, 1, 1, false, {}, 1, false, false, false, config
    );
    return kai::ops::gemm<int8_t, int8_t, int32_t>(args);
}

size_t
PackedSize(size_t n, size_t payload_size)
{
    size_t size = 0;
    if (MlasMultiplyOverflowsSizeT(n, sizeof(uint32_t), &size) ||
        MlasAddOverflowsSizeT(size, sizeof(PackedHeader), &size) ||
        MlasAddOverflowsSizeT(size, kAlignment - 1, &size) ||
        MlasAddOverflowsSizeT(size, payload_size, &size)) {
        return 0;
    }
    return size;
}

std::unique_ptr<Gemm>
SelectPackingGemm(size_t n, size_t k, bool a_signed, bool b_signed)
{
    if (!a_signed || !b_signed) {
        return nullptr;
    }
    // PackB has no capacity argument, so its ISA-dependent sizing contract cannot change between calls.
    static const uint32_t sve_length = VectorLength("sve_");
    static const uint32_t sme_length = VectorLength("sme_");
    if (sve_length != VectorLength("sve_") || sme_length != VectorLength("sme_")) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 packing requires the original vector length.");
    }
    auto gemm = SelectGemm(1, n, k);
    if (gemm == nullptr || !gemm->B_is_pretransposed() || gemm->get_B_pretransposed_array_size() == 0 ||
        gemm->get_config().filter.empty() || gemm->get_config().filter.size() >= sizeof(PackedHeader::kernel_name)) {
        return nullptr;
    }
    return gemm;
}

bool
PayloadSizeMatches(const PackedHeader& header, size_t runtime_size, size_t m)
{
    if (runtime_size == header.payload_size) {
        return true;
    }
    const std::string_view kernel(header.kernel_name);
    size_t expected_size = 0;
    // HybridIndirect's reported size includes an unused M-sized result-pointer tail.
    return kernel.find("_hybrid_") != std::string_view::npos && kernel.find("smallK") == std::string_view::npos &&
           !MlasAddOverflowsSizeT(header.payload_size, (m - 1) * sizeof(const int32_t*), &expected_size) &&
           runtime_size == expected_size;
}

struct PreparedGemm {
    PackedHeader header;
    kai::ops::GemmConfig config;
    std::unique_ptr<Gemm> gemm;
    size_t working_size = 0;
};

void
PrepareGemm(const MLAS_GEMM_QUANT_SHAPE_PARAMS& shape, const MLAS_GEMM_QUANT_DATA_PARAMS& data, PreparedGemm& prepared)
{
    if (data.B == nullptr || data.A == nullptr || data.C == nullptr || data.ZeroPointB == nullptr ||
        !FitsMatrixSpan(shape.M, shape.K, data.lda) || !FitsMatrixSpan(shape.M, shape.N, data.ldc)) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 requires valid matrices, strides and zero points.");
    }
    auto& header = prepared.header;
    uint64_t magic = 0;
    std::memcpy(&magic, data.B, sizeof(magic));
    if (magic != kPackedMagic) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 requires its own packed RHS format.");
    }
    std::memcpy(&header, data.B, sizeof(header));
    const size_t metadata_size = sizeof(header) + shape.N * sizeof(uint32_t);
    if (header.version != kPackedVersion || header.n != shape.N || header.k != shape.K ||
        header.kernel_name[0] == '\0' || std::memchr(header.kernel_name, '\0', sizeof(header.kernel_name)) == nullptr ||
        header.inner_block_size == 0 || header.inner_block_size > shape.K + 64 ||
        header.outer_block_size == 0 || header.outer_block_size > shape.N + 1024 ||
        header.payload_offset < metadata_size || header.payload_offset > metadata_size + kAlignment - 1 ||
        (reinterpret_cast<uintptr_t>(data.B) + header.payload_offset) % kAlignment != 0 ||
        header.vector_length != VectorLength(header.kernel_name)) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 packed RHS metadata or vector length is incompatible.");
    }
    auto& config = prepared.config;
    config.filter = header.kernel_name;
    config.inner_block_size = header.inner_block_size;
    config.outer_block_size = header.outer_block_size;
    config.weight_format = static_cast<kai::ops::WeightFormat>(header.weight_format);
    prepared.gemm = SelectGemm(shape.M, shape.N, shape.K, &config);
    if (prepared.gemm == nullptr || !prepared.gemm->B_is_pretransposed()) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 cannot execute this shape with its packed RHS.");
    }
    const auto actual_config = prepared.gemm->get_config();
    if (actual_config.filter != config.filter || actual_config.inner_block_size != config.inner_block_size ||
        actual_config.outer_block_size != config.outer_block_size || actual_config.weight_format != config.weight_format ||
        !PayloadSizeMatches(header, prepared.gemm->get_B_pretransposed_array_size(), shape.M)) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 execution would change the packed RHS layout.");
    }
    prepared.working_size = prepared.gemm->get_working_size();
    if (prepared.working_size > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 workspace size overflow.");
    }
}

}  // namespace

namespace ArmKleidiAI
{

size_t MLASCALL
MlasQGemmPackBSizeKleidiAIOps(size_t N, size_t K, bool AIsSigned, bool BIsSigned)
{
    auto gemm = SelectPackingGemm(N, K, AIsSigned, BIsSigned);
    return gemm != nullptr ? PackedSize(N, gemm->get_B_pretransposed_array_size()) : 0;
}

void MLASCALL
MlasQGemmPackBKleidiAIOps(size_t N, size_t K, const uint8_t* B, size_t ldb, bool AIsSigned, bool BIsSigned, void* PackedB)
{
    auto gemm = SelectPackingGemm(N, K, AIsSigned, BIsSigned);
    if (gemm == nullptr || B == nullptr || PackedB == nullptr || !FitsMatrixSpan(K, N, ldb)) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 cannot pack this RHS.");
    }
    PackedHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic = kPackedMagic;
    header.version = kPackedVersion;
    header.n = N;
    header.k = K;
    header.payload_size = gemm->get_B_pretransposed_array_size();
    const auto config = gemm->get_config();
    header.inner_block_size = config.inner_block_size;
    header.outer_block_size = config.outer_block_size;
    header.weight_format = static_cast<uint32_t>(config.weight_format);
    header.vector_length = VectorLength(config.filter);
    std::memcpy(header.kernel_name, config.filter.data(), config.filter.size());
    auto* bytes = static_cast<std::byte*>(PackedB);
    const size_t packed_size = PackedSize(N, header.payload_size);
    if (packed_size == 0) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 packed RHS size overflow.");
    }
    std::memset(PackedB, 0, packed_size);
    const size_t metadata_size = sizeof(header) + N * sizeof(uint32_t);
    const uintptr_t address = reinterpret_cast<uintptr_t>(bytes + metadata_size);
    header.payload_offset = metadata_size + (kAlignment - address % kAlignment) % kAlignment;
    const auto* source = reinterpret_cast<const int8_t*>(B);
    for (size_t n = 0; n < N; ++n) {
        uint32_t sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += static_cast<uint32_t>(static_cast<int32_t>(source[k * ldb + n]));
        }
        std::memcpy(bytes + sizeof(header) + n * sizeof(sum), &sum, sizeof(sum));
    }
    KLEIDIAI_KERNEL_LOG("kai::ops prepack kernel=" << header.kernel_name << " type=s8s8 N=" << N << " K=" << K);
    gemm->pretranspose_B_array(bytes + header.payload_offset, source, static_cast<int>(ldb), 0, false);
    std::memcpy(PackedB, &header, sizeof(header));
}

bool MLASCALL
MlasQGemmBatchKleidiAIOps(const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape, const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams, size_t BatchN, MLAS_THREADPOOL* ThreadPool)
{
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
    if (!Shape.AIsSigned || !Shape.BIsSigned || !FitsOpsDimensions(1, Shape.N, Shape.K)) {
        return false;
    }
    if (BatchN == 0 || Shape.M == 0) {
        return true;
    }
    if (DataParams == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 requires valid batch parameters.");
    }
    for (size_t batch = 0; batch < BatchN; ++batch) {
        if (!DataParams[batch].BIsPacked) {
            return false;
        }
        if (DataParams[batch].B == nullptr) {
            MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops INT8 requires a valid packed RHS.");
        }
        uint64_t magic = 0;
        std::memcpy(&magic, DataParams[batch].B, sizeof(magic));
        if (magic != kPackedMagic) {
            return false;
        }
    }
    std::vector<PreparedGemm> prepared(BatchN);
    for (size_t batch = 0; batch < BatchN; ++batch) {
        PrepareGemm(Shape, DataParams[batch], prepared[batch]);
    }
    for (size_t batch = 0; batch < BatchN; ++batch) {
        const auto& data = DataParams[batch];
        auto& plan = prepared[batch];
        auto& gemm = *plan.gemm;
        const auto* packed = static_cast<const std::byte*>(data.B);
        std::vector<int32_t> product;
        if (Shape.IsAccumulateMode) {
            product.resize(Shape.M * Shape.N);
        }
        int32_t* output = Shape.IsAccumulateMode ? product.data() : data.C;
        const size_t ldc = Shape.IsAccumulateMode ? Shape.N : data.ldc;
        gemm.set_arrays(reinterpret_cast<const int8_t*>(data.A), static_cast<int>(data.lda), 0, 0, nullptr, 0, 0, output, static_cast<int>(ldc), 0, 0, nullptr, 0);
        gemm.set_pretransposed_B_data(const_cast<std::byte*>(packed) + plan.header.payload_offset);
        std::vector<std::byte> workspace;
        if (plan.working_size != 0) {
            workspace.resize(plan.working_size + kAlignment - 1);
            const uintptr_t address = reinterpret_cast<uintptr_t>(workspace.data());
            gemm.set_working_space(workspace.data() + (kAlignment - address % kAlignment) % kAlignment);
        }
        // shortcut: serial execution freezes packed blocking; add contract-aware scheduling before performance tuning.
        gemm.set_nthreads(1);
        const auto window = gemm.get_window_size();
        KLEIDIAI_KERNEL_LOG("kai::ops execute kernel=" << plan.header.kernel_name << " type=s8s8 M=" << Shape.M << " N=" << Shape.N << " K=" << Shape.K << " threads=1");
        gemm.execute({{0, window.get_size(0)}, {0, window.get_size(1)}}, {{0, 1}, {0, 1}}, 0);
        const uint32_t zp_a = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(data.ZeroPointA)));
        const bool needs_row_sums = data.PerColumnZeroPoints || *data.ZeroPointB != 0;
        if (Shape.IsAccumulateMode || zp_a != 0 || needs_row_sums) {
            const auto* a = reinterpret_cast<const int8_t*>(data.A);
            for (size_t m = 0; m < Shape.M; ++m) {
                uint32_t row_sum = 0;
                if (needs_row_sums) {
                    for (size_t k = 0; k < Shape.K; ++k) {
                        row_sum += static_cast<uint32_t>(static_cast<int32_t>(a[m * data.lda + k]));
                    }
                }
                for (size_t n = 0; n < Shape.N; ++n) {
                    const uint32_t zp_b = static_cast<uint32_t>(
                        static_cast<int32_t>(static_cast<int8_t>(data.ZeroPointB[data.PerColumnZeroPoints ? n : 0]))
                    );
                    uint32_t column_sum = 0;
                    if (zp_a != 0) {
                        std::memcpy(&column_sum, packed + sizeof(PackedHeader) + n * sizeof(column_sum), sizeof(column_sum));
                    }
                    uint32_t value = static_cast<uint32_t>(output[m * ldc + n]);
                    value += static_cast<uint32_t>(Shape.K) * zp_a * zp_b - zp_a * column_sum - zp_b * row_sum;
                    if (Shape.IsAccumulateMode) {
                        value += static_cast<uint32_t>(data.C[m * data.ldc + n]);
                    }
                    std::memcpy(data.C + m * data.ldc + n, &value, sizeof(value));
                }
            }
        }
        if (data.OutputProcessor != nullptr) {
            data.OutputProcessor->Process(data.C, 0, 0, Shape.M, Shape.N, data.ldc);
        }
    }
    return true;
}

}  // namespace ArmKleidiAI
