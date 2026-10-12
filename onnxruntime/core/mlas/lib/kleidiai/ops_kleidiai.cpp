//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common_internal/utils.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "mlasi_kleidiai.h"

namespace
{

constexpr uint64_t kPackedMagic = UINT64_C(0x504f54524f49414b);  // "KAIORTOP"
constexpr uint16_t kPackedVersion = 2;
constexpr size_t kPackedAlignment = 64;
constexpr size_t kKernelNameCapacity = 96;

enum class PackedDataType : uint16_t {
    Fp32 = 1,
};

struct PackedHeader {
    uint64_t magic{};
    uint16_t version{};
    PackedDataType data_type{};
    uint32_t trans_b{};
    uint32_t weight_format{};
    uint32_t inner_block_size{};
    uint32_t outer_block_size{};
    uint32_t vector_length_bytes{};
    size_t n{};
    size_t k{};
    size_t column_sums_offset{};
    size_t payload_offset{};
    size_t payload_size{};
    char kernel_name[kKernelNameCapacity]{};
};

struct AlignedByteBuffer {
    std::vector<std::byte> storage;

    bool Resize(size_t size)
    {
        if (size > std::numeric_limits<size_t>::max() - (kPackedAlignment - 1)) {
            return false;
        }
        storage.resize(size + kPackedAlignment - 1);
        return true;
    }

    void* Data()
    {
        if (storage.empty()) {
            return nullptr;
        }
        const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data());
        const size_t padding = (kPackedAlignment - address % kPackedAlignment) % kPackedAlignment;
        return storage.data() + padding;
    }
};

struct ThreadScratch {
    std::vector<float> fp32_output;
    std::vector<PackedHeader> packed_headers;
    AlignedByteBuffer working;
};

ThreadScratch&
GetThreadScratch()
{
    thread_local ThreadScratch scratch;
    return scratch;
}

constexpr size_t kAlphaBetaNeonMinElements = 32;

void
ApplyAlphaBeta(
    const float* product,
    size_t rows,
    size_t columns,
    float alpha,
    float beta,
    float* output,
    size_t output_ldc
)
{
    if (rows == 0 || columns == 0) {
        return;
    }

    const float32x4_t alpha_vector = vdupq_n_f32(alpha);
    const float32x4_t beta_vector = vdupq_n_f32(beta);
    for (size_t row = 0; row < rows; ++row) {
        const float* source = product + row * columns;
        float* destination = output + row * output_ldc;
        if (alpha == 1.0f && beta == 0.0f) {
            std::memcpy(destination, source, columns * sizeof(float));
            continue;
        }

        size_t column = 0;
        if (columns >= kAlphaBetaNeonMinElements) {
            if (alpha == 1.0f) {
                for (; column + 4 <= columns; column += 4) {
                    const float32x4_t product_vector = vld1q_f32(source + column);
                    const float32x4_t output_vector = vld1q_f32(destination + column);
                    vst1q_f32(
                        destination + column,
                        vmlaq_f32(product_vector, beta_vector, output_vector)
                    );
                }
            } else if (beta == 0.0f) {
                for (; column + 4 <= columns; column += 4) {
                    vst1q_f32(
                        destination + column,
                        vmulq_f32(alpha_vector, vld1q_f32(source + column))
                    );
                }
            } else {
                for (; column + 4 <= columns; column += 4) {
                    const float32x4_t product_vector = vmulq_f32(
                        alpha_vector, vld1q_f32(source + column)
                    );
                    const float32x4_t output_vector = vld1q_f32(destination + column);
                    vst1q_f32(
                        destination + column,
                        vmlaq_f32(product_vector, beta_vector, output_vector)
                    );
                }
            }
        }

        for (; column < columns; ++column) {
            float value = alpha == 1.0f ? source[column] : alpha * source[column];
            if (beta != 0.0f) {
                value += beta * destination[column];
            }
            destination[column] = value;
        }
    }
}

void
ApplyBeta(
    float* output,
    size_t rows,
    size_t columns,
    size_t output_ldc,
    float beta
)
{
    if (beta == 1.0f) {
        return;
    }
    for (size_t row = 0; row < rows; ++row) {
        float* destination = output + row * output_ldc;
        if (beta == 0.0f) {
            std::fill_n(destination, columns, 0.0f);
            continue;
        }
        const float32x4_t beta_vector = vdupq_n_f32(beta);
        size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            vst1q_f32(
                destination + column,
                vmulq_f32(beta_vector, vld1q_f32(destination + column))
            );
        }
        for (; column < columns; ++column) {
            destination[column] *= beta;
        }
    }
}

const CPUInfo&
GetCPUInfo()
{
    // kai::ops declares get_CPUInfo(), but that function is supplied only by
    // its benchmark executable. Constructing CPUInfo performs the same HWCAP
    // probing without introducing an undefined benchmark symbol.
    static const CPUInfo cpu_info;
    return cpu_info;
}

bool
IsRuntimeSupported()
{
#if defined(__APPLE__)
    // KleidiAI Ops v1.30 can build its A64 GEMM path on Darwin, but its kernels
    // are not supported there. In particular, execution can report success
    // without writing the output, including on Apple CPUs that advertise SME.
    // Decline Ops before PackB sizing so MLAS creates and consumes its native
    // packed format instead.
    return false;
#else
    return true;
#endif
}

bool
AddOverflows(size_t lhs, size_t rhs)
{
    return lhs > std::numeric_limits<size_t>::max() - rhs;
}

bool
FitsInUnsigned(size_t value)
{
    return value <= static_cast<size_t>(std::numeric_limits<unsigned int>::max());
}

bool
FitsInInt(size_t value)
{
    return value <= static_cast<size_t>(std::numeric_limits<int>::max());
}

std::string
RequestedKernelFilter()
{
    // Keep PackBSize, PackB, and execution on one process-wide selection even
    // if another thread mutates the environment between those calls.
    static const std::string requested_filter = []() {
        const char* requested = std::getenv("ORT_KLEIDIAI_OPS_GEMM_FILTER");
        return requested != nullptr && requested[0] != '\0'
                   ? std::string(requested)
                   : std::string{};
    }();
    // An empty filter delegates ISA and kernel selection to KleidiAI Ops.
    return requested_filter;
}

bool
CopyKernelName(PackedHeader& header, const std::string& name)
{
    if (name.empty() || name.size() >= kKernelNameCapacity) {
        return false;
    }
    std::memset(header.kernel_name, 0, sizeof(header.kernel_name));
    std::memcpy(header.kernel_name, name.data(), name.size());
    return true;
}

bool
ReadKernelName(const PackedHeader& header, std::string& name)
{
    const void* terminator = std::memchr(header.kernel_name, '\0', sizeof(header.kernel_name));
    if (terminator == nullptr) {
        return false;
    }
    const auto* end = static_cast<const char*>(terminator);
    if (end == header.kernel_name) {
        return false;
    }
    name.assign(header.kernel_name, static_cast<size_t>(end - header.kernel_name));
    return true;
}

uint32_t
CurrentVectorLengthBytes(std::string_view kernel_name)
{
    if (kernel_name.compare(0, 4, "sme_") == 0 || kernel_name.compare(0, 5, "sme2_") == 0) {
        return GetCPUInfo().has_sme()
                   ? static_cast<uint32_t>(kai::ops::utils::sme::get_vector_length<uint8_t>())
                   : 0;
    }
    if (kernel_name.compare(0, 4, "sve_") != 0) {
        return 0;
    }
    // get_vector_length() executes CNTB directly. Never call it unless runtime
    // feature probing has established that SVE instructions are available.
    if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        return 0;
    }
    const auto vector_length = kai::ops::utils::get_vector_length<uint8_t>();
    return vector_length <= std::numeric_limits<uint32_t>::max()
               ? static_cast<uint32_t>(vector_length)
               : 0;
}

bool
AlignAddressOffset(const void* base, size_t initial_offset, size_t alignment, size_t& result)
{
    const uintptr_t address = reinterpret_cast<uintptr_t>(base);
    if (address > std::numeric_limits<uintptr_t>::max() - initial_offset) {
        return false;
    }
    const uintptr_t unaligned = address + initial_offset;
    const size_t padding = (alignment - unaligned % alignment) % alignment;
    if (AddOverflows(initial_offset, padding)) {
        return false;
    }
    result = initial_offset + padding;
    return true;
}

bool
ComputePackedLayout(
    const void* base,
    size_t payload_size,
    size_t& column_sums_offset,
    size_t& payload_offset,
    size_t& total_size
)
{
    size_t offset = sizeof(PackedHeader);
    column_sums_offset = 0;

    if (!AlignAddressOffset(base, offset, kPackedAlignment, payload_offset) ||
        AddOverflows(payload_offset, payload_size)) {
        return false;
    }
    total_size = payload_offset + payload_size;
    return true;
}

size_t
MaximumPackedSize(size_t payload_size)
{
    size_t size = sizeof(PackedHeader);
    if (AddOverflows(size, kPackedAlignment - 1) ||
        AddOverflows(size + kPackedAlignment - 1, payload_size)) {
        return 0;
    }
    return size + kPackedAlignment - 1 + payload_size;
}

kai::ops::GemmArgs
MakeGemmArgs(
    size_t m,
    size_t n,
    size_t k,
    const kai::ops::GemmConfig* config,
    bool accumulate,
    int max_threads
)
{
    return kai::ops::GemmArgs(
        &GetCPUInfo(),
        static_cast<unsigned int>(m),
        static_cast<unsigned int>(n),
        static_cast<unsigned int>(k),
        1,
        1,
        1,
        false,
        kai::ops::Activation{},
        max_threads,
        false,
        false,
        accumulate,
        config
    );
}

template <typename Lhs, typename Rhs, typename Output>
std::unique_ptr<kai::ops::GemmCommon<Lhs, Rhs, Output>>
SelectGemm(
    size_t m,
    size_t n,
    size_t k,
    const kai::ops::GemmConfig* config,
    bool accumulate = false,
    int max_threads = 1
)
{
    if (!FitsInUnsigned(m) || !FitsInUnsigned(n) || !FitsInUnsigned(k) || max_threads < 1) {
        return nullptr;
    }
    return kai::ops::gemm<Lhs, Rhs, Output>(MakeGemmArgs(m, n, k, config, accumulate, max_threads));
}

bool
ConfigFromHeader(const PackedHeader& header, kai::ops::GemmConfig& config)
{
    std::string kernel_name;
    if (!ReadKernelName(header, kernel_name)) {
        return false;
    }
    config.filter = std::move(kernel_name);
    config.weight_format = static_cast<kai::ops::WeightFormat>(header.weight_format);
    config.inner_block_size = header.inner_block_size;
    config.outer_block_size = header.outer_block_size;
    return true;
}

bool
IsPackedPayloadCompatible(
    std::string_view kernel_name,
    size_t packed_size,
    size_t runtime_size,
    size_t m
)
{
    if (runtime_size == packed_size) {
        return true;
    }

    // GemmHybridIndirect includes an obsolete M-sized output-pointer tail in
    // its reported pretranspose size. The pretranspose and execute paths do not
    // access that tail, so a buffer packed with M=1 remains valid for any M.
    // The small-K GemmHybrid implementation does not include this tail.
    if (kernel_name.find("_hybrid_") == std::string_view::npos ||
        kernel_name.find("smallK") != std::string_view::npos || m == 0) {
        return false;
    }
    size_t additional_size = 0;
    return !MlasMultiplyOverflowsSizeT(m - 1, sizeof(const void*), &additional_size) &&
           !AddOverflows(packed_size, additional_size) &&
           runtime_size == packed_size + additional_size;
}

template <typename Lhs, typename Rhs, typename Output>
bool
SelectPackedGemm(
    size_t n,
    size_t k,
    kai::ops::GemmConfig& selected_config,
    size_t& payload_size
)
{
    kai::ops::GemmConfig constraints;
    constraints.filter = RequestedKernelFilter();
    // MLAS's reusable PackB API has no M or thread-count argument. Select a
    // general GEMM (M=2 excludes M=1-only GEMV kernels), then retain that kernel
    // and its packing parameters for every subsequent use of this buffer.
    // Full-width N blocking keeps interleaved packing identical when Ops
    // switches between row-only and 2D scheduling at execution time.
    if (!FitsInUnsigned(n)) {
        return false;
    }
    constraints.outer_block_size = static_cast<unsigned int>(n);
    auto gemm = SelectGemm<Lhs, Rhs, Output>(2, n, k, &constraints);
    if (gemm == nullptr || !gemm->B_is_pretransposed()) {
        return false;
    }

    selected_config = gemm->get_config();
    gemm = SelectGemm<Lhs, Rhs, Output>(1, n, k, &selected_config);
    if (gemm == nullptr || !gemm->B_is_pretransposed()) {
        return false;
    }
    selected_config = gemm->get_config();
    payload_size = gemm->get_B_pretransposed_array_size();
    return payload_size != 0;
}

bool
HasPackedHeaderPrefix(const void* packed_b, PackedDataType data_type)
{
    if (packed_b == nullptr) {
        return false;
    }
    uint64_t magic = 0;
    uint16_t version = 0;
    PackedDataType actual_type{};
    std::memcpy(&magic, packed_b, sizeof(magic));
    std::memcpy(&version, static_cast<const std::byte*>(packed_b) + offsetof(PackedHeader, version), sizeof(version));
    std::memcpy(
        &actual_type,
        static_cast<const std::byte*>(packed_b) + offsetof(PackedHeader, data_type),
        sizeof(actual_type)
    );
    return magic == kPackedMagic && version == kPackedVersion && actual_type == data_type;
}

bool
ReadHeader(
    const void* packed_b,
    PackedDataType data_type,
    size_t n,
    size_t k,
    PackedHeader& header
)
{
    // BIsPacked is shared with the generic MLAS packing format. Check only
    // the small Ops prefix before reading the complete private header so a
    // small foreign PackedB allocation cannot be read out of bounds.
    if (!HasPackedHeaderPrefix(packed_b, data_type)) {
        return false;
    }
    std::memcpy(&header, packed_b, sizeof(header));
    std::string kernel_name;
    if (header.magic != kPackedMagic ||
        header.version != kPackedVersion ||
        header.data_type != data_type ||
        header.n != n || header.k != k ||
        header.payload_size == 0 ||
        !ReadKernelName(header, kernel_name)) {
        return false;
    }
    const bool is_scalable_kernel = kernel_name.compare(0, 4, "sve_") == 0 ||
                                    kernel_name.compare(0, 4, "sme_") == 0 ||
                                    kernel_name.compare(0, 5, "sme2_") == 0;
    const uint32_t current_vector_length = CurrentVectorLengthBytes(kernel_name);
    if ((is_scalable_kernel && current_vector_length == 0) ||
        header.vector_length_bytes != current_vector_length) {
        return false;
    }

    size_t expected_column_sums_offset = 0;
    size_t expected_payload_offset = 0;
    size_t expected_total_size = 0;
    return ComputePackedLayout(
               packed_b, header.payload_size,
               expected_column_sums_offset, expected_payload_offset,
               expected_total_size
           ) &&
           header.column_sums_offset == expected_column_sums_offset &&
           header.payload_offset == expected_payload_offset &&
           expected_total_size <= MaximumPackedSize(header.payload_size);
}

bool
IsPackedType(const void* packed_b, PackedDataType data_type)
{
    return HasPackedHeaderPrefix(packed_b, data_type);
}

template <typename Lhs, typename Rhs, typename Output>
bool
PackB(
    PackedDataType data_type,
    size_t n,
    size_t k,
    const Rhs* b,
    size_t ldb,
    bool transposed,
    void* packed_b
)
{
    if (b == nullptr || packed_b == nullptr || !FitsInInt(ldb)) {
        return false;
    }

    kai::ops::GemmConfig selected_config;
    size_t payload_size = 0;
    if (!SelectPackedGemm<Lhs, Rhs, Output>(n, k, selected_config, payload_size)) {
        return false;
    }

    auto gemm = SelectGemm<Lhs, Rhs, Output>(1, n, k, &selected_config);
    if (gemm == nullptr || !gemm->B_is_pretransposed() ||
        gemm->get_B_pretransposed_array_size() != payload_size ||
        (transposed && !gemm->B_pretranspose_supports_transpose())) {
        return false;
    }

    size_t column_sums_offset = 0;
    size_t payload_offset = 0;
    size_t total_size = 0;
    if (!ComputePackedLayout(
            packed_b, payload_size,
            column_sums_offset, payload_offset, total_size
        )) {
        return false;
    }
    const size_t maximum_size = MaximumPackedSize(payload_size);
    if (maximum_size == 0 || total_size > maximum_size) {
        return false;
    }

    PackedHeader header{};
    header.magic = kPackedMagic;
    header.version = kPackedVersion;
    header.data_type = data_type;
    header.trans_b = transposed ? static_cast<uint32_t>(CblasTrans) : static_cast<uint32_t>(CblasNoTrans);
    header.weight_format = static_cast<uint32_t>(selected_config.weight_format);
    header.inner_block_size = selected_config.inner_block_size;
    header.outer_block_size = selected_config.outer_block_size;
    header.vector_length_bytes = CurrentVectorLengthBytes(selected_config.filter);
    header.n = n;
    header.k = k;
    header.column_sums_offset = column_sums_offset;
    header.payload_offset = payload_offset;
    header.payload_size = payload_size;
    if (!CopyKernelName(header, selected_config.filter)) {
        return false;
    }

    std::memcpy(packed_b, &header, sizeof(header));
    KLEIDIAI_KERNEL_LOG("kai::ops pack kernel=" << selected_config.filter << " N=" << n << " K=" << k << " payload_bytes=" << payload_size);
    gemm->pretranspose_B_array(
        static_cast<std::byte*>(packed_b) + payload_offset,
        b,
        static_cast<int>(ldb),
        0,
        transposed
    );
    return true;
}

int
GetTargetThreadCount(MLAS_THREADPOOL* thread_pool, double complexity, double thread_complexity)
{
    const ptrdiff_t maximum_count = MlasGetMaximumThreadCount(thread_pool);
    if (maximum_count <= 1) {
        return 1;
    }
    const ptrdiff_t bounded_maximum = std::min<ptrdiff_t>(maximum_count, std::numeric_limits<int>::max());
    if (complexity >= thread_complexity * static_cast<double>(bounded_maximum)) {
        return static_cast<int>(bounded_maximum);
    }
    return static_cast<int>(complexity / thread_complexity) + 1;
}

struct WorkPartition {
    size_t dim0_parts{1};
    size_t dim1_parts{1};
    ptrdiff_t task_count{1};
};

WorkPartition
PartitionWindow(const kai::ops::ndrange_t& window, int max_threads)
{
    const size_t dim0_size = window.get_size(0);
    const size_t dim1_size = window.get_size(1);
    const size_t thread_limit = static_cast<size_t>(max_threads);

    WorkPartition best{};
    long double best_imbalance = std::numeric_limits<long double>::max();
    const size_t max_dim0_parts = std::min(dim0_size, thread_limit);
    for (size_t dim0_parts = 1; dim0_parts <= max_dim0_parts; ++dim0_parts) {
        const size_t dim1_parts = std::min(dim1_size, thread_limit / dim0_parts);
        const size_t task_count = dim0_parts * dim1_parts;
        const long double imbalance = std::fabs(
            static_cast<long double>(dim0_parts) * dim1_size -
            static_cast<long double>(dim1_parts) * dim0_size
        );
        if (task_count > static_cast<size_t>(best.task_count) ||
            (task_count == static_cast<size_t>(best.task_count) && imbalance < best_imbalance)) {
            best.dim0_parts = dim0_parts;
            best.dim1_parts = dim1_parts;
            best.task_count = static_cast<ptrdiff_t>(task_count);
            best_imbalance = imbalance;
        }
    }
    return best;
}

template <typename Lhs, typename Rhs, typename Output>
struct CachedGemmPlan {
    PackedHeader header{};
    size_t m{};
    size_t n{};
    size_t k{};
    bool accumulate{};
    int max_threads{};
    std::unique_ptr<kai::ops::GemmCommon<Lhs, Rhs, Output>> gemm;
    WorkPartition partition{};
    size_t window_dim0{};
    size_t window_dim1{};
    size_t working_size{};
    uint32_t vector_length{};
    std::string kernel_name;
};

template <typename Lhs, typename Rhs, typename Output>
struct GemmPlanCache {
    // Plans contain mutable array/workspace pointers, so keep them thread-local
    // instead of sharing them between concurrent MLAS callers. Bound the cache
    // while retaining enough entries for models with many distinct GEMM shapes.
    static constexpr size_t kMaximumEntries = 64;
    std::vector<CachedGemmPlan<Lhs, Rhs, Output>> entries;
    size_t next_replacement{};
};

bool
SamePlanKey(
    const PackedHeader& lhs,
    const PackedHeader& rhs
)
{
    return lhs.data_type == rhs.data_type &&
           lhs.weight_format == rhs.weight_format &&
           lhs.inner_block_size == rhs.inner_block_size &&
           lhs.outer_block_size == rhs.outer_block_size &&
           lhs.vector_length_bytes == rhs.vector_length_bytes &&
           lhs.payload_size == rhs.payload_size &&
           std::memcmp(lhs.kernel_name, rhs.kernel_name, sizeof(lhs.kernel_name)) == 0;
}

template <typename Lhs, typename Rhs, typename Output>
GemmPlanCache<Lhs, Rhs, Output>&
GetGemmPlanCache()
{
    thread_local GemmPlanCache<Lhs, Rhs, Output> cache;
    return cache;
}

template <typename Lhs, typename Rhs, typename Output>
CachedGemmPlan<Lhs, Rhs, Output>*
PrepareGemm(
    const PackedHeader& header,
    size_t m,
    size_t n,
    size_t k,
    bool accumulate,
    int max_threads
)
{
    auto& cache = GetGemmPlanCache<Lhs, Rhs, Output>();
    for (auto& plan : cache.entries) {
        if (plan.m == m && plan.n == n && plan.k == k &&
            plan.accumulate == accumulate && plan.max_threads == max_threads &&
            SamePlanKey(plan.header, header)) {
            return &plan;
        }
    }

    kai::ops::GemmConfig config;
    if (!ConfigFromHeader(header, config)) {
        return nullptr;
    }
    auto gemm = SelectGemm<Lhs, Rhs, Output>(m, n, k, &config, accumulate, max_threads);
    if (gemm == nullptr || !gemm->B_is_pretransposed()) {
        return nullptr;
    }
    const auto& selected_config = gemm->get_config();
    if (selected_config.filter != config.filter ||
        selected_config.weight_format != config.weight_format ||
        selected_config.inner_block_size != config.inner_block_size ||
        selected_config.outer_block_size != config.outer_block_size ||
        !IsPackedPayloadCompatible(
            config.filter, header.payload_size,
            gemm->get_B_pretransposed_array_size(), m
        )) {
        return nullptr;
    }

    CachedGemmPlan<Lhs, Rhs, Output> new_plan;
    new_plan.header = header;
    new_plan.m = m;
    new_plan.n = n;
    new_plan.k = k;
    new_plan.accumulate = accumulate;
    new_plan.max_threads = max_threads;
    new_plan.gemm = std::move(gemm);
    new_plan.kernel_name = selected_config.filter;
    new_plan.vector_length = CurrentVectorLengthBytes(new_plan.kernel_name);
    const auto window = new_plan.gemm->get_window_size();
    new_plan.window_dim0 = window.get_size(0);
    new_plan.window_dim1 = window.get_size(1);
    new_plan.partition = PartitionWindow(window, max_threads);
    new_plan.gemm->set_nthreads(static_cast<int>(new_plan.partition.task_count));
    new_plan.working_size = new_plan.gemm->get_working_size();

    if (cache.entries.size() < cache.kMaximumEntries) {
        cache.entries.emplace_back(std::move(new_plan));
        return &cache.entries.back();
    }
    auto& replacement = cache.entries[cache.next_replacement];
    replacement = std::move(new_plan);
    cache.next_replacement = (cache.next_replacement + 1) % cache.kMaximumEntries;
    return &replacement;
}

template <typename Lhs, typename Rhs, typename Output>
bool
ExecuteGemm(
    CachedGemmPlan<Lhs, Rhs, Output>& plan,
    const Lhs* a,
    size_t lda,
    Output* c,
    size_t ldc,
    void* packed_payload,
    size_t m,
    size_t n,
    size_t k,
    bool accumulate,
    const char* data_type,
    MLAS_THREADPOOL* thread_pool
)
{
    if (a == nullptr || c == nullptr || !FitsInInt(lda) || !FitsInInt(ldc)) {
        return false;
    }

    auto& gemm = *plan.gemm;
    gemm.set_arrays(
        a, static_cast<int>(lda), 0, 0,
        nullptr, 0, 0,
        c, static_cast<int>(ldc), 0, 0,
        nullptr, 0
    );
    gemm.set_pretransposed_B_data(packed_payload);

    if (plan.working_size != 0) {
        auto& working = GetThreadScratch().working;
        if (!working.Resize(plan.working_size)) {
            return false;
        }
        // Some kernels use workspace as an accumulation buffer and do not
        // overwrite every byte before reading it. Plan caching keeps the size
        // lookup off the hot path, but each execution must start from the
        // kernel API's required zero-initialized state.
        std::memset(working.Data(), 0, plan.working_size);
        gemm.set_working_space(working.Data());
    }

    KLEIDIAI_KERNEL_LOG(
        "kai::ops execute kernel=" << plan.kernel_name << " type=" << data_type
                                    << " M=" << m << " N=" << n << " K=" << k
                                    << " accumulate=" << accumulate << " threads=" << plan.partition.task_count
    );
    MlasTrySimpleParallel(thread_pool, plan.partition.task_count, [&](ptrdiff_t task_id) {
        // SVE VL and SME streaming VL are per-thread state on Linux/Android.
        // Never execute a scalable kernel against weights packed at another VL.
        if (CurrentVectorLengthBytes(plan.kernel_name) != plan.vector_length) {
            MLAS_THROW_EX(std::runtime_error, "KleidiAI Ops worker vector length differs from PackedB.");
        }
        const size_t dim0_id = static_cast<size_t>(task_id) % plan.partition.dim0_parts;
        const size_t dim1_id = static_cast<size_t>(task_id) / plan.partition.dim0_parts;
        const size_t dim0_start = dim0_id * plan.window_dim0 / plan.partition.dim0_parts;
        const size_t dim0_end = (dim0_id + 1) * plan.window_dim0 / plan.partition.dim0_parts;
        const size_t dim1_start = dim1_id * plan.window_dim1 / plan.partition.dim1_parts;
        const size_t dim1_end = (dim1_id + 1) * plan.window_dim1 / plan.partition.dim1_parts;
        const kai::ops::ndcoord_t work_range = {
            {dim0_start, dim0_end - dim0_start},
            {dim1_start, dim1_end - dim1_start}
        };
        const kai::ops::ndcoord_t thread_locator = {
            {dim0_id, plan.partition.dim0_parts},
            {dim1_id, plan.partition.dim1_parts}
        };
        gemm.execute(work_range, thread_locator, static_cast<int>(task_id));
    });
    return true;
}

}  // namespace

size_t
    MLASCALL
    ArmKleidiAI::MlasGemmPackBSizeKleidiAIOps(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K
    )
{
    if (!IsRuntimeSupported() ||
        TransA != CblasNoTrans || TransB != CblasNoTrans || N == 0 || K == 0) {
        return 0;
    }
    kai::ops::GemmConfig config;
    size_t payload_size = 0;
    if (!SelectPackedGemm<float, float, float>(N, K, config, payload_size)) {
        return 0;
    }
    return MaximumPackedSize(payload_size);
}

bool
    MLASCALL
    ArmKleidiAI::MlasGemmPackBKleidiAIOps(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K,
        const float* B,
        size_t ldb,
        void* PackedB
    )
{
    return IsRuntimeSupported() &&
           TransA == CblasNoTrans && TransB == CblasNoTrans &&
           PackB<float, float, float>(PackedDataType::Fp32, N, K, B, ldb, false, PackedB);
}

bool
    MLASCALL
    ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(const void* PackedB)
{
    return IsPackedType(PackedB, PackedDataType::Fp32);
}

bool
    MLASCALL
    ArmKleidiAI::MlasGemmBatchKleidiAIOps(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        const MLAS_SGEMM_DATA_PARAMS* Data,
        size_t BatchSize,
        MLAS_THREADPOOL* ThreadPool
    )
{
    if (!IsRuntimeSupported()) {
        return false;
    }
    if (M == 0 || N == 0 || BatchSize == 0) {
        return true;
    }
    if (TransA != CblasNoTrans || K == 0 || Data == nullptr) {
        return false;
    }

    auto& scratch = GetThreadScratch();
    scratch.packed_headers.resize(BatchSize);
    for (size_t batch = 0; batch < BatchSize; ++batch) {
        if (Data[batch].A == nullptr || Data[batch].C == nullptr ||
            !Data[batch].BIsPacked) {
            return false;
        }
        auto& header = scratch.packed_headers[batch];
        if (!ReadHeader(Data[batch].B, PackedDataType::Fp32, N, K, header)) {
            return false;
        }
        // MLAS deliberately passes CblasTrans for any opaque PackedB. The actual
        // source orientation is recorded in our header.
        if (header.trans_b != static_cast<uint32_t>(CblasNoTrans) ||
            (TransB != CblasNoTrans && TransB != CblasTrans)) {
            return false;
        }
    }

    size_t output_size = 0;
    if (MlasMultiplyOverflowsSizeT(M, N, &output_size)) {
        return false;
    }
    const int max_threads = GetTargetThreadCount(
        ThreadPool, double(M) * double(N) * double(K), double(MLAS_SGEMM_THREAD_COMPLEXITY)
    );

    for (size_t batch = 0; batch < BatchSize; ++batch) {
        const auto& params = Data[batch];
        const auto& header = scratch.packed_headers[batch];
        if (params.alpha == 0.0f) {
            ApplyBeta(params.C, M, N, params.ldc, params.beta);
            continue;
        }
        const bool can_write_directly =
            params.alpha == 1.0f && (params.beta == 0.0f || params.beta == 1.0f);
        const bool accumulate = params.beta == 1.0f;
        CachedGemmPlan<float, float, float>* plan = nullptr;
        if (can_write_directly) {
            plan = PrepareGemm<float, float, float>(header, M, N, K, accumulate, max_threads);
        }
        if (plan != nullptr &&
            ExecuteGemm(
                *plan, params.A, params.lda,
                params.C, params.ldc,
                const_cast<std::byte*>(reinterpret_cast<const std::byte*>(params.B)) + header.payload_offset,
                M, N, K, accumulate, "fp32", ThreadPool
            )) {
            continue;
        }

        // General alpha/beta values need a temporary product. This also provides
        // a correctness fallback if a selected implementation cannot accumulate.
        scratch.fp32_output.resize(output_size);
        plan = PrepareGemm<float, float, float>(header, M, N, K, false, max_threads);
        if (plan == nullptr ||
            !ExecuteGemm(
                *plan, params.A, params.lda,
                scratch.fp32_output.data(), N,
                const_cast<std::byte*>(reinterpret_cast<const std::byte*>(params.B)) + header.payload_offset,
                M, N, K, false, "fp32", ThreadPool
            )) {
            return false;
        }
        ApplyAlphaBeta(
            scratch.fp32_output.data(), M, N,
            params.alpha, params.beta, params.C, params.ldc
        );
    }
    return true;
}
