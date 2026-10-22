//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common_internal/utils.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "mlasi_kleidiai.h"
#include "sgemm_common.h"

namespace
{

constexpr uint64_t kPackedMagic = UINT64_C(0x504f54524f49414b);  // "KAIORTOP"
constexpr uint16_t kPackedVersion = 3;
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
    size_t payload_offset{};
    size_t payload_size{};
    char kernel_name[kKernelNameCapacity]{};
};

struct AlignedByteBuffer {
    std::vector<std::byte> storage;

    bool Resize(size_t size)
    {
        size_t allocation_size = 0;
        if (MlasAddOverflowsSizeT(size, kPackedAlignment - 1, &allocation_size)) {
            return false;
        }
        storage.resize(allocation_size);
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
    std::vector<uint8_t> skipped_tasks;
    AlignedByteBuffer working;
    AlignedByteBuffer packed_rhs;
    bool in_use{};
};

struct ScratchLease {
    ThreadScratch& scratch;

    explicit ScratchLease(ThreadScratch& value) : scratch(value) { scratch.in_use = true; }
    ~ScratchLease()
    {
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(scratch.fp32_output);
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(scratch.working.storage);
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(scratch.packed_rhs.storage);
        scratch.in_use = false;
    }
    ScratchLease(const ScratchLease&) = delete;
    ScratchLease& operator=(const ScratchLease&) = delete;
};

ThreadScratch&
GetThreadScratch()
{
    thread_local ThreadScratch scratch;
    return scratch;
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
FitsInInt(size_t value)
{
    return value <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool
FitsOpsMatrixSpan(size_t rows, size_t columns, size_t stride)
{
    size_t offset = 0;
    return rows != 0 && stride >= columns && FitsInInt(stride) &&
           !MlasMultiplyOverflowsSizeT(rows - 1, stride, &offset) &&
           !MlasAddOverflowsSizeT(offset, columns, &offset) && FitsInInt(offset);
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
PackingVectorLengthsAreCompatible()
{
    struct VectorLengths {
        uint32_t sve;
        uint32_t sme;
    };
    static const VectorLengths packing_lengths = {
        CurrentVectorLengthBytes("sve_"), CurrentVectorLengthBytes("sme_")
    };
    static std::atomic<bool> packing_enabled{true};
    if (!packing_enabled.load()) {
        return false;
    }
    if (CurrentVectorLengthBytes("sve_") != packing_lengths.sve ||
        CurrentVectorLengthBytes("sme_") != packing_lengths.sme) {
        // PackB has no allocation-capacity argument. Keep one process-wide
        // packing VL contract and permanently decline new Ops packs after a
        // mismatch, including after restoring the original VL. Otherwise a
        // header-only PackBSize result could later be followed by a full pack.
        // Existing packs remain executable at their recorded vector length.
        packing_enabled.store(false);
        return false;
    }
    return true;
}

constexpr size_t kPayloadOffset =
    (sizeof(PackedHeader) + kPackedAlignment - 1) & ~(kPackedAlignment - 1);

size_t
MaximumPackedSize(size_t payload_size)
{
    size_t aligned_size = 0;
    if (MlasAddOverflowsSizeT(kPayloadOffset, payload_size, &aligned_size) ||
        MlasAddOverflowsSizeT(aligned_size, kPackedAlignment - 1, &aligned_size)) {
        return 0;
    }
    // Relative offsets and aligned lengths survive shared-pack copies and
    // concatenated allocations without depending on the allocation address.
    return aligned_size & ~(kPackedAlignment - 1);
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
    if (!FitsInInt(m) || !FitsInInt(n) || !FitsInInt(k) || max_threads < 1 ||
        m == 0 || n == 0 || k == 0) {
        return nullptr;
    }
    // Ops v1.30 rounds and multiplies dimensions in unsigned arithmetic.
    // Leave oversized matrices to MLAS rather than accepting a wrapped pack
    // size or window. These margins cover FP32 tiles at the maximum Arm VL.
    const uint64_t padded_n = static_cast<uint64_t>(n) + 1024;
    const uint64_t padded_k = static_cast<uint64_t>(k) + 64;
    const uint64_t padded_m = static_cast<uint64_t>(m) + 64;
    const uint64_t unsigned_limit = std::numeric_limits<unsigned int>::max();
    if (padded_n * padded_k > unsigned_limit || padded_m * padded_n > unsigned_limit) {
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
    size_t expected_size = 0;
    return !MlasMultiplyOverflowsSizeT(m - 1, sizeof(const void*), &additional_size) &&
           !MlasAddOverflowsSizeT(packed_size, additional_size, &expected_size) &&
           runtime_size == expected_size;
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
    if (!PackingVectorLengthsAreCompatible()) {
        return false;
    }
    kai::ops::GemmConfig constraints;
    constraints.filter = RequestedKernelFilter();
    // MLAS's reusable PackB API has no M or thread-count argument. Select a
    // general GEMM (M=2 excludes M=1-only GEMV kernels), then retain that kernel
    // and its packing parameters for every subsequent use of this buffer.
    // Full-width N blocking keeps interleaved packing identical when Ops
    // switches between row-only and 2D scheduling at execution time.
    if (!FitsInInt(n)) {
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
    // Every public PackB allocation reserves this extension header, including
    // native-only packs. Its initialized zero prefix means Ops is unavailable.
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

    return header.payload_offset == kPayloadOffset && MaximumPackedSize(header.payload_size) != 0;
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
    if (b == nullptr || packed_b == nullptr ||
        !FitsOpsMatrixSpan(transposed ? n : k, transposed ? k : n, ldb)) {
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

    const size_t maximum_size = MaximumPackedSize(payload_size);
    if (maximum_size == 0) {
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
    header.payload_offset = kPayloadOffset;
    header.payload_size = payload_size;
    if (!CopyKernelName(header, selected_config.filter)) {
        return false;
    }

    std::memset(packed_b, 0, maximum_size);
    KLEIDIAI_KERNEL_LOG("kai::ops pack kernel=" << selected_config.filter << " N=" << n << " K=" << k << " payload_bytes=" << payload_size);
    gemm->pretranspose_B_array(
        static_cast<std::byte*>(packed_b) + kPayloadOffset,
        b,
        static_cast<int>(ldb),
        0,
        transposed
    );
    // Publish the header only after a complete pack has been produced.
    std::memcpy(packed_b, &header, sizeof(header));
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
    bool unpacked_rhs{};
    bool transposed_rhs{};
    int max_threads{};
    std::string requested_filter;
    std::unique_ptr<kai::ops::GemmCommon<Lhs, Rhs, Output>> gemm;
    WorkPartition partition{};
    size_t window_dim0{};
    size_t window_dim1{};
    size_t working_size{};
    size_t rhs_packed_size{};
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
CacheGemmPlan(CachedGemmPlan<Lhs, Rhs, Output>&& new_plan)
{
    new_plan.kernel_name = new_plan.gemm->get_config().filter;
    new_plan.vector_length = CurrentVectorLengthBytes(new_plan.kernel_name);
    const auto window = new_plan.gemm->get_window_size();
    new_plan.window_dim0 = window.get_size(0);
    new_plan.window_dim1 = window.get_size(1);
    if (new_plan.window_dim0 == 0 || new_plan.window_dim1 == 0) {
        return nullptr;
    }
    new_plan.partition = PartitionWindow(window, new_plan.max_threads);
    new_plan.gemm->set_nthreads(static_cast<int>(new_plan.partition.task_count));
    new_plan.working_size = new_plan.gemm->get_working_size();
    new_plan.rhs_packed_size = new_plan.gemm->get_B_pretransposed_array_size();

    auto& cache = GetGemmPlanCache<Lhs, Rhs, Output>();
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
        if (!plan.unpacked_rhs && plan.m == m && plan.n == n && plan.k == k &&
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
    return CacheGemmPlan(std::move(new_plan));
}

template <typename Lhs, typename Rhs, typename Output>
CachedGemmPlan<Lhs, Rhs, Output>*
PrepareUnpackedGemm(size_t m, size_t n, size_t k, bool transposed, bool accumulate, int max_threads)
{
    const std::string requested_filter = RequestedKernelFilter();
    auto& cache = GetGemmPlanCache<Lhs, Rhs, Output>();
    for (auto& plan : cache.entries) {
        if (plan.unpacked_rhs && plan.m == m && plan.n == n && plan.k == k &&
            plan.transposed_rhs == transposed && plan.accumulate == accumulate &&
            plan.max_threads == max_threads && plan.requested_filter == requested_filter &&
            plan.vector_length == CurrentVectorLengthBytes(plan.kernel_name)) {
            return &plan;
        }
    }

    kai::ops::GemmConfig config;
    config.filter = requested_filter;
    auto gemm = SelectGemm<Lhs, Rhs, Output>(m, n, k, &config, accumulate, max_threads);
    // The pinned non-fixed FP32 implementations require RHS packing. Fixed
    // weight formats are not raw RHS layouts and cannot be substituted here.
    if (gemm == nullptr || !gemm->B_is_pretransposed() ||
        (transposed && !gemm->B_pretranspose_supports_transpose())) {
        return nullptr;
    }
    CachedGemmPlan<Lhs, Rhs, Output> new_plan;
    new_plan.m = m;
    new_plan.n = n;
    new_plan.k = k;
    new_plan.accumulate = accumulate;
    new_plan.unpacked_rhs = true;
    new_plan.transposed_rhs = transposed;
    new_plan.max_threads = max_threads;
    new_plan.requested_filter = requested_filter;
    new_plan.gemm = std::move(gemm);
    return CacheGemmPlan(std::move(new_plan));
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
    if (a == nullptr || c == nullptr || !FitsInInt(lda) || !FitsInInt(ldc) ||
        CurrentVectorLengthBytes(plan.kernel_name) != plan.vector_length) {
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
    auto run_task = [&](ptrdiff_t task_id) {
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
    };
    auto& skipped = GetThreadScratch().skipped_tasks;
    skipped.assign(static_cast<size_t>(plan.partition.task_count), 0);
    MlasTrySimpleParallel(thread_pool, plan.partition.task_count, [&](ptrdiff_t task_id) {
        // Worker VL is per-thread. Execute incompatible ranges on the validated
        // caller after the pool completes; replaying completed ranges would
        // double-accumulate when beta=1. Each byte has exactly one writer.
        if (CurrentVectorLengthBytes(plan.kernel_name) != plan.vector_length) {
            skipped[static_cast<size_t>(task_id)] = 1;
            return;
        }
        run_task(task_id);
    });
    for (size_t task_id = 0; task_id < skipped.size(); ++task_id) {
        if (skipped[task_id] != 0) {
            run_task(static_cast<ptrdiff_t>(task_id));
        }
    }
    return true;
}

bool
ExecuteUnpackedGemm(
    size_t m,
    size_t n,
    size_t k,
    const MLAS_SGEMM_DATA_PARAMS& params,
    bool transposed,
    int max_threads,
    MLAS_THREADPOOL* thread_pool
)
{
    if (!FitsOpsMatrixSpan(transposed ? n : k, transposed ? k : n, params.ldb)) {
        return false;
    }
    if (params.alpha == 0.0f) {
        ArmKleidiAI::MlasGemmApplyBeta(params.C, m, n, params.ldc, params.beta);
        return true;
    }

    const bool can_write_directly = params.alpha == 1.0f && (params.beta == 0.0f || params.beta == 1.0f);
    auto* plan = can_write_directly
                     ? PrepareUnpackedGemm<float, float, float>(m, n, k, transposed, params.beta == 1.0f, max_threads)
                     : nullptr;
    const bool temporary_output = plan == nullptr;
    if (temporary_output) {
        plan = PrepareUnpackedGemm<float, float, float>(m, n, k, transposed, false, max_threads);
    }
    if (plan == nullptr || plan->rhs_packed_size == 0) {
        return false;
    }

    size_t output_size = 0;
    if (temporary_output && MlasMultiplyOverflowsSizeT(m, n, &output_size)) {
        return false;
    }
    auto& scratch = GetThreadScratch();
    if (!scratch.packed_rhs.Resize(plan->rhs_packed_size)) {
        return false;
    }
    std::memset(scratch.packed_rhs.Data(), 0, plan->rhs_packed_size);
    // Only the execution plan is cached: a reused B address can contain new
    // values. Pack with this invocation's actual shape and scheduling state.
    plan->gemm->pretranspose_B_array(
        scratch.packed_rhs.Data(), params.B, static_cast<int>(params.ldb), 0, transposed
    );
    KLEIDIAI_KERNEL_LOG(
        "kai::ops transient pack kernel=" << plan->kernel_name << " M=" << m << " N=" << n << " K=" << k
                                          << " payload_bytes=" << plan->rhs_packed_size
    );
    if (temporary_output) {
        scratch.fp32_output.resize(output_size);
    }
    if (!ExecuteGemm(
            *plan, params.A, params.lda,
            temporary_output ? scratch.fp32_output.data() : params.C,
            temporary_output ? n : params.ldc,
            scratch.packed_rhs.Data(), m, n, k, plan->accumulate, "fp32", thread_pool
        )) {
        return false;
    }
    if (temporary_output) {
        ArmKleidiAI::MlasGemmApplyAlphaBeta(
            scratch.fp32_output.data(), m, n, params.alpha, params.beta, params.C, params.ldc, false
        );
    }
    return true;
}

}  // namespace

size_t
    MLASCALL
    ArmKleidiAI::MlasGemmPackBExtensionSizeKleidiAIOps(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K,
        bool UseOps
    )
{
    if (N == 0 || K == 0) {
        return 0;
    }
    // A zero header marks native-only packs, even when Ops is disabled or the
    // input layout is unsupported. Execution can always inspect that header.
    if (!UseOps || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        return kPayloadOffset;
    }
    kai::ops::GemmConfig config;
    size_t payload_size = 0;
    if (!SelectPackedGemm<float, float, float>(N, K, config, payload_size)) {
        return kPayloadOffset;
    }
    const size_t size = MaximumPackedSize(payload_size);
    return size != 0 ? size : kPayloadOffset;
}

bool
    MLASCALL
    ArmKleidiAI::MlasGemmPackBExtensionKleidiAIOps(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K,
        const float* B,
        size_t ldb,
        void* Extension,
        bool UseOps
    )
{
    if (N == 0 || K == 0 || Extension == nullptr) {
        return false;
    }
    const size_t size = MlasGemmPackBExtensionSizeKleidiAIOps(TransA, TransB, N, K, UseOps);
    std::memset(Extension, 0, size);
    return UseOps && TransA == CblasNoTrans && TransB == CblasNoTrans &&
           PackB<float, float, float>(PackedDataType::Fp32, N, K, B, ldb, false, Extension);
}

bool
    MLASCALL
    ArmKleidiAI::MlasGemmPackedBIsKleidiAIOps(const void* PackedB, size_t N, size_t K)
{
    return PackedB != nullptr && N != 0 && K != 0 &&
           IsPackedType(static_cast<const std::byte*>(PackedB) + MlasSgemmNativePackBSize(N, K), PackedDataType::Fp32);
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
    if (M == 0 || N == 0 || BatchSize == 0) {
        return true;
    }
    // False means no output was changed; the dispatcher falls back per entry.
    if (BatchSize != 1 || TransA != CblasNoTrans || K == 0 || Data == nullptr ||
        Data->A == nullptr || Data->C == nullptr ||
        (TransB != CblasNoTrans && TransB != CblasTrans) ||
        !FitsOpsMatrixSpan(M, K, Data->lda) || !FitsOpsMatrixSpan(M, N, Data->ldc)) {
        return false;
    }
    const int max_threads = GetTargetThreadCount(
        ThreadPool, double(M) * double(N) * double(K), double(MLAS_SGEMM_THREAD_COMPLEXITY)
    );
    auto& scratch = GetThreadScratch();
    if (scratch.in_use || Data->B == nullptr) {
        return false;
    }
    ScratchLease lease(scratch);
    if (!Data->BIsPacked) {
        return ExecuteUnpackedGemm(M, N, K, *Data, TransB == CblasTrans, max_threads, ThreadPool);
    }
    const auto* extension = reinterpret_cast<const std::byte*>(Data->B) + MlasSgemmNativePackBSize(N, K);
    PackedHeader header{};
    if (!ReadHeader(extension, PackedDataType::Fp32, N, K, header) ||
        header.trans_b != static_cast<uint32_t>(CblasNoTrans)) {
        return false;
    }

    size_t output_size = 0;
    if (MlasMultiplyOverflowsSizeT(M, N, &output_size)) {
        return false;
    }
    const auto& params = *Data;
    if (params.alpha == 0.0f) {
        ArmKleidiAI::MlasGemmApplyBeta(params.C, M, N, params.ldc, params.beta);
        return true;
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
            const_cast<std::byte*>(extension) + header.payload_offset,
            M, N, K, accumulate, "fp32", ThreadPool
        )) {
        return true;
    }

    // General alpha/beta values need a temporary product. This also provides
    // a correctness fallback if a selected implementation cannot accumulate.
    scratch.fp32_output.resize(output_size);
    plan = PrepareGemm<float, float, float>(header, M, N, K, false, max_threads);
    if (plan == nullptr ||
        !ExecuteGemm(
            *plan, params.A, params.lda,
            scratch.fp32_output.data(), N,
            const_cast<std::byte*>(extension) + header.payload_offset,
            M, N, K, false, "fp32", ThreadPool
        )) {
        return false;
    }
    ArmKleidiAI::MlasGemmApplyAlphaBeta(
        scratch.fp32_output.data(), M, N,
        params.alpha, params.beta, params.C, params.ldc, false
    );
    return true;
}
