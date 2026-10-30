// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Session admission tests for the QMoE offload planner. They drive QMoESessionController with mock offload nodes that
// only report byte sizes, so they need neither a model nor expert weights nor any large allocation.

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <vector>
#endif

#include "gtest/gtest.h"

#include "core/common/status.h"
#include "core/framework/moe_offload_plan.h"
#include "core/providers/cuda/qmoe_session_controller.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

using moe_offload::BudgetSpec;
using moe_offload::MoeOffloadSessionPolicy;

constexpr uint64_t kFixedTableBytes = 64;
constexpr uint64_t kDeviceSlotBytes = 100;
constexpr uint64_t kHostSlotBytes = 1000;

// Reports a 64 byte lookup table, 100 bytes per device expert slot and 1000 bytes per host expert.
class MockOffloadNode final : public moe_offload::IMoeOffloadNode {
 public:
  MockOffloadNode(int32_t layer_index, int32_t num_experts) {
    descriptor_.stable_node_id = "moe_" + std::to_string(layer_index);
    descriptor_.layer_index = layer_index;
    descriptor_.num_experts = num_experts;
    descriptor_.hidden_size = 16;
    descriptor_.inter_size = 32;
    capabilities_.gpu_fixed_table_bytes = kFixedTableBytes;
    capabilities_.gpu_slot_bytes = kDeviceSlotBytes;
    capabilities_.cpu_slot_bytes = kHostSlotBytes;
  }

  const std::string& Name() const { return descriptor_.stable_node_id; }
  moe_offload::MoeExpertSourceDescriptor& Descriptor() { return descriptor_; }
  moe_offload::MoeBackendCapabilities& Capabilities() { return capabilities_; }
  const moe_offload::MoeNodeExecutionPlan& Plan() const { return plan_; }
  int InitializeCalls() const { return initialize_calls_; }

  const moe_offload::MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept override { return descriptor_; }
  const moe_offload::MoeBackendCapabilities& GetBackendCapabilities() const noexcept override { return capabilities_; }
  common::Status InitializeAdmittedPlan(const MoeOffloadSessionPolicy&,
                                        const moe_offload::MoeNodeExecutionPlan& plan) override {
    plan_ = plan;
    ++initialize_calls_;
    return common::Status::OK();
  }

 private:
  moe_offload::MoeExpertSourceDescriptor descriptor_;
  moe_offload::MoeBackendCapabilities capabilities_;
  moe_offload::MoeNodeExecutionPlan plan_;
  int initialize_calls_ = 0;
};

// Static placement with an explicit device budget gives a deterministic plan that does not depend on free GPU memory.
MoeOffloadSessionPolicy StaticPolicy(uint64_t device_expert_bytes) {
  MoeOffloadSessionPolicy policy;
  policy.mode = moe_offload::OffloadMode::kRequired;
  policy.placement = moe_offload::PlacementPolicy::kStatic;
  policy.device_expert_bytes = BudgetSpec{false, device_expert_bytes};
  return policy;
}

void AddNodeOverride(MoeOffloadSessionPolicy& policy, const std::string& node_name,
                     std::optional<uint64_t> device_expert_bytes,
                     std::optional<int32_t> max_resident_slots = std::nullopt) {
  moe_offload::NodeOverrideSpec spec;
  spec.node_name = node_name;
  spec.device_expert_bytes = device_expert_bytes;
  spec.max_resident_slots = max_resident_slots;
  policy.node_overrides.push_back(std::move(spec));
}

common::Status PlanSession(const MoeOffloadSessionPolicy& policy, std::initializer_list<MockOffloadNode*> nodes) {
  QMoESessionController controller(/*device_id=*/0);
  for (MockOffloadNode* node : nodes) {
    int32_t layer_index = node->GetSourceDescriptor().layer_index;
    controller.RegisterMoeOffloadNode(node, policy, layer_index);
  }
  return controller.PlanAndAdmitSession();
}

void ExpectAdmissionFailure(const common::Status& status, const std::string& expected_message) {
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find(expected_message), std::string::npos)
      << "expected \"" << expected_message << "\" in: " << status.ErrorMessage();
}

// 8 experts with a device budget for exactly 2 of them: 6 cold experts, i.e. 6000 host bytes.
constexpr int32_t kPartialExperts = 8;
constexpr uint64_t kTwoSlotDeviceBudget = kFixedTableBytes + 2 * kDeviceSlotBytes;
constexpr uint64_t kColdExpertHostBytes = (kPartialExperts - 2) * kHostSlotBytes;

MoeOffloadSessionPolicy PartiallyResidentPolicy(uint64_t host_expert_bytes) {
  MoeOffloadSessionPolicy policy = StaticPolicy(kTwoSlotDeviceBudget);
  policy.host_expert_bytes = BudgetSpec{false, host_expert_bytes};
  return policy;
}

void EnableBoundedHostCache(MockOffloadNode& node, size_t min_host_cache_slots) {
  node.Capabilities().supports_file_backed_bounded_host_cache = true;
  node.Capabilities().min_host_cache_slots = min_host_cache_slots;
}

}  // namespace

TEST(QMoEOffloadAdmissionTest, NodeDeviceCapBelowFixedTableOverheadIsRejected) {
  for (const uint64_t cap : {uint64_t{0}, uint64_t{1}, kFixedTableBytes - 1}) {
    SCOPED_TRACE(cap);
    MockOffloadNode node(0, 4);
    MoeOffloadSessionPolicy policy = StaticPolicy(1000);
    AddNodeOverride(policy, node.Name(), cap);

    ExpectAdmissionFailure(PlanSession(policy, {&node}),
                           "smaller than the fixed lookup table and dummy slot overhead (64 bytes) of node 'moe_0'");
    EXPECT_EQ(node.InitializeCalls(), 0);
  }
}

TEST(QMoEOffloadAdmissionTest, NodeDeviceCapEqualToFixedTableOverheadAdmitsZeroResidentSlots) {
  MockOffloadNode node(0, 4);
  MoeOffloadSessionPolicy policy = StaticPolicy(1000);
  AddNodeOverride(policy, node.Name(), kFixedTableBytes);

  const common::Status status = PlanSession(policy, {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(node.InitializeCalls(), 1);
  EXPECT_EQ(node.Plan().resident_slots, 0);
  EXPECT_EQ(node.Plan().admitted_gpu_expert_bytes, kFixedTableBytes);
}

TEST(QMoEOffloadAdmissionTest, NodeDeviceCapAboveFixedTableOverheadAdmitsOnlyWholeSlots) {
  struct Case {
    uint64_t cap;
    int32_t expected_resident_slots;
  };
  // The 64 byte fixed table is paid first; each further 100 bytes buys one slot, up to the node's 4 experts.
  for (const Case& c : {Case{163, 0}, Case{164, 1}, Case{263, 1}, Case{264, 2}, Case{1064, 4}, Case{uint64_t{1} << 20, 4}}) {
    SCOPED_TRACE(c.cap);
    MockOffloadNode node(0, 4);
    MoeOffloadSessionPolicy policy = StaticPolicy(1000);
    AddNodeOverride(policy, node.Name(), c.cap);

    const common::Status status = PlanSession(policy, {&node});
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(node.Plan().resident_slots, c.expected_resident_slots);
    EXPECT_EQ(node.Plan().admitted_gpu_expert_bytes,
              kFixedTableBytes + static_cast<uint64_t>(c.expected_resident_slots) * kDeviceSlotBytes);
  }
}

TEST(QMoEOffloadAdmissionTest, NodeDeviceCapCoversDummySlotWhenDecodeKernelCannotSkipNegativeSlots) {
  constexpr uint64_t kBaseBytes = kFixedTableBytes + kDeviceSlotBytes;

  {
    MockOffloadNode node(0, 4);
    node.Capabilities().decode_kernel_skips_negative_slots = false;
    MoeOffloadSessionPolicy policy = StaticPolicy(1000);
    AddNodeOverride(policy, node.Name(), kBaseBytes - 1);
    ExpectAdmissionFailure(PlanSession(policy, {&node}), "dummy slot overhead (164 bytes) of node 'moe_0'");
    EXPECT_EQ(node.InitializeCalls(), 0);
  }
  {
    MockOffloadNode node(0, 4);
    node.Capabilities().decode_kernel_skips_negative_slots = false;
    MoeOffloadSessionPolicy policy = StaticPolicy(1000);
    AddNodeOverride(policy, node.Name(), kBaseBytes);
    const common::Status status = PlanSession(policy, {&node});
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(node.Plan().dummy_slots, 1);
    EXPECT_EQ(node.Plan().resident_slots, 0);
    EXPECT_EQ(node.Plan().admitted_gpu_expert_bytes, kBaseBytes);
  }
  {
    MockOffloadNode node(0, 4);
    node.Capabilities().decode_kernel_skips_negative_slots = false;
    MoeOffloadSessionPolicy policy = StaticPolicy(1000);
    AddNodeOverride(policy, node.Name(), kBaseBytes + kDeviceSlotBytes);
    const common::Status status = PlanSession(policy, {&node});
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(node.Plan().dummy_slots, 1);
    EXPECT_EQ(node.Plan().resident_slots, 1);
    EXPECT_EQ(node.Plan().admitted_gpu_expert_bytes, kBaseBytes + kDeviceSlotBytes);
  }
}

TEST(QMoEOffloadAdmissionTest, NodeDeviceCapLimitsOnlyThatNode) {
  MockOffloadNode capped(0, 4);
  MockOffloadNode open(1, 4);
  // Two tables and five slots: the capped node takes one slot and the other node takes the remaining four.
  MoeOffloadSessionPolicy policy = StaticPolicy(2 * kFixedTableBytes + 5 * kDeviceSlotBytes);
  AddNodeOverride(policy, capped.Name(), kFixedTableBytes + kDeviceSlotBytes);

  const common::Status status = PlanSession(policy, {&capped, &open});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(capped.Plan().resident_slots, 1);
  EXPECT_EQ(capped.Plan().admitted_gpu_expert_bytes, kFixedTableBytes + kDeviceSlotBytes);
  EXPECT_EQ(open.Plan().resident_slots, 4);
  EXPECT_EQ(open.Plan().admitted_gpu_expert_bytes, kFixedTableBytes + 4 * kDeviceSlotBytes);
}

TEST(QMoEOffloadAdmissionTest, DeviceBudgetBelowFixedTableOverheadIsRejected) {
  MockOffloadNode node(0, 4);
  ExpectAdmissionFailure(PlanSession(StaticPolicy(kFixedTableBytes - 1), {&node}),
                         "device_expert_bytes (63) is smaller than fixed lookup table and dummy slot overhead "
                         "(64 bytes across 1 layers)");
  EXPECT_EQ(node.InitializeCalls(), 0);
}

TEST(QMoEOffloadAdmissionTest, ExplicitZeroDeviceBudgetInRequiredModeAdmitsFixedOverheadOnly) {
  MockOffloadNode node(0, 4);
  const common::Status status = PlanSession(StaticPolicy(0), {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(node.Plan().resident_slots, 0);
  EXPECT_EQ(node.Plan().admitted_gpu_expert_bytes, kFixedTableBytes);
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCapRequiresBackendSupport) {
  MockOffloadNode node(0, kPartialExperts);
  ASSERT_FALSE(node.Capabilities().supports_file_backed_bounded_host_cache);

  ExpectAdmissionFailure(PlanSession(PartiallyResidentPolicy(4000), {&node}),
                         "requires a file-backed bounded host cache, which node 'moe_0' does not support");
  EXPECT_EQ(node.InitializeCalls(), 0);
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCapBelowMinimumCacheCapacityIsRejected) {
  MockOffloadNode node(0, kPartialExperts);
  EnableBoundedHostCache(node, /*min_host_cache_slots=*/3);

  ExpectAdmissionFailure(PlanSession(PartiallyResidentPolicy(3 * kHostSlotBytes - 1), {&node}),
                         "smaller than minimum bounded host cache capacity (3000 bytes across 1 partially "
                         "offloaded layers)");
  EXPECT_EQ(node.InitializeCalls(), 0);
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCapAtMinimumCacheCapacityAdmitsMinimumSlots) {
  MockOffloadNode node(0, kPartialExperts);
  EnableBoundedHostCache(node, /*min_host_cache_slots=*/3);

  const common::Status status = PlanSession(PartiallyResidentPolicy(3 * kHostSlotBytes), {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(node.Plan().resident_slots, 2);
  EXPECT_TRUE(node.Plan().use_file_backing);
  EXPECT_EQ(node.Plan().host_cache_slots, 3);
  EXPECT_EQ(node.Plan().host_cache_bytes, 3 * kHostSlotBytes);
  EXPECT_EQ(node.Plan().admitted_host_expert_bytes, 3 * kHostSlotBytes);
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCapBeyondMinimumIsSpentOnWholeCacheSlots) {
  struct Case {
    uint64_t host_expert_bytes;
    int32_t expected_cache_slots;
  };
  // With a minimum of 2 slots, every further 1000 bytes of the cap buys one more cached expert.
  for (const Case& c : {Case{2000, 2}, Case{2999, 2}, Case{4000, 4}, Case{5500, 5}}) {
    SCOPED_TRACE(c.host_expert_bytes);
    MockOffloadNode node(0, kPartialExperts);
    EnableBoundedHostCache(node, /*min_host_cache_slots=*/2);

    const common::Status status = PlanSession(PartiallyResidentPolicy(c.host_expert_bytes), {&node});
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_TRUE(node.Plan().use_file_backing);
    EXPECT_EQ(node.Plan().host_cache_slots, c.expected_cache_slots);
    EXPECT_EQ(node.Plan().host_cache_bytes, static_cast<uint64_t>(c.expected_cache_slots) * kHostSlotBytes);
    EXPECT_LE(node.Plan().host_cache_bytes, c.host_expert_bytes);
  }
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCacheMinimumIsClampedToTheColdExpertsOfEachNode) {
  // Node 0 keeps 2 of 8 experts (6 cold) and node 1 keeps 2 of 3 (1 cold). Both ask for at least 4 cache slots, but
  // node 1 can never need more than its single cold expert, so 4 + 1 slots (5000 bytes) is enough for the session.
  MockOffloadNode wide(0, 8);
  MockOffloadNode narrow(1, 3);
  EnableBoundedHostCache(wide, /*min_host_cache_slots=*/4);
  EnableBoundedHostCache(narrow, /*min_host_cache_slots=*/4);

  MoeOffloadSessionPolicy policy = StaticPolicy(2 * kFixedTableBytes + 4 * kDeviceSlotBytes);
  policy.host_expert_bytes = BudgetSpec{false, 5 * kHostSlotBytes};
  AddNodeOverride(policy, wide.Name(), std::nullopt, /*max_resident_slots=*/2);
  AddNodeOverride(policy, narrow.Name(), std::nullopt, /*max_resident_slots=*/2);

  const common::Status status = PlanSession(policy, {&wide, &narrow});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(wide.Plan().resident_slots, 2);
  EXPECT_EQ(narrow.Plan().resident_slots, 2);
  EXPECT_EQ(wide.Plan().host_cache_slots, 4);
  EXPECT_EQ(narrow.Plan().host_cache_slots, 1);
  EXPECT_EQ(wide.Plan().host_cache_bytes + narrow.Plan().host_cache_bytes, 5 * kHostSlotBytes);
}

TEST(QMoEOffloadAdmissionTest, BoundedHostCacheReservesEachNodesMinimumBeforeSharingTheRest) {
  // Both nodes have 6 cold experts and the cap holds exactly the two minimums (4 + 1 slots), so nothing is left to
  // spread. Sharing the cap evenly would leave the first node below the 4 slots its backend needs.
  MockOffloadNode demanding(0, 8);
  MockOffloadNode relaxed(1, 8);
  EnableBoundedHostCache(demanding, /*min_host_cache_slots=*/4);
  EnableBoundedHostCache(relaxed, /*min_host_cache_slots=*/1);

  MoeOffloadSessionPolicy policy = StaticPolicy(2 * kFixedTableBytes + 4 * kDeviceSlotBytes);
  policy.host_expert_bytes = BudgetSpec{false, 5 * kHostSlotBytes};
  AddNodeOverride(policy, demanding.Name(), std::nullopt, /*max_resident_slots=*/2);
  AddNodeOverride(policy, relaxed.Name(), std::nullopt, /*max_resident_slots=*/2);

  const common::Status status = PlanSession(policy, {&demanding, &relaxed});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(demanding.Plan().host_cache_slots, 4);
  EXPECT_EQ(relaxed.Plan().host_cache_slots, 1);
  EXPECT_EQ(demanding.Plan().host_cache_bytes + relaxed.Plan().host_cache_bytes, 5 * kHostSlotBytes);
}

TEST(QMoEOffloadAdmissionTest, FullyResidentNodesDoNotNeedBoundedHostCacheSupport) {
  // Node 0 ends up fully resident (2 of 2) so only node 1 (2 of 8) has cold experts and needs the bounded cache.
  MockOffloadNode resident(0, 2);
  MockOffloadNode partial(1, kPartialExperts);
  ASSERT_FALSE(resident.Capabilities().supports_file_backed_bounded_host_cache);
  EnableBoundedHostCache(partial, /*min_host_cache_slots=*/2);

  MoeOffloadSessionPolicy policy = StaticPolicy(2 * kFixedTableBytes + 4 * kDeviceSlotBytes);
  policy.host_expert_bytes = BudgetSpec{false, 4 * kHostSlotBytes};

  const common::Status status = PlanSession(policy, {&resident, &partial});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(resident.Plan().resident_slots, 2);
  EXPECT_FALSE(resident.Plan().use_file_backing);
  EXPECT_EQ(partial.Plan().resident_slots, 2);
  EXPECT_TRUE(partial.Plan().use_file_backing);
  EXPECT_EQ(partial.Plan().host_cache_slots, 4);
}

TEST(QMoEOffloadAdmissionTest, HostCapCoveringTheColdComplementNeedsNoBoundedCache) {
  for (const uint64_t host_expert_bytes : {kColdExpertHostBytes, kColdExpertHostBytes + 1, 2 * kColdExpertHostBytes}) {
    SCOPED_TRACE(host_expert_bytes);
    MockOffloadNode node(0, kPartialExperts);
    ASSERT_FALSE(node.Capabilities().supports_file_backed_bounded_host_cache);

    const common::Status status = PlanSession(PartiallyResidentPolicy(host_expert_bytes), {&node});
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(node.Plan().resident_slots, 2);
    EXPECT_FALSE(node.Plan().use_file_backing);
    EXPECT_EQ(node.Plan().host_cache_slots, 0);
    EXPECT_EQ(node.Plan().admitted_host_expert_bytes, kColdExpertHostBytes);
  }
}

TEST(QMoEOffloadAdmissionTest, AutoDeviceBudgetDoesNotWrapWhenReservesOverflow) {
  uint64_t usable_free_bytes = 0;
  QMoESessionController::QueryPlatformUsableDeviceMemory(/*device_id=*/0, &usable_free_bytes);
  if (usable_free_bytes < (uint64_t{1} << 20)) {
    GTEST_SKIP() << "No usable CUDA device memory to derive an automatic device budget from.";
  }

  struct Case {
    uint64_t reserve_device_bytes;
    uint64_t device_working_bytes;
  };
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  // Each pair sums to a value that wraps around uint64_t (to 1, 1 and 0), which used to turn an impossible reserve
  // into a tiny one and admitted the whole free device memory.
  for (const Case& c : {Case{kMax, 2}, Case{2, kMax}, Case{uint64_t{1} << 63, uint64_t{1} << 63}}) {
    SCOPED_TRACE(::testing::Message() << "reserve=" << c.reserve_device_bytes << " working=" << c.device_working_bytes);
    MockOffloadNode node(0, 4);
    MoeOffloadSessionPolicy policy;
    policy.mode = moe_offload::OffloadMode::kRequired;
    policy.placement = moe_offload::PlacementPolicy::kStatic;
    policy.reserve_device_bytes = c.reserve_device_bytes;
    policy.device_working_bytes = c.device_working_bytes;

    ExpectAdmissionFailure(PlanSession(policy, {&node}),
                           "device_expert_bytes (0) is smaller than fixed lookup table and dummy slot overhead");
    EXPECT_EQ(node.InitializeCalls(), 0);
  }
}

TEST(QMoEOffloadAdmissionTest, AutoDeviceBudgetUsesFreeDeviceMemoryWhenNothingIsReserved) {
  uint64_t usable_free_bytes = 0;
  QMoESessionController::QueryPlatformUsableDeviceMemory(/*device_id=*/0, &usable_free_bytes);
  if (usable_free_bytes < (uint64_t{1} << 20)) {
    GTEST_SKIP() << "No usable CUDA device memory to derive an automatic device budget from.";
  }

  MockOffloadNode node(0, 4);
  MoeOffloadSessionPolicy policy;
  policy.mode = moe_offload::OffloadMode::kRequired;
  policy.placement = moe_offload::PlacementPolicy::kStatic;
  policy.reserve_device_bytes = 0;
  policy.device_working_bytes = 0;

  const common::Status status = PlanSession(policy, {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(node.Plan().resident_slots, 4);
}

#if defined(__linux__)
namespace {

size_t PageSize() {
  const long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 4096u;
}

// A file of `bytes` copies of `fill` in the temp directory, removed on destruction.
class ScopedTempFile {
 public:
  ScopedTempFile(const std::string& tag, size_t bytes, uint8_t fill)
      : path_(std::filesystem::temp_directory_path() /
              ("ort_qmoe_admission_" + tag + "_" + std::to_string(getpid()) + ".bin")) {
    {
      std::ofstream out(path_, std::ios::binary | std::ios::trunc);
      const std::vector<char> content(bytes, static_cast<char>(fill));
      out.write(content.data(), static_cast<std::streamsize>(content.size()));
      if (!out.good()) return;
    }
    fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  }

  ~ScopedTempFile() {
    if (fd_ >= 0) close(fd_);
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  ScopedTempFile(const ScopedTempFile&) = delete;
  ScopedTempFile& operator=(const ScopedTempFile&) = delete;

  int fd() const { return fd_; }

 private:
  std::filesystem::path path_;
  int fd_ = -1;
};

class ScopedMapping {
 public:
  ScopedMapping(int fd, size_t bytes, int prot, int flags)
      : bytes_(bytes), data_(mmap(nullptr, bytes, prot, flags, fd, 0)) {}

  ~ScopedMapping() {
    if (data_ != MAP_FAILED) munmap(data_, bytes_);
  }

  ScopedMapping(const ScopedMapping&) = delete;
  ScopedMapping& operator=(const ScopedMapping&) = delete;

  bool ok() const { return data_ != MAP_FAILED; }
  void* data() const { return data_; }

 private:
  size_t bytes_;
  void* data_;
};

// Whether the page holding `address` is currently mapped into this process, from /proc/self/pagemap.
std::optional<bool> IsPageMapped(const void* address) {
  const int pagemap_fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (pagemap_fd < 0) return std::nullopt;
  uint64_t entry = 0;
  const off_t offset = static_cast<off_t>(reinterpret_cast<uintptr_t>(address) / PageSize() * sizeof(entry));
  const ssize_t bytes_read = pread(pagemap_fd, &entry, sizeof(entry), offset);
  close(pagemap_fd);
  if (bytes_read != static_cast<ssize_t>(sizeof(entry))) return std::nullopt;
  return ((entry >> 63) & 1ULL) != 0;
}

void SetFc1WeightBank(MockOffloadNode& node, const void* host_data, size_t bytes_per_expert, size_t total_bytes) {
  moe_offload::MoeExpertBankSlice bank;
  bank.kind = moe_offload::ExpertBankKind::kFc1Weights;
  bank.bytes_per_expert = bytes_per_expert;
  bank.host_data = host_data;
  bank.total_bytes = total_bytes;
  node.Descriptor().banks[static_cast<size_t>(moe_offload::ExpertBankKind::kFc1Weights)] = std::move(bank);
}

constexpr int32_t kMappedExperts = 3;
constexpr uint8_t kFileByte = 0x11;
constexpr uint8_t kPrivateByte = 0x22;

}  // namespace

// With static placement and every expert resident, the planner discards the host pages of the uploaded experts. That
// is only valid for pages that can be reloaded from the file; a private modification would silently be reverted.
TEST(QMoEOffloadAdmissionTest, FullyResidentPlanKeepsPrivatelyModifiedHostPages) {
  const size_t page_size = PageSize();
  const size_t total_bytes = page_size * kMappedExperts;
  ScopedTempFile file("modified", total_bytes, kFileByte);
  ASSERT_GE(file.fd(), 0);
  ScopedMapping mapping(file.fd(), total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE);
  ASSERT_TRUE(mapping.ok());

  auto* bytes = static_cast<uint8_t*>(mapping.data());
  const size_t modified_offset = page_size + 7;
  bytes[modified_offset] = kPrivateByte;
  ASSERT_EQ(mprotect(mapping.data(), total_bytes, PROT_READ), 0);

  MockOffloadNode node(0, kMappedExperts);
  SetFc1WeightBank(node, mapping.data(), page_size, total_bytes);
  const common::Status status =
      PlanSession(StaticPolicy(kFixedTableBytes + kMappedExperts * kDeviceSlotBytes), {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(node.Plan().resident_slots, kMappedExperts);

  EXPECT_EQ(bytes[modified_offset], kPrivateByte);
  EXPECT_EQ(bytes[0], kFileByte);
}

TEST(QMoEOffloadAdmissionTest, FullyResidentPlanDiscardsHostPagesOfCleanSharedMapping) {
  const size_t page_size = PageSize();
  const size_t total_bytes = page_size * kMappedExperts;
  ScopedTempFile file("clean", total_bytes, kFileByte);
  ASSERT_GE(file.fd(), 0);
  ScopedMapping mapping(file.fd(), total_bytes, PROT_READ, MAP_SHARED);
  ASSERT_TRUE(mapping.ok());

  // Fault every page in so that the planner discarding them is observable.
  const auto* bytes = static_cast<const uint8_t*>(mapping.data());
  uint8_t checksum = 0;
  for (int32_t page = 0; page < kMappedExperts; ++page) {
    checksum ^= bytes[static_cast<size_t>(page) * page_size];
  }
  EXPECT_EQ(checksum, kFileByte);
  for (int32_t page = 0; page < kMappedExperts; ++page) {
    const std::optional<bool> mapped = IsPageMapped(bytes + static_cast<size_t>(page) * page_size);
    if (!mapped.has_value()) GTEST_SKIP() << "/proc/self/pagemap is not readable.";
    ASSERT_TRUE(*mapped);
  }

  MockOffloadNode node(0, kMappedExperts);
  SetFc1WeightBank(node, mapping.data(), page_size, total_bytes);
  const common::Status status =
      PlanSession(StaticPolicy(kFixedTableBytes + kMappedExperts * kDeviceSlotBytes), {&node});
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(node.Plan().resident_slots, kMappedExperts);

  for (int32_t page = 0; page < kMappedExperts; ++page) {
    const std::optional<bool> mapped = IsPageMapped(bytes + static_cast<size_t>(page) * page_size);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_FALSE(*mapped) << "page " << page;
  }
  EXPECT_EQ(bytes[0], kFileByte);
}
#endif  // defined(__linux__)

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
