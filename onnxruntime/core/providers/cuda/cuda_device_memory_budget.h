#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace onnxruntime {

class SessionBudgetAccount;
class BudgetGroup;

// Physical device ledger and session/group budget accounts (v3 Section 9.1 / F4).
// Supports:
//  - Transactional Reserve -> Commit / Abort
//  - Lock-free Freeing state (BeginFree -> FinishFree) so cudaFree runs outside allocator locks
//    and failed driver frees retain their charge in quarantine rather than leaking accounting
//  - Safe reconfiguration when all prior allocations/pending/freeing bytes are discharged
//  - Independent per-session limits on the same physical device via SessionBudgetAccount
//  - Explicit BudgetGroup for target/head session sharing.
class CudaDeviceMemoryBudget {
 public:
  struct Stats {
    size_t limit;
    size_t committed;
    size_t pending;
    size_t peak;
    size_t freeing{0};
    size_t quarantined{0};
  };

  class Reservation {
   public:
    Reservation() noexcept : budget_(nullptr), bytes_(0) {}
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept
        : budget_(std::exchange(other.budget_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    Reservation& operator=(Reservation&& other) noexcept {
      if (this != &other) {
        if (budget_) budget_->Abort(bytes_);
        budget_ = std::exchange(other.budget_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
      }
      return *this;
    }
    ~Reservation() {
      if (budget_) budget_->Abort(bytes_);
    }

    void Commit() {
      if (!budget_) throw std::logic_error("CUDA budget reservation already consumed");
      budget_->Commit(bytes_);
      budget_ = nullptr;
      bytes_ = 0;
    }

    size_t Bytes() const noexcept { return bytes_; }
    bool Active() const noexcept { return budget_ != nullptr; }

   private:
    friend class CudaDeviceMemoryBudget;
    Reservation(CudaDeviceMemoryBudget& budget, size_t bytes) : budget_(&budget), bytes_(bytes) {}
    CudaDeviceMemoryBudget* budget_;
    size_t bytes_;
  };

  CudaDeviceMemoryBudget() = default;
  CudaDeviceMemoryBudget(const CudaDeviceMemoryBudget&) = delete;
  CudaDeviceMemoryBudget& operator=(const CudaDeviceMemoryBudget&) = delete;
  CudaDeviceMemoryBudget(CudaDeviceMemoryBudget&&) = delete;
  CudaDeviceMemoryBudget& operator=(CudaDeviceMemoryBudget&&) = delete;

  void Configure(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_session_limit_ = (limit > 0) ? limit : std::numeric_limits<size_t>::max();
    if (limit == 0 || limit_ == limit) return;
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (in_use == 0 && active_owners_ <= 1) {
      limit_ = limit;
      return;
    }
    if (!hard_physical_ceiling_) {
      if (limit_ == std::numeric_limits<size_t>::max()) {
        limit_ = std::max(limit, in_use + limit);
      } else {
        // Expand shared physical ledger for additional concurrent session while each session's
        // SessionBudgetAccount enforces its own per-session limit independently (v3 Section 9.1 / F4).
        limit_ = std::max(limit_, in_use) + limit;
      }
    }
  }

  // Strictly configure a hard physical ceiling across all sessions on the device.
  void ConfigurePhysicalCeiling(size_t physical_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t in_use = committed_ + pending_ + freeing_ + quarantined_;
    if (physical_limit < in_use) {
      throw std::runtime_error("Physical device ceiling is smaller than currently committed bytes");
    }
    limit_ = physical_limit;
    hard_physical_ceiling_ = (physical_limit != std::numeric_limits<size_t>::max());
  }

  size_t ConsumePendingSessionLimit() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_session_limit_;
  }

  void ClearPendingSessionLimit() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_session_limit_ = std::numeric_limits<size_t>::max();
  }

  size_t PeekPendingSessionLimit() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_session_limit_;
  }

  void ResetWhenDischarged() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0) {
      limit_ = std::numeric_limits<size_t>::max();
      pending_session_limit_ = std::numeric_limits<size_t>::max();
      hard_physical_ceiling_ = false;
      peak_ = 0;
    }
  }

  void AddOwner() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_owners_;
  }

  void RemoveOwner() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_owners_ > 0) --active_owners_;
    if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0) {
      limit_ = std::numeric_limits<size_t>::max();
      pending_session_limit_ = std::numeric_limits<size_t>::max();
      hard_physical_ceiling_ = false;
    }
  }

  Reservation Reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t used = committed_ + pending_ + freeing_ + quarantined_;
    const size_t remaining = (limit_ >= used) ? (limit_ - used) : 0;
    if (bytes > remaining) {
      throw std::runtime_error("CUDA device memory budget exhausted: requested " + std::to_string(bytes) +
                               " bytes, remaining " + std::to_string(remaining) + " bytes");
    }
    pending_ += bytes;
    return Reservation(*this, bytes);
  }

  // Transition committed bytes into Freeing state before calling cudaFree outside locks (F4).
  void BeginFree(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > committed_) std::terminate();
    committed_ -= bytes;
    freeing_ += bytes;
  }

  // Complete a BeginFree transition after cudaFree returns. If driver_ok is false, retain the charge
  // in quarantined_ rather than dropping accounting (F4).
  void FinishFree(size_t bytes, bool driver_ok) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > freeing_) std::terminate();
    freeing_ -= bytes;
    if (!driver_ok) {
      quarantined_ += bytes;
    }
  }

  void Release(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > committed_) std::terminate();
    committed_ -= bytes;
  }

  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {limit_, committed_ + freeing_ + quarantined_, pending_, peak_, freeing_, quarantined_};
  }

 private:
  void Abort(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ -= bytes;
  }

  void Commit(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ -= bytes;
    committed_ += bytes;
    peak_ = std::max(peak_, committed_ + freeing_ + quarantined_);
  }

  mutable std::mutex mutex_;
  size_t limit_{std::numeric_limits<size_t>::max()};
  size_t pending_session_limit_{std::numeric_limits<size_t>::max()};
  bool hard_physical_ceiling_{false};
  size_t committed_{};
  size_t pending_{};
  size_t freeing_{};
  size_t quarantined_{};
  size_t peak_{};
  size_t active_owners_{};
};

inline std::shared_ptr<CudaDeviceMemoryBudget> GetCudaDeviceMemoryBudgetByOrdinal(int device_id, size_t limit = 0) {
  static std::mutex registry_mutex;
  static std::unordered_map<int, std::shared_ptr<CudaDeviceMemoryBudget>> budgets;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto& budget = budgets[device_id];
  if (!budget) budget = std::make_shared<CudaDeviceMemoryBudget>();
  if (limit > 0) {
    budget->Configure(limit);
  }
  return budget;
}

// Explicit group budget account for related sessions such as target + MTP head (v3 Section 9.1 / F4).
class BudgetGroup {
 public:
  explicit BudgetGroup(std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger,
                       size_t group_limit = std::numeric_limits<size_t>::max())
      : physical_ledger_(std::move(physical_ledger)),
        group_budget_() {
    if (group_limit != 0 && group_limit != std::numeric_limits<size_t>::max()) {
      group_budget_.Configure(group_limit);
    }
    if (physical_ledger_) physical_ledger_->AddOwner();
  }

  ~BudgetGroup() {
    if (physical_ledger_) physical_ledger_->RemoveOwner();
  }

  BudgetGroup(const BudgetGroup&) = delete;
  BudgetGroup& operator=(const BudgetGroup&) = delete;

  CudaDeviceMemoryBudget& GroupBudget() noexcept { return group_budget_; }
  const std::shared_ptr<CudaDeviceMemoryBudget>& PhysicalLedger() const noexcept { return physical_ledger_; }
  CudaDeviceMemoryBudget::Stats GetGroupStats() const { return group_budget_.GetStats(); }

 private:
  std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger_;
  CudaDeviceMemoryBudget group_budget_;
};

// Explicit session/group budget account layered over a physical ledger and optional BudgetGroup (v3 Section 9.1 / F4).
class SessionBudgetAccount {
 public:
  explicit SessionBudgetAccount(std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger,
                                size_t session_limit = std::numeric_limits<size_t>::max(),
                                std::shared_ptr<BudgetGroup> group = nullptr)
      : physical_ledger_(std::move(physical_ledger)),
        group_(std::move(group)),
        local_budget_() {
    if (session_limit != 0 && session_limit != std::numeric_limits<size_t>::max()) {
      local_budget_.Configure(session_limit);
    }
    if (physical_ledger_) physical_ledger_->AddOwner();
  }

  ~SessionBudgetAccount() {
    if (physical_ledger_) physical_ledger_->RemoveOwner();
  }

  SessionBudgetAccount(const SessionBudgetAccount&) = delete;
  SessionBudgetAccount& operator=(const SessionBudgetAccount&) = delete;

  void ConfigureSessionLimit(size_t session_limit) {
    if (session_limit != 0 && session_limit != std::numeric_limits<size_t>::max()) {
      local_budget_.Configure(session_limit);
    }
  }

  class CombinedReservation {
   public:
    CombinedReservation() = default;
    CombinedReservation(CudaDeviceMemoryBudget::Reservation local_res,
                        CudaDeviceMemoryBudget::Reservation group_res,
                        CudaDeviceMemoryBudget::Reservation phys_res)
        : local_res_(std::move(local_res)),
          group_res_(std::move(group_res)),
          phys_res_(std::move(phys_res)) {}

    void Commit() {
      if (local_res_.Active()) local_res_.Commit();
      if (group_res_.Active()) group_res_.Commit();
      if (phys_res_.Active()) phys_res_.Commit();
    }

   private:
    CudaDeviceMemoryBudget::Reservation local_res_;
    CudaDeviceMemoryBudget::Reservation group_res_;
    CudaDeviceMemoryBudget::Reservation phys_res_;
  };

  CombinedReservation Reserve(size_t bytes) {
    auto local_res = local_budget_.Reserve(bytes);
    auto group_res = group_ ? group_->GroupBudget().Reserve(bytes) : CudaDeviceMemoryBudget::Reservation{};
    auto phys_res = physical_ledger_ ? physical_ledger_->Reserve(bytes) : CudaDeviceMemoryBudget::Reservation{};
    return CombinedReservation(std::move(local_res), std::move(group_res), std::move(phys_res));
  }

  void BeginFree(size_t bytes) noexcept {
    local_budget_.BeginFree(bytes);
    if (group_) group_->GroupBudget().BeginFree(bytes);
    if (physical_ledger_) physical_ledger_->BeginFree(bytes);
  }

  void FinishFree(size_t bytes, bool driver_ok) noexcept {
    local_budget_.FinishFree(bytes, driver_ok);
    if (group_) group_->GroupBudget().FinishFree(bytes, driver_ok);
    if (physical_ledger_) physical_ledger_->FinishFree(bytes, driver_ok);
  }

  CudaDeviceMemoryBudget::Stats GetAccountStats() const { return local_budget_.GetStats(); }
  CudaDeviceMemoryBudget::Stats GetGroupStats() const {
    return group_ ? group_->GetGroupStats() : local_budget_.GetStats();
  }
  CudaDeviceMemoryBudget::Stats GetPhysicalStats() const {
    return physical_ledger_ ? physical_ledger_->GetStats() : local_budget_.GetStats();
  }

 private:
  std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger_;
  std::shared_ptr<BudgetGroup> group_;
  CudaDeviceMemoryBudget local_budget_;
};

// Host pinned and registered range ledger (v3 Section 9.2).
// Deduplicates registered host ranges across target/head sessions so shared initializers are charged once.
class HostPinnedBudgetLedger {
 public:
  CudaDeviceMemoryBudget::Reservation ReserveAllocated(size_t bytes) {
    return budget_.Reserve(bytes);
  }

  void BeginFreeAllocated(size_t bytes) noexcept {
    budget_.BeginFree(bytes);
  }

  void FinishFreeAllocated(size_t bytes, bool driver_ok = true) noexcept {
    budget_.FinishFree(bytes, driver_ok);
  }

  // Register a host pointer range; returns true if newly registered in the ledger, false if already tracked.
  bool RegisterHostRange(const void* base, size_t bytes) {
    if (!base || bytes == 0) return false;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto& entry = registered_ranges_[base];
    if (entry.ref_count > 0) {
      ++entry.ref_count;
      return false;
    }
    auto res = budget_.Reserve(bytes);
    res.Commit();
    entry.bytes = bytes;
    entry.ref_count = 1;
    return true;
  }

  // Decrements ref count for a registered range; returns true when the last owner releases and cudaHostUnregister should run.
  bool UnregisterHostRange(const void* base, bool driver_ok = true) noexcept {
    if (!base) return false;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = registered_ranges_.find(base);
    if (it == registered_ranges_.end()) return false;
    if (--it->second.ref_count > 0) {
      return false;
    }
    const size_t bytes = it->second.bytes;
    registered_ranges_.erase(it);
    budget_.BeginFree(bytes);
    budget_.FinishFree(bytes, driver_ok);
    return true;
  }

  void QuarantineHostRange(const void* base) noexcept {
    if (!base) return;
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = registered_ranges_.find(base);
    if (it == registered_ranges_.end()) return;
    const size_t bytes = it->second.bytes;
    registered_ranges_.erase(it);
    budget_.BeginFree(bytes);
    budget_.FinishFree(bytes, false);
  }

  CudaDeviceMemoryBudget::Stats GetStats() const { return budget_.GetStats(); }

 private:
  struct RangeEntry {
    size_t bytes = 0;
    size_t ref_count = 0;
  };
  mutable std::mutex reg_mutex_;
  std::unordered_map<const void*, RangeEntry> registered_ranges_;
  CudaDeviceMemoryBudget budget_;
};

inline std::shared_ptr<HostPinnedBudgetLedger> GetCudaHostPinnedMemoryBudget() {
  static auto ledger = std::make_shared<HostPinnedBudgetLedger>();
  return ledger;
}

}  // namespace onnxruntime
