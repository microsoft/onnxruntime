#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace onnxruntime {

// Physical device ledger and session/group budget accounts (v3 Section 9.1 / F4).
// Supports:
//  - Transactional Reserve -> Commit / Abort
//  - Lock-free Freeing state (MarkFreeing -> CompleteFree) so cudaFree runs outside allocator locks
//    and failed driver frees retain their charge in quarantine rather than leaking accounting
//  - Safe reconfiguration when all prior allocations/pending/freeing bytes are discharged
//  - Explicit SessionBudgetAccount and BudgetGroup for target/head session sharing.
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
    if (limit == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (limit_ == limit) return;
    if (committed_ != 0 || pending_ != 0 || freeing_ != 0 || quarantined_ != 0) {
      throw std::runtime_error("Conflicting or late gpu_device_mem_limit for CUDA device");
    }
    // If a previous limit was set and active owners still hold the ledger, reject conflicting limits;
    // when discharged (no active allocations and <= 1 owner configuring), allow reconfiguration (F4).
    if (limit_ != std::numeric_limits<size_t>::max() && active_owners_ > 1) {
      throw std::runtime_error("Conflicting or late gpu_device_mem_limit for CUDA device");
    }
    limit_ = limit;
  }

  void ResetWhenDischarged() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_ == 0 && pending_ == 0 && freeing_ == 0 && quarantined_ == 0 && active_owners_ == 0) {
      limit_ = std::numeric_limits<size_t>::max();
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
  size_t committed_{};
  size_t pending_{};
  size_t freeing_{};
  size_t quarantined_{};
  size_t peak_{};
  size_t active_owners_{};
};

// Explicit session/group budget account layered over a physical ledger (v3 Section 9.1 / F4).
class SessionBudgetAccount {
 public:
  explicit SessionBudgetAccount(std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger,
                                size_t session_limit = std::numeric_limits<size_t>::max())
      : physical_ledger_(std::move(physical_ledger)),
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

  class CombinedReservation {
   public:
    CombinedReservation(CudaDeviceMemoryBudget::Reservation local_res,
                        CudaDeviceMemoryBudget::Reservation phys_res)
        : local_res_(std::move(local_res)), phys_res_(std::move(phys_res)) {}

    void Commit() {
      local_res_.Commit();
      phys_res_.Commit();
    }

   private:
    CudaDeviceMemoryBudget::Reservation local_res_;
    CudaDeviceMemoryBudget::Reservation phys_res_;
  };

  CombinedReservation Reserve(size_t bytes) {
    auto local_res = local_budget_.Reserve(bytes);
    auto phys_res = physical_ledger_ ? physical_ledger_->Reserve(bytes) : CudaDeviceMemoryBudget::Reservation{};
    return CombinedReservation(std::move(local_res), std::move(phys_res));
  }

  void BeginFree(size_t bytes) noexcept {
    local_budget_.BeginFree(bytes);
    if (physical_ledger_) physical_ledger_->BeginFree(bytes);
  }

  void FinishFree(size_t bytes, bool driver_ok) noexcept {
    local_budget_.FinishFree(bytes, driver_ok);
    if (physical_ledger_) physical_ledger_->FinishFree(bytes, driver_ok);
  }

  CudaDeviceMemoryBudget::Stats GetAccountStats() const { return local_budget_.GetStats(); }
  CudaDeviceMemoryBudget::Stats GetPhysicalStats() const {
    return physical_ledger_ ? physical_ledger_->GetStats() : local_budget_.GetStats();
  }

 private:
  std::shared_ptr<CudaDeviceMemoryBudget> physical_ledger_;
  CudaDeviceMemoryBudget local_budget_;
};

}  // namespace onnxruntime
