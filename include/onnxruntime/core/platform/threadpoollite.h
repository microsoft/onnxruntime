#pragma once
#include "threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include <thread>
#include <atomic>

#define MAX_NUM_TASK 8

namespace onnxruntime {

namespace concurrency {

using Fn = std::function<void(std::ptrdiff_t, std::ptrdiff_t)>;
using SimpleFn = std::function<void(std::ptrdiff_t)>;
using SchdFn = std::function<void()>;

class ThreadPoolLite final : public ThreadPool {
 public:
  ThreadPoolLite(Env*,
                 const ThreadOptions&,
                 const NAME_CHAR_TYPE*,
                 int num_threads,
                 bool);
  ~ThreadPoolLite();

 private:

  struct Task {
    std::ptrdiff_t fn_ = 0;
    std::int16_t progress_ = 0;
    std::int16_t step_ = 0;
    std::int16_t done_ = 0;
  };

  int num_tasks = 0;
  std::atomic<Task> tasks_[MAX_NUM_TASK];
  int NumThreads() const override { return num_sub_threads_; }
  void ParallelFor(std::ptrdiff_t, double, const Fn&) override;
  void ParallelFor(std::ptrdiff_t, const TensorOpCost&, const Fn&) override;
  void SimpleParallelFor(std::ptrdiff_t, const SimpleFn&) override;
  void Schedule(SchdFn) override;
  void StartProfiling() override;
  void MainLoop(int);
  std::string StopProfiling() override;
  int num_sub_threads_;
  std::vector<std::thread> sub_threads_;
  bool exit_ = false;
  ThreadPoolProfiler profiler_;
};

template<int32_t ThreadPerPool, int32_t PoolSize>
class ThreadPoolLite2 final : public ThreadPool {
 public:
  ThreadPoolLite2(Env*,
                  const ThreadOptions&,
                  const NAME_CHAR_TYPE*,
                  int num_threads,
                  bool);
  ~ThreadPoolLite2();

 private:
  struct Slot {
    std::atomic_int32_t progress_{-1};
    std::atomic_int32_t done_{0};
    SchdFn schd_fn_;
  };

  std::unique_ptr<Slot[]> slots_;
  int NumThreads() const override { return static_cast<int>(sub_threads_.size()); }
  void ParallelFor(std::ptrdiff_t, double, const Fn&) override;
  void ParallelFor(std::ptrdiff_t, const TensorOpCost&, const Fn&) override;
  void SimpleParallelFor(std::ptrdiff_t, const SimpleFn&) override;
  void SimpleParallelForImpl(const SchdFn&);
  void Schedule(SchdFn) override;
  void StartProfiling() override;
  void MainLoop(int);
  std::string StopProfiling() override;
  std::vector<std::thread> sub_threads_;
  bool exit_ = false;
  ThreadPoolProfiler profiler_;
  int32_t num_pools_{0};
  int32_t num_slots_{0};
  int32_t num_sub_threads_{0};
  bool set_denormal_as_zero_{false};
};

template <int32_t PoolSize = 16>
class ThreadPoolLite3 final : public ThreadPool {
 public:
  ThreadPoolLite3(Env*,
                  const ThreadOptions&,
                  const NAME_CHAR_TYPE*,
                  int num_threads,
                  bool);
  ~ThreadPoolLite3();

 private:

  enum Stage {
      empty = 0,
      ready,
      running,
      done
  };
  using AtomicStage = std::atomic<Stage>;
  struct Slot {
    std::atomic<Stage> slot_stage_{empty};
    SchdFn schd_fn_;
    std::unique_ptr<AtomicStage[]> thread_stages_;
  };

  Slot slots_[PoolSize];
  int NumThreads() const override { return static_cast<int>(sub_threads_.size()); }
  void ParallelFor(std::ptrdiff_t, double, const Fn&) override;
  void ParallelFor(std::ptrdiff_t, const TensorOpCost&, const Fn&) override;
  void SimpleParallelFor(std::ptrdiff_t, const SimpleFn&) override;
  void Schedule(SchdFn) override;
  void StartProfiling() override;
  void MainLoop(int);
  std::string StopProfiling() override;
  std::vector<std::thread> sub_threads_;
  bool exit_ = false;
  ThreadPoolProfiler profiler_;
  int num_sub_threads_{0};
};
}  // namespace concurrency
}  // namespace onnxruntime