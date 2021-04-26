// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* Modifications Copyright (c) Microsoft. */

#include <type_traits>

#pragma once
#include "onnxruntime_config.h"
// build/external/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:162:71:
// error: ignoring attributes on template argument "Eigen::PacketType<const float, Eigen::DefaultDevice>::type {aka
// __vector(4) float}" [-Werror=ignored-attributes]
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#pragma warning(disable : 4805)
#endif

#include "unsupported/Eigen/CXX11/ThreadPool"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
#include "core/common/denormal.h"
#include "core/common/make_unique.h"
#include "core/common/spin_pause.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/Barrier.h"

// ORT thread pool overview
// ------------------------
//
// The ORT thread pool implementation is split into two layers.  This
// file provides the low-level component.  See the accompanying
// comments in threadpool.h for the high-level component.
//
// The code here is derived from the Eigen non-blocking thread pool,
// although many parts have been updated over time.  The main
// abstractions used here are:
//
// - The thread pool maintains a set of OS threads running
//   ThreadPoolTempl::WorkerLoop.
//
//   Each thread has its own RunQueue object, holding a queue of tasks
//   that have been pushed to the thread for execution.  The main work
//   loop is to pop a task from the head of the queue, and to execute
//   it to completion.  If the worker's run queue is empty then it
//   will spin waiting for work, then attempt to steal tasks from
//   other threads' queues, and then block in the OS if it cannot find
//   work.
//
//   This spin-then-block behavior is configured via a flag provided
//   when creating the thread pool, and by the constant spin_count.
//
// - Although all tasks are simple void()->void functions,
//   conceptually there are three different kinds:
//
//   - One-shot tasks submitted externally via the Schedule() method.
//     These tasks are used to support asynchronous work.  These are
//     used in the parallel executor, but otherwise are not widely
//     used outside of test harnesses (see threadpool_test.cc for some
//     examples).
//
//   - Tasks for running a parallel loop.
//
//     The tasks themselves are defined in threadpool.cc, and are
//     submitted to the run queues via RunInParallel->SummonWorkers.
//     Each task will loop internally, picking off iterations from the
//     user's code via atoic-fetch-and-add, until the loop is
//     complete.
//
//     This two-layer approach lets us separate out the
//     super-lightweight per-iteration-batch work from the more
//     costsly per-loop work of managing Task objects.
//
//   - Tasks for running a parallel section.  This is an extension of
//     the approach taken for parallel loops.  However, the Tasks are
//     defined in this file, and can pick up iterations from a series
//     of different parallel loops.  The tasks are defined in
//     RunInParallelSection->SummonWorkers.
//
//     The additional layer of parallel sections is a further way to
//     amortize costs: the work done creating the tasks can be
//     performed once, and then exploited over a series of loops.
//
// There are a few aspects of the modified Eigen thread pool to
// highlight:
//
// - The run queues follow the usual approach of having push/pop
//   operations on the front/back, and optimizing the PopFront case
//   for single-threaded use by the thread owning the run queue.
//
//   However, we support an additional Revoke operation to replace an
//   item in the middle of a queue with a tombstone.  This operation
//   is used at the end of parallel loops and parallel sections to
//   remove any tasks that were created but not yet executed.  Once
//   revoked, a thread can rely on the fact that the task will no
//   longer execute.  Revocation helps manage captured state in
//   parallel loops: the alternatives would be (i) waiting for all
//   tasks that captured state to reach the head of their queues and
//   execute, or (ii) use heap-allocated state in tasks, and use a
//   technique such as reference counting to de-allocate it.
//
//   To support revoation, each thread has a unique "Tag" to identify
//   the items that it adds to the work queues.  A thread can revoke
//   an item only if it has the thread's own tag.
//
// - The worker threads maintain a best-effort bitmap in
//   good_worker_hints_ of which threads to push work to.  A thread
//   controls its status via SetGoodWorkerHint.  A thread is a "good"
//   worker when it is actively spinning for work, meaning both that
//   it is not blocked in the OS, and that it is not busy with work
//   already.
//
//   This heuristic aims to avoid waking additional sleeping threads
//   where possible, and in a series of parallel loops or parallel
//   sections to push the work to the same set of threads each time.

namespace onnxruntime {
namespace concurrency {

#ifdef _WIN32
using CHAR_TYPE = wchar_t;
#else
using CHAR_TYPE = char;
#endif

class ThreadPoolParallelSection;
class ThreadPoolLoop;

// Align to avoid false sharing with prior fields.  If required,
// alignment or padding must be added subsequently to avoid false
// sharing with later fields.  Note that:
//
// - The __x86_64__ value is twice the line size (64 bytes).  This
//   accounts for 2-line prefetch behavior on some cores.
//
// - Ideally, ORT_ALIGN_TO_AVOID_FALSE_SHARING is used.  However, the
//   definition of ThreadPoolParallelSection uses naive padding
//   because C++11 does not support alignment constraints on
//   allocation or expose stdlib.h aligned_alloc.  C++17 introduces
//   support for aligned allocation which we could use here.

#if defined(__x86_64__)
#define ORT_FALSE_SHARING_BYTES 128
#else
#define ORT_FALSE_SHARING_BYTES 64
#endif

#define ORT_ALIGN_TO_AVOID_FALSE_SHARING alignas(ORT_FALSE_SHARING_BYTES)

struct PaddingToAvoidFalseSharing {
  char padding[ORT_FALSE_SHARING_BYTES];
};

/* Usage:
1. In executor, call Start() before profiling and Stop() to get profiled numbers;
2. Inside thread pool, call LogStart() before interested section and LogEnd... after to log elapsed time;
3. To extend, just add more events in enum Event before "All", and update GetEventName(...) accordingly;
4. Note LogStart must pair with either LogEnd or LogEndAndStart, otherwise ORT_ENFORCE will fail;
5. ThreadPoolProfiler is thread-safe.
*/
#ifdef ORT_MINIMAL_BUILD
class ThreadPoolProfiler {
 public:
  enum ThreadPoolEvent {
    DISTRIBUTION = 0,
    DISTRIBUTION_ENQUEUE,
    RUN,
    WAIT,
    WAIT_REVOKE,
    MAX_EVENT
  };
  ThreadPoolProfiler(int, const CHAR_TYPE*){};
  ~ThreadPoolProfiler() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThreadPoolProfiler);
  void Start(){};
  std::string Stop() { return "not available for minimal build"; }
  void LogStart(){};
  void LogEnd(ThreadPoolEvent){};
  void LogEndAndStart(ThreadPoolEvent){};
  void LogStartAndCoreAndBlock(std::ptrdiff_t){};
  void LogCoreAndBlock(std::ptrdiff_t){};
  void LogThreadId(int){};
  void LogRun(int){};
  std::string DumpChildThreadStat() { return {}; }
};
#else
class ThreadPoolProfiler {
 public:
  enum ThreadPoolEvent {
    DISTRIBUTION = 0,
    DISTRIBUTION_ENQUEUE,
    RUN,
    WAIT,
    WAIT_REVOKE,
    MAX_EVENT
  };
  ThreadPoolProfiler(int num_threads, const CHAR_TYPE* threal_pool_name);
  ~ThreadPoolProfiler();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThreadPoolProfiler);
  using Clock = std::chrono::high_resolution_clock;
  void Start();                  //called by executor to start profiling
  std::string Stop();            //called by executor to stop profiling and return collected numbers
  void LogStart();               //called in main thread to record the starting time point
  void LogEnd(ThreadPoolEvent);  //called in main thread to calculate and save the time elapsed from last start point
  void LogEndAndStart(ThreadPoolEvent);
  void LogStartAndCoreAndBlock(std::ptrdiff_t block_size);
  void LogCoreAndBlock(std::ptrdiff_t block_size);  //called in main thread to log core and block size for task breakdown
  void LogThreadId(int thread_idx);                 //called in child thread to log its id
  void LogRun(int thread_idx);                      //called in child thread to log num of run
  std::string DumpChildThreadStat();                //return all child statitics collected so far

 private:
  static const char* GetEventName(ThreadPoolEvent);
  struct MainThreadStat {
    uint64_t events_[MAX_EVENT] = {};
    int32_t core_ = -1;
    std::vector<std::ptrdiff_t> blocks_;  //block size determined by cost model
    std::vector<onnxruntime::TimePoint> points_;
    void LogCore();
    void LogBlockSize(std::ptrdiff_t block_size);
    void LogStart();
    void LogEnd(ThreadPoolEvent);
    void LogEndAndStart(ThreadPoolEvent);
    std::string Reset();
  };
  bool enabled_ = false;
  MainThreadStat& GetMainThreadStat(); //return thread local stat
  int num_threads_;
  struct ChildThreadStat {
    std::thread::id thread_id_;
    uint64_t num_run_ = 0;
    onnxruntime::TimePoint last_logged_point_ = Clock::now();
    int32_t core_ = -1;  //core that the child thread is running on
    PaddingToAvoidFalseSharing padding_; //to prevent false sharing
  };
  std::vector<ChildThreadStat> child_thread_stats_;
  std::string thread_pool_name_;
};
#endif

// Extended Eigen thread pool interface, avoiding the need to modify
// the ThreadPoolInterface.h header from the external Eigen
// repository.

class ExtendedThreadPoolInterface : public Eigen::ThreadPoolInterface {
 public:
  // Start/end a parallel section, within which calls to
  // RunInParallelSection may be made.  Parallel sections are
  // non-nesting.
  virtual std::unique_ptr<ThreadPoolParallelSection, void(*)(ThreadPoolParallelSection*)> AllocateParallelSection() = 0;
  virtual void StartParallelSection(ThreadPoolParallelSection &ps) = 0;
  virtual void EndParallelSection(ThreadPoolParallelSection &ps) = 0;

  // Run fn with up to n degree-of-parallelism enlisting the thread
  // pool for help.  The degree-of-parallelism includes the caller,
  // and so if n==1 then the function will run directly in the caller.
  //
  // The fork-join synchronization is handled in the thread pool, and
  // so any state captured by fn() is safe from concurrent access once
  // RunInParallelSection returns.
  //
  // The parameter idx provides a loop-local thread ID in the range
  // [0,k) where k<=n.
  virtual void RunInParallelSection(ThreadPoolParallelSection &ps,
                                    std::function<void(unsigned idx)> fn,
                                    unsigned n, std::ptrdiff_t block_size) = 0;

  // Special case alternative to RunInParallelSection for use without
  // an existing parallel section.  Ideally we would use a single
  // iplemenation and a stack-allocated ThreadPoolParallelSection.
  //
  // However, on the BM_ThreadPoolParallelFor microbenchmark I saw
  // ~20% overhead on the resulting single-loop parallel sections.
  // There are some additional costs (~5%) for additional invocations
  // through lambda functions on loop entry.  Most significantly, on
  // loop exit, we incurred ~15% cost by no longer being able to
  // overlap clean-up of unused Task objects in EndParallelSection
  // with waiting for loop iterations to complete.
  //
  // [ Note that this 20% overhead is more than paid for when we have
  // two loops execute in series in a parallel section. ]
  virtual void RunInParallel(std::function<void(unsigned idx)> fn,
                             unsigned n, std::ptrdiff_t block_size) = 0;
  virtual void StartProfiling()  = 0;
  virtual std::string StopProfiling() = 0;
};


class ThreadPoolParallelSection {
 public:
  // State accessed only by the main thread
  // --------------------------------------

  // Tasks successfully submitted to the work queues.  This sets the
  // maximum degree of parallelism that the section will support.
  std::vector<std::pair<int,unsigned>> tasks;

  // State shared between the main thread and worker threads
  // -------------------------------------------------------

  // Flag to signal termination of the parallel section
  std::atomic<bool> active{false};

  std::atomic<unsigned> worker_idx{0};

  // Count of the number of tasks that completed normally.  Other
  // tasks may be running currently, or may be present in work queues,
  // or may have been removed from the queues by
  // RunQueue::RevokeWithTag.
  PaddingToAvoidFalseSharing padding_1;
  std::atomic<unsigned> tasks_finished{0};
  PaddingToAvoidFalseSharing padding_2;

  // If non-null, the current loop that tasks should be executing.  We
  // need to be careful on access to the contents of current_loop
  // because it can be stack allocated on the thread entering the
  // loop:
  //
  // - Readers increment workers_in_loop and then read current_loop
  //
  // - Writers wishing to deallocate *current_loop must first clear
  //   current_loop and then wait for workers_in_loop==0
  std::atomic<ThreadPoolLoop *> current_loop{nullptr};
  std::atomic<unsigned> workers_in_loop{0};
};

class ThreadPoolLoop {
 public:
   ThreadPoolLoop(std::function<void(unsigned)> f, unsigned t) : fn(std::move(f)), threads_needed(t) {
   }

  const std::function<void(unsigned)> fn;
  const unsigned threads_needed;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ThreadPoolLoop);
};

template <typename Elem, int32_t Capacity, int32_t Base = 8>
class RunQueue {
 public:
  RunQueue() {
    assert(Base > 0 && (Base & Base - 1) == 0);
    assert(Capacity >= Base && (Capacity & Capacity - 1) == 0);
  }
  ~RunQueue() {
    assert(Empty());
  }
  Elem Push(Elem e, size_t tag, unsigned& at) {
    if (Size() < Capacity) {
      int32_t base = base_.load(std::memory_order_relaxed);
      for (int32_t i = 0; i < base; i++) {
        Signature signature = slots_[i].signature_.load(std::memory_order_relaxed);
        if (State::empty == signature.state_) {
          if (slots_[i].signature_.compare_exchange_weak(signature, {State::loading, 0}, std::memory_order_relaxed)) {
            slots_[i].elem_ = std::move(e);
            size_.fetch_add(1, std::memory_order_relaxed);
            at = i;
            slots_[i].signature_.store({State::ready, 0}, std::memory_order_release);
            return {};
          }
        }
      }
      int32_t doubled_base = base << 1;
      if (base < Capacity && base_.compare_exchange_weak(base, doubled_base, std::memory_order_relaxed)) {
        for (int32_t i = base; i < doubled_base; i++) {
          Signature signature = slots_[i].signature_.load(std::memory_order_relaxed);
          if (State::empty == signature.state_) {
            if (slots_[i].signature_.compare_exchange_weak(signature, {State::loading, tag}, std::memory_order_relaxed)) {
              slots_[i].elem_ = std::move(e);
              size_.fetch_add(1, std::memory_order_relaxed);
              at = i;
              slots_[i].signature_.store({State::ready, tag}, std::memory_order_release);
              return {};
            }
          }
        }
      }
    }
    return e;
  }
  Elem Pop() {
    if (!Empty()) {
      int32_t base = base_.load(std::memory_order_relaxed);
      int32_t half_base = base >> 1;
      for (int32_t i = base - 1; i > -1; i--) {
        Signature signature = slots_[i].signature_.load(std::memory_order_relaxed);
        if (State::ready == signature.state_) {
          if (slots_[i].signature_.compare_exchange_weak(signature, {State::unloading, 0}, std::memory_order_acquire)) {
            Elem e = slots_[i].elem_;
            size_.fetch_sub(1, std::memory_order_relaxed);
            slots_[i].signature_.store({State::empty, 0}, std::memory_order_release);
            if (i < half_base && half_base >= Base) {
              base_.compare_exchange_weak(base, half_base, std::memory_order_relaxed);
            }
            return e;
          }
        }
      }
    }
    return {};
  }
  enum State {
    empty = 0,
    loading,
    unloading,
    ready,
  };
  struct Signature {
    State state_{State::empty};
    size_t tag_{0};
  };
  struct Slot {
    std::atomic<Signature> signature_;
    Elem elem_;
  };
  Elem PushFront(Elem e) {
    unsigned at = 0;
    return Push(e, 0, at);
  }
  Elem PushBack(Elem e) {
    unsigned at = 0;
    return Push(e, 0, at);
  }
  Elem PopFront() {
    return Pop();
  }
  Elem PopBack() {
    return Pop();
  }
  Elem PushBackWithTag(Elem e, size_t tag, unsigned& at) {
    return Push(e, tag, at);
  }
  bool RevokeWithTag(size_t tag, unsigned at) {
    Signature signature{State::ready, tag};
    if (at < Capacity &&
        slots_[at].signature_.compare_exchange_weak(signature, {State::empty, 0}, std::memory_order_relaxed)) {
      size_.fetch_sub(1, std::memory_order_relaxed);
      return true;
    } else {
      return false;
    }
  }
  unsigned Size() const {
    return size_.load(std::memory_order_relaxed);
  }
  bool Empty() const {
    return Size() == 0;
  }
  void Flush() {
    while (!Empty()) {
      PopFront();
    }
  }
 private:
  ORT_ALIGN_TO_AVOID_FALSE_SHARING Slot slots_[Capacity];
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic_int32_t size_{0};
  ORT_ALIGN_TO_AVOID_FALSE_SHARING std::atomic_int32_t base_{Base};
};

static std::atomic<uint32_t> next_tag{1};

template <typename Environment>
class ThreadPoolTempl : public onnxruntime::concurrency::ExtendedThreadPoolInterface {

 private:
  struct PerThread;

  static unsigned WorkerLoop(int id, Eigen::ThreadPoolInterface* param) {
    // unsafe downcast
    ThreadPoolTempl* this_ptr = (ThreadPoolTempl*)param;
    this_ptr->WorkerLoop(id);
    return 0;
  }

  ThreadPoolProfiler profiler_;

 public:

  void StartProfiling() override {
    profiler_.Start();
  }

  std::string StopProfiling() override {
    return profiler_.Stop();
  }

  /*
  struct Tag {
    constexpr Tag() : v_(0) {
    }

    Tag(uint32_t v) : v_(v) {
    }

    // Allocate a new tag to use to identify work items from a given
    // thread in a parallel section.  Ideally, threads will have
    // unique tags, but re-use is not incorrect if the counter wraps
    // (for intsance, if a long-running workload is calling into ORT
    // from a fresh thread for each request).  We must not re-use the
    // default tag 0 which is used to identify work items added via
    // Schedule as opposed to requests for help in parallel sections.

    static Tag GetNext() {
      Tag t = Tag(next_tag++);
      if (t.v_ == 0) {
        t = Tag(next_tag++);
      }
      return t;
    }

    uint32_t Get() const {
      return v_;
    }

    bool operator==(Tag& other) const {
      return v_ == other.v_;
    }

    uint32_t v_ = 0;
  };*/

  typedef std::function<void()> Task;
  //typedef RunQueue<Task, Tag, 1024> Queue;
  typedef RunQueue<Task, 1024> Queue;

  ThreadPoolTempl(const CHAR_TYPE* name, int num_threads, bool allow_spinning, Environment& env,
                  const ThreadOptions& thread_options)
      : profiler_(num_threads, name),
        env_(env),
        num_threads_(num_threads),
        allow_spinning_(allow_spinning),
        set_denormal_as_zero_(thread_options.set_denormal_as_zero),
        worker_data_(num_threads),
        all_coprimes_(num_threads),
        blocked_(0),
        done_(false),
        cancelled_(false) {

    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    for (int i = 1; i <= num_threads_; ++i) {
      all_coprimes_.emplace_back(i);
      ComputeCoprimes(i, &all_coprimes_.back());
    }

    // Allocate space for per-thread bits to indicate which threads to consider
    // preferable for pushing work.  We use a regular array given that a std::vector
    // cannot contain std::atomic.
    num_hint_words_ = static_cast<int>((num_threads_ + bits_per_hint_word_ - 1) / bits_per_hint_word_);
    good_worker_hints_ = onnxruntime::make_unique<std::atomic<uint64_t>[]>(num_hint_words_);

    worker_data_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
      worker_data_[i].thread.reset(env_.CreateThread(name, i, WorkerLoop, this, thread_options));
    }
  }

  ~ThreadPoolTempl() override {
    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!cancelled_) {
      WakeAllWorkersForExit();
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < worker_data_.size(); i++) {
        worker_data_[i].queue.Flush();
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < worker_data_.size(); ++i) worker_data_[i].thread.reset();
  }

  // Run fn().  Ordinarily, the function will be added to the thread pool and executed
  // by a worker thread.  If the thread pool rejects the work then fn() will instead
  // execute synchronously during Schedule(fn).  Currently the thread pool will only
  // reject work if the queue of pending work is full.

  void Schedule(std::function<void()> fn) override {
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = worker_data_[pt->thread_id].queue;
      fn = q.PushFront(std::move(fn));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      int q_idx = Rand(&pt->rand) % num_threads_;
      WorkerData &td = worker_data_[q_idx];
      Queue& q = td.queue;
      fn = q.PushBack(std::move(fn));
      if (!fn) {
        // The queue accepted the work; ensure that the thread will pick it up
        td.EnsureAwake();
      }
    }

    // Run the work directly if the queue rejected the work
    if (fn) fn();
  }

// The thread pool maintains a set of hints for which threads will be good to distribute
// work to.  A thread is considered "good" if it is actively spinning, meaning both that
// it is not busy with existing work, and that it should respond quickly to the addition
// of new work.

void SetGoodWorkerHint(int idx, bool is_good) {
  assert(idx >= 0 && idx < num_threads_);
  std::atomic<uint64_t>& u64 = good_worker_hints_[idx / bits_per_hint_word_];
  uint64_t bit = 1ull << (idx % bits_per_hint_word_);
  uint64_t saw, want;
  do {
    saw = u64.load();
    want = is_good ? (saw|bit) : (saw&~bit);
  } while (!u64.compare_exchange_weak(saw, want));
}

// Retrieve hints for up to n threads to distribute work to.  Threads in good_hints
// pass a best-effort check to identify spinning threads via the good_worker_hints_
// bitmap.  Threads in alt_hint do not pass that test, but are distinct from those in
// good_hints, letting the caller avoid distributing more than one work item to
// any individual thread.

void GetGoodWorkerHints(unsigned n, std::vector<unsigned>& good_hints, std::vector<unsigned>& alt_hints) {
  PerThread* pt = GetPerThread();
  unsigned need_alt = n;
  good_hints.clear();
  alt_hints.clear();

  // Iterate through the words of hints, starting from a pseudo-randomly chosen
  // base.  This aims to distribute work across large machines in cases we
  // have multiple threads scheduling work concurrently.

  unsigned base = Rand(&pt->rand) % num_hint_words_;
  for (unsigned i = 0u; n && (i < num_hint_words_); i++) {
    int u64_idx = (base + i) % num_hint_words_;
    std::atomic<uint64_t>* u64 = &good_worker_hints_[u64_idx];
    uint64_t saw = u64->load();
    uint64_t want = saw;

    // Pick up to n bits that are set in the current word
    for (unsigned j = 0u; n && (j < bits_per_hint_word_); j++) {
      uint64_t bit = 1ull << j;
      int thread = u64_idx * bits_per_hint_word_ + j;
      if (saw & bit) {
        good_hints.push_back(thread);
        want &= ~bit;
        n--;
      } else if (need_alt && thread < num_threads_) {
        alt_hints.push_back(thread);
	      need_alt--;
      }
    }

    // Best-effort attempt to remove the hints.  We should measure the impact of
    // contention here, but the intuition is that if we conflict on the CAS then the
    // machine is likely to be busy in any case, and we will have queuing on the
    // work items.
    u64->compare_exchange_strong(saw, want);
  }
}

//......................................................................
//
// Parallel sections
// -----------------
//
// Allocate a new ThreadPoolParallelSection, owned by the returned
// unique_ptr.  The explicit deleter avoids the Eigen-specific
// definition of ThreadPoolParallelSection needing to be avilable in
// threadpool.h where the user-facing parallel section API is defined.

std::unique_ptr<ThreadPoolParallelSection, void(*)(ThreadPoolParallelSection*)> AllocateParallelSection() override {
  return std::unique_ptr<ThreadPoolParallelSection, void(*)(ThreadPoolParallelSection*)>
    (new ThreadPoolParallelSection,
     [](ThreadPoolParallelSection *tps) {
      delete tps;
    });
}

// Start a parallel section, using a caller-provided
// ThreadPoolParallelSection for maintaining the per-section state.
// Starting a parallel section is just book-keeping; threads are
// "summoned" to help with the parallel section once it enters
// parallel loops.  The threads are then retained until the end of the
// section, being re-used over subsequent loops.

void StartParallelSectionInternal(PerThread &pt,
                                  ThreadPoolParallelSection &ps) {
  assert((!pt.leading_par_section) && "Nested parallelism not supported");
  assert((!ps.active) && "Starting parallel section, but active already");
  pt.leading_par_section = true;
  /*
  if (!pt.tag.Get()) {
    pt.tag = Tag::GetNext();
  }*/
  ps.active = true;
}

void StartParallelSection(ThreadPoolParallelSection &ps) override {
  PerThread* pt = GetPerThread();
  StartParallelSectionInternal(*pt, ps);
}

// End a parallel section, waiting for all worker threads to exit from
// section.  Hence, on return, the ThreadPoolParallelSection object
// can be dealloacted.

void EndParallelSectionInternal(PerThread &pt,
                                ThreadPoolParallelSection &ps) {
  assert((pt.leading_par_section) && "Ending parallel section, but none started");
  assert((ps.active) && "Ending parallel section, but not active");
  pt.leading_par_section = false;

  // Notify workers to exit from the section
  ps.active = false;

  profiler_.LogStart();
  // Attempt to revoke any tasks that were sent to workers but not
  // started.
  unsigned tasks_started = static_cast<unsigned>(ps.tasks.size());
  unsigned tasks_revoked = 0;
  while (!ps.tasks.empty()) {
    const auto& item = ps.tasks.back();
    Queue& q = worker_data_[item.first].queue;
    if (q.RevokeWithTag(pt.tag, item.second)) {
      tasks_revoked++;
    }
    ps.tasks.pop_back();
  }
  profiler_.LogEnd(ThreadPoolProfiler::WAIT_REVOKE);

  // Wait for workers to exit ParLoopWorker
  auto tasks_to_wait_for = tasks_started - tasks_revoked;
  while (ps.tasks_finished < tasks_to_wait_for) {
    onnxruntime::concurrency::SpinPause();
  }
  // Clear status to allow the ThreadPoolParallelSection to be
  // re-used.
  ps.tasks_finished = 0;
}

void EndParallelSection(ThreadPoolParallelSection &ps) override {
  PerThread* pt = GetPerThread();
  EndParallelSectionInternal(*pt, ps);
}

//......................................................................
//
// Parallel loops
// --------------
//
// Ensure that the ThreadPoolParallelSection has sufficient workers to
// execute a loop with degree of parallelism n.  We track the number
// of workers already avaiable to the parallel section, prior to
// submitting tasks to the work queues to make up the total.
//
// Each worker will call in to worker_fn(idx) with a per-worker thread
// ID.  Note there are different levels of indirection here:
//
// - In a single-loop parallel section, worker_fn will directly
//   execute the threadpool.cc code that implements the parallel loop.
//
// - In a multi-loop parallel section, worker_fn is an intermediate
//   function that is long-lived (i.e., that lasts until the end of
//   the parallel section, as opposed to just a single loop's
//   duration).

void SummonWorkers(PerThread &pt,
                   ThreadPoolParallelSection &ps,
                   unsigned n,
                   const std::function<void(unsigned)> &worker_fn) {
  // Wrap the user's worker function with one that allocates a unique
  // worker index for the loop, and synchronizes (as the last step)
  // with the exit path in EndParallelSection.  In principle we could
  // allocate worker IDs during the loop below and capture them by
  // value.  However, the costs of creating distinct lambda for each
  // iteration appeared more costly than the cost of synchronization
  // on a shared counter.
  auto call_worker_fn = [&ps, worker_fn]() {
    unsigned my_idx = ++ps.worker_idx;
    worker_fn(my_idx);
    // After the assignment to ps.tasks_finished, the stack-allocated
    // ThreadPoolParallelSection object may be destroyed.
    ps.tasks_finished++;
  };

  // Identify whether we need to create additional workers.
  // Throughout the threadpool implementation, degrees of parallelism
  // ("n" here) refer to the total parallelism including the main
  // thread.  Hence we consider the number of existing tasks + 1.
  unsigned current_dop = static_cast<unsigned>(ps.tasks.size()) + 1;
  if (n > current_dop) {
    unsigned extra_needed = n - current_dop;

    // Obtain hints for which worker threads to push the tasks to.
    // This uses a best-effort assessment of which threads are
    // spinning.
    std::vector<unsigned> good_hints, alt_hints;
    GetGoodWorkerHints(extra_needed, good_hints, alt_hints);
    profiler_.LogStart();

    // Create the additional tasks, and push them to workers.
    for (auto i = 0u; i < extra_needed; i++) {
      Task t;
      int q_idx;
      if (i < good_hints.size()) {
        q_idx = good_hints[i];
      } else {
        auto alt_i = i - static_cast<unsigned>(good_hints.size());
        if (alt_i < alt_hints.size()) {
          q_idx = alt_hints[alt_i];
        } else {
          q_idx = Rand(&pt.rand) % num_threads_;
        }
      }

      // If the worker's queue accepts the task, then record it in
      // the vector of tasks that we will need to synchronize with on
      // exiting the parallel section.  If the queue rejects the task
      // (perhaps because it is full) then we take no further action:
      // in a parallel loop we will always be running work on the
      // main thread, providing progress.
      WorkerData& td = worker_data_[q_idx];
      Queue& q = td.queue;
      unsigned w_idx;
      t = q.PushBackWithTag(call_worker_fn, pt.tag, w_idx);
      if (!t) {
        ps.tasks.push_back({q_idx, w_idx});
        td.EnsureAwake();
      }
    }
    profiler_.LogEnd(ThreadPoolProfiler::DISTRIBUTION_ENQUEUE);
  }
}

// Run a single parallel loop in an existing parallel section.  This
// maps directly onto SummonWorkers to create sufficient worker
// threads for the desired degree of parallelism, followed by
// dispatching the loop to those workers.

void RunInParallelSection(ThreadPoolParallelSection &ps,
                          std::function<void(unsigned idx)> fn,
                          unsigned n, std::ptrdiff_t block_size) override {
  profiler_.LogStartAndCoreAndBlock(block_size);
  PerThread* pt = GetPerThread();
  assert(pt->leading_par_section && "RunInParallel, but not in parallel section");
  assert((n > 1) && "Trivial parallel section; should be avoided by caller");

  // Publish the work to any existing workers in the parallel
  // section, and ensure it is visible to any new threads created
  // below.
  assert((!ps.current_loop) && "RunInParallelSection, but loop already active");
  ThreadPoolLoop loop{std::move(fn), n};
  ps.current_loop = &loop;

  // Increase the worker count if needed.  Each worker will pick up
  // loops to execute from the current parallel section.
  const auto worker_fn = [&ps](unsigned my_idx) {
    while (ps.active) {
      if (!ps.current_loop) {
        onnxruntime::concurrency::SpinPause();
      } else {
        ps.workers_in_loop++;
        ThreadPoolLoop *work_item = ps.current_loop;
        if (work_item && my_idx < work_item->threads_needed) {
          work_item->fn(my_idx);
        }
        ps.workers_in_loop--;
      }
    }
  };
  SummonWorkers(*pt, ps, n, worker_fn);
  profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);

  // Run work in the main thread
  loop.fn(0);
  profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);

  // Wait for workers to exit the loop
  ps.current_loop = 0;
  while (ps.workers_in_loop) {
    onnxruntime::concurrency::SpinPause();
  }
  profiler_.LogEnd(ThreadPoolProfiler::WAIT);
}

// Run a single parallel loop _without_ a parallel section.  This is a
// special case of RunInParallelSection, avoiding code paths for
// handing off multiple loops to the pool of workers.

void RunInParallel(std::function<void(unsigned idx)> fn, unsigned n, std::ptrdiff_t block_size) override {
  profiler_.LogStartAndCoreAndBlock(block_size);
  PerThread *pt = GetPerThread();
  ThreadPoolParallelSection ps;
  StartParallelSectionInternal(*pt, ps);

  // Summon workers to run the function (n is the desired maximum
  // degree of parallelism, including the main thread).  Unlike the
  // multi-loop RunInParallelSection, this single-loop worker can run
  // fn directly without needing to receive it via ps.current_loop.
  SummonWorkers(*pt, ps, n, fn);
  profiler_.LogEndAndStart(ThreadPoolProfiler::DISTRIBUTION);

  // Run work in the main thread
  fn(0);
  profiler_.LogEndAndStart(ThreadPoolProfiler::RUN);

  // Wait for workers to exit the parallel section and hence to have
  // completed the loop (i.e., ps.tasks_finished matches the number of
  // tasks that have been created less the number successfully
  // revoked).
  EndParallelSectionInternal(*pt, ps);
  profiler_.LogEnd(ThreadPoolProfiler::WAIT);
}

void Cancel() override {
  cancelled_ = true;
  // If done_ is true, which means this object is being destructing.
  // Therefore worker_data_[i].thread could be NULL.
  if (!done_) {
    done_ = true;
    // Let each thread know it's been cancelled.
    for (size_t i = 0; i < worker_data_.size(); i++) {
      assert(worker_data_[i].thread != nullptr);
      worker_data_[i].thread->OnCancel();
    }
  }

  // Wake up the threads without work to let them exit on their own.
  WakeAllWorkersForExit();
}

int NumThreads() const EIGEN_FINAL {
  return num_threads_;
}

int CurrentThreadId() const EIGEN_FINAL {
  const PerThread* pt = const_cast<ThreadPoolTempl*>(this)->GetPerThread();
  if (pt->pool == this) {
    return pt->thread_id;
  }
  return -1;
}

 private:

#ifdef NDEBUG
  void AssertBounds(int, int) {
  }
#else
  void AssertBounds(int start, int end) {
    assert(start >= 0);
    assert(start < end);  // non-zero sized partition
    assert(end <= num_threads_);
  }
#endif

  void ComputeCoprimes(int N, Eigen::MaxSizeVector<unsigned>* coprimes) {
    for (int i = 1; i <= N; i++) {
      unsigned a = i;
      unsigned b = N;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes->push_back(i);
      }
    }
  }

  typedef typename Environment::EnvThread Thread;
  struct WorkerData;

  // PerThread objects are allocated in thread-local storage and allocated
  // on the thread's first call to GetPerThread.  The object should
  // remain trivially-destructable, with other state placed in the
  // WorkerData objects that are allocated and cleaned-up explicitly.
  //
  // PerThread objects are allocated for all threads that submit work to
  // the thread pool, in addition to threads within the pool.
  //
  // In contrast, the WorkerData objects are allocated only for the
  // threads in the pool, and their lifetime is managed along with the
  // pool.

  struct PerThread {
    constexpr PerThread() : pool(nullptr) {
      tag = reinterpret_cast<size_t>(this);
    }
    ThreadPoolTempl* pool;            // Parent pool, or null for normal threads.
    uint64_t rand{0};                 // Random generator state.
    int thread_id{-1};                // Worker thread index in pool.
    // Tag tag{};                        // Work item tag used to identify this thread.
    size_t tag;
    bool leading_par_section{false};  // Leading a parallel section (used only for asserts)
  };

  static_assert(std::is_trivially_destructible<PerThread>::value,
                "Per-thread state should be trivially destructible");

  struct WorkerData {
    constexpr WorkerData() : thread(), queue() {
    }
    std::unique_ptr<Thread> thread;
    Queue queue;

    // Each thread has a status, available read-only without locking, and protected
    // by the mutex field below for updates.  The status is used for three
    // purposes:
    //
    // 1. To identify threads that are good candidates to push work to.
    //    We prefer to push work to threads that are actively spinning (no need
    //    for an OS wake-up, and no need for current work to finish).  After that, we
    //    prefer to push work to threads that are blocked (no need to wait for the
    //    current work to finish).
    //
    // 2. To identify threads that are good candidates to steal work from.  We
    //    prefer to steal work from threads that are active outside the worker loop.
    //    This avoids "snatching" new work away from a thread that has just been
    //    given it but not yet noticed.
    //
    // 3. When pushing work to a thread, we use the status read-only to identify
    //    when we need to wake the thread.  This read-only check avoids the
    //    need for mutex / condvar operations in the case where the thread pool
    //    remains busy.

    enum class ThreadStatus : uint8_t {
      Spinning,  // Spinning in the work loop, and other cases (initialization) where
                 // the thread will soon be in the loop
      Active,    // Running user code, not waiting for work
      Blocking,  // In the process of blocking; may no longer notice work pushed to it
      Blocked,   // Blocked on cv
      Waking,    // Not yet back in the worker loop, but wake-up notification sent
    };

    ThreadStatus GetStatus() const {
      return status;
    }

    // State transitions, called from other threads

    void EnsureAwake() {
      ThreadStatus seen = status;
      if (seen == ThreadStatus::Blocking ||
          seen == ThreadStatus::Blocked) {
        std::unique_lock<OrtMutex> lk(mutex);
        // Blocking state exists only transiently during the SetBlock() method
        // while holding the lock.  We may observe it at the start of this
        // function, but after acquiring the lock then the target thread
        // will either be blocked or not.
        seen = status;
        assert(seen != ThreadStatus::Blocking);
        if (seen == ThreadStatus::Blocked) {
          status = ThreadStatus::Waking;
          cv.notify_one();
        }
      }
    }

    // State transitions, called only from the thread itself

    void SetActive() {
      std::unique_lock<OrtMutex> lk(mutex);
      status = ThreadStatus::Active;
    }

    void SetSpinning() {
      std::unique_lock<OrtMutex> lk(mutex);
      status = ThreadStatus::Spinning;
    }

    void SetBlocked(std::function<bool()> should_block,
                    std::function<void()> post_block) {
      std::unique_lock<OrtMutex> lk(mutex);
      assert(status == ThreadStatus::Spinning);
      status = ThreadStatus::Blocking;
      if (should_block()) {
        status = ThreadStatus::Blocked;
        while (status == ThreadStatus::Blocked) {
          cv.wait(lk);
        }
        post_block();
      }
      status = ThreadStatus::Spinning;
    }

  private:
    std::atomic<ThreadStatus> status{ThreadStatus::Spinning};
    OrtMutex mutex;
    OrtCondVar cv;
  };

  Environment& env_;
  const int num_threads_;
  const bool allow_spinning_;
  const bool set_denormal_as_zero_;
  Eigen::MaxSizeVector<WorkerData> worker_data_;
  Eigen::MaxSizeVector<Eigen::MaxSizeVector<unsigned>> all_coprimes_;
  std::atomic<unsigned> blocked_;  // Count of blocked workers, used as a termination condition
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;

  // Allow control over how many bits to use in each entry in good_worker_hints_.
  // We reduce this below the full 64-bit word size for two reasons.  First, it
  // helps test coverage on machines without 64 vCPUS.  Second, it lets us
  // reduce contention by having different threads start work searching for hints
  // at different locations in the bitmap.

  static const unsigned bits_per_hint_word_ = 4;
  unsigned num_hint_words_;
  std::unique_ptr<std::atomic<uint64_t>[]> good_worker_hints_;

  // Wake any blocked workers so that they can cleanly exit WorkerLoop().  For an
  // abrupt exit, cancelled_==true and threads will exit their worker loops.  For
  // a clean exit, each thread will observe (1) done_ set, indicating that the
  // destructor has been called, (2) all threads blocked, and (3) no
  // items in the work queues.

  void WakeAllWorkersForExit() {
    for (auto &td: worker_data_) {
      td.EnsureAwake();
    }
  }

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    PerThread* pt = GetPerThread();
    WorkerData& td = worker_data_[thread_id];
    Queue& q = td.queue;
    bool should_exit = false;
    pt->pool = this;
    pt->rand = GlobalThreadIdHash();
    pt->thread_id = thread_id;

    assert(td.GetStatus() == WorkerData::ThreadStatus::Spinning);
    SetGoodWorkerHint(thread_id, true /* Is good */);

    const int log2_spin = 20;
    const int spin_count = allow_spinning_ ? (1ull<<log2_spin) : 0;
    const int steal_count = spin_count/100;

    SetDenormalAsZero(set_denormal_as_zero_);
    profiler_.LogThreadId(thread_id);

    while (!cancelled_ && !should_exit) {
        Task t = q.PopFront();
        if (!t) {
          // Spin waiting for work.  We indicate, via SetGOodWorkerHint that we are
          // spinning.  This will bias other threads toward pushing work to our queue.
          // In addition, priodically make a best-effort attempt to steal from other
          // threads which are not themselves spinning.

          SetGoodWorkerHint(thread_id, true);
          for (int i = 0; i < spin_count && !t && !cancelled_ && !done_; i++) {
            t = ((i + 1) % steal_count == 0) ? TrySteal() : q.PopFront();
            onnxruntime::concurrency::SpinPause();
          }
          SetGoodWorkerHint(thread_id, false);

          if (!t) {
            // No work passed to us while spinning; make a further full attempt to
            // steal work from other threads prior to blocking.
            if (num_threads_ != 1) {
              t = Steal(true /* true => check all queues */);
            }
            if (!t) {
              td.SetBlocked(
                  // Pre-block test
                  [&]() -> bool {
                    bool should_block = true;
                    // We already did a best-effort emptiness check when stealing; now
                    // do a full check prior to blocking.
                    int victim = NonEmptyQueueIndex();
                    if (victim != -1) {
                      should_block = false;
                      if (!cancelled_) {
                        t = worker_data_[victim].queue.PopBack();
                      }
                    }
                    // Number of blocked threads is used as termination condition.
                    // If we are shutting down and all worker threads blocked without work,
                    // that's we are done.
                    if (should_block) {
                      blocked_++;
                      if (done_ && blocked_ == static_cast<unsigned>(num_threads_)) {
                        should_block = false;
                        // Almost done, but need to re-check queues.
                        // Consider that all queues are empty and all worker threads are preempted
                        // right after incrementing blocked_ above. Now a free-standing thread
                        // submits work and calls destructor (which sets done_). If we don't
                        // re-check queues, we will exit leaving the work unexecuted.
                        if (NonEmptyQueueIndex() != -1) {
                          // Note: we must not pop from queues before we decrement blocked_,
                          // otherwise the following scenario is possible. Consider that instead
                          // of checking for emptiness we popped the only element from queues.
                          // Now other worker threads can start exiting, which is bad if the
                          // work item submits other work. So we just check emptiness here,
                          // which ensures that all worker threads exit at the same time.
                          blocked_--;
                        } else {
                          should_exit = true;
                        }
                      }
                    }
                    return should_block;
                  },
                  // Post-block update (executed only if we blocked)
                  [&]() {
                    blocked_--;
                  });
            }
          }
        }
        if (t) {
          td.SetActive();
          t();
          profiler_.LogRun(thread_id);
          td.SetSpinning();
        }
      }

      // Whichever thread(s) observe the termination conditions are responsible for waking
      // any other threads that have remained blocked.
      if (should_exit) {
        WakeAllWorkersForExit();
      }
    }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.  We make two passes over the threads:
  //
  // - round 0 : we attempt to steal from threads that are running in
  //   user code (ThreadStatus::Active).  The intuition behind this is that
  //   the thread is busy with other work, and that by preferring to
  //   steel from busy victims we will avoid "snatching" work from a
  //   thread which is just about to notice the work itself.
  //
  // - round 1 : we steal work from any thread, including those which claim
  //   to be spinning.  In these cases, even though the victim thread is
  //   looking for work itself, it may have been pre-empted.

  Task Steal(bool check_all) {
    PerThread* pt = GetPerThread();
    unsigned size = static_cast<unsigned>(num_threads_);
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];

    for (int round = 0; round < 2; round++) {
      unsigned victim = r % size;
      for (unsigned i = 0; i < size; i++) {
        assert(victim < size);
        if (round == 1 ||
            worker_data_[victim].GetStatus() == WorkerData::ThreadStatus::Active) {
          Task t = worker_data_[victim].queue.PopBack();
          if (t) {
            return t;
          }
        }
        if (!check_all) {
          return Task();
        }
        victim += inc;
        if (victim >= size) {
          victim -= size;
        }
      }
    }

    return Task();
  }

  Task TrySteal() {
    return Steal(false);
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    const unsigned size = static_cast<unsigned>(worker_data_.size());
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!worker_data_[victim].queue.Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static EIGEN_STRONG_INLINE uint64_t GlobalThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }

  static EIGEN_STRONG_INLINE PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
  }
};

 }  // namespace concurrency

}  // namespace onnxruntime
