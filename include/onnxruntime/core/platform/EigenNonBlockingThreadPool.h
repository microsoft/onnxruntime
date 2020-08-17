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
#include "core/common/make_unique.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/Barrier.h"

namespace onnxruntime {

namespace concurrency {

// Extended Eigen thread pool interface, avoiding the need to modify the ThreadPoolInterface.h
// header from the external Eigen repository.

class ExtendedThreadPoolInterface : public Eigen::ThreadPoolInterface {
 public:
  // Run fn with up to n degree-of-parallelism enlisting the thread pool for
  // help.  The degree-of-parallelism includes the caller, and so if n==1
  // then the function will run directly in the caller.  The fork-join
  // synchronization is handled in the thread pool, and so any state captured
  // by fn() is safe from concurrent access once RunInParallel returns.
  virtual void RunInParallel(std::function<void()> fn, unsigned n) = 0;
};

}  // namespace concurrency

template <typename Work, typename Tag, unsigned kSize>
class RunQueue {
 public:
  RunQueue() : front_(0), back_(0) {
    // require power-of-two for fast masking
    assert((kSize & (kSize - 1)) == 0);
    assert(kSize > 2);            // why would you do this?
    assert(kSize <= (64 << 10));  // leave enough space for counter
    for (unsigned i = 0; i < kSize; i++) array_[i].state.store(ElemState::kEmpty, std::memory_order_relaxed);
  }

  ~RunQueue() {
    assert(Size() == 0);
  }

  // PushFront inserts w at the beginning of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushFront(Work w) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem& e = array_[front & kMask];
    ElemState s = e.state.load(std::memory_order_relaxed);
    if (s != ElemState::kEmpty ||
        !e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
      return w;
    front_.store(front + 1 + (kSize << 1), std::memory_order_relaxed);
    e.w = std::move(w);
    e.tag = Tag();
    e.state.store(ElemState::kReady, std::memory_order_release);
    return Work();
  }

  // PopFront removes and returns the first element in the queue.
  // If the queue was empty returns default-constructed Work.
  Work PopFront() {
    unsigned front;
    Elem *e;
    ElemState s;

    // Drain revoked items from the front of the queue.  CAS to busy to synchronize with
    // any attempt to take the same item from the back of the queue.
    do {
      front = front_.load(std::memory_order_relaxed);
      e = &array_[(front - 1) & kMask];
      s = e->state.load(std::memory_order_relaxed);
      if (s == ElemState::kRevoked &&
          e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
        e->state.store(ElemState::kEmpty, std::memory_order_release);
        front = ((front - 1) & kMask2) | (front & ~kMask2);
        front_.store(front, std::memory_order_relaxed);
      }
    } while (s == ElemState::kRevoked);

    // Attempt to take next item.  State kEmpty shows the queue is empty, kBusy shows
    // the work is in progress on the item at the front of the queue.
    if (s != ElemState::kReady ||
        !e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->tag = Tag();
    e->state.store(ElemState::kEmpty, std::memory_order_release);
    front = ((front - 1) & kMask2) | (front & ~kMask2);
    front_.store(front, std::memory_order_relaxed);
    return w;
  }

  // PushBack adds w at the end of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushBack(Work w) {
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem& e = array_[(back - 1) & kMask];
    ElemState s = e.state.load(std::memory_order_relaxed);
    if (s != ElemState::kEmpty ||
        !e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
      return w;
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e.w = std::move(w);
    e.tag = Tag();
    e.state.store(ElemState::kReady, std::memory_order_release);
    return Work();
  }

  // PushBackWithTag adds w at the end of the queue.  The tag value can be used on a 
  // subsequent call to RevokeWithTag to remove the item from the queue in combination
  // with w_idx.  Typically the tag will be a per-thread ID to distinguish work
  // submitted from different threads.
  //
  // If the queue is full, returns w, otherwise returns default-constructed work.
  Work PushBackWithTag(Work w, Tag tag, unsigned &w_idx) {
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    w_idx = (back-1) & kMask;
    Elem& e = array_[w_idx];
    ElemState s = e.state.load(std::memory_order_relaxed);
    if (s != ElemState::kEmpty ||
        !e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
      return w;
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e.w = std::move(w);
    e.tag = tag;
    e.state.store(ElemState::kReady, std::memory_order_release);
    return Work();
  }

  // PopBack removes and returns the last elements in the queue.
  Work PopBack() {
    if (Empty())
      return Work();
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back;
    Elem *e;
    ElemState s;

    // Drain revoked items from the back of the queue.  CAS to busy to synchronize with
    // any attempt to take the same item from the front of the queue.
    do {
      back = back_.load(std::memory_order_relaxed);
      e = &array_[back & kMask];
      s = e->state.load(std::memory_order_relaxed);
      if (s == ElemState::kRevoked &&
          e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
        e->state.store(ElemState::kEmpty, std::memory_order_release);
        back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
      }
    } while (s == ElemState::kRevoked);

    if (s != ElemState::kReady ||
        !e->state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->tag = Tag();
    e->state.store(ElemState::kEmpty, std::memory_order_release);
    back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
    return w;
  }

  // RevokeItem removes a work item from the queue.  Items are identified positionally,
  // and so a tag is used to detect whether the same position is occupied by a 
  // different work item at the time of removal.  RevokeWithTags lets threads offer work
  // for parallel execution, and then revoke the offer prior to the work executing (for 
  // instance if the thread itself completes all of the work).  Revoking the work 
  // lets the thread deallocate state that might otherwise have been captured by the work item
  // and accessed by it.
  //
  // Return true iff the item is successfully revoked.  If the item is not revoked then
  // the caller must assume that it may still execute, for instance because it
  // has been pop'd from the queue concurrent with the revocation request.

  bool RevokeWithTag(Tag tag, unsigned w_idx) {
    bool revoked = false;
    std::unique_lock<OrtMutex> lock(mutex_);
    Elem& e = array_[w_idx];
    ElemState s = e.state.load(std::memory_order_relaxed);
    if (s == ElemState::kReady &&
        e.state.compare_exchange_strong(s, ElemState::kBusy, std::memory_order_acquire)) {
      if (e.tag == tag) {
        unsigned back = back_.load(std::memory_order_relaxed);
        unsigned back_idx = back & kMask;
        if (back_idx != w_idx) {
          // Item is not at the back of the queue, mark it in-place as revoked
          e.tag = Tag();
          e.w = Work();
          e.state.store(ElemState::kRevoked, std::memory_order_release);
          revoked = true;
        } else {
          // Item being removed as still at the back; shift the back pointer over it,
          // and bump the version number.
          e.tag = Tag();
          e.w = Work();
          e.state.store(ElemState::kEmpty, std::memory_order_release);
          back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
          revoked = true;
        }
      } else {
        // Tag mismatch, i.e. work queue slot re-used
        e.state.store(ElemState::kReady, std::memory_order_release);
      }
    }
    return revoked;
  }

  // Size returns current queue size.
  // Can be called by any thread at any time.
  unsigned Size() const {
    return SizeOrNotEmpty<true>();
  }

  // Empty tests whether container is empty.
  // Can be called by any thread at any time.
  bool Empty() const {
    return SizeOrNotEmpty<false>() == 0;
  }

  // Delete all the elements from the queue.
  void Flush() {
    while (!Empty()) {
      PopFront();
    }
  }

 private:
  static const unsigned kMask = kSize - 1;
  static const unsigned kMask2 = (kSize << 1) - 1;

  enum class ElemState : uint8_t {
    kEmpty,
    kBusy,
    kReady,
    kRevoked,
  };

  // Updates to an element are bracketed by a std::memory_order_acquire
  // load from the state, and a std::memory_order_release store.  Accesses
  // to the front/back indices for the work queue use relaxed semantics,
  // with the state of the elements being authoritative.
  //
  // TODO: Revisit whether there is a significant benefit for the current
  // workloads in the complexity here.
  struct Elem {
    std::atomic<ElemState> state;
    Tag tag;
    Work w;
  };

  OrtMutex mutex_;
  // Low log(kSize) + 1 bits in front_ and back_ contain rolling index of
  // front/back, respectively. The remaining bits contain modification counters
  // that are incremented on Push operations. This allows us to (1) distinguish
  // between empty and full conditions (if we would use log(kSize) bits for
  // position, these conditions would be indistinguishable); (2) obtain
  // consistent snapshot of front_/back_ for Size operation using the
  // modification counters.
  std::atomic<unsigned> front_;
  std::atomic<unsigned> back_;
  Elem array_[kSize];

  // SizeOrNotEmpty returns current queue size; if NeedSizeEstimate is false,
  // only whether the size is 0 is guaranteed to be correct.
  // Can be called by any thread at any time.
  template <bool NeedSizeEstimate>
  unsigned SizeOrNotEmpty() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    unsigned front = front_.load(std::memory_order_acquire);
    for (;;) {
      // Capture a consistent snapshot of front/tail.
      unsigned back = back_.load(std::memory_order_acquire);
      unsigned front1 = front_.load(std::memory_order_relaxed);
      if (front != front1) {
        front = front1;
        std::atomic_thread_fence(std::memory_order_acquire);
        continue;
      }
      if (NeedSizeEstimate) {
        return CalculateSize(front, back);
      }
        // This value will be 0 if the queue is empty, and undefined otherwise.
        unsigned maybe_zero = ((front ^ back) & kMask2);
        // Queue size estimate must agree with maybe zero check on the queue
        // empty/non-empty state.
        eigen_assert((CalculateSize(front, back) == 0) == (maybe_zero == 0));
        return maybe_zero;
    }
  }

  EIGEN_ALWAYS_INLINE
  unsigned CalculateSize(unsigned front, unsigned back) const {
    int size = (front & kMask2) - (back & kMask2);
    // Fix overflow.
    if (size < 0)
      size += 2 * kSize;
    // Order of modification in push/pop is crafted to make the queue look
    // larger than it is during concurrent modifications. E.g. push can
    // increment size before the corresponding pop has decremented it.
    // So the computed size can be up to kSize + 1, fix it.
    if (size > static_cast<int>(kSize))
      size = kSize;
    return static_cast<unsigned>(size);
  }

  RunQueue(const RunQueue&) = delete;
  void operator=(const RunQueue&) = delete;
};

static std::atomic<uint32_t> next_tag{1};

template <typename Environment>
class ThreadPoolTempl : public onnxruntime::concurrency::ExtendedThreadPoolInterface {

 private:
  static unsigned WorkerLoop(int id, Eigen::ThreadPoolInterface* param) {
    // unsafe downcast
    ThreadPoolTempl* this_ptr = (ThreadPoolTempl*)param;
    this_ptr->WorkerLoop(id);
    return 0;
  }

 public:
  typedef typename Environment::Task Task;

  struct Tag {
    constexpr Tag() : v_(0) {
    }

    Tag(uint32_t v) : v_(v) {
    }

    // Allocate a new tag to use to identify work items from a given thread
    // in RunInParallel.  Ideally, threads will have unique tags, but re-use
    // is not incorrect if the counter wraps (for intsance, if a long-running
    // workload is calling into ORT from a fresh thread for each request).
    // We must not re-use the default tag 0 which is used to identify work
    // items added via Schedule as opposed to requests for help in RunInParallel.

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
  };

  static Tag GetNextTag() {
    return Tag(next_tag++);
  }

  typedef RunQueue<Task, Tag, 1024> Queue;
#ifdef _WIN32
  using CHAR_TYPE = wchar_t;
#else
  using CHAR_TYPE = char;
#endif
  ThreadPoolTempl(const CHAR_TYPE* name, int num_threads, bool allow_spinning, Environment& env,
                  const ThreadOptions& thread_options)
      : env_(env),
        num_threads_(num_threads),
        allow_spinning_(allow_spinning),
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
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = worker_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      int q_idx = Rand(&pt->rand) % num_threads_;
      WorkerData &td = worker_data_[q_idx];
      Queue& q = td.queue;
      t = q.PushBack(std::move(t));
      if (!t.f) {
        // The queue accepted the work; ensure that the thread will pick it up
        td.EnsureAwake();
      }
    }

    // Run the work directly if the queue rejected the work
    if (t.f) {
      env_.ExecuteTask(t);
    }
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

void GetGoodWorkerHints(int n, std::vector<unsigned>& good_hints, std::vector<unsigned>& alt_hints) {
  PerThread* pt = GetPerThread();
  int need_alt = n;
  good_hints.clear();
  alt_hints.clear();

  // Iterate through the words of hints, starting from a pseudo-randomly chosen
  // base.  This aims to distribute work across large machines in cases we
  // have multiple threads scheduling work concurrently.

  unsigned base = Rand(&pt->rand) % num_hint_words_;
  for (int i = 0; n && (i < num_hint_words_); i++) {
    int u64_idx = (base + i) % num_hint_words_;
    std::atomic<uint64_t>* u64 = &good_worker_hints_[u64_idx];
    uint64_t saw = u64->load();
    uint64_t want = saw;

    // Pick up to n bits that are set in the current word
    for (int j = 0; n && (j < bits_per_hint_word_); j++) {
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

void RunInParallel(std::function<void()> fn, unsigned n) override {
  PerThread* my_pt = GetPerThread();
  assert(n>=1);
  if (n == 1 || my_pt->in_parallel) {
    fn();
  } else {
    // We build a list of <thread,idx> pairs for each of the queues that accepts a work
    // item.  This lets us remove any work items that do not get executed by the threads
    // that we push them to.
    std::vector<std::pair<int, unsigned>> pending_items;
    Barrier b(n, allow_spinning_);

    my_pt->in_parallel = true;
    if (!my_pt->tag.Get()) {
      my_pt->tag = Tag::GetNext();
    }

    // Push up to n-1 copies of the work item into the queues
    std::vector<unsigned> good_hints, alt_hints;
    GetGoodWorkerHints(n - 1, good_hints, alt_hints);
    for (unsigned i = 0; i < n - 1; i++) {
      Task t = env_.CreateTask([&b, &fn]() {
        fn();
        b.Notify(1);
      });
      int q_idx;
      if (i < good_hints.size()) {
        q_idx = good_hints[i];
      } else {
        auto alt_i = i - static_cast<unsigned>(good_hints.size());
        if (alt_i < alt_hints.size()) {
          q_idx = alt_hints[alt_i];
        } else {
          q_idx = Rand(&my_pt->rand) % num_threads_;
        }
      }
      WorkerData& td = worker_data_[q_idx];
      Queue& q = td.queue;
      unsigned w_idx;
      t = q.PushBackWithTag(std::move(t), my_pt->tag, w_idx);
      if (t.f) {
        // The queue rejected the work.  Account for the missing capacity for work
        // on the synchronization barrier.  The semantics for RunInParallel are that
        // the function is called with up to n-way parallelism, and so the
        // work itself will be performed in the current thread's call to fn()
        // after finishing adding work to the pool.
        b.Notify(1);
      } else {
        // The queue accepted the work, ensure that the thread is servicing the queue
        pending_items.push_back({q_idx, w_idx});
        td.EnsureAwake();
      }
    }

    // Run the final copy ourselves, for the total of n degree-of-parallelism
    fn();

    // Notify the barrier for the work we completed, plus any work that we successfully
    // revoke from the work queues
    int notifications_needed = 1;
    for (auto& item : pending_items) {
      Queue& q = worker_data_[item.first].queue;
      if (q.RevokeWithTag(my_pt->tag, item.second)) {
        notifications_needed++;
      }
    }
    b.Notify(notifications_needed);

    // Synchronize with any work items that are still running
    b.Wait();
    my_pt->in_parallel = false;
  }
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
    }
    ThreadPoolTempl* pool;             // Parent pool, or null for normal threads.
    uint64_t rand{0};                  // Random generator state.
    int thread_id{-1};                 // Worker thread index in pool.
    Tag tag{};                         // Work item tag used to identify this thread.
    bool in_parallel{false};           // Inside a parallel section (hence tag not unique if we re-use)
  };

  static_assert(std::is_trivially_destructible<PerThread>::value, "Per-thread state should be trivially destructible");

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

  static const int bits_per_hint_word_ = 4;
  int num_hint_words_;
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

    while (!cancelled_ && !should_exit) {
        Task t = q.PopFront();
        if (!t.f) {
          // Spin waiting for work.  We indicate, via SetGOodWorkerHint that we are
          // spinning.  This will bias other threads toward pushing work to our queue.
          // In addition, priodically make a best-effort attempt to steal from other
          // threads which are not themselves spinning.

          SetGoodWorkerHint(thread_id, true);
          for (int i = 0; i < spin_count && !t.f && !cancelled_ && !done_; i++) {
            t = (i%steal_count == 0) ? TrySteal() : q.PopFront();
          }
          SetGoodWorkerHint(thread_id, false);

          if (!t.f) {
            // No work passed to us while spinning; make a further full attempt to
            // steal work from other threads prior to blocking.
            if (num_threads_ != 1) {
              t = Steal(true /* true => check all queues */);
            }
            if (!t.f) {
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
        if (t.f) {
          td.SetActive();
          env_.ExecuteTask(t);
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
          if (t.f) {
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

  EIGEN_STRONG_INLINE PerThread* GetPerThread() {
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

}  // namespace onnxruntime
