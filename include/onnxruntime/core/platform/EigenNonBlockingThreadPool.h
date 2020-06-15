// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* Modifications Copyright (c) Microsoft. */

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
#include "core/platform/ort_mutex.h"
#include "core/platform/Barrier.h"

namespace onnxruntime {


class EventCount {
 public:
  class Waiter;

  explicit EventCount(Eigen::MaxSizeVector<Waiter>& waiters) : state_(kStackMask), waiters_(waiters) {
    assert(waiters.size() < (1 << kWaiterBits) - 1);
  }

#ifdef NDEBUG
  ~EventCount() = default;
#else
  ~EventCount() {
    // Ensure there are no waiters.
    assert(state_.load() == kStackMask);
  }
#endif
  // Prewait prepares for waiting.
  // After calling Prewait, the thread must re-check the wait predicate
  // and then call either CancelWait or CommitWait.
  void Prewait() {
    uint64_t state = state_.load(std::memory_order_relaxed);
    for (;;) {
      CheckState(state);
      uint64_t newstate = state + kWaiterInc;
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_seq_cst))
        return;
    }
  }

  // CommitWait commits waiting after Prewait.
  void CommitWait(Waiter* w) {
    assert((w->epoch & ~kEpochMask) == 0);
    w->state = Waiter::kNotSignaled;
    const uint64_t me = (w - &waiters_[0]) | w->epoch;
    uint64_t state = state_.load(std::memory_order_seq_cst);
    for (;;) {
      CheckState(state, true);
      uint64_t newstate;
      if ((state & kSignalMask) != 0) {
        // Consume the signal and return immidiately.
        newstate = state - kWaiterInc - kSignalInc;
      } else {
        // Remove this thread from pre-wait counter and add to the waiter stack.
        newstate = ((state & kWaiterMask) - kWaiterInc) | me;
        w->next.store(state & (kStackMask | kEpochMask), std::memory_order_relaxed);
      }
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel)) {
        if ((state & kSignalMask) == 0) {
          w->epoch += kEpochInc;
          Park(w);
        }
        return;
      }
    }
  }

  // CancelWait cancels effects of the previous Prewait call.
  void CancelWait() {
    uint64_t state = state_.load(std::memory_order_relaxed);
    for (;;) {
      CheckState(state, true);
      uint64_t newstate = state - kWaiterInc;
      // We don't know if the thread was also notified or not,
      // so we should not consume a signal unconditionaly.
      // Only if number of waiters is equal to number of signals,
      // we know that the thread was notified and we must take away the signal.
      if (((state & kWaiterMask) >> kWaiterShift) == ((state & kSignalMask) >> kSignalShift))
        newstate -= kSignalInc;
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel))
        return;
    }
  }

  // Notify wakes one or all waiting threads.
  // Must be called after changing the associated wait predicate.
  void Notify(bool notifyAll) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t state = state_.load(std::memory_order_acquire);
    for (;;) {
      CheckState(state);
      const uint64_t waiters = (state & kWaiterMask) >> kWaiterShift;
      const uint64_t signals = (state & kSignalMask) >> kSignalShift;
      // Easy case: no waiters.
      if ((state & kStackMask) == kStackMask && waiters == signals)
        return;
      uint64_t newstate;
      if (notifyAll) {
        // Empty wait stack and set signal to number of pre-wait threads.
        newstate = (state & kWaiterMask) | (waiters << kSignalShift) | kStackMask;
      } else if (signals < waiters) {
        // There is a thread in pre-wait state, unblock it.
        newstate = state + kSignalInc;
      } else {
        // Pop a waiter from list and unpark it.
        Waiter* w = &waiters_[state & kStackMask];
        uint64_t next = w->next.load(std::memory_order_relaxed);
        newstate = (state & (kWaiterMask | kSignalMask)) | next;
      }
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel)) {
        if (!notifyAll && (signals < waiters))
          return;  // unblocked pre-wait thread
        if ((state & kStackMask) == kStackMask)
          return;
        Waiter* w = &waiters_[state & kStackMask];
        if (!notifyAll)
          w->next.store(kStackMask, std::memory_order_relaxed);
        Unpark(w);
        return;
      }
    }
  }

  class Waiter {
    friend class EventCount;
    // Align to 128 byte boundary to prevent false sharing with other Waiter
    // objects in the same vector.
    EIGEN_ALIGN_TO_BOUNDARY(128) std::atomic<uint64_t> next;
    OrtMutex mu;
    OrtCondVar cv;
    uint64_t epoch = 0;
    unsigned state = kNotSignaled;
    enum {
      kNotSignaled,
      kWaiting,
      kSignaled,
    };
  };

 private:
  // State_ layout:
  // - low kWaiterBits is a stack of waiters committed wait
  //   (indexes in waiters_ array are used as stack elements,
  //   kStackMask means empty stack).
  // - next kWaiterBits is count of waiters in prewait state.
  // - next kWaiterBits is count of pending signals.
  // - remaining bits are ABA counter for the stack.
  //   (stored in Waiter node and incremented on push).
  static constexpr uint64_t kWaiterBits = 14;
  static constexpr uint64_t kStackMask = (1ull << kWaiterBits) - 1;
  static constexpr uint64_t kWaiterShift = kWaiterBits;
  static constexpr uint64_t kWaiterMask = ((1ull << kWaiterBits) - 1) << kWaiterShift;
  static constexpr uint64_t kWaiterInc = 1ull << kWaiterShift;
  static constexpr uint64_t kSignalShift = 2 * kWaiterBits;
  static constexpr uint64_t kSignalMask = ((1ull << kWaiterBits) - 1) << kSignalShift;
  static constexpr uint64_t kSignalInc = 1ull << kSignalShift;
  static constexpr uint64_t kEpochShift = 3 * kWaiterBits;
  static constexpr uint64_t kEpochBits = 64 - kEpochShift;
  static constexpr uint64_t kEpochMask = ((1ull << kEpochBits) - 1) << kEpochShift;
  static constexpr uint64_t kEpochInc = 1ull << kEpochShift;
  std::atomic<uint64_t> state_;
  Eigen::MaxSizeVector<Waiter>& waiters_;

#ifdef NDEBUG
  static void CheckState(uint64_t, bool) {
  }
  static void CheckState(uint64_t) {
  }
#else
  static void CheckState(uint64_t state, bool waiter = false) {
    static_assert(kEpochBits >= 20, "not enough bits to prevent ABA problem");
    const uint64_t waiters = (state & kWaiterMask) >> kWaiterShift;
    const uint64_t signals = (state & kSignalMask) >> kSignalShift;
    assert(waiters >= signals);
    assert(waiters < (1 << kWaiterBits) - 1);
    assert(!waiter || waiters > 0);
    (void)waiters;
    (void)signals;
  }
#endif
  static void Park(Waiter* w) {
    std::unique_lock<OrtMutex> lock(w->mu);
    while (w->state != Waiter::kSignaled) {
      w->state = Waiter::kWaiting;
      w->cv.wait(lock);
    }
  }

  void Unpark(Waiter* w) {
    for (Waiter* next; w; w = next) {
      uint64_t wnext = w->next.load(std::memory_order_relaxed) & kStackMask;
      next = wnext == kStackMask ? nullptr : &waiters_[static_cast<size_t>(wnext)];
      unsigned state;
      {
        std::unique_lock<OrtMutex> lock(w->mu);
        state = w->state;
        w->state = Waiter::kSignaled;
      }
      // Avoid notifying if it wasn't waiting.
      if (state == Waiter::kWaiting)
        w->cv.notify_one();
    }
  }

 public:
  EventCount(const EventCount&) = delete;
  void operator=(const EventCount&) = delete;
};
template <typename Work, unsigned kSize>
class RunQueue {
 public:
  RunQueue() : front_(0), back_(0) {
    // require power-of-two for fast masking
    assert((kSize & (kSize - 1)) == 0);
    assert(kSize > 2);            // why would you do this?
    assert(kSize <= (64 << 10));  // leave enough space for counter
    for (unsigned i = 0; i < kSize; i++) array_[i].state.store(kEmpty, std::memory_order_relaxed);
  }

  ~RunQueue() {
    assert(Size() == 0);
  }

  // PushFront inserts w at the beginning of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushFront(Work w) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[front & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    front_.store(front + 1 + (kSize << 1), std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopFront removes and returns the first element in the queue.
  // If the queue was empty returns default-constructed Work.
  Work PopFront() {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[(front - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    front = ((front - 1) & kMask2) | (front & ~kMask2);
    front_.store(front, std::memory_order_relaxed);
    return w;
  }

  // PushBack adds w at the end of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushBack(Work w) {
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[(back - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopBack removes and returns the last elements in the queue.
  Work PopBack() {
    if (Empty())
      return Work();
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[back & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
    return w;
  }

  // PopBackHalf removes and returns half last elements in the queue.
  // Returns number of elements removed.
  unsigned PopBackHalf(std::vector<Work>* result) {
    if (Empty())
      return 0;
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    unsigned size = Size();
    unsigned mid = back;
    if (size > 1)
      mid = back + (size - 1) / 2;
    unsigned n = 0;
    unsigned start = 0;
    for (; static_cast<int>(mid - back) >= 0; mid--) {
      Elem* e = &array_[mid & kMask];
      uint8_t s = e->state.load(std::memory_order_relaxed);
      if (n == 0) {
        if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
          continue;
        start = mid;
      } else {
        // Note: no need to store temporal kBusy, we exclusively own these
        // elements.
        assert(s == kReady);
      }
      result->push_back(std::move(e->w));
      e->state.store(kEmpty, std::memory_order_release);
      n++;
    }
    if (n != 0)
      back_.store(start + 1 + (kSize << 1), std::memory_order_relaxed);
    return n;
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
  struct Elem {
    std::atomic<uint8_t> state;
    Work w;
  };
  enum {
    kEmpty,
    kBusy,
    kReady,
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

template <typename Environment>
class ThreadPoolTempl : public Eigen::ThreadPoolInterface {
 private:
  static unsigned WorkerLoop(int id, Eigen::ThreadPoolInterface* param) {
    // unsafe downcast
    ThreadPoolTempl* this_ptr = (ThreadPoolTempl*)param;
    this_ptr->WorkerLoop(id);
    return 0;
  }

 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;
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
        thread_data_(num_threads),
        all_coprimes_(num_threads),
        waiters_(num_threads),
        global_steal_partition_(EncodePartition(0, num_threads_)),
        blocked_(0),
        spinning_(false),
        done_(false),
        cancelled_(false),
        ec_(waiters_) {
    waiters_.resize(num_threads_);
    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    assert(num_threads_ < kMaxThreads);
    for (int i = 1; i <= num_threads_; ++i) {
      all_coprimes_.emplace_back(i);
      ComputeCoprimes(i, &all_coprimes_.back());
    }

    thread_data_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
      SetStealPartition(i, EncodePartition(0, num_threads_));
      thread_data_[i].thread.reset(env_.CreateThread(name, i, WorkerLoop, this, thread_options));
    }
  }

  ~ThreadPoolTempl() override {
    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!cancelled_) {
      ec_.Notify(true);
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < thread_data_.size(); i++) {
        thread_data_[i].queue.Flush();
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < thread_data_.size(); ++i) thread_data_[i].thread.reset();
  }

  void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
    assert(partitions.size() == static_cast<std::size_t>(num_threads_));

    // Pass this information to each thread queue.
    for (int i = 0; i < num_threads_; i++) {
      const auto& pair = partitions[i];
      unsigned start = pair.first;
      unsigned end = pair.second;
      AssertBounds(start, end);
      unsigned val = EncodePartition(start, end);
      SetStealPartition(i, val);
    }
  }

  void Schedule(std::function<void()> fn) override {
    ScheduleWithHint(std::move(fn), 0, num_threads_);
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int limit) override {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = thread_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      assert(start < limit);
      assert(limit <= num_threads_);
      int num_queues = limit - start;
      int rnd = Rand(&pt->rand) % num_queues;
      assert(start + rnd < limit);
      Queue& q = thread_data_[start + rnd].queue;
      t = q.PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f) {
      ec_.Notify(false);
    } else {
      env_.ExecuteTask(t);  // Push failed, execute directly.
    }
  }

  void Cancel() override {
    cancelled_ = true;
    // If done_ is true, which means this object is being destructing.
    // Therefore thread_data_[i].thread could be NULL.
    if (!done_) {
      done_ = true;
      // Let each thread know it's been cancelled.
      for (size_t i = 0; i < thread_data_.size(); i++) {
        assert(thread_data_[i].thread != nullptr);
        thread_data_[i].thread->OnCancel();
      }
    }

    // Wake up the threads without work to let them exit on their own.
    ec_.Notify(true);
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
  // Create a single atomic<int> that encodes start and limit information for
  // each thread.
  // We expect num_threads_ < 65536, so we can store them in a single
  // std::atomic<unsigned>.
  // Exposed publicly as static functions so that external callers can reuse
  // this encode/decode logic for maintaining their own thread-safe copies of
  // scheduling and steal domain(s).
  static const int kMaxPartitionBits = 16;
  static const int kMaxThreads = 1 << kMaxPartitionBits;

  inline unsigned EncodePartition(unsigned start, unsigned limit) {
    return (start << kMaxPartitionBits) | limit;
  }

  inline void DecodePartition(unsigned val, unsigned* start, unsigned* limit) {
    *limit = val & (kMaxThreads - 1);
    val >>= kMaxPartitionBits;
    *start = val;
  }
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
  inline void SetStealPartition(size_t i, unsigned val) {
    thread_data_[i].steal_partition.store(val, std::memory_order_relaxed);
  }

  inline unsigned GetStealPartition(int i) {
    return thread_data_[i].steal_partition.load(std::memory_order_relaxed);
  }

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

  struct PerThread {
    constexpr PerThread() : pool(nullptr) {
    }
    ThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand{0};       // Random generator state.
    int thread_id{-1};      // Worker thread index in pool.
  };

  struct ThreadData {
    constexpr ThreadData() : thread(), steal_partition(0), queue() {
    }
    std::unique_ptr<Thread> thread;
    std::atomic<unsigned> steal_partition;
    Queue queue;
  };

  Environment& env_;
  const int num_threads_;
  const bool allow_spinning_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  Eigen::MaxSizeVector<Eigen::MaxSizeVector<unsigned>> all_coprimes_;
  Eigen::MaxSizeVector<EventCount::Waiter> waiters_;
  unsigned global_steal_partition_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;
  EventCount ec_;

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = GlobalThreadIdHash();
    pt->thread_id = thread_id;
    Queue& q = thread_data_[thread_id].queue;
    EventCount::Waiter* waiter = &waiters_[thread_id];
    // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
    // proportional to num_threads_ and we assume that new work is scheduled at
    // a constant rate, so we set spin_count to 5000 / num_threads_. The
    // constant was picked based on a fair dice roll, tune it.
    const int spin_count = allow_spinning_ && num_threads_ > 0 ? 5000 / num_threads_ : 0;
    if (num_threads_ == 1) {
      // For num_threads_ == 1 there is no point in going through the expensive
      // steal loop. Moreover, since NonEmptyQueueIndex() calls PopBack() on the
      // victim queues it might reverse the order in which ops are executed
      // compared to the order in which they are scheduled, which tends to be
      // counter-productive for the types of I/O workloads the single thread
      // pools tend to be used for.
      while (!cancelled_) {
        Task t = q.PopFront();
        for (int i = 0; i < spin_count && !t.f; i++) {
          if (!cancelled_.load(std::memory_order_relaxed)) {
            t = q.PopFront();
          }
        }
        if (!t.f) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    } else {
      while (!cancelled_) {
        Task t = q.PopFront();
        if (!t.f) {
          t = LocalSteal();
          if (!t.f) {
            t = GlobalSteal();
            if (!t.f) {
              // Leave one thread spinning. This reduces latency.
              if (allow_spinning_ && !spinning_ && !spinning_.exchange(true)) {
                for (int i = 0; i < spin_count && !t.f; i++) {
                  if (!cancelled_.load(std::memory_order_relaxed)) {
                    t = GlobalSteal();
                  } else {
                    return;
                  }
                }
                spinning_ = false;
              }
              if (!t.f) {
                if (!WaitForWork(waiter, &t)) {
                  return;
                }
              }
            }
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    }
  }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.
  Task Steal(unsigned start, unsigned limit) {
    PerThread* pt = GetPerThread();
    const unsigned size = static_cast<unsigned>(limit - start);
    unsigned r = Rand(&pt->rand);
    unsigned victim = r % size;
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];

    for (unsigned i = 0; i < size; i++) {
      assert(start + victim < limit);
      Task t = thread_data_[start + victim].queue.PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }

  // Steals work within threads belonging to the partition.
  Task LocalSteal() {
    PerThread* pt = GetPerThread();
    unsigned partition = GetStealPartition(pt->thread_id);
    // If thread steal partition is the same as global partition, there is no
    // need to go through the steal loop twice.
    if (global_steal_partition_ == partition)
      return Task();
    unsigned start;
    unsigned limit;
    DecodePartition(partition, &start, &limit);
    AssertBounds(start, limit);

    return Steal(start, limit);
  }

  // Steals work from any other thread in the pool.
  Task GlobalSteal() {
    return Steal(0, num_threads_);
  }

  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    assert(!t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait();
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait();
      if (cancelled_) {
        return false;
      }
        *t = thread_data_[victim].queue.PopBack();
        return true;
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    // TODO is blocked_ required to be unsigned?
    if (done_ && blocked_ == static_cast<unsigned>(num_threads_)) {
      ec_.CancelWait();
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
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    // We intentionally design NonEmptyQueueIndex to steal work from
    // anywhere in the queue so threads don't block in WaitForWork() forever
    // when all threads in their partition go to sleep. Steal is still local.
    const unsigned size = static_cast<unsigned>(thread_data_.size());
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!thread_data_[victim].queue.Empty()) {
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
