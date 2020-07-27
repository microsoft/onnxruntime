#include <benchmark/benchmark.h>
#include <core/platform/threadpool.h>
#include <core/util/thread_utils.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/platform/Barrier.h>

#ifdef _WIN32
#include <Windows.h>
#endif
using namespace onnxruntime;
using namespace onnxruntime::concurrency;

static void BM_CreateThreadPool(benchmark::State& state) {
  for (auto _ : state) {
    ThreadPool tp(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), ORT_TSTR(""), 48, true);
  }
}
BENCHMARK(BM_CreateThreadPool)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMillisecond);

//On Xeon W-2123 CPU, it takes about 2ns for each iteration
#ifdef _WIN32
#pragma optimize("", off)
#else
#pragma GCC push_options
#pragma GCC optimize("O0")
#endif
void SimpleForLoop(ptrdiff_t first, ptrdiff_t last) {
  size_t sum = 0;
  for (; first != last; ++first) {
    ++sum;
  }
}
#ifdef _WIN32
#pragma optimize("", on)
#else
#pragma GCC pop_options
#endif

static void BM_ThreadPoolParallelFor(benchmark::State& state) {
  const size_t len = state.range(0);
  const int cost = static_cast<int>(state.range(1));
  OrtThreadPoolParams tpo;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, ThreadPoolType::INTRA_OP));
  for (auto _ : state) {
    tp->ParallelFor(len, cost, SimpleForLoop);
  }
}
BENCHMARK(BM_ThreadPoolParallelFor)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({100, 1})
    ->Args({100, 10})
    ->Args({100, 100})
    ->Args({100, 200})
    ->Args({100, 400})
    ->Args({1000, 1})
    ->Args({1000, 10})
    ->Args({1000, 100})
    ->Args({1000, 200})
    ->Args({1000, 400})
    ->Args({10000, 1})
    ->Args({10000, 10})
    ->Args({10000, 100})
    ->Args({10000, 200})
    ->Args({10000, 400})
    ->Args({15000, 1})
    ->Args({15000, 10})
    ->Args({15000, 100})
    ->Args({15000, 200})
    ->Args({15000, 400})
    ->Args({20000, 1})
    ->Args({20000, 10})
    ->Args({20000, 100})
    ->Args({20000, 200})
    ->Args({20000, 400})
    ->Args({40000, 200})
    ->Args({80000, 200})
    ->Args({160000, 200});

static void BM_SimpleForLoop(benchmark::State& state) {
  const size_t len = state.range(0);
  for (auto _ : state) {
    SimpleForLoop(0, len);
  }
}
BENCHMARK(BM_SimpleForLoop)
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100);

static void TestPartitionWork(std::ptrdiff_t ThreadId, std::ptrdiff_t ThreadCount, std::ptrdiff_t TotalWork,
                              std::ptrdiff_t* WorkIndex, std::ptrdiff_t* WorkRemaining) {
  const std::ptrdiff_t WorkPerThread = TotalWork / ThreadCount;
  const std::ptrdiff_t WorkPerThreadExtra = TotalWork % ThreadCount;

  if (ThreadId < WorkPerThreadExtra) {
    *WorkIndex = (WorkPerThread + 1) * ThreadId;
    *WorkRemaining = WorkPerThread + 1;
  } else {
    *WorkIndex = WorkPerThread * ThreadId + WorkPerThreadExtra;
    *WorkRemaining = WorkPerThread;
  }
}

static void BM_SimpleScheduleWait(benchmark::State& state) {
  const size_t len = state.range(0);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  tpo.thread_pool_size = 0;
  std::unique_ptr<concurrency::ThreadPool> tp(concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, ThreadPoolType::INTRA_OP));
  size_t threads = tp->NumThreads();

  for (auto _ : state) {
    onnxruntime::Barrier barrier(static_cast<unsigned int>(threads));
    for (std::ptrdiff_t id = 0; id < static_cast<std::ptrdiff_t>(threads); ++id) {
      tp->Schedule([id, threads, len, &barrier]() {
        std::ptrdiff_t start, work_remaining;
        TestPartitionWork(id, threads, len, &start, &work_remaining);
        SimpleForLoop(start, start + work_remaining);
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
}
BENCHMARK(BM_SimpleScheduleWait)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000);

#ifdef _WIN32
struct Param {
  std::atomic<int> id = 0;
  std::ptrdiff_t threads;
  std::ptrdiff_t len;
  onnxruntime::Barrier* barrier;
};
VOID NTAPI SimpleCalc(_Inout_ PTP_CALLBACK_INSTANCE, _Inout_ PVOID Context, _Inout_ PTP_WORK) {
  Param* p = (Param*)Context;
  std::ptrdiff_t start, work_remaining;
  int id = p->id++;
  TestPartitionWork(id, p->threads, p->len, &start, &work_remaining);
  SimpleForLoop(start, start + work_remaining);
  p->barrier->Notify();
}

static void BM_SimpleScheduleWaitWindowsTP(benchmark::State& state) {
  const size_t len = state.range(0);
  const int threads = 4;
  Param p;
  for (auto _ : state) {
    onnxruntime::Barrier barrier(static_cast<unsigned int>(threads));
    p.len = len;
    p.threads = threads;
    p.barrier = &barrier;
    PTP_WORK works = CreateThreadpoolWork(SimpleCalc, &p, nullptr);
    for (int i = 0; i != threads; ++i) {
      SubmitThreadpoolWork(works);
    }
    barrier.Wait();
    CloseThreadpoolWork(works);
  }
}

BENCHMARK(BM_SimpleScheduleWaitWindowsTP)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000);
#endif