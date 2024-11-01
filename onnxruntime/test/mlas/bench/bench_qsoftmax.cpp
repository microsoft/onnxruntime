#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/util/thread_utils.h"

static const std::vector<std::string> qsoftmax_bench_arg_names = {"N", "D", "is_signed"};
//(const void* Input, void* Output, size_t N, size_t D, const float* LoopupTable,float Scale, int ZeroPoint, bool is_signed, MLAS_THREADPOOL* ThreadPool);
void BM_Qsoftmax(benchmark::State& state) {
  size_t N = static_cast<size_t>(state.range(0));
  size_t D = static_cast<size_t>(state.range(1));
  bool is_signed = static_cast<bool>(state.range(0));
  const size_t count = N * D;
  auto src = RandomVectorUniform<int8_t>(count, 0, 127);
  auto LoopupTable = RandomVectorUniform<float>(count, 0.f, 1e10);
  auto dst = std::vector<int8_t>(count + 16);
  int8_t* dst_start = dst.data();
  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = 8;
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  // Warm up
  MlasComputeQSoftmax(src.data(), dst_start, N, D, LoopupTable.data(), 0.1f, 1.0f, 0, is_signed, tp.get());

  for (auto _ : state) {
    MlasComputeQSoftmax(src.data(), dst_start, N, D, LoopupTable.data(), 0.1f, 1.0f, 0, is_signed, tp.get());
  }
}

BENCHMARK(BM_Qsoftmax)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames(qsoftmax_bench_arg_names);
      b->ArgsProduct({{1971, 20}, {81, 1000}, {0, 1}});
    });
