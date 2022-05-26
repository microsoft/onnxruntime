#include <limits>

#include "common.h"

#include <benchmark/benchmark.h>
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/mlas/lib/mlasi.h"
#include "core/providers/cpu/tensor/upsample.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/util/thread_utils.h"

using namespace onnxruntime;

template <typename T>
static void BM_NhwcUpsampleBilinear(benchmark::State& state) {
  const int64_t output_height = static_cast<int64_t>(state.range(0));
  const int64_t output_width = static_cast<int64_t>(state.range(1));
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_channels = 256;
  constexpr int64_t input_height = 32;
  constexpr int64_t input_width = 32;
  const int64_t height_scale = output_height / input_height;
  const int64_t width_scale = output_width / input_width;
  const std::vector<float> roi{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  constexpr bool use_extrapolation = false;
  constexpr float extrapolation_value = 0;
  constexpr size_t XdataBaseSize = batch_size * num_channels * input_height * input_width;
  const T* const XdataBase = GenerateArrayWithRandomValue<T>(XdataBaseSize, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  const size_t YdataBaseSize = batch_size * num_channels * output_height * output_width;
  T* const YdataBase = (T*)aligned_alloc(sizeof(T) * YdataBaseSize, 64);
  AllocatorPtr alloc = std::make_shared<CPUAllocator>();
  const GetOriginalCoordinateFunc& get_original_coordinate =
      [](float x_resized, float x_scale, float, float, float, float) {
        return x_resized / x_scale;
      };
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));

  for (auto _ : state) {
    NhwcUpsampleBilinear(batch_size, num_channels, input_height, input_width, output_height, output_width,
                         static_cast<float>(height_scale), static_cast<float>(width_scale), roi,
                         use_extrapolation, extrapolation_value, XdataBase,
                         YdataBase, alloc, get_original_coordinate,
                         output_height * output_width * num_channels > 64 ? tp.get() : nullptr);
  }
}

BENCHMARK_TEMPLATE(BM_NhwcUpsampleBilinear, uint8_t)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({96, 96})
    ->Args({128, 128})
    ->Args({160, 160})
    ->Args({1, 1000000});

BENCHMARK_TEMPLATE(BM_NhwcUpsampleBilinear, int8_t)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({96, 96})
    ->Args({128, 128})
    ->Args({160, 160})
    ->Args({1, 1000000});

BENCHMARK_TEMPLATE(BM_NhwcUpsampleBilinear, float)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({32, 32})
    ->Args({64, 64})
    ->Args({96, 96})
    ->Args({128, 128})
    ->Args({160, 160})
    ->Args({1, 1000000});
