// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Offline autotuner for the Intel subgroup-matrix MatMul kernel.
//
// This is NOT a correctness test. It sweeps a grid of (M, N, K) problems and,
// for each problem, benchmarks every valid tile + split-K config (enumerated by
// EnumerateSgMatMulConfigs) on the real device, then writes the fastest config
// per problem to a CSV. A companion Python script
// (tools/python/gen_sgmm_tuned_table.py) turns that CSV into a generated lookup
// table consumed by the kernel's SelectConfig.
//
// It only runs when the ORT_WEBGPU_SGMM_TUNE environment variable is set, since
// it is long-running and only produces useful data on Intel Xe2/Xe3 hardware
// that exposes the 8x16x16 F16 subgroup-matrix config. Set ORT_WEBGPU_SGMM_TUNE
// to 1 to enable, and optionally ORT_WEBGPU_SGMM_TUNE_OUT to a CSV path.
//
// Example:
//   set ORT_WEBGPU_SGMM_TUNE=1
//   set ORT_WEBGPU_SGMM_TUNE_OUT=D:\tmp\sgmm_tuning.csv
//   onnxruntime_provider_test.exe --gtest_filter=WebGpuSgMatMulTuning.*

#if defined(__wasm__)
// The Intel subgroup-matrix kernel and its tuning hooks are not built for wasm.
#else

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gsl/gsl>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"

#include "core/common/common.h"
#include "core/framework/run_options.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/platform/env_var.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_matmul_tuning.h"
#include "core/session/inference_session.h"

#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
namespace {

namespace wi = onnxruntime::webgpu::intel;

// Benchmark knobs.
constexpr int kWarmupRuns = 5;
constexpr int kTimedRuns = 25;

// (M, N, K) sweep over (0, 4096]. Includes powers of two, odd/unaligned sizes,
// and a few skewed shapes so the table covers the regimes the heuristic guesses
// at (few large tiles -> split-K vs. many tiles -> no split).
constexpr uint32_t kDimGrid[] = {8, 16, 32, 48, 64, 96, 128, 256, 384,
                                 512, 768, 1024, 1536, 2048, 3072, 4096};

struct Measurement {
  wi::SgMatMulConfig config;
  double gpu_us = -1.0;   // median GPU kernel time (us); -1 if timestamps absent
  double wall_us = -1.0;  // median wall-clock per Run() (us), incl. readback
};

double Median(std::vector<double>& v) {
  if (v.empty()) {
    return -1.0;
  }
  std::sort(v.begin(), v.end());
  const size_t mid = v.size() / 2;
  return (v.size() % 2 == 0) ? 0.5 * (v[mid - 1] + v[mid]) : v[mid];
}

// Builds a serialized single-node f16 MatMul model: A[M,K] @ B[K,N] -> Y[M,N],
// with B as a constant initializer (matching the inference prepack path). The
// activation input A is returned in `feeds` ready to pass to Session::Run.
std::string BuildMatMulModel(uint32_t M, uint32_t N, uint32_t K,
                             NameMLValMap& feeds,
                             std::vector<std::string>& output_names) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 14;

  Model model("SgMatMulTuner", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  std::vector<MLFloat16> a_data(static_cast<size_t>(M) * K);
  std::vector<MLFloat16> b_data(static_cast<size_t>(K) * N);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = MLFloat16(static_cast<float>((i % 7) - 3) * 0.1f);
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = MLFloat16(static_cast<float>((i % 5) - 2) * 0.1f);
  }

  auto* a = builder.MakeInput<MLFloat16>({static_cast<int64_t>(M), static_cast<int64_t>(K)}, a_data);
  auto* b = builder.MakeInitializer<MLFloat16>({static_cast<int64_t>(K), static_cast<int64_t>(N)}, b_data);
  auto* y = builder.MakeOutput();
  builder.AddNode("MatMul", {a, b}, {y});
  builder.SetGraphOutputs();

  EXPECT_STATUS_OK(graph.Resolve());
  feeds = builder.feeds_;
  output_names = builder.output_names_;

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// Sums GPU kernel durations (profiler "Api" category events) per Run, returning
// one duration per timed run. Empty if the device produced no timestamp events.
std::vector<double> ParseGpuDurations(const std::string& profile_path) {
  std::vector<double> per_event;
  std::ifstream stream(profile_path);
  if (!stream.is_open()) {
    return per_event;
  }
  nlohmann::json events;
  try {
    stream >> events;
  } catch (const std::exception&) {
    return per_event;
  }
  if (!events.is_array()) {
    return per_event;
  }
  for (const auto& e : events) {
    if (!e.is_object() || !e.contains("cat") || !e.contains("dur")) {
      continue;
    }
    if (e["cat"].get<std::string>() == "Api") {
      per_event.push_back(static_cast<double>(e["dur"].get<long long>()));
    }
  }
  return per_event;
}

// Times one already-configured MatMul on a fresh session + EP (sharing a WebGPU
// EP across sessions can dangle the session profiler pointer). The caller is
// responsible for installing/clearing any subgroup-matrix config override or
// disable toggle before/after the call. Returns {gpu_us, wall_us}; either is
// -1 if the device produced no timestamps / the run failed.
struct TimingResult {
  double gpu_us = -1.0;
  double wall_us = -1.0;
};

TimingResult RunTimedSession(const std::string& model_data,
                             const NameMLValMap& feeds,
                             const std::vector<std::string>& output_names) {
  TimingResult r;

  // Idle briefly before each bench so the GPU settles (avoids thermal/clock
  // carry-over from the previous run skewing the measurement).
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  auto profile_prefix = std::filesystem::temp_directory_path() / ORT_TSTR("sgmm_tune");

  SessionOptions so;
  so.enable_profiling = true;
  so.profile_file_prefix = profile_prefix.native();

  InferenceSessionWrapper session{so, GetEnvironment()};
  auto ep = DefaultWebGpuExecutionProvider();
  if (ep == nullptr || !session.RegisterExecutionProvider(std::move(ep)).IsOK()) {
    return r;
  }
  if (!session.Load(model_data.data(), static_cast<int>(model_data.size())).IsOK() ||
      !session.Initialize().IsOK()) {
    return r;
  }

  RunOptions run_options;
  std::vector<OrtValue> fetches;

  for (int i = 0; i < kWarmupRuns; ++i) {
    if (!session.Run(run_options, feeds, output_names, &fetches).IsOK()) {
      return r;
    }
  }

  std::vector<double> wall;
  wall.reserve(kTimedRuns);
  for (int i = 0; i < kTimedRuns; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!session.Run(run_options, feeds, output_names, &fetches).IsOK()) {
      return r;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  const std::string profile_path = session.EndProfiling();
  auto cleanup = gsl::finally([&profile_path] {
    std::error_code ec;
    std::filesystem::remove(profile_path, ec);
  });

  // GPU events accumulate over warmup + timed runs; drop the warmup prefix.
  std::vector<double> gpu = ParseGpuDurations(profile_path);
  if (gpu.size() > static_cast<size_t>(kWarmupRuns)) {
    gpu.erase(gpu.begin(), gpu.begin() + kWarmupRuns);
    r.gpu_us = Median(gpu);
  }
  r.wall_us = Median(wall);
  return r;
}

// Benchmarks one subgroup-matrix config on one problem.
Measurement BenchmarkConfig(const std::string& model_data,
                            const NameMLValMap& feeds,
                            const std::vector<std::string>& output_names,
                            const wi::SgMatMulConfig& config) {
  Measurement m;
  m.config = config;

  wi::SetSgMatMulConfigOverride(config);
  auto clear_override = gsl::finally([] { wi::SetSgMatMulConfigOverride(std::nullopt); });

  const TimingResult r = RunTimedSession(model_data, feeds, output_names);
  m.gpu_us = r.gpu_us;
  m.wall_us = r.wall_us;
  return m;
}

// Benchmarks the generic (non-subgroup-matrix) MatMul path as a baseline by
// disabling the subgroup-matrix kernel for the duration of the run.
TimingResult BenchmarkBaseline(const std::string& model_data,
                               const NameMLValMap& feeds,
                               const std::vector<std::string>& output_names) {
  wi::SetSgMatMulDisabled(true);
  auto reenable = gsl::finally([] { wi::SetSgMatMulDisabled(false); });
  return RunTimedSession(model_data, feeds, output_names);
}

}  // namespace

// Sweeps the (M, N, K) grid, benchmarks every valid config per problem plus the
// generic (non-subgroup-matrix) baseline, and writes the best config (lowest GPU
// time, falling back to wall time) alongside the baseline and speedup to a CSV.
TEST(WebGpuSgMatMulTuning, SweepAndEmitCsv) {
  if (onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE").empty()) {
    GTEST_SKIP() << "Set ORT_WEBGPU_SGMM_TUNE=1 to run the subgroup-matrix MatMul autotuner.";
  }

  std::string out_path = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE_OUT");
  if (out_path.empty()) {
    out_path = "sgmm_tuning.csv";
  }

  // ORT_WEBGPU_SGMM_TUNE=2 also logs every candidate config (very chatty).
  const bool verbose = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SGMM_TUNE") == "2";

  std::ofstream csv(out_path, std::ios::trunc);
  ASSERT_TRUE(csv.is_open()) << "Cannot open output CSV: " << out_path;
  csv << "arch,M,N,K,tile_m,tile_n,split_k,gpu_us,wall_us,baseline_gpu_us,baseline_wall_us,speedup\n";

  // Pre-count problems that have at least one valid config, for progress %.
  size_t total_problems = 0;
  for (uint32_t M : kDimGrid) {
    for (uint32_t N : kDimGrid) {
      for (uint32_t K : kDimGrid) {
        if (!wi::EnumerateSgMatMulConfigs(M, N, K).empty()) {
          ++total_problems;
        }
      }
    }
  }

  std::cout << "[sgmm-tune] sweeping " << total_problems << " problems over a "
            << std::size(kDimGrid) << "^3 (M,N,K) grid; output -> " << out_path << std::endl;

  const auto sweep_start = std::chrono::steady_clock::now();
  size_t problem_index = 0;
  // Device architecture (e.g. "xe-2lpg"), captured after the first MatMul runs;
  // each tuned table is keyed by this so distinct GPU arches get distinct tables.
  std::string arch;
  // Best-config speedup over the generic baseline per problem, for the summary
  // distribution printed at the end.
  std::vector<double> speedups;

  for (uint32_t M : kDimGrid) {
    for (uint32_t N : kDimGrid) {
      for (uint32_t K : kDimGrid) {
        const std::vector<wi::SgMatMulConfig> configs = wi::EnumerateSgMatMulConfigs(M, N, K);
        if (configs.empty()) {
          continue;
        }

        ++problem_index;
        const double elapsed_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - sweep_start).count();
        std::cout << "[sgmm-tune] (" << problem_index << "/" << total_problems << ", "
                  << static_cast<int>(100.0 * problem_index / total_problems) << "%, "
                  << static_cast<int>(elapsed_s) << "s) M=" << M << " N=" << N << " K=" << K
                  << " : benchmarking " << configs.size() << " configs..." << std::endl;

        NameMLValMap feeds;
        std::vector<std::string> output_names;
        const std::string model_data = BuildMatMulModel(M, N, K, feeds, output_names);
        ASSERT_FALSE(model_data.empty());
        ASSERT_FALSE(feeds.empty());

        // Baseline: the generic WebGPU MatMul path (subgroup-matrix disabled).
        const TimingResult baseline = BenchmarkBaseline(model_data, feeds, output_names);
        const double baseline_metric = (baseline.gpu_us > 0.0) ? baseline.gpu_us : baseline.wall_us;

        // The arch is known once at least one MatMul has executed on the device.
        if (arch.empty()) {
          arch = wi::GetSgMatMulDeviceArch();
          if (arch.empty()) {
            arch = "unknown";
          }
          std::cout << "[sgmm-tune] device architecture: " << arch << std::endl;
        }

        if (verbose) {
          std::cout << "[sgmm-tune]     baseline (generic) -> gpu=" << baseline.gpu_us
                    << "us wall=" << baseline.wall_us << "us"
                    << (baseline_metric <= 0.0 ? " (FAILED)" : "") << std::endl;
        }

        Measurement best;
        bool have_best = false;
        for (const auto& config : configs) {
          const Measurement m = BenchmarkConfig(model_data, feeds, output_names, config);
          const double m_metric = (m.gpu_us > 0.0) ? m.gpu_us : m.wall_us;
          if (verbose) {
            std::cout << "[sgmm-tune]     tile " << config.tile_m << "x" << config.tile_n
                      << " split_k=" << config.split_k << " -> gpu=" << m.gpu_us
                      << "us wall=" << m.wall_us << "us"
                      << (m_metric <= 0.0 ? " (FAILED)" : "") << std::endl;
          }
          if (m_metric <= 0.0) {
            continue;  // config failed to run; skip
          }
          const double best_metric = (best.gpu_us > 0.0) ? best.gpu_us : best.wall_us;
          if (!have_best || m_metric < best_metric) {
            best = m;
            have_best = true;
          }
        }

        if (have_best) {
          const double best_metric = (best.gpu_us > 0.0) ? best.gpu_us : best.wall_us;
          const double speedup =
              (baseline_metric > 0.0 && best_metric > 0.0) ? baseline_metric / best_metric : -1.0;
          if (speedup > 0.0) {
            speedups.push_back(speedup);
          }
          csv << arch << "," << M << "," << N << "," << K << ","
              << best.config.tile_m << "," << best.config.tile_n << "," << best.config.split_k << ","
              << best.gpu_us << "," << best.wall_us << ","
              << baseline.gpu_us << "," << baseline.wall_us << "," << speedup << "\n";
          csv.flush();
          std::cout << "[sgmm-tune]   -> best tile " << best.config.tile_m << "x" << best.config.tile_n
                    << " split_k=" << best.config.split_k << " (gpu=" << best.gpu_us
                    << "us wall=" << best.wall_us << "us) vs baseline gpu=" << baseline.gpu_us
                    << "us wall=" << baseline.wall_us << "us";
          if (speedup > 0.0) {
            std::cout << " => " << speedup << "x";
          }
          std::cout << std::endl;
        } else {
          std::cout << "[sgmm-tune]   -> no config ran successfully (skipped)" << std::endl;
        }
      }
    }
  }

  csv.close();
  const double total_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - sweep_start).count();
  std::cout << "[sgmm-tune] done: " << problem_index << " problems in "
            << static_cast<int>(total_s) << "s -> " << out_path << std::endl;

  // Speedup distribution: how many problems fall into each best-vs-baseline
  // speedup bucket. Buckets are half-open [lo, hi); the last is [10x, inf).
  if (!speedups.empty()) {
    constexpr double kEdges[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0};
    constexpr size_t kNumEdges = std::size(kEdges);
    size_t counts[kNumEdges + 1] = {0};  // one extra bucket for >= last edge
    for (double s : speedups) {
      size_t bucket = kNumEdges;  // default: >= last edge
      for (size_t i = 0; i < kNumEdges; ++i) {
        if (s < kEdges[i]) {
          bucket = i;
          break;
        }
      }
      ++counts[bucket];
    }

    const size_t total = speedups.size();
    std::cout << "[sgmm-tune] speedup distribution over " << total << " problems:" << std::endl;
    for (size_t i = 0; i <= kNumEdges; ++i) {
      std::ostringstream label;
      if (i == 0) {
        label << "      < " << kEdges[0] << "x";
      } else if (i == kNumEdges) {
        label << "    >= " << kEdges[kNumEdges - 1] << "x";
      } else {
        label << "[" << kEdges[i - 1] << "x, " << kEdges[i] << "x)";
      }
      const double pct = 100.0 * static_cast<double>(counts[i]) / static_cast<double>(total);
      std::cout << "[sgmm-tune]   " << label.str() << " : " << counts[i]
                << " (" << static_cast<int>(pct + 0.5) << "%)" << std::endl;
    }
  }

  GTEST_LOG_(INFO) << "Subgroup-matrix MatMul tuning written to " << out_path;
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
