/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm_riscv_bench.cpp

Abstract:

    This module implements a standalone SGEMM benchmark used while tuning the
    RISC-V MLAS path. It is intentionally separate from the Google Benchmark
    suite so it can print pack time, compute time, checksum, and compare RVV
    against scalar execution via ORT_MLAS_RISCV_FORCE_SCALAR.

--*/

#include "mlas.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

struct Options {
  size_t m = 128;
  size_t n = 3072;
  size_t k = 768;
  size_t iters = 20;
  size_t warmup = 3;
  bool pack_b = false;
  bool trans_a = false;
  bool trans_b = false;
  float alpha = 1.0f;
  float beta = 0.0f;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [--m=N] [--n=N] [--k=N] [--iters=N] [--warmup=N]\n"
      << "       [--pack_b=0|1] [--trans_a=0|1] [--trans_b=0|1]\n"
      << "       [--alpha=F] [--beta=F]\n";
}

bool ParseBool(std::string_view value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

float MakeValue(size_t index) {
  uint32_t x = static_cast<uint32_t>(index * 747796405u + 2891336453u);
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  const uint32_t bucket = x % 2048u;
  return (static_cast<float>(bucket) / 1024.0f) - 1.0f;
}

Options ParseArgs(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }

    const auto split = arg.find('=');
    if (split == std::string_view::npos || split == 0 || split + 1 >= arg.size()) {
      continue;
    }

    const std::string_view key = arg.substr(0, split);
    const std::string_view value = arg.substr(split + 1);

    if (key == "--m") {
      options.m = std::strtoull(value.data(), nullptr, 10);
    } else if (key == "--n") {
      options.n = std::strtoull(value.data(), nullptr, 10);
    } else if (key == "--k") {
      options.k = std::strtoull(value.data(), nullptr, 10);
    } else if (key == "--iters") {
      options.iters = std::strtoull(value.data(), nullptr, 10);
    } else if (key == "--warmup") {
      options.warmup = std::strtoull(value.data(), nullptr, 10);
    } else if (key == "--pack_b") {
      options.pack_b = ParseBool(value);
    } else if (key == "--trans_a") {
      options.trans_a = ParseBool(value);
    } else if (key == "--trans_b") {
      options.trans_b = ParseBool(value);
    } else if (key == "--alpha") {
      options.alpha = std::strtof(value.data(), nullptr);
    } else if (key == "--beta") {
      options.beta = std::strtof(value.data(), nullptr);
    }
  }

  return options;
}

template <typename Fn>
double TimeLoop(size_t iterations, Fn&& fn) {
  const auto begin = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    fn();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = ParseArgs(argc, argv);

  if (options.m == 0 || options.n == 0 || options.k == 0 || options.iters == 0) {
    std::cerr << "m, n, k, and iters must be > 0" << std::endl;
    return 1;
  }

  const size_t a_size = options.m * options.k;
  const size_t b_size = options.n * options.k;
  const size_t c_size = options.m * options.n;

  std::vector<float> a(a_size);
  std::vector<float> b(b_size);
  std::vector<float> c(c_size, 0.0f);

  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = MakeValue(i);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = MakeValue(i + a.size());
  }

  const CBLAS_TRANSPOSE trans_a = options.trans_a ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_b = options.trans_b ? CblasTrans : CblasNoTrans;
  const size_t lda = options.trans_a ? options.m : options.k;
  const size_t ldb = options.trans_b ? options.k : options.n;
  const size_t ldc = options.n;

  std::vector<uint8_t> packed_b;
  double pack_ms = 0.0;

  if (options.pack_b) {
    const size_t packed_b_size = MlasGemmPackBSize(trans_a, trans_b, options.n, options.k, nullptr);
    if (packed_b_size == 0) {
      std::cerr << "packing is not supported for this configuration" << std::endl;
      return 2;
    }

    packed_b.resize(packed_b_size);

    pack_ms = TimeLoop(options.iters, [&]() {
      MlasGemmPackB(trans_a, trans_b, options.n, options.k, b.data(), ldb, packed_b.data(), nullptr);
    });

    MlasGemmPackB(trans_a, trans_b, options.n, options.k, b.data(), ldb, packed_b.data(), nullptr);
  }

  auto run_once = [&]() {
    if (options.beta == 0.0f) {
      std::fill(c.begin(), c.end(), 0.0f);
    }

    if (options.pack_b) {
      MlasGemm(
          trans_a,
          options.m,
          options.n,
          options.k,
          options.alpha,
          a.data(),
          lda,
          packed_b.data(),
          options.beta,
          c.data(),
          ldc,
          nullptr,
          nullptr);
    } else {
      MlasGemm(
          trans_a,
          trans_b,
          options.m,
          options.n,
          options.k,
          options.alpha,
          a.data(),
          lda,
          b.data(),
          ldb,
          options.beta,
          c.data(),
          ldc,
          nullptr,
          nullptr);
    }
  };

  for (size_t i = 0; i < options.warmup; ++i) {
    run_once();
  }

  const double compute_ms = TimeLoop(options.iters, run_once);
  const double avg_compute_ms = compute_ms / static_cast<double>(options.iters);
  const double avg_pack_ms = pack_ms / static_cast<double>(options.iters);
  const double flops = 2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
                       static_cast<double>(options.k);
  const double gflops = flops / (avg_compute_ms * 1.0e6);
  const double checksum = std::accumulate(c.begin(), c.end(), 0.0);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "M=" << options.m
            << " N=" << options.n
            << " K=" << options.k
            << " pack_b=" << (options.pack_b ? 1 : 0)
            << " trans_a=" << (options.trans_a ? 1 : 0)
            << " trans_b=" << (options.trans_b ? 1 : 0)
            << " iters=" << options.iters
            << " warmup=" << options.warmup << '\n';
  if (options.pack_b) {
    std::cout << "pack_total_ms=" << pack_ms << " pack_avg_ms=" << avg_pack_ms << '\n';
  }
  std::cout << "compute_total_ms=" << compute_ms
            << " compute_avg_ms=" << avg_compute_ms
            << " gflops=" << gflops << '\n';
  std::cout << "checksum=" << checksum << std::endl;

  return 0;
}
