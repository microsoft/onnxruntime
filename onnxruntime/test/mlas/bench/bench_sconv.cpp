// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"

#include <stdexcept>
#include <numeric>

static std::vector<std::string> BuildArgNamesForConv(size_t rank) {
  std::vector<std::string> names = {"Rank", "N", "G", "Cpg", "Fpg"};

  size_t arg_position = names.size();
  names.resize(arg_position + rank * 6, std::string(""));

  names[arg_position] = "I";
  arg_position += rank;

  names[arg_position] = "K";
  arg_position += rank;

  names[arg_position] = "P";
  arg_position += rank * 2;

  names[arg_position] = "S";
  arg_position += rank;

  names[arg_position] = "D";

  return names;
}

static const std::vector<std::string>& ArgNamesForConv(size_t rank) {
  static std::map<size_t, std::vector<std::string>> rank_to_args_name;
  if (rank_to_args_name.find(rank) == rank_to_args_name.end()) {
    rank_to_args_name.emplace(std::make_pair(rank, BuildArgNamesForConv(rank)));
  }
  return rank_to_args_name[rank];
}

// dummy for some strange build error when using Bench capture
void SCONV_NCHW(benchmark::State& state, const char* /*dummy*/) {
  const int64_t rank = state.range(0);                       // Rank
  const int64_t batch_size = state.range(1);                 // N
  const int64_t groups = state.range(2);                     // G
  const int64_t input_channels_per_group = state.range(3);   // Cpg
  const int64_t output_channels_per_group = state.range(4);  // Fpg

  if (rank <= 0) throw std::invalid_argument("Kernel rank must greater than 0!");
  if (batch_size <= 0) throw std::invalid_argument("Batch size must greater than 0!");
  if (groups <= 0) throw std::invalid_argument("Group count must greater than 0!");
  if (input_channels_per_group <= 0) throw std::invalid_argument("input_channels_per_group must greater than 0!");
  if (output_channels_per_group <= 0) throw std::invalid_argument("output_channels_per_group must greater than 0!");

  size_t arg_position = 5;
  const auto input_shape = BenchArgsVector(state, arg_position, rank);
  const auto kernel_shape = BenchArgsVector(state, arg_position, rank);
  const auto paddings = BenchArgsVector(state, arg_position, rank * 2);
  const auto strides = BenchArgsVector(state, arg_position, rank);
  const auto dilations = BenchArgsVector(state, arg_position, rank);

  // do not check the size of each vector as they are forced from args.
  if (std::any_of(input_shape.begin(), input_shape.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all input image dim must > 0");
  }

  if (std::any_of(kernel_shape.begin(), kernel_shape.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all kernel dim must > 0");
  }

  if (std::any_of(strides.begin(), strides.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all strides dim must > 0");
  }

  if (std::any_of(dilations.begin(), dilations.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all dilations dim must > 0");
  }

  const int64_t GC = groups * input_channels_per_group;
  const int64_t GF = groups * output_channels_per_group;
  std::vector<int64_t> x_shape = {batch_size, GC};
  x_shape.insert(x_shape.end(), input_shape.begin(), input_shape.end());
  std::vector<int64_t> f_shape = {GF, input_channels_per_group};
  f_shape.insert(f_shape.end(), kernel_shape.begin(), kernel_shape.end());

  std::vector<int64_t> output_shape((size_t)rank);
  for (int64_t i = 0; i < rank; ++i) {
    auto km = 1 + dilations[i] * (kernel_shape[i] - 1);
    output_shape[i] = (paddings[i] + paddings[i + rank] + input_shape[i] - km) / strides[i] + 1;
  }
  std::vector<int64_t> y_shape = {batch_size, GF};
  y_shape.insert(y_shape.end(), output_shape.begin(), output_shape.end());

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;
  MLAS_CONV_PARAMETERS Parameters;
  size_t WorkingBufferSize = 0;
  MlasConvPrepare(&Parameters,
                  static_cast<size_t>(rank),
                  static_cast<size_t>(batch_size),
                  static_cast<size_t>(groups),
                  static_cast<size_t>(input_channels_per_group),
                  input_shape.data(),
                  kernel_shape.data(),
                  dilations.data(),
                  paddings.data(),
                  strides.data(),
                  output_shape.data(),
                  static_cast<size_t>(output_channels_per_group),
                  &activation,
                  &WorkingBufferSize,
                  0.0f,
                  nullptr);

  auto X = RandomVectorUniform(x_shape, -2.0, 2.0);
  auto F = RandomVectorUniform(f_shape, -1.0, 1.0);
  int64_t y_size = std::accumulate(y_shape.begin(), y_shape.end(), 1LL, std::multiplies<int64_t>());
  std::vector<float> Y(static_cast<size_t>(y_size));
  std::vector<float> working_buffer(WorkingBufferSize);

  // warm up first round.
  MlasConv(&Parameters,
           X.data(),
           F.data(),
           nullptr,
           working_buffer.data(),
           Y.data(),
           nullptr);

  for (auto _ : state) {
    MlasConv(&Parameters,
             X.data(),
             F.data(),
             nullptr,
             working_buffer.data(),
             Y.data(),
             nullptr);
  }
}

static void ResNet50(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForConv(2));

  //************************* Conv 1 *************************
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 3, 64, 224, 224, 7, 7, 3, 3, 3, 3, 2, 2, 1, 1});

  //************************ Conv 2.1 ************************
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 64, 64, 56, 56, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({2, 1, 1, 64, 256, 56, 56, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  // b->Args({2, 1, 1, 64,256, 56, 56, 1,1, 0,0,0,0, 1,1, 1,1});

  //************************ Conv 2.X ************************
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 256, 64, 56, 56, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  // b->Args({2, 1, 1, 64, 64, 56, 56, 3,3, 1,1,1,1, 1,1, 1,1});
  // b->Args({2, 1, 1, 64,256, 56, 56, 1,1, 0,0,0,0, 1,1, 1,1});

  /************************ Conv 3.1 ************************/
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 256, 128, 56, 56, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({2, 1, 1, 128, 512, 28, 28, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 256, 512, 56, 56, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1});

  /************************ Conv 3.X ************************/
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 512, 128, 28, 28, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  // b->Args({2, 1, 1,128,512, 28, 28, 1,1, 0,0,0,0, 1,1, 1,1});

  /************************ Conv 4.1 ************************/
  //    Rank, N, G,Cpg,Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 512, 256, 28, 28, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({2, 1, 1, 256, 1024, 14, 14, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 512, 1024, 28, 28, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1});

  /************************ Conv 4.X ************************/
  //    Rank, N, G,  Cpg, Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 1024, 256, 14, 14, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  // b->Args({2, 1, 1, 256, 1024, 14, 14, 1,1, 0,0,0,0, 1,1, 1,1});

  /************************ Conv 5.1 ************************/
  //    Rank, N, G,  Cpg, Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 1024, 512, 14, 14, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 512, 512, 14, 14, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({2, 1, 1, 512, 2048, 7, 7, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 1024, 2048, 14, 14, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1});

  /************************ Conv 5.X ************************/
  //    Rank, N, G,  Cpg, Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 2048, 512, 7, 7, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});
  b->Args({2, 1, 1, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  // b->Args({2, 1, 1,  512,2048,  7,  7, 1,1, 0,0,0,0, 1,1, 1,1});
}

BENCHMARK_CAPTURE(SCONV_NCHW, ResNet50, "")->Apply(ResNet50)->UseRealTime();

static void TeamsModel(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForConv(2));
  //    Rank, N, G, Cpg, Fpg,  I,   , K, , P, , , , S, , D, ,
  b->Args({2, 1, 1, 40, 24, 24, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});   // fused conv_349 => 24x40
  b->Args({2, 1, 1, 24, 24, 24, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});   // fused Conv_367 => 24x40
  b->Args({2, 1, 1, 4, 24, 96, 160, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1});   // fused Conv_15 => 48x80
  b->Args({2, 1, 1, 12, 72, 48, 80, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});   // fused Conv_38 => 48x80
  b->Args({2, 1, 1, 12, 8, 48, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});    // fused Conv_395 => 48x80
  b->Args({2, 1, 24, 1, 1, 48, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});    // fused Conv_33 => 48x80
  b->Args({2, 1, 1, 8, 8, 48, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});     // fused Conv_413 => 48x80
  b->Args({2, 1, 72, 1, 1, 48, 80, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1});    // fused Conv_56 => 24x40
  b->Args({2, 1, 72, 1, 1, 24, 40, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});    // fused Conv_79 => 24x40
  b->Args({2, 1, 1, 24, 12, 48, 80, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});   // Conv_36 => 48x80
  b->Args({2, 1, 1, 12, 72, 24, 40, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});   // fused Conv_61/85 => 24x40
  b->Args({2, 1, 1, 24, 144, 12, 20, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});  // fused Conv_108/132 => 12x20

  b->Args({2, 1, 1, 12, 12, 48, 80, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});  // fused Conv_376 => 48x80
  b->Args({2, 1, 1, 12, 72, 48, 80, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1});  // Conv_59 => 24x40
}

BENCHMARK_CAPTURE(SCONV_NCHW, TeamsModel, "")->Apply(TeamsModel)->UseRealTime();

static void General_Conv2d(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForConv(2));
  b->ArgsProduct(
      {{2},       // Rank,
       {1},       // N
       {1, 2},    // Groups
       {3, 12},   // Cpg
       {6},       // Fpg
       {24, 72},  // Input Image Shape
       {36},
       {3},  // kernel shape
       {3},
       {0},  // paddings
       {0},
       {0},
       {0},
       {1},  // strides
       {1},
       {1},  // dilations
       {1}});
}

BENCHMARK_CAPTURE(SCONV_NCHW, 2d, "")->Apply(General_Conv2d)->UseRealTime();
