// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <map>
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

struct ConvBenchmarkArgs {
  int64_t rank;
  int64_t batch_size;
  int64_t groups;
  int64_t input_channels_per_group;
  int64_t output_channels_per_group;
  int64_t total_input_channels;
  int64_t total_output_channels;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> paddings;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> output_shape;
};

static ConvBenchmarkArgs ParseConvBenchmarkArgs(benchmark::State& state) {
  ConvBenchmarkArgs args{
      state.range(0),  // rank
      state.range(1),  // batch_size
      state.range(2),  // groups
      state.range(3),  // input_channels_per_group
      state.range(4),  // output_channels_per_group
      0,
      0,
      {},
      {},
      {},
      {},
      {},
      {}};

  if (args.rank <= 0) throw std::invalid_argument("Kernel rank must be greater than 0");
  if (args.batch_size <= 0) throw std::invalid_argument("Batch size must be greater than 0");
  if (args.groups <= 0) throw std::invalid_argument("Group count must be greater than 0");
  if (args.input_channels_per_group <= 0) throw std::invalid_argument("input_channels_per_group must be greater than 0");
  if (args.output_channels_per_group <= 0) throw std::invalid_argument("output_channels_per_group must be greater than 0");

  size_t arg_position = 5;
  args.input_shape = BenchArgsVector(state, arg_position, args.rank);
  args.kernel_shape = BenchArgsVector(state, arg_position, args.rank);
  args.paddings = BenchArgsVector(state, arg_position, args.rank * 2);
  args.strides = BenchArgsVector(state, arg_position, args.rank);
  args.dilations = BenchArgsVector(state, arg_position, args.rank);

  if (std::any_of(args.input_shape.begin(), args.input_shape.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all input image dim must > 0");
  }

  if (std::any_of(args.kernel_shape.begin(), args.kernel_shape.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all kernel dim must > 0");
  }

  if (std::any_of(args.strides.begin(), args.strides.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all strides dim must > 0");
  }

  if (std::any_of(args.dilations.begin(), args.dilations.end(), [](const int64_t& dim) { return dim <= 0; })) {
    throw std::invalid_argument("all dilations dim must > 0");
  }

  args.total_input_channels = args.groups * args.input_channels_per_group;
  args.total_output_channels = args.groups * args.output_channels_per_group;
  args.output_shape.resize(static_cast<size_t>(args.rank));
  for (int64_t i = 0; i < args.rank; ++i) {
    const auto index = static_cast<size_t>(i);
    const auto km = 1 + args.dilations[index] * (args.kernel_shape[index] - 1);
    args.output_shape[index] =
        (args.paddings[index] + args.paddings[index + static_cast<size_t>(args.rank)] + args.input_shape[index] - km) /
            args.strides[index] +
        1;
  }

  return args;
}

static std::vector<int64_t> MakeInputShape(const ConvBenchmarkArgs& args, bool channels_last) {
  std::vector<int64_t> shape = {args.batch_size};
  if (channels_last) {
    shape.insert(shape.end(), args.input_shape.begin(), args.input_shape.end());
    shape.push_back(args.total_input_channels);
  } else {
    shape.push_back(args.total_input_channels);
    shape.insert(shape.end(), args.input_shape.begin(), args.input_shape.end());
  }

  return shape;
}

static std::vector<int64_t> MakeFilterShape(const ConvBenchmarkArgs& args) {
  std::vector<int64_t> shape = {args.total_output_channels, args.input_channels_per_group};
  shape.insert(shape.end(), args.kernel_shape.begin(), args.kernel_shape.end());
  return shape;
}

static std::vector<int64_t> MakeOutputShape(const ConvBenchmarkArgs& args, bool channels_last) {
  std::vector<int64_t> shape = {args.batch_size};
  if (channels_last) {
    shape.insert(shape.end(), args.output_shape.begin(), args.output_shape.end());
    shape.push_back(args.total_output_channels);
  } else {
    shape.push_back(args.total_output_channels);
    shape.insert(shape.end(), args.output_shape.begin(), args.output_shape.end());
  }

  return shape;
}

static std::vector<size_t> ToSizeT(const std::vector<int64_t>& values) {
  std::vector<size_t> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(), [](int64_t value) {
    return static_cast<size_t>(value);
  });
  return result;
}

static void PrepareConvParameters(const ConvBenchmarkArgs& args,
                                  bool channels_last,
                                  MLAS_THREADPOOL* thread_pool,
                                  MLAS_CONV_PARAMETERS* parameters,
                                  size_t* working_buffer_size) {
  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;
  MlasConvPrepare(parameters,
                  static_cast<size_t>(args.rank),
                  static_cast<size_t>(args.batch_size),
                  static_cast<size_t>(args.groups),
                  static_cast<size_t>(args.input_channels_per_group),
                  args.input_shape.data(),
                  args.kernel_shape.data(),
                  args.dilations.data(),
                  args.paddings.data(),
                  args.strides.data(),
                  args.output_shape.data(),
                  static_cast<size_t>(args.output_channels_per_group),
                  &activation,
                  working_buffer_size,
                  channels_last,
                  0.0f,
                  thread_pool);
}

// dummy for some strange build error when using Bench capture
void SCONV_NCHW(benchmark::State& state, const char* /*dummy*/) {
  const auto args = ParseConvBenchmarkArgs(state);
  const auto x_shape = MakeInputShape(args, false);
  const auto f_shape = MakeFilterShape(args);
  const auto y_shape = MakeOutputShape(args, false);

  MLAS_CONV_PARAMETERS Parameters;
  size_t WorkingBufferSize = 0;
  PrepareConvParameters(args, false, nullptr, &Parameters, &WorkingBufferSize);

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

static MLAS_THREADPOOL* GetMlasThreadPoolForConvBenchmark(void) {
  static auto threadpool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 4, true);
  return threadpool.get();
}

void SCONV_NHWC_KLEIDIAI(benchmark::State& state, const char* /*dummy*/) {
  const auto args = ParseConvBenchmarkArgs(state);
  if (args.rank != 2 || args.batch_size != 1) {
    state.SkipWithError("KleidiAI NHWC benchmark requires 2D convolution with batch size 1.");
    return;
  }

  const auto input_shape_size_t = ToSizeT(args.input_shape);
  const auto kernel_shape_size_t = ToSizeT(args.kernel_shape);
  const auto paddings_size_t = ToSizeT(args.paddings);
  const auto strides_size_t = ToSizeT(args.strides);
  const auto dilations_size_t = ToSizeT(args.dilations);

  if (!MlasConvSupportsSymmetricChannelsLast2DFloatKernel(
          static_cast<size_t>(args.rank),
          static_cast<size_t>(args.batch_size),
          static_cast<size_t>(args.groups),
          input_shape_size_t.data(),
          kernel_shape_size_t.data(),
          dilations_size_t.data(),
          paddings_size_t.data(),
          strides_size_t.data(),
          static_cast<size_t>(args.output_channels_per_group),
          0.0f)) {
    state.SkipWithError("KleidiAI NHWC kernel is not supported for this benchmark shape on the current platform.");
    return;
  }

  const auto x_shape = MakeInputShape(args, true);
  const auto f_shape = MakeFilterShape(args);
  const auto y_shape = MakeOutputShape(args, true);

  MLAS_CONV_PARAMETERS Parameters;
  size_t WorkingBufferSize = 0;
  PrepareConvParameters(args, true, nullptr, &Parameters, &WorkingBufferSize);

  auto X = RandomVectorUniform(x_shape, -2.0, 2.0);
  auto F = RandomVectorUniform(f_shape, -1.0, 1.0);
  int64_t y_size = std::accumulate(y_shape.begin(), y_shape.end(), 1LL, std::multiplies<int64_t>());
  std::vector<float> Y(static_cast<size_t>(y_size));
  std::vector<float> working_buffer(WorkingBufferSize);

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

void SCONV_NCHW_THREADED(benchmark::State& state, const char* /*dummy*/) {
  MLAS_THREADPOOL* tp = GetMlasThreadPoolForConvBenchmark();
  const auto args = ParseConvBenchmarkArgs(state);
  const auto x_shape = MakeInputShape(args, false);
  const auto f_shape = MakeFilterShape(args);
  const auto y_shape = MakeOutputShape(args, false);

  MLAS_CONV_PARAMETERS Parameters;
  size_t WorkingBufferSize = 0;
  PrepareConvParameters(args, false, tp, &Parameters, &WorkingBufferSize);

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
           tp);

  for (auto _ : state) {
    MlasConv(&Parameters,
             X.data(),
             F.data(),
             nullptr,
             working_buffer.data(),
             Y.data(),
             tp);
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

  b->Args({2, 1, 256, 1, 1, 378, 378, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});  // External customer model
  b->Args({2, 1, 512, 1, 1, 378, 378, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});  // External customer model
  b->Args({2, 1, 960, 1, 1, 378, 378, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});  // External customer model
}

BENCHMARK_CAPTURE(SCONV_NCHW, TeamsModel, "")->Apply(TeamsModel)->UseRealTime();
BENCHMARK_CAPTURE(SCONV_NCHW_THREADED, TeamsModel, "")->Apply(TeamsModel)->UseRealTime();

static void MobileClip(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForConv(2));

  // 7x7 grouped depthwise-multiplier-2 shapes.
  // Input: 1x64x64x64, Filter: 128x1x7x7, Groups: 64, Pad: 3, Stride: 2, Dilation: 1.
  b->Args({2, 1, 64, 1, 2, 64, 64, 7, 7, 3, 3, 3, 3, 2, 2, 1, 1});

  // Input: 1x128x32x32, Filter: 256x1x7x7, Groups: 128, Pad: 3, Stride: 2, Dilation: 1.
  b->Args({2, 1, 128, 1, 2, 32, 32, 7, 7, 3, 3, 3, 3, 2, 2, 1, 1});

  // Input: 1x256x16x16, Filter: 512x1x7x7, Groups: 256, Pad: 3, Stride: 2, Dilation: 1.
  b->Args({2, 1, 256, 1, 2, 16, 16, 7, 7, 3, 3, 3, 3, 2, 2, 1, 1});
}

BENCHMARK_CAPTURE(SCONV_NCHW, MobileClip, "")->Apply(MobileClip)->UseRealTime();
BENCHMARK_CAPTURE(SCONV_NCHW_THREADED, MobileClip, "")->Apply(MobileClip)->UseRealTime();

static void KleidiAiNhwcComparison(benchmark::internal::Benchmark* b) {
  b->ArgNames(ArgNamesForConv(2));

  // Dense 3x3 conv shapes that fit the Arm SME / KleidiAI NHWC fast-path envelope.
  b->Args({2, 1, 1, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({2, 1, 1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});

  // Classic depthwise shapes now supported by the NHWC helper gate.
  b->Args({2, 1, 64, 1, 1, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({2, 1, 72, 1, 1, 48, 80, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
}

BENCHMARK_CAPTURE(SCONV_NCHW, KleidiAiNhwcComparison_NchwBaseline, "")->Apply(KleidiAiNhwcComparison)->UseRealTime();
BENCHMARK_CAPTURE(SCONV_NHWC_KLEIDIAI, KleidiAiNhwcComparison_NhwcFastPath, "")->Apply(KleidiAiNhwcComparison)->UseRealTime();

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
