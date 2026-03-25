// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#include <array>
#include <vector>

#if defined(MLAS_TARGET_AMD64)

namespace {

constexpr float kSiluAbsoluteTolerance = 1.0e-4f;
constexpr float kSiluRelativeTolerance = 1.0e-4f;
constexpr float kGeluAbsoluteTolerance = 1.0e-4f;
constexpr float kGeluRelativeTolerance = 1.0e-4f;
constexpr float kReluAbsoluteTolerance = 0.0f;
constexpr float kReluRelativeTolerance = 0.0f;

bool IsFusedPointwiseActivationAvailable() {
  return MlasNchwcGetBlockSize() > 1 &&
         GetMlasPlatform().ConvPointwiseFloatKernel == MlasConvPointwiseFloatKernelAvx512F;
}

bool OutputsMatch(float actual, float expected, float absolute_tolerance, float relative_tolerance) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }

  if (std::isinf(expected)) {
    return std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
  }

  if (actual == 0.0f && expected == 0.0f) {
    return std::signbit(actual) == std::signbit(expected);
  }

  const float diff = std::fabs(actual - expected);
  if (diff <= absolute_tolerance) {
    return true;
  }

  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return scale > 0.0f && diff <= scale * relative_tolerance;
}

void FillSignedPattern(float* buffer, size_t count, int seed) {
  for (size_t i = 0; i < count; ++i) {
    const int value = static_cast<int>((i * 17 + seed * 13) % 29) - 14;
    buffer[i] = static_cast<float>(value) * 0.125f;
  }
}

void ReferencePointwiseConv(size_t batch_count,
                            size_t input_channels,
                            size_t input_height,
                            size_t input_width,
                            size_t output_channels,
                            const float* input,
                            const float* filter,
                            const float* bias,
                            float* output) {
  const size_t spatial_size = input_height * input_width;

  for (size_t n = 0; n < batch_count; ++n) {
    for (size_t oc = 0; oc < output_channels; ++oc) {
      for (size_t hw = 0; hw < spatial_size; ++hw) {
        float sum = bias == nullptr ? 0.0f : bias[oc];
        for (size_t ic = 0; ic < input_channels; ++ic) {
          const size_t input_index = n * input_channels * spatial_size + ic * spatial_size + hw;
          const size_t filter_index = oc * input_channels + ic;
          sum += input[input_index] * filter[filter_index];
        }
        output[n * output_channels * spatial_size + oc * spatial_size + hw] = sum;
      }
    }
  }
}

void ApplyReferenceActivation(MLAS_ACTIVATION_KIND activation_kind,
                              const float* input,
                              float* output,
                              size_t count) {
  switch (activation_kind) {
    case MlasReluActivation:
      for (size_t i = 0; i < count; ++i) {
        output[i] = std::max(input[i], 0.0f);
      }
      break;

    case MlasSiluActivation:
      MlasComputeSilu(input, output, count);
      break;

    case MlasGeluErfActivation:
      MlasComputeGeluErf(input, output, count);
      break;

    default:
      FAIL() << "Unsupported activation kind";
      break;
  }
}

class MlasNchwcPointwiseActivationTestBase : public MlasTestBase {
 public:
  void ExecuteShort() override {
    ExecuteCommon();
  }

  void ExecuteLong() override {
    ExecuteCommon();
  }

 private:
  virtual MLAS_ACTIVATION_KIND GetActivationKind() const = 0;
  virtual float GetAbsoluteTolerance() const = 0;
  virtual float GetRelativeTolerance() const = 0;

  void ExecuteCommon() {
    if (!IsFusedPointwiseActivationAvailable()) {
      GTEST_SKIP() << "Fused AVX512 pointwise activation path is not available on this machine.";
    }

    const size_t block_size = MlasNchwcGetBlockSize();
    const std::vector<size_t> output_widths = {1, 2, 3, 6, 7, 8, 9};
    const std::vector<size_t> output_channel_blocks = {1, 2, 3, 4};
    const MLAS_ACTIVATION_KIND activation_kind = GetActivationKind();

    for (size_t output_channel_blocks_value : output_channel_blocks) {
      for (size_t output_width : output_widths) {
        RunPointwiseCase(block_size,
                         activation_kind,
                         output_channel_blocks_value * block_size,
                         output_width);
      }
    }
  }

  void RunPointwiseCase(size_t block_size,
                        MLAS_ACTIVATION_KIND activation_kind,
                        size_t output_channels,
                        size_t output_width) {
    SCOPED_TRACE(testing::Message()
           << "activation=" << static_cast<int>(activation_kind)
           << ", block_size=" << block_size
           << ", output_channels=" << output_channels
           << ", output_width=" << output_width
           << ", kernel=1x1");

    const size_t batch_count = 1;
    const size_t input_channels = block_size;
    const size_t input_height = 1;
    const size_t input_width = output_width;
    const size_t output_height = 1;
    const size_t output_elements = batch_count * output_channels * output_height * output_width;
    const size_t input_elements = batch_count * input_channels * input_height * input_width;
    const size_t filter_elements = output_channels * input_channels;
    const size_t bias_elements = output_channels;

    float* input = input_buffer_.GetFilledBuffer(input_elements, [](float* data, size_t count) {
      FillSignedPattern(data, count, 1);
    });
    float* filter = filter_buffer_.GetFilledBuffer(filter_elements, [](float* data, size_t count) {
      FillSignedPattern(data, count, 3);
    });
    float* bias = bias_buffer_.GetFilledBuffer(bias_elements, [](float* data, size_t count) {
      FillSignedPattern(data, count, 5);
    });

    float* reference_conv_output = reference_conv_output_buffer_.GetBuffer(output_elements, true);
    float* reference_output = reference_output_buffer_.GetBuffer(output_elements, true);
    float* unfused_conv_output = unfused_conv_output_buffer_.GetBuffer(output_elements, true);
    float* unfused_output = unfused_output_buffer_.GetBuffer(output_elements, true);
    float* fused_output = actual_output_buffer_.GetBuffer(output_elements, true);

    ReferencePointwiseConv(batch_count,
                           input_channels,
                           input_height,
                           input_width,
                           output_channels,
                           input,
                           filter,
                           bias,
                           reference_conv_output);
    ApplyReferenceActivation(activation_kind, reference_conv_output, reference_output, output_elements);

    InvokeNchwcPointwise(batch_count,
               input_channels,
               input_height,
               input_width,
               output_channels,
               input,
               filter,
               bias,
               MlasIdentityActivation,
               unfused_conv_output);

    ApplyReferenceActivation(activation_kind, unfused_conv_output, unfused_output, output_elements);

    InvokeNchwcPointwise(batch_count,
               input_channels,
               input_height,
               input_width,
               output_channels,
               input,
               filter,
               bias,
               activation_kind,
               fused_output);

    const float absolute_tolerance = GetAbsoluteTolerance();
    const float relative_tolerance = GetRelativeTolerance();

    for (size_t i = 0; i < output_elements; ++i) {
      ASSERT_TRUE(OutputsMatch(unfused_output[i], reference_output[i], absolute_tolerance, relative_tolerance))
        << "Unfused activation=" << static_cast<int>(activation_kind)
        << ", output_channels=" << output_channels
        << ", output_width=" << output_width
        << ", index=" << i
        << ", unfused=" << unfused_output[i]
        << ", expected=" << reference_output[i]
        << ", abs_diff=" << std::fabs(unfused_output[i] - reference_output[i]);

      ASSERT_TRUE(OutputsMatch(fused_output[i], unfused_output[i], absolute_tolerance, relative_tolerance))
        << "Fused-vs-unfused activation=" << static_cast<int>(activation_kind)
        << ", output_channels=" << output_channels
        << ", output_width=" << output_width
        << ", index=" << i
        << ", fused=" << fused_output[i]
        << ", unfused=" << unfused_output[i]
        << ", abs_diff=" << std::fabs(fused_output[i] - unfused_output[i]);

      ASSERT_TRUE(OutputsMatch(fused_output[i], reference_output[i], absolute_tolerance, relative_tolerance))
          << "Activation=" << static_cast<int>(activation_kind)
          << ", output_channels=" << output_channels
          << ", output_width=" << output_width
          << ", index=" << i
        << ", fused=" << fused_output[i]
          << ", expected=" << reference_output[i]
        << ", abs_diff=" << std::fabs(fused_output[i] - reference_output[i]);
    }
  }

  void InvokeNchwcPointwise(size_t batch_count,
                            size_t input_channels,
                            size_t input_height,
                            size_t input_width,
                            size_t output_channels,
                            const float* input,
                            const float* filter,
                            const float* bias,
                            MLAS_ACTIVATION_KIND activation_kind,
                            float* output) {
    const size_t block_size = MlasNchwcGetBlockSize();
    const size_t nchwc_input_channels = (input_channels + block_size - 1) & ~(block_size - 1);
    const size_t nchwc_output_channels = (output_channels + block_size - 1) & ~(block_size - 1);
    const size_t output_height = input_height;
    const size_t output_width = input_width;

    std::array<int64_t, 4> nchw_input_shape = {
        static_cast<int64_t>(batch_count), static_cast<int64_t>(input_channels),
        static_cast<int64_t>(input_height), static_cast<int64_t>(input_width)};
    int64_t input_shape[] = {static_cast<int64_t>(batch_count), static_cast<int64_t>(nchwc_input_channels),
                             static_cast<int64_t>(input_height), static_cast<int64_t>(input_width)};
    int64_t filter_shape[] = {static_cast<int64_t>(output_channels), static_cast<int64_t>(input_channels), 1, 1};
    int64_t kernel_shape[] = {1, 1};
    int64_t dilation_shape[] = {1, 1};
    int64_t padding[] = {0, 0, 0, 0};
    int64_t stride_shape[] = {1, 1};
    int64_t output_shape[] = {static_cast<int64_t>(batch_count), static_cast<int64_t>(nchwc_output_channels),
                              static_cast<int64_t>(output_height), static_cast<int64_t>(output_width)};
    int64_t nchw_output_shape[] = {static_cast<int64_t>(batch_count), static_cast<int64_t>(output_channels),
                                   static_cast<int64_t>(output_height), static_cast<int64_t>(output_width)};

    float* nchwc_input = nchwc_input_buffer_.GetBuffer(batch_count * nchwc_input_channels * input_height * input_width, true);
    float* reordered_filter = nchwc_filter_buffer_.GetBuffer(nchwc_output_channels * nchwc_input_channels, true);
    float* aligned_bias = nchwc_bias_buffer_.GetBuffer(nchwc_output_channels, true);
    float* nchwc_output = nchwc_output_buffer_.GetBuffer(batch_count * nchwc_output_channels * output_height * output_width, true);

    ReorderInputNchw(nchw_input_shape.data(), input, nchwc_input);
    MlasReorderFilterOIHWBiBo(filter_shape, filter, reordered_filter);

    for (size_t oc = 0; oc < output_channels; ++oc) {
      aligned_bias[oc] = bias[oc];
    }
    for (size_t oc = output_channels; oc < nchwc_output_channels; ++oc) {
      aligned_bias[oc] = 0.0f;
    }

    MLAS_ACTIVATION activation;
    activation.ActivationKind = activation_kind;

    MlasNchwcConv(input_shape,
                  kernel_shape,
                  dilation_shape,
                  padding,
                  stride_shape,
                  output_shape,
                  1,
                  nchwc_input,
                  reordered_filter,
                  aligned_bias,
                  nchwc_output,
                  &activation,
                  true,
                  nullptr,
                  nullptr,
                  false);

    MlasReorderOutputNchw(nchw_output_shape, nchwc_output, output, nullptr);
  }

  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> filter_buffer_;
  MatrixGuardBuffer<float> bias_buffer_;
  MatrixGuardBuffer<float> reference_conv_output_buffer_;
  MatrixGuardBuffer<float> reference_output_buffer_;
  MatrixGuardBuffer<float> unfused_conv_output_buffer_;
  MatrixGuardBuffer<float> unfused_output_buffer_;
  MatrixGuardBuffer<float> actual_output_buffer_;
  MatrixGuardBuffer<float> nchwc_input_buffer_;
  MatrixGuardBuffer<float> nchwc_filter_buffer_;
  MatrixGuardBuffer<float> nchwc_bias_buffer_;
  MatrixGuardBuffer<float> nchwc_output_buffer_;
};

class MlasNchwcPointwiseReluActivationTest : public MlasNchwcPointwiseActivationTestBase {
 public:
  static const char* GetTestSuiteName() {
    return "Conv2dNchwcPointwiseReluActivation";
  }

 private:
  MLAS_ACTIVATION_KIND GetActivationKind() const override {
    return MlasReluActivation;
  }

  float GetAbsoluteTolerance() const override {
    return kReluAbsoluteTolerance;
  }

  float GetRelativeTolerance() const override {
    return kReluRelativeTolerance;
  }
};

class MlasNchwcPointwiseSiluActivationTest : public MlasNchwcPointwiseActivationTestBase {
 public:
  static const char* GetTestSuiteName() {
    return "Conv2dNchwcPointwiseSiluActivation";
  }

 private:
  MLAS_ACTIVATION_KIND GetActivationKind() const override {
    return MlasSiluActivation;
  }

  float GetAbsoluteTolerance() const override {
    return kSiluAbsoluteTolerance;
  }

  float GetRelativeTolerance() const override {
    return kSiluRelativeTolerance;
  }
};

class MlasNchwcPointwiseGeluActivationTest : public MlasNchwcPointwiseActivationTestBase {
 public:
  static const char* GetTestSuiteName() {
    return "Conv2dNchwcPointwiseGeluActivation";
  }

 private:
  MLAS_ACTIVATION_KIND GetActivationKind() const override {
    return MlasGeluErfActivation;
  }

  float GetAbsoluteTolerance() const override {
    return kGeluAbsoluteTolerance;
  }

  float GetRelativeTolerance() const override {
    return kGeluRelativeTolerance;
  }
};

}  // namespace

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;

  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNchwcPointwiseReluActivationTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNchwcPointwiseSiluActivationTest>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNchwcPointwiseGeluActivationTest>::RegisterShortExecute();
  } else {
    count += MlasLongExecuteTests<MlasNchwcPointwiseReluActivationTest>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasNchwcPointwiseSiluActivationTest>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasNchwcPointwiseGeluActivationTest>::RegisterLongExecute();
  }

  return count;
});

#else

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool) {
  return size_t{0};
});

#endif
