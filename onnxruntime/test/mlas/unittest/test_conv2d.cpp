// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_conv2d.h"
#include "test_conv2d_fixture.h"

TEST(Conv2d_HalfConv, PrepareRespectsBackendSelectorConfig) {
  const int64_t input_shape[] = {5, 5};
  const int64_t kernel_shape[] = {3, 3};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t padding[] = {1, 1, 1, 1};
  const int64_t stride_shape[] = {1, 1};
  const int64_t output_shape[] = {5, 5};

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  MLAS_CONV_PARAMETERS parameters{};
  size_t working_buffer_size = 0;
  if (!MlasHalfConvPrepare(&parameters,
                           2,
                           1,
                           1,
                           4,
                           input_shape,
                           kernel_shape,
                           dilation_shape,
                           padding,
                           stride_shape,
                           output_shape,
                           8,
                           &activation,
                           &working_buffer_size,
                           0.0f,
                           false,
                           nullptr,
                           nullptr)) {
    GTEST_SKIP() << "HalfConv prepare path unavailable";
  }

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config;
  selector_config.use_kleidiai = false;

  MLAS_CONV_PARAMETERS disabled_parameters{};
  EXPECT_FALSE(MlasHalfConvPrepare(&disabled_parameters,
                                   2,
                                   1,
                                   1,
                                   4,
                                   input_shape,
                                   kernel_shape,
                                   dilation_shape,
                                   padding,
                                   stride_shape,
                                   output_shape,
                                   8,
                                   &activation,
                                   &working_buffer_size,
                                   0.0f,
                                   false,
                                   nullptr,
                                   &selector_config));
}

TEST(Conv2d_HalfConv, PrepareReportsWorkingBufferSizeInBytes) {
  const int64_t input_shape[] = {5, 5};
  const int64_t kernel_shape[] = {3, 3};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t padding[] = {1, 1, 1, 1};
  const int64_t stride_shape[] = {1, 1};
  const int64_t output_shape[] = {5, 5};

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  MLAS_CONV_PARAMETERS parameters{};
  size_t working_buffer_size = 0;
  if (!MlasHalfConvPrepare(&parameters,
                           2,
                           1,
                           1,
                           4,
                           input_shape,
                           kernel_shape,
                           dilation_shape,
                           padding,
                           stride_shape,
                           output_shape,
                           8,
                           &activation,
                           &working_buffer_size,
                           0.0f,
                           false,
                           nullptr,
                           nullptr)) {
    GTEST_SKIP() << "HalfConv prepare path unavailable";
  }

  EXPECT_EQ(working_buffer_size, parameters.OutputSize * parameters.FilterCount * sizeof(uint16_t));
}

TEST(Conv2d_HalfConv, PrepareInitializesLayoutFlags) {
  const int64_t input_shape[] = {5, 5};
  const int64_t kernel_shape[] = {3, 3};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t padding[] = {1, 1, 1, 1};
  const int64_t stride_shape[] = {1, 1};
  const int64_t output_shape[] = {5, 5};

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  MLAS_CONV_PARAMETERS nchw_parameters{};
  nchw_parameters.ChannelsLast = true;
  nchw_parameters.InputOutputChannelsLast = true;
  size_t nchw_working_buffer_size = 0;
  if (!MlasHalfConvPrepare(&nchw_parameters,
                           2,
                           1,
                           1,
                           4,
                           input_shape,
                           kernel_shape,
                           dilation_shape,
                           padding,
                           stride_shape,
                           output_shape,
                           8,
                           &activation,
                           &nchw_working_buffer_size,
                           0.0f,
                           false,
                           nullptr,
                           nullptr)) {
    GTEST_SKIP() << "HalfConv prepare path unavailable";
  }

  EXPECT_FALSE(nchw_parameters.ChannelsLast);
  EXPECT_FALSE(nchw_parameters.InputOutputChannelsLast);

  MLAS_CONV_PARAMETERS nhwc_parameters{};
  size_t nhwc_working_buffer_size = 0;
  ASSERT_TRUE(MlasHalfConvPrepare(&nhwc_parameters,
                                  2,
                                  1,
                                  1,
                                  4,
                                  input_shape,
                                  kernel_shape,
                                  dilation_shape,
                                  padding,
                                  stride_shape,
                                  output_shape,
                                  8,
                                  &activation,
                                  &nhwc_working_buffer_size,
                                  0.0f,
                                  true,
                                  nullptr,
                                  nullptr));

  EXPECT_TRUE(nhwc_parameters.ChannelsLast);
  EXPECT_TRUE(nhwc_parameters.InputOutputChannelsLast);
  EXPECT_EQ(nhwc_working_buffer_size, size_t{0});
}

static size_t Conv2dRegistLongExecute() {
  size_t count = MlasLongExecuteTests<MlasConv2DTest<false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasConv2DTest<true>>::RegisterLongExecute();
  }
  return count;
}

static size_t Conv2dRegistShortExecute() {
  size_t count = Conv2dShortExecuteTest<MlasConv2DTest<false>>::RegisterShortExecuteTests();
  count += MlasDirectShortExecuteTests<MlasConv2DTest<false>>::RegisterShortExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += Conv2dShortExecuteTest<MlasConv2DTest<true>>::RegisterShortExecuteTests();
    count += MlasDirectShortExecuteTests<MlasConv2DTest<true>>::RegisterShortExecute();
  }
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Conv2dRegistShortExecute() : Conv2dRegistLongExecute();
});
