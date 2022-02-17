// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4389)
#endif

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

// sin(1, 2, ..., 100)
constexpr std::initializer_list<float> kRandomValuesVectorAligned = {
    0.8414709848078965, 0.9092974268256817, 0.1411200080598672, -0.7568024953079282, -0.9589242746631385,
    -0.27941549819892586, 0.6569865987187891, 0.9893582466233818, 0.4121184852417566, -0.5440211108893698,
    -0.9999902065507035, -0.5365729180004349, 0.4201670368266409, 0.9906073556948704, 0.6502878401571168,
    -0.2879033166650653, -0.9613974918795568, -0.750987246771676, 0.14987720966295234, 0.9129452507276277,
    0.8366556385360561, -0.008851309290403876, -0.8462204041751706, -0.9055783620066239, -0.13235175009777303,
    0.7625584504796027, 0.956375928404503, 0.27090578830786904, -0.6636338842129675, -0.9880316240928618,
    -0.404037645323065, 0.5514266812416906, 0.9999118601072672, 0.5290826861200238, -0.428182669496151,
    -0.9917788534431158, -0.6435381333569995, 0.2963685787093853, 0.9637953862840878, 0.7451131604793488,
    -0.158622668804709, -0.9165215479156338, -0.8317747426285983, 0.017701925105413577, 0.8509035245341184,
    0.9017883476488092, 0.123573122745224, -0.7682546613236668, -0.9537526527594719, -0.26237485370392877,
    0.6702291758433747, 0.9866275920404853, 0.39592515018183416, -0.5587890488516163, -0.9997551733586199,
    -0.5215510020869119, 0.43616475524782494, 0.9928726480845371, 0.6367380071391379, -0.3048106211022167,
    -0.9661177700083929, -0.7391806966492228, 0.16735570030280691, 0.9200260381967906, 0.8268286794901034,
    -0.026551154023966794, -0.8555199789753223, -0.8979276806892913, -0.11478481378318722, 0.7738906815578891,
    0.9510546532543747, 0.25382336276203626, -0.6767719568873076, -0.9851462604682474, -0.38778163540943045,
    0.5661076368981803, 0.9995201585807313, 0.5139784559875352, -0.4441126687075084, -0.9938886539233752,
    -0.6298879942744539, 0.31322878243308516, 0.9683644611001854, 0.7331903200732922, -0.1760756199485871,
    -0.9234584470040598, -0.8218178366308225, 0.03539830273366068, 0.8600694058124533, 0.8939966636005579,
    0.10598751175115685, -0.7794660696158047, -0.9482821412699473, -0.24525198546765434, 0.683261714736121,
    0.9835877454343449, 0.3796077390275217, -0.5733818719904229, -0.9992068341863537, -0.5063656411097588};

// >>> import random
// >>> [bool(random.getrandbits(1) for _ in range(1, 101)]
constexpr std::initializer_list<bool> kRandomMaskVectorAligned = {
    true, true, false, true, false, false, false, false, false, false, false, false, true, true, false, true,
    false, true, true, false, false, false, true, true, true, false, true, false, true, false, false, true,
    true, false, false, true, false, true, true, true, false, true, false, true, false, true, true, true, true,
    true, true, true, false, true, false, true, false, true, false, false, false, true, false, false, true,
    false, true, false, false, false, false, true, true, true, false, false, false, true, false, true, false,
    false, true, false, true, true, true, false, false, true, true, false, false, true, false, false, true,
    true, false, true};
constexpr std::initializer_list<int64_t> kRandomValuesVectorAlignedDims = {10, 10};

// sin(1, 2, ..., 99)
constexpr std::initializer_list<float> kRandomValuesVectorUnaligned = {
    0.8414709848078965, 0.9092974268256817, 0.1411200080598672, -0.7568024953079282, -0.9589242746631385,
    -0.27941549819892586, 0.6569865987187891, 0.9893582466233818, 0.4121184852417566, -0.5440211108893698,
    -0.9999902065507035, -0.5365729180004349, 0.4201670368266409, 0.9906073556948704, 0.6502878401571168,
    -0.2879033166650653, -0.9613974918795568, -0.750987246771676, 0.14987720966295234, 0.9129452507276277,
    0.8366556385360561, -0.008851309290403876, -0.8462204041751706, -0.9055783620066239, -0.13235175009777303,
    0.7625584504796027, 0.956375928404503, 0.27090578830786904, -0.6636338842129675, -0.9880316240928618,
    -0.404037645323065, 0.5514266812416906, 0.9999118601072672, 0.5290826861200238, -0.428182669496151,
    -0.9917788534431158, -0.6435381333569995, 0.2963685787093853, 0.9637953862840878, 0.7451131604793488,
    -0.158622668804709, -0.9165215479156338, -0.8317747426285983, 0.017701925105413577, 0.8509035245341184,
    0.9017883476488092, 0.123573122745224, -0.7682546613236668, -0.9537526527594719, -0.26237485370392877,
    0.6702291758433747, 0.9866275920404853, 0.39592515018183416, -0.5587890488516163, -0.9997551733586199,
    -0.5215510020869119, 0.43616475524782494, 0.9928726480845371, 0.6367380071391379, -0.3048106211022167,
    -0.9661177700083929, -0.7391806966492228, 0.16735570030280691, 0.9200260381967906, 0.8268286794901034,
    -0.026551154023966794, -0.8555199789753223, -0.8979276806892913, -0.11478481378318722, 0.7738906815578891,
    0.9510546532543747, 0.25382336276203626, -0.6767719568873076, -0.9851462604682474, -0.38778163540943045,
    0.5661076368981803, 0.9995201585807313, 0.5139784559875352, -0.4441126687075084, -0.9938886539233752,
    -0.6298879942744539, 0.31322878243308516, 0.9683644611001854, 0.7331903200732922, -0.1760756199485871,
    -0.9234584470040598, -0.8218178366308225, 0.03539830273366068, 0.8600694058124533, 0.8939966636005579,
    0.10598751175115685, -0.7794660696158047, -0.9482821412699473, -0.24525198546765434, 0.683261714736121,
    0.9835877454343449, 0.3796077390275217, -0.5733818719904229, -0.9992068341863537};

// >>> import random
// >>> [bool(random.getrandbits(1) for _ in range(1, 100)]
constexpr std::initializer_list<bool> kRandomMaskVectorUnaligned = {
    true, false, false, true, true, false, false, true, false, true, false, false, true, false, true, false,
    false, true, false, true, false, false, true, true, true, false, true, true, false, true, false, false,
    false, true, true, true, false, false, false, false, true, false, true, true, false, false, true, true,
    true, false, true, false, true, false, false, true, true, false, false, true, true, true, true, false,
    false, true, true, true, false, false, true, true, true, false, false, true, true, true, false, true,
    true, false, false, false, false, false, false, false, true, false, true, true, true, true, true, true,
    false, false, true};
constexpr std::initializer_list<int64_t> kRandomValuesVectorUnalignedDims = {9, 11};

constexpr int64_t kMaskSeed = 42;
constexpr std::initializer_list<float> kTestedRatios = {0.00f, 0.25f, 0.50f, 0.75f, 0.99f};

namespace {

std::vector<uint32_t> maskToBitmask(const std::vector<bool>& mask) {
  std::vector<uint32_t> result;

  for (size_t i = 0; i < mask.size(); i++) {
    size_t bitmask_idx = i / 32;
    size_t bit_idx = i % 32;

    while (bitmask_idx >= result.size()) {
      result.push_back(0);
    }

    if (mask[i]) {
      result[bitmask_idx] |= (1 << bit_idx);
    }
  }

  return result;
}

std::vector<uint32_t> maskToBitmask(const bool* data, size_t size) {
  std::vector<uint32_t> result;

  for (size_t i = 0; i < size; i++) {
    size_t bitmask_idx = i / 32;
    size_t bit_idx = i % 32;

    while (bitmask_idx >= result.size()) {
      result.push_back(0);
    }

    if (data[i]) {
      result[bitmask_idx] |= (1 << bit_idx);
    }
  }

  return result;
}

template <typename T>
void dropoutGradCpu(const std::vector<T>& data,
                    const std::vector<bool>& mask,
                    const float ratio,
                    std::vector<T>& result) {
  ASSERT_EQ(data.size(), mask.size());

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  result.clear();

  for (size_t i = 0; i < data.size(); i++) {
    result.push_back(data[i] * mask[i] * scale);
  }
}

}  // namespace

// BiasDropout kernel is only implemented for CUDA
#if defined(USE_CUDA)

// test aligned
TEST(BitmaskDropoutGradTest, InferenceFullMaskAligned) {
  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<bool> mask(kRandomValuesVectorAligned.size(), true);
  std::vector<uint32_t> bitmask = maskToBitmask(mask);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.Run();
}

TEST(BitmaskDropoutGradTest, InferenceEmptyMaskAligned) {
  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<bool> mask(kRandomValuesVectorAligned.size(), false);
  std::vector<uint32_t> bitmask = maskToBitmask(mask);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);

  std::vector<float> expected_output(kRandomValuesVectorAligned.size(), 0.00f);
  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, expected_output);
  test.Run();
}

TEST(BitmaskDropoutGradTest, InferenceRatioAligned) {
  for (float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<bool> mask(kRandomValuesVectorAligned.size(), true);
    std::vector<uint32_t> bitmask = maskToBitmask(mask);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    // ratio should be ignored in inferencing mode
    test.AddInput<float>("ratio", {}, {ratio});

    test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, InferenceRatioExplicitTrainingModeAligned) {
  for (float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<bool> mask(kRandomValuesVectorAligned.size(), true);
    std::vector<uint32_t> bitmask = maskToBitmask(mask);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    // ratio should be ignored in inferencing mode
    test.AddInput<float>("ratio", {}, {ratio});
    test.AddInput<bool>("training_mode", {}, {false});

    test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, TrainingDefaultRatioAligned) {
  constexpr float default_ratio = 0.5;

  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorAligned);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
  test.AddOptionalInputEdge<float>();  // ratio
  test.AddInput<bool>("training_mode", {}, {true});

  std::vector<float> expected_output;
  dropoutGradCpu(std::vector(kRandomValuesVectorAligned),
                 std::vector(kRandomMaskVectorAligned),
                 default_ratio,
                 expected_output);

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, expected_output);
  test.Run();
}

TEST(BitmaskDropoutGradTest, TrainingSpecifiedRatioAligned) {
  for (const float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorAligned);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    test.AddInput<float>("ratio", {}, {ratio});
    test.AddInput<bool>("training_mode", {}, {true});

    std::vector<float> expected_output;
    dropoutGradCpu(std::vector(kRandomValuesVectorAligned),
                   std::vector(kRandomMaskVectorAligned),
                   ratio,
                   expected_output);

    test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, expected_output);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, TrainingCompareDropoutGradAligned) {
  for (const float ratio : kTestedRatios) {
    // TODO: Original dropout has improper handling when given a ratio of 0.0f. If a non-full mask
    // is provided, the mask will not actually be applied.
    //
    // This will not show up in E2E testing scenarios (as a non-zero ratio would be required
    // to have generated a non-full mask), but this must be accounted for when doing more fine-grained
    // integration testing.
    //
    // For now, don't perform any comparison for a ratio of 0.0f.
    if (ratio == 0.00f) {
      continue;
    }

    OpTester dropout_grad("DropoutGrad", 1, kMSDomain, false);

    dropout_grad.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    dropout_grad.AddInput<bool>("mask", kRandomValuesVectorAlignedDims, kRandomMaskVectorAligned);
    dropout_grad.AddInput<float>("ratio", {}, {ratio});
    dropout_grad.AddInput<bool>("training_mode", {}, {true});

    // This output will not be used as output verification is disabled. Dimensions still must match.
    dropout_grad.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

    dropout_grad.Run();

    std::vector<OrtValue> dropout_grad_output = dropout_grad.GetFetches();

    const Tensor& output_tensor = dropout_grad_output[0].Get<Tensor>();
    const float* output_values = output_tensor.Data<float>();
    const size_t output_size = output_tensor.Shape().Size();

    OpTester bitmask_dropout_grad("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorAligned);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    bitmask_dropout_grad.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    bitmask_dropout_grad.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    bitmask_dropout_grad.AddInput<float>("ratio", {}, {ratio});
    bitmask_dropout_grad.AddInput<bool>("training_mode", {}, {true});

    bitmask_dropout_grad.AddOutput<float>("output", kRandomValuesVectorAlignedDims, output_values, output_size);

    bitmask_dropout_grad.Run();
  }
}

// test unaligned

TEST(BitmaskDropoutGradTest, InferenceFullMaskUnaligned) {
  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<bool> mask(kRandomValuesVectorUnaligned.size(), true);
  std::vector<uint32_t> bitmask = maskToBitmask(mask);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.Run();
}

TEST(BitmaskDropoutGradTest, InferenceEmptyMaskUnaligned) {
  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<bool> mask(kRandomValuesVectorUnaligned.size(), false);
  std::vector<uint32_t> bitmask = maskToBitmask(mask);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);

  std::vector<float> expected_output(kRandomValuesVectorUnaligned.size(), 0.00f);
  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, expected_output);
  test.Run();
}

TEST(BitmaskDropoutGradTest, InferenceRatioUnaligned) {
  for (float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<bool> mask(kRandomValuesVectorUnaligned.size(), true);
    std::vector<uint32_t> bitmask = maskToBitmask(mask);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    // ratio should be ignored in inferencing mode
    test.AddInput<float>("ratio", {}, {ratio});

    test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, InferenceRatioExplicitTrainingModeUnaligned) {
  for (float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<bool> mask(kRandomValuesVectorUnaligned.size(), true);
    std::vector<uint32_t> bitmask = maskToBitmask(mask);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    // ratio should be ignored in inferencing mode
    test.AddInput<float>("ratio", {}, {ratio});
    test.AddInput<bool>("training_mode", {}, {false});

    test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, TrainingDefaultRatioUnaligned) {
  constexpr float default_ratio = 0.5;

  OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

  std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorUnaligned);
  const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
  test.AddOptionalInputEdge<float>();  // ratio
  test.AddInput<bool>("training_mode", {}, {true});

  std::vector<float> expected_output;
  dropoutGradCpu(std::vector(kRandomValuesVectorUnaligned),
                 std::vector(kRandomMaskVectorUnaligned),
                 default_ratio,
                 expected_output);

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, expected_output);
  test.Run();
}

TEST(BitmaskDropoutGradTest, TrainingSpecifiedRatioUnaligned) {
  for (const float ratio : kTestedRatios) {
    OpTester test("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorUnaligned);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    test.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    test.AddInput<float>("ratio", {}, {ratio});
    test.AddInput<bool>("training_mode", {}, {true});

    std::vector<float> expected_output;
    dropoutGradCpu(std::vector(kRandomValuesVectorUnaligned),
                   std::vector(kRandomMaskVectorUnaligned),
                   ratio,
                   expected_output);

    test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, expected_output);
    test.Run();
  }
}

TEST(BitmaskDropoutGradTest, TrainingCompareDropoutGradUnaligned) {
  for (const float ratio : kTestedRatios) {
    // TODO: Original dropout has improper handling when given a ratio of 0.0f. If a non-full mask
    // is provided, the mask will not actually be applied.
    //
    // This will not show up in E2E testing scenarios (as a non-zero ratio would be required
    // to have generated a non-full mask), but this must be accounted for when doing more fine-grained
    // integration testing.
    //
    // For now, don't perform any comparison for a ratio of 0.0f.
    if (ratio == 0.00f) {
      continue;
    }

    OpTester dropout_grad("DropoutGrad", 1, kMSDomain, false);

    dropout_grad.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    dropout_grad.AddInput<bool>("mask", kRandomValuesVectorUnalignedDims, kRandomMaskVectorUnaligned);
    dropout_grad.AddInput<float>("ratio", {}, {ratio});
    dropout_grad.AddInput<bool>("training_mode", {}, {true});

    // This output will not be used as output verification is disabled. Dimensions still must match.
    dropout_grad.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

    dropout_grad.Run();

    std::vector<OrtValue> dropout_grad_output = dropout_grad.GetFetches();

    const Tensor& output_tensor = dropout_grad_output[0].Get<Tensor>();
    const float* output_values = output_tensor.Data<float>();
    const size_t output_size = output_tensor.Shape().Size();

    OpTester bitmask_dropout_grad("BitmaskDropoutGrad", 1, kMSDomain);

    std::vector<uint32_t> bitmask = maskToBitmask(kRandomMaskVectorUnaligned);
    const int64_t bitmask_elements = static_cast<int64_t>(bitmask.size());

    bitmask_dropout_grad.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    bitmask_dropout_grad.AddInput<uint32_t>("mask", {bitmask_elements}, bitmask);
    bitmask_dropout_grad.AddInput<float>("ratio", {}, {ratio});
    bitmask_dropout_grad.AddInput<bool>("training_mode", {}, {true});

    bitmask_dropout_grad.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, output_values, output_size);

    bitmask_dropout_grad.Run();
  }
}

#endif

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime