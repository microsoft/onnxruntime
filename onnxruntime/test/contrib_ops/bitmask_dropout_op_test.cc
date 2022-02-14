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

namespace {

// sin(1, 2, ..., 101)
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
constexpr std::initializer_list<int64_t> kRandomValuesVectorAlignedDims = {10, 10};

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
constexpr std::initializer_list<int64_t> kRandomValuesVectorUnalignedDims = {9, 11};

constexpr int64_t kMaskSeed = 42;
constexpr std::initializer_list<float> kTestedRatios = {0.00f, 0.25f, 0.50f, 0.75f, 0.99f};

static std::vector<uint32_t> maskToBitmask(const std::vector<bool>& mask) {
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

static std::vector<uint32_t> maskToBitmask(const bool* data, size_t size) {
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

}  // namespace

// BiasDropout kernel is only implemented for CUDA
#if defined(USE_CUDA)

// Test Aligned
TEST(BitmaskDropoutTest, InferenceAligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceRatioAligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  // Ratio is ignored in inference mode
  test.AddInput<float>("ratio", {}, {0.25f});

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceSeedAligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  // Seed is ignored in inference mode.
  test.AddAttribute<int64_t>("seed", kMaskSeed);

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceExplicitAligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  test.AddOptionalInputEdge<float>();

  // Training mode may be explicitly passed as false, as well.
  test.AddInput<bool>("training_mode", {}, {false});

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceMaskOutputAligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  test.AddOutput<float>("output", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);

  std::vector<bool> expected_mask(kRandomValuesVectorAligned.size(), true);
  std::vector<uint32_t> expected_bitmask = maskToBitmask(expected_mask);
  int64_t bitmask_elements = static_cast<int64_t>(expected_bitmask.size());
  test.AddOutput<uint32_t>("mask", {bitmask_elements}, expected_bitmask);
  test.Run();
}

TEST(BitmaskDropoutTest, DropoutSameOutputValuesAligned) {
  for (float ratio : kTestedRatios) {
    OpTester dropout("Dropout", 13, kOnnxDomain, false);

    dropout.AddAttribute<int64_t>("seed", kMaskSeed);

    dropout.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    dropout.AddInput<float>("ratio", {}, {ratio});
    dropout.AddInput<bool>("training_mode", {}, {true});

    // Outputs need to be populated with a value (even if incorrect), test output is not checked.
    // We need the output value for this test, but OpTester has no way to populate an output
    // while explicitly ignoring its value.
    constexpr size_t kFakeOutputSize = kRandomValuesVectorAligned.size();
    constexpr std::initializer_list<int64_t> kFakeOutputDims = kRandomValuesVectorAlignedDims;
    std::array<float, kFakeOutputSize> fake_output;
    fake_output.fill(0.0f);
    std::array<bool, kFakeOutputSize> fake_mask;
    fake_mask.fill(false);

    dropout.AddOutput<float>("output", kFakeOutputDims, fake_output.data(), fake_output.size());
    dropout.AddOutput<bool>("mask", kFakeOutputDims, fake_mask.data(), fake_mask.size());

    // Fake output will not be verified, we run the op solely to get output.
    dropout.Run();

    std::vector<OrtValue> dropout_output = dropout.GetFetches();

    const float* output_values = dropout_output[0].Get<Tensor>().Data<float>();
    const bool* mask_values = dropout_output[1].Get<Tensor>().Data<bool>();

    std::vector<uint32_t> bitmask_values = maskToBitmask(mask_values, kFakeOutputSize);

    OpTester bitmask_dropout("BitmaskDropout", 1, kMSDomain);

    bitmask_dropout.AddAttribute<int64_t>("seed", kMaskSeed);

    bitmask_dropout.AddInput<float>("data", kRandomValuesVectorAlignedDims, kRandomValuesVectorAligned);
    bitmask_dropout.AddInput<float>("ratio", {}, {ratio});
    bitmask_dropout.AddInput<bool>("training_mode", {}, {true});

    bitmask_dropout.AddOutput<float>("output", kFakeOutputDims, output_values, kFakeOutputSize);
    int64_t bitmask_elements = bitmask_values.size();
    bitmask_dropout.AddOutput<uint32_t>("mask", {bitmask_elements}, bitmask_values);

    bitmask_dropout.Run();
  }
}

// Test unaligned
TEST(BitmaskDropoutTest, InferenceUnaligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceRatioUnaligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  // Ratio is ignored in inference mode
  test.AddInput<float>("ratio", {}, {0.25f});

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceSeedUnaligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddAttribute<int64_t>("seed", kMaskSeed);

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceExplicitUnaligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  test.AddOptionalInputEdge<float>();

  // Training mode may be explicitly passed as false, as well.
  test.AddInput<bool>("training_mode", {}, {false});

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
  test.Run();
}

TEST(BitmaskDropoutTest, InferenceMaskOutputUnaligned) {
  OpTester test("BitmaskDropout", 1, kMSDomain);

  test.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  test.AddOutput<float>("output", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);

  std::vector<bool> expected_mask(kRandomValuesVectorUnaligned.size(), true);
  std::vector<uint32_t> expected_bitmask = maskToBitmask(expected_mask);
  int64_t bitmask_elements = static_cast<int64_t>(expected_bitmask.size());
  test.AddOutput<uint32_t>("mask", {bitmask_elements}, expected_bitmask);
  test.Run();
}

TEST(BitmaskDropoutTest, DropoutSameOutputValuesUnaligned) {
  for (float ratio : kTestedRatios) {
    OpTester dropout("Dropout", 13, kOnnxDomain, false);

    dropout.AddAttribute<int64_t>("seed", kMaskSeed);

    dropout.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    dropout.AddInput<float>("ratio", {}, {ratio});
    dropout.AddInput<bool>("training_mode", {}, {true});

    // Outputs need to be populated with a value (even if incorrect), Test output won't actually be checked.
    // We need the output value for this test, but OpTester has no way to populate an output
    // while explicitly ignoring its value.
    constexpr size_t kFakeOutputSize = kRandomValuesVectorUnaligned.size();
    constexpr std::initializer_list<int64_t> kFakeOutputDims = kRandomValuesVectorUnalignedDims;
    std::array<float, kFakeOutputSize> fake_output;
    fake_output.fill(0.0f);
    std::array<bool, kFakeOutputSize> fake_mask;
    fake_mask.fill(false);

    dropout.AddOutput<float>("output", kFakeOutputDims, fake_output.data(), fake_output.size());
    dropout.AddOutput<bool>("mask", kFakeOutputDims, fake_mask.data(), fake_mask.size());

    // Fake output will not be verified, we run the op solely to get output.
    dropout.Run();

    std::vector<OrtValue> dropout_output = dropout.GetFetches();

    const float* output_values = dropout_output[0].Get<Tensor>().Data<float>();
    const bool* mask_values = dropout_output[1].Get<Tensor>().Data<bool>();

    std::vector<uint32_t> bitmask_values = maskToBitmask(mask_values, kFakeOutputSize);

    OpTester bitmask_dropout("BitmaskDropout", 1, kMSDomain);

    bitmask_dropout.AddAttribute<int64_t>("seed", kMaskSeed);

    bitmask_dropout.AddInput<float>("data", kRandomValuesVectorUnalignedDims, kRandomValuesVectorUnaligned);
    bitmask_dropout.AddInput<float>("ratio", {}, {ratio});
    bitmask_dropout.AddInput<bool>("training_mode", {}, {true});

    bitmask_dropout.AddOutput<float>("output", kFakeOutputDims, output_values, kFakeOutputSize);
    int64_t bitmask_elements = bitmask_values.size();
    bitmask_dropout.AddOutput<uint32_t>("mask", {bitmask_elements}, bitmask_values);

    bitmask_dropout.Run();
  }
}
#endif

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime