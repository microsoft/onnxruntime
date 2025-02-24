/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_rope.h

Abstract:

    Tests for MLAS RoPE.

--*/

#include "test_util.h"
#include "mlas.h"
#include "core/mlas/lib/rotary_embedding.h"

class MlasRoPETest : public MlasTestBase {
  const float Pi = 2 * std::acos(0.0f);

 public:
  void Test(size_t rotary_emb_dim, bool interleaved) {
    std::vector<float> input(rotary_emb_dim);
    size_t table_len = interleaved ? rotary_emb_dim / 2 : rotary_emb_dim;
    std::vector<float> sin_data(table_len);
    std::vector<float> cos_data(table_len);
    std::vector<float> output_ref(rotary_emb_dim), output_impl(rotary_emb_dim);

    for (size_t i = 0; i < rotary_emb_dim; ++i) {
      input[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < table_len; ++i) {
      float theta = (float)i / 1000 * Pi;
      sin_data[i] = std::sin(theta);
      cos_data[i] = std::cos(theta);
    }

    // Call the function
    MlasRotaryEmbedOneRow_FallBack<float>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_ref[0]);
    MlasRotaryEmbedOneRow<float>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);

    for (size_t i = 0; i < rotary_emb_dim; i++) {
      ASSERT_TRUE(CloseEnough(output_impl[i], output_ref[i]))
          << "Expected: " << output_ref[i] << " Actual: " << output_impl[i] << "@[" << i << "], "
          << "rotary_emb_dim=" << rotary_emb_dim << ", interleaved=" << interleaved;
    }
  }

 public:
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
class RoPEShortExecuteTest : public MlasTestFixture<MlasRoPETest> {
 public:
  explicit RoPEShortExecuteTest(size_t rotary_emb_dim, bool interleaved)
      : rotary_emb_dim_(rotary_emb_dim),
        interleaved_(interleaved) {}

  void TestBody() override {
    MlasTestFixture<MlasRoPETest>::mlas_tester->Test(rotary_emb_dim_, interleaved_);
  }

  static size_t RegisterSingleTest(size_t rotary_emb_dim, bool interleaved) {
    size_t tests_registered = 0;

    std::stringstream ss;
    ss << "/rotary_emb_dim" << rotary_emb_dim << "/interleaved" << interleaved;
    auto test_name = ss.str();

    testing::RegisterTest(
        "RoPE",
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasRoPETest>* {
          return new RoPEShortExecuteTest(rotary_emb_dim, interleaved);
        });

    tests_registered += 1;

    return tests_registered;
  }

  static size_t RegisterShortExecuteTests() {
    size_t tests_registered = 0;
    tests_registered += RegisterSingleTest(6, false);
    tests_registered += RegisterSingleTest(6, true);
    tests_registered += RegisterSingleTest(16, false);
    tests_registered += RegisterSingleTest(16, true);
    tests_registered += RegisterSingleTest(24, false);
    tests_registered += RegisterSingleTest(24, true);
    tests_registered += RegisterSingleTest(32, false);
    tests_registered += RegisterSingleTest(32, true);
    tests_registered += RegisterSingleTest(42, false);
    tests_registered += RegisterSingleTest(42, true);
    tests_registered += RegisterSingleTest(64, false);
    tests_registered += RegisterSingleTest(64, true);
    tests_registered += RegisterSingleTest(70, false);
    tests_registered += RegisterSingleTest(70, true);
    return tests_registered;
  }

 private:
  size_t rotary_emb_dim_;
  bool interleaved_;
};

// only test float RoPE with avx2 where RopeDispatch is assigned at this moment.
#ifdef MLAS_TARGET_AMD64
static size_t RoPERegisterAllShortExecuteTests() {
  return RoPEShortExecuteTest::RegisterShortExecuteTests();
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return RoPERegisterAllShortExecuteTests();
      }
      return 0;
    });
#endif
