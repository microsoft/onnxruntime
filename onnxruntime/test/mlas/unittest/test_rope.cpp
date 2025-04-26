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
#include "core/framework/float16.h"
#include "core/mlas/lib/rotary_embedding.h"

using namespace onnxruntime;

template <typename T>
class MlasRoPETest : public MlasTestBase {
 public:
  void Test(size_t rotary_emb_dim, bool interleaved) {
    std::vector<T> input(rotary_emb_dim);
    size_t table_len = interleaved ? rotary_emb_dim / 2 : rotary_emb_dim;
    std::vector<T> sin_data(table_len);
    std::vector<T> cos_data(table_len);
    std::vector<T> output_ref(rotary_emb_dim), output_impl(rotary_emb_dim);

    for (size_t i = 0; i < rotary_emb_dim; ++i) {
      input[i] = static_cast<T>(i + 1.0f);
    }
    for (size_t i = 0; i < table_len; ++i) {
      // https://arxiv.org/pdf/2104.09864 section 3.4.3
      float theta_i = static_cast<float>(pow(10000, -2.0f * i / rotary_emb_dim));
      sin_data[i] = static_cast<T>(std::sin(theta_i));
      cos_data[i] = static_cast<T>(std::cos(theta_i));
    }

    // Call the function
    MlasRotaryEmbedOneRow_FallBack<T>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_ref[0]);
    MlasRotaryEmbedOneRow<T>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);

    for (size_t i = 0; i < rotary_emb_dim; i++) {
      ASSERT_TRUE(CloseEnough(output_impl[i], output_ref[i]))
          << "Expected: " << output_ref[i] << " Actual: " << output_impl[i] << "@[" << i << "], "
          << "rotary_emb_dim=" << rotary_emb_dim << ", interleaved=" << interleaved;
    }
  }
};

//
// Short Execute() test helper to register each test separately by all parameters.
//
template <typename T>
class RoPEShortExecuteTest : public MlasTestFixture<MlasRoPETest<T>> {
 public:
  explicit RoPEShortExecuteTest(size_t rotary_emb_dim, bool interleaved)
      : rotary_emb_dim_(rotary_emb_dim),
        interleaved_(interleaved) {}

  void TestBody() override {
    MlasTestFixture<MlasRoPETest<T>>::mlas_tester->Test(rotary_emb_dim_, interleaved_);
  }

  static size_t RegisterSingleTest(size_t rotary_emb_dim, bool interleaved) {
    size_t tests_registered = 0;

    std::string test_suite_name{"RoPE_"};
    if (std::is_same<T, float>::value) {
      test_suite_name += "fp32";
    } else if (std::is_same<T, MLFloat16>::value) {
      test_suite_name += "fp16";
    } else {
      ADD_FAILURE() << "Unknown type passed to test: " << test_suite_name;
      return 0;  // Return 0 since no test is registered
    }

    std::stringstream ss;
    ss << "/rotary_emb_dim" << rotary_emb_dim << "/interleaved" << interleaved;
    auto test_name = ss.str();

    testing::RegisterTest(
        test_suite_name.c_str(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasRoPETest<T>>* {
          return new RoPEShortExecuteTest<T>(rotary_emb_dim, interleaved);
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
  return RoPEShortExecuteTest<float>::RegisterShortExecuteTests() + RoPEShortExecuteTest<MLFloat16>::RegisterShortExecuteTests();
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister(
    [](bool is_short_execute) -> size_t {
      if (is_short_execute) {
        return RoPERegisterAllShortExecuteTests();
      }
      return 0;
    });
#endif
