// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

using Attributes = std::vector<std::pair<std::string, int64_t>>;

void Configure(OpTester& test, const Attributes& attributes) {
  test.SetAllowUnreleasedOnnxOpset();
  SessionOptions options;
  options.use_per_session_threads = false;
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options);
  test.ConfigEp(DefaultCpuExecutionProvider());
  for (const auto& [name, value] : attributes) {
    test.AddAttribute(name, value);
  }
}

template <typename Word>
void TestPack(const std::vector<int64_t>& shape, const std::vector<uint8_t>& input,
              const std::vector<int64_t>& packed_shape, const std::vector<Word>& expected,
              Attributes attributes, bool success = true, const std::string& error = "") {
  OpTester test("Pack", 29);
  if constexpr (sizeof(Word) == 4) {
    attributes.emplace_back("output_type", ONNX_NAMESPACE::TensorProto::UINT32);
  }
  Configure(test, attributes);
  test.AddInput<uint8_t>("X", shape, input);
  test.AddOutput<Word>("Y", packed_shape, expected);
  test.Config(success ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure, error);
  test.RunWithConfig();
  if (success) {
    EXPECT_EQ(test.GetNumberOfNodesAfterRun(), 1);
  }
}

template <typename Word>
void TestUnpack(const std::vector<int64_t>& shape, const std::vector<Word>& input,
                int64_t count, const std::vector<int64_t>& output_shape, const std::vector<uint8_t>& expected,
                const Attributes& attributes, bool success = true, const std::string& error = "",
                bool constant_count = false) {
  OpTester test("Unpack", 29);
  Configure(test, attributes);
  test.AddInput<Word>("X", shape, input);
  test.AddInput<int64_t>("count", {}, {count}, constant_count);
  test.AddOutput<uint8_t>("Y", output_shape, expected);
  test.Config(success ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure, error);
  test.RunWithConfig();
  if (success) {
    EXPECT_EQ(test.GetNumberOfNodesAfterRun(), 1);
  }
}

// Deliberately encode one bit at a time, independently of the kernels' word-chunk implementation.
template <typename Word>
void TestReference(int64_t radix, int64_t group_size, int64_t count, int64_t axis, bool bits_spelling) {
  const size_t normalized_axis = static_cast<size_t>(axis < 0 ? axis + 3 : axis);
  std::vector<int64_t> shape{2, 3, 2};
  shape[normalized_axis] = count;
  uint64_t power = 1;
  for (int64_t i = 0; i < group_size; ++i) {
    power *= static_cast<uint64_t>(radix);
  }
  int width = 0;
  for (uint64_t value = power - 1; value != 0; value >>= 1) {
    ++width;
  }
  constexpr int word_bits = sizeof(Word) * 8;
  const int64_t groups = count / group_size + (count % group_size != 0);
  const int64_t words = (groups * width + word_bits - 1) / word_bits;
  auto packed_shape = shape;
  packed_shape[normalized_axis] = words;
  const int64_t outer = std::accumulate(shape.begin(), shape.begin() + normalized_axis,
                                        int64_t{1}, std::multiplies<int64_t>());
  const int64_t inner = std::accumulate(shape.begin() + normalized_axis + 1, shape.end(),
                                        int64_t{1}, std::multiplies<int64_t>());
  std::vector<uint8_t> input(static_cast<size_t>(outer * count * inner));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>((i * 37 + 11) % radix);
  }
  std::vector<Word> packed(static_cast<size_t>(outer * words * inner), 0);
  for (int64_t row = 0; row < outer; ++row) {
    for (int64_t lane = 0; lane < inner; ++lane) {
      std::vector<bool> stream;
      for (int64_t group = 0; group < groups; ++group) {
        uint64_t index = 0;
        uint64_t place = 1;
        for (int64_t digit = 0; digit < group_size && group * group_size + digit < count; ++digit) {
          index += input[static_cast<size_t>((row * count + group * group_size + digit) * inner + lane)] * place;
          place *= static_cast<uint64_t>(radix);
        }
        for (int bit = 0; bit < width; ++bit) {
          stream.push_back(((index >> bit) & 1) != 0);
        }
      }
      for (size_t bit = 0; bit < stream.size(); ++bit) {
        if (stream[bit]) {
          const auto offset = static_cast<size_t>(row * words * inner + lane) + (bit / word_bits) * inner;
          packed[offset] |= static_cast<Word>(uint64_t{1} << (bit % word_bits));
        }
      }
    }
  }
  Attributes attributes{{"axis", axis}};
  if (bits_spelling) {
    attributes.emplace_back("bits", width);
  } else {
    attributes.emplace_back("radix", radix);
    attributes.emplace_back("group_size", group_size);
  }
  TestPack<Word>(shape, input, packed_shape, packed, attributes);
  TestUnpack<Word>(packed_shape, packed, count, shape, input, attributes);
}

TEST(PackUnpackTest, ProposalExamples) {
  TestPack<uint8_t>({8}, {0, 1, 2, 3, 4, 5, 6, 7}, {3}, {136, 198, 250}, {{"bits", 3}});
  TestUnpack<uint8_t>({3}, {136, 198, 250}, 8, {8}, {0, 1, 2, 3, 4, 5, 6, 7}, {{"bits", 3}});
  TestPack<uint8_t>({2, 3}, {1, 2, 3, 31, 0, 16}, {2, 2}, {65, 12, 31, 64}, {{"bits", 5}});
  TestUnpack<uint8_t>({2, 2}, {65, 12, 31, 64}, 3, {2, 3}, {1, 2, 3, 31, 0, 16}, {{"bits", 5}});
  TestPack<uint32_t>({3}, {1, 2, 3}, {1}, {3137}, {{"bits", 5}});
  TestUnpack<uint32_t>({1}, {3137}, 3, {3}, {1, 2, 3}, {{"bits", 5}}, true, "", true);
  TestPack<uint8_t>({5}, {0, 1, 2, 0, 1}, {1}, {102}, {{"radix", 3}, {"group_size", 5}});
  TestUnpack<uint8_t>({1}, {102}, 5, {5}, {0, 1, 2, 0, 1}, {{"radix", 3}, {"group_size", 5}});
  TestPack<uint8_t>({5}, {2, 2, 2, 2, 2}, {1}, {242}, {{"radix", 3}, {"group_size", 5}});
  TestPack<uint8_t>({2}, {1, 2}, {1}, {7}, {{"radix", 3}, {"group_size", 5}});
  TestUnpack<uint8_t>({1}, {7}, 2, {2}, {1, 2}, {{"radix", 3}, {"group_size", 5}});
}

TEST(PackUnpackTest, AllBitWidthsAndAxes) {
  for (int bits = 1; bits <= 8; ++bits) {
    for (int64_t count : {0, 1, 7, 9, 65}) {
      for (int64_t axis : {0, 1, 2, -1, -2, -3}) {
        SCOPED_TRACE(MakeString("bits=", bits, " count=", count, " axis=", axis));
        TestReference<uint8_t>(int64_t{1} << bits, 1, count, axis, true);
        TestReference<uint32_t>(int64_t{1} << bits, 1, count, axis, true);
      }
    }
  }
}

TEST(PackUnpackTest, GroupedPackingAndWideIndices) {
  for (const auto& [radix, group] : std::vector<std::pair<int64_t, int64_t>>{
           {3, 5}, {3, 7}, {2, 62}, {5, 27}, {7, 20}, {16, 15}, {255, 7}, {256, 7}, {256, 1}}) {
    for (int64_t count : {int64_t{0}, int64_t{1}, group - 1, group, group + 1, group * 2 + 3}) {
      for (int64_t axis : {0, 1, -1}) {
        SCOPED_TRACE(MakeString("radix=", radix, " group=", group, " count=", count, " axis=", axis));
        TestReference<uint8_t>(radix, group, count, axis, false);
        TestReference<uint32_t>(radix, group, count, axis, false);
      }
    }
  }
}

TEST(PackUnpackTest, PaddingIsIgnored) {
  TestUnpack<uint8_t>({1}, {255}, 2, {2}, {7, 7}, {{"bits", 3}});
  TestUnpack<uint32_t>({1}, {0xffffffffU}, 2, {2}, {7, 7}, {{"bits", 3}});
  TestUnpack<uint8_t>({1}, {242}, 2, {2}, {2, 2}, {{"radix", 3}, {"group_size", 5}});
  TestUnpack<uint8_t>({1}, {255}, 2, {2}, {1, 1}, {{"radix", 2}, {"group_size", 3}});
}

TEST(PackUnpackTest, EmptyNonAxisDimensions) {
  TestPack<uint8_t>({0, 5, 3}, {}, {0, 2, 3}, {}, {{"bits", 3}, {"axis", 1}});
  TestUnpack<uint8_t>({0, 2, 3}, {}, 5, {0, 5, 3}, {}, {{"bits", 3}, {"axis", 1}});
  TestPack<uint32_t>({2, 5, 0}, {}, {2, 1, 0}, {}, {{"radix", 3}, {"group_size", 5}, {"axis", 1}});
  TestUnpack<uint32_t>({2, 1, 0}, {}, 5, {2, 5, 0}, {}, {{"radix", 3}, {"group_size", 5}, {"axis", 1}});
}

TEST(PackUnpackTest, RejectsInvalidAttributes) {
  const std::vector<Attributes> invalid{
      {}, {{"bits", 0}}, {{"bits", 9}}, {{"radix", 3}}, {{"group_size", 5}}, {{"bits", 3}, {"radix", 3}, {"group_size", 5}}, {{"bits", 3}, {"radix", 3}}, {{"bits", 3}, {"group_size", 5}}, {{"radix", 1}, {"group_size", 1}}, {{"radix", 257}, {"group_size", 1}}, {{"radix", 3}, {"group_size", 0}}, {{"radix", 3}, {"group_size", -1}}, {{"radix", 2}, {"group_size", 63}}, {{"radix", 256}, {"group_size", 8}}, {{"radix", 2}, {"group_size", std::numeric_limits<int64_t>::max()}}, {{"bits", 1}, {"axis", 1}}, {{"bits", 1}, {"axis", -2}}};
  for (const auto& attributes : invalid) {
    TestPack<uint8_t>({1}, {0}, {1}, {0}, attributes, false);
    TestUnpack<uint8_t>({1}, {0}, 1, {1}, {0}, attributes, false);
  }
  TestPack<uint8_t>({1}, {0}, {1}, {0}, {{"bits", 1}, {"output_type", ONNX_NAMESPACE::TensorProto::INT8}}, false);
}

TEST(PackUnpackTest, RejectsInvalidRuntimeInputs) {
  TestPack<uint8_t>({1}, {8}, {1}, {0}, {{"bits", 3}}, false, "input code must be less than radix");
  TestPack<uint32_t>({1}, {3}, {1}, {0}, {{"radix", 3}, {"group_size", 5}}, false,
                     "input code must be less than radix");
  TestUnpack<uint8_t>({1}, {243}, 5, {5}, {0, 0, 0, 0, 0}, {{"radix", 3}, {"group_size", 5}}, false,
                      "group index must be less than radix^group_size");
  TestUnpack<uint8_t>({1}, {0}, -1, {1}, {0}, {{"bits", 3}}, false, "count must be nonnegative");
  TestUnpack<uint8_t>({1}, {0}, 3, {3}, {0, 0, 0}, {{"bits", 3}}, false, "packed word count");
  TestUnpack<uint8_t>({2}, {0, 0}, 1, {1}, {0}, {{"bits", 3}}, false, "packed word count");
  TestUnpack<uint8_t>({1}, {0}, 0, {0}, {}, {{"bits", 3}}, false, "packed word count");
  TestUnpack<uint8_t>({0}, {}, 1, {1}, {0}, {{"bits", 3}}, false, "packed word count");
  TestUnpack<uint8_t>({0, 1}, {}, 3, {0, 3}, {}, {{"bits", 3}}, false, "packed word count");
  TestUnpack<uint8_t>({0, 0}, {}, std::numeric_limits<int64_t>::max(), {0, 1}, {},
                      {{"bits", 8}}, false, "packed word count");
  TestPack<uint8_t>({}, {0}, {}, {0}, {{"bits", 3}}, false);
  TestUnpack<uint8_t>({}, {0}, 1, {}, {0}, {{"bits", 3}}, false);
}

TEST(PackUnpackTest, RejectsNonScalarCount) {
  OpTester test("Unpack", 29);
  Configure(test, {{"bits", 3}});
  test.AddInput<uint8_t>("X", {1}, {0});
  test.AddInput<int64_t>("count", {1}, {1});
  test.AddOutput<uint8_t>("Y", {1}, {0});
  test.Config(OpTester::ExpectResult::kExpectFailure, "");
  test.RunWithConfig();
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
