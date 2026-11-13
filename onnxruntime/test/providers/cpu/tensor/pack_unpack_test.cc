// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/allocator.h"
#include "core/framework/customregistry.h"
#include "core/providers/cpu/tensor/pack_unpack.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

using Attrs = std::map<std::string, int64_t>;

std::shared_ptr<CustomRegistry> TestSchemas() {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas;
  for (bool unpack : {false, true}) {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetName(unpack ? "Unpack" : "Pack").SetDomain(kOnnxDomain).SinceVersion(29).Input(0, "X", "", unpack ? "TStorage" : "T").Output(0, "Y", "", unpack ? "T" : "TStorage").Attr("axis", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{-1}).Attr("bits", "", ONNX_NAMESPACE::AttributeProto::INT, false).Attr("radix", "", ONNX_NAMESPACE::AttributeProto::INT, false).Attr("group_size", "", ONNX_NAMESPACE::AttributeProto::INT, false).TypeConstraint("T", {"tensor(uint8)"}, "").TypeConstraint("TStorage", {"tensor(uint8)", "tensor(uint32)"}, "");
    if (unpack) {
      schema.Input(1, "count", "", "tensor(int64)");
    } else {
      schema.Attr("output_type", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{2});
    }
    schemas.push_back(std::move(schema));
  }
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 28, 29));
  return registry;
}

void Run(OpTester& test, bool cuda, OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
         const std::string& error = "") {
  test.SetAllowUnreleasedOnnxOpset();
  test.AddCustomOpRegistry(TestSchemas());
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  if (cuda) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider());
  test.Config(options).ConfigEps(std::move(providers)).Config(expect, error).RunWithConfig();
}

template <typename Word>
void CheckPack(const std::vector<int64_t>& shape, const std::vector<uint8_t>& codes,
               const std::vector<int64_t>& packed_shape, const std::vector<Word>& expected,
               Attrs attrs, bool cuda) {
  OpTester test("Pack", 29);
  attrs["output_type"] = sizeof(Word) == 4 ? 12 : 2;
  for (const auto& attr : attrs) test.AddAttribute(attr.first, attr.second);
  test.AddInput<uint8_t>("X", shape, codes);
  test.AddOutput<Word>("Y", packed_shape, expected);
  Run(test, cuda);
}

template <typename Word>
void CheckUnpack(const std::vector<int64_t>& packed_shape, const std::vector<Word>& words,
                 const std::vector<int64_t>& shape, const std::vector<uint8_t>& expected,
                 Attrs attrs, bool cuda) {
  OpTester test("Unpack", 29);
  for (const auto& attr : attrs) test.AddAttribute(attr.first, attr.second);
  const int64_t axis = attrs.count("axis") ? attrs.at("axis") : -1;
  test.AddInput<Word>("X", packed_shape, words);
  test.AddInput<int64_t>("count", {}, {shape[static_cast<size_t>(axis < 0 ? axis + shape.size() : axis)]});
  test.AddOutput<uint8_t>("Y", shape, expected);
  Run(test, cuda);
}

// Independent oracle: materialize each group's bits and place them one at a time.
template <typename Word>
std::vector<Word> Oracle(const std::vector<int64_t>& shape, const std::vector<uint8_t>& codes,
                         int axis, int radix, int group_size, std::vector<int64_t>& packed_shape) {
  if (axis < 0) axis += static_cast<int>(shape.size());
  uint64_t limit = 1;
  for (int i = 0; i < group_size; ++i) limit *= radix;
  int width = 0;
  for (uint64_t v = limit - 1; v; v >>= 1) ++width;
  const int64_t n = shape[axis];
  const int64_t groups = (n + group_size - 1) / group_size;
  const int64_t words = (groups * width + sizeof(Word) * 8 - 1) / (sizeof(Word) * 8);
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; ++i) outer *= shape[i];
  for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
  packed_shape = shape;
  packed_shape[axis] = words;
  std::vector<Word> result(static_cast<size_t>(outer * words * inner), 0);
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t k = 0; k < inner; ++k) {
      for (int64_t g = 0; g < groups; ++g) {
        uint64_t value = 0, power = 1;
        for (int j = 0; j < group_size; ++j) {
          if (g * group_size + j < n) {
            value += codes[(o * n + g * group_size + j) * inner + k] * power;
          }
          if (j + 1 < group_size) power *= radix;
        }
        for (int bit = 0; bit < width; ++bit) {
          const int64_t position = g * width + bit;
          result[(o * words + position / (sizeof(Word) * 8)) * inner + k] |=
              static_cast<Word>(((value >> bit) & 1) << (position % (sizeof(Word) * 8)));
        }
      }
    }
  }
  return result;
}

template <typename Word>
void CheckOracle(const std::vector<int64_t>& shape, int axis, int radix, int group_size,
                 const Attrs& attrs, bool cuda) {
  const size_t size = static_cast<size_t>(TensorShape(shape).Size());
  std::vector<uint8_t> codes(size);
  for (size_t i = 0; i < size; ++i) codes[i] = static_cast<uint8_t>((i * 73 + i / 7 + 11) % radix);
  std::vector<int64_t> packed_shape;
  const auto words = Oracle<Word>(shape, codes, axis, radix, group_size, packed_shape);
  CheckPack(shape, codes, packed_shape, words, attrs, cuda);
  CheckUnpack(packed_shape, words, shape, codes, attrs, cuda);
}

void KnownLayouts(bool cuda) {
  CheckPack<uint8_t>({8}, {0, 1, 2, 3, 4, 5, 6, 7}, {3}, {136, 198, 250}, {{"bits", 3}}, cuda);
  CheckPack<uint8_t>({2, 3}, {1, 2, 3, 31, 0, 16}, {2, 2}, {65, 12, 31, 64}, {{"bits", 5}}, cuda);
  CheckPack<uint32_t>({3}, {1, 2, 3}, {1}, {3137}, {{"bits", 5}}, cuda);
  CheckPack<uint32_t>({7}, {31, 31, 31, 31, 31, 31, 31}, {2}, {0xffffffff, 7}, {{"bits", 5}}, cuda);
  CheckPack<uint8_t>({5}, {0, 1, 2, 0, 1}, {1}, {102}, {{"radix", 3}, {"group_size", 5}}, cuda);
  CheckPack<uint8_t>({2}, {1, 2}, {1}, {7}, {{"radix", 3}, {"group_size", 5}}, cuda);
  CheckUnpack<uint8_t>({1}, {255}, {2}, {7, 7}, {{"bits", 3}}, cuda);
  CheckUnpack<uint32_t>({2}, {0xffffffff, 0xffffffff}, {7}, {31, 31, 31, 31, 31, 31, 31}, {{"bits", 5}}, cuda);
  CheckUnpack<uint8_t>({1}, {242}, {2}, {2, 2}, {{"radix", 3}, {"group_size", 5}}, cuda);
  CheckUnpack<uint8_t>({1}, {0xf8}, {2}, {2, 2}, {{"radix", 3}, {"group_size", 2}}, cuda);
}

void BitsAndAxes(bool cuda) {
  for (int bits = 1; bits <= 8; ++bits) {
    for (int axis : {0, -2, -1}) {
      const Attrs attrs{{"bits", bits}, {"axis", axis}};
      CheckOracle<uint8_t>({3, 17, 5}, axis, 1 << bits, 1, attrs, cuda);
      CheckOracle<uint32_t>({3, 17, 5}, axis, 1 << bits, 1, attrs, cuda);
    }
    CheckOracle<uint8_t>({67}, -1, 1 << bits, 1, {{"radix", 1 << bits}, {"group_size", 1}}, cuda);
  }
}

void GroupedAndLargeIndices(bool cuda) {
  for (const auto& pair : std::array<std::array<int, 2>, 11>{{{3, 5}, {3, 3}, {5, 3}, {7, 2}, {16, 2}, {256, 1}, {2, 62}, {3, 39}, {7, 22}, {5, 27}, {255, 7}}}) {
    for (int axis : {0, 1, -1}) {
      const Attrs attrs{{"radix", pair[0]}, {"group_size", pair[1]}, {"axis", axis}};
      CheckOracle<uint8_t>({3, 41, 67}, axis, pair[0], pair[1], attrs, cuda);
      CheckOracle<uint32_t>({3, 41, 67}, axis, pair[0], pair[1], attrs, cuda);
    }
  }
  // 3^39 - 1 = 4052555153018976266: exceeds exact floating-point integer precision.
  CheckPack<uint32_t>({39}, std::vector<uint8_t>(39, 2), {2}, {0xb85ff80a, 0x383d9170},
                      {{"radix", 3}, {"group_size", 39}}, cuda);
  CheckUnpack<uint32_t>({2}, {0xb85ff80a, 0x383d9170}, {39}, std::vector<uint8_t>(39, 2),
                        {{"radix", 3}, {"group_size", 39}}, cuda);
  CheckPack<uint32_t>({62}, std::vector<uint8_t>(62, 1), {2}, {0xffffffff, 0x3fffffff},
                      {{"radix", 2}, {"group_size", 62}}, cuda);
}

void EmptyAndLongRows(bool cuda) {
  const int64_t maximum = std::numeric_limits<int64_t>::max();
  CheckPack<uint8_t>({maximum, 1, 0}, {}, {maximum, 7, 0}, {},
                     {{"radix", 256}, {"group_size", 7}, {"axis", 1}}, cuda);
  CheckUnpack<uint8_t>({maximum, 7, 0}, {}, {maximum, 1, 0}, {},
                       {{"radix", 256}, {"group_size", 7}, {"axis", 1}}, cuda);
  for (const auto& shape : std::vector<std::vector<int64_t>>{{0}, {0, 17, 3}, {2, 0, 3}, {2, 17, 0}}) {
    for (int axis = 0; axis < static_cast<int>(shape.size()); ++axis) {
      CheckOracle<uint8_t>(shape, axis, 3, 39, {{"radix", 3}, {"group_size", 39}, {"axis", axis}}, cuda);
      CheckOracle<uint32_t>(shape, axis, 32, 1, {{"bits", 5}, {"axis", axis}}, cuda);
    }
  }
  CheckOracle<uint8_t>({2, 100001}, -1, 3, 5, {{"radix", 3}, {"group_size", 5}}, cuda);
  CheckOracle<uint32_t>({2, 100001}, -1, 32, 1, {{"bits", 5}}, cuda);
  CheckOracle<uint8_t>({2, 100001}, -1, 2, 1, {{"bits", 1}}, cuda);
  CheckOracle<uint32_t>({2, 100001}, -1, 2, 1, {{"bits", 1}}, cuda);
}

void InvalidMetadata(bool cuda) {
  for (const Attrs& attrs : std::vector<Attrs>{
           {}, {{"radix", 3}}, {{"group_size", 5}}, {{"bits", 3}, {"radix", 3}}, {{"bits", 3}, {"group_size", 5}}, {{"bits", 3}, {"radix", 3}, {"group_size", 5}}, {{"bits", 0}}, {{"bits", 9}}, {{"bits", -1}}, {{"radix", 1}, {"group_size", 5}}, {{"radix", 257}, {"group_size", 1}}, {{"radix", 3}, {"group_size", 0}}, {{"radix", 3}, {"group_size", -1}}, {{"radix", 256}, {"group_size", 8}}, {{"radix", 2}, {"group_size", 63}}, {{"radix", 3}, {"group_size", 40}}}) {
    for (bool unpack : {false, true}) {
      OpTester test(unpack ? "Unpack" : "Pack", 29);
      for (const auto& attr : attrs) test.AddAttribute(attr.first, attr.second);
      test.AddInput<uint8_t>("X", {1}, {0});
      if (unpack) test.AddInput<int64_t>("count", {}, {1});
      test.AddOutput<uint8_t>("Y", {1}, {0});
      Run(test, cuda, OpTester::ExpectResult::kExpectFailure, "Pack/Unpack:");
    }
  }
  for (bool unpack : {false, true}) {
    for (int64_t axis : {-3, 2}) {
      OpTester test(unpack ? "Unpack" : "Pack", 29);
      test.AddAttribute("bits", int64_t{3});
      test.AddAttribute("axis", axis);
      test.AddInput<uint8_t>("X", {1, 1}, {0});
      if (unpack) test.AddInput<int64_t>("count", {}, {1});
      test.AddOutput<uint8_t>("Y", {1, 1}, {0});
      Run(test, cuda, OpTester::ExpectResult::kExpectFailure, "axis must be");
    }
    OpTester scalar(unpack ? "Unpack" : "Pack", 29);
    scalar.AddAttribute("bits", int64_t{3});
    scalar.AddInput<uint8_t>("X", {}, {0});
    if (unpack) scalar.AddInput<int64_t>("count", {}, {1});
    scalar.AddOutput<uint8_t>("Y", {}, {0});
    Run(scalar, cuda, OpTester::ExpectResult::kExpectFailure, "rank at least 1");
  }
  for (int64_t count : {-1, 0, 6, 11}) {
    OpTester test("Unpack", 29);
    test.AddAttribute("radix", int64_t{3});
    test.AddAttribute("group_size", int64_t{5});
    test.AddInput<uint8_t>("X", {1}, {0});
    test.AddInput<int64_t>("count", {}, {count});
    test.AddOutput<uint8_t>("Y", {1}, {0});
    Run(test, cuda, OpTester::ExpectResult::kExpectFailure, count < 0 ? "nonnegative" : "required word count");
  }
  OpTester vector_count("Unpack", 29);
  vector_count.AddAttribute("bits", int64_t{3});
  vector_count.AddInput<uint8_t>("X", {1}, {0});
  vector_count.AddInput<int64_t>("count", {1}, {1});
  vector_count.AddOutput<uint8_t>("Y", {1}, {0});
  Run(vector_count, cuda, OpTester::ExpectResult::kExpectFailure, "scalar INT64");
  OpTester output_type("Pack", 29);
  output_type.AddAttribute("bits", int64_t{3});
  output_type.AddAttribute("output_type", int64_t{6});
  output_type.AddInput<uint8_t>("X", {1}, {0});
  output_type.AddOutput<uint8_t>("Y", {1}, {0});
  Run(output_type, cuda, OpTester::ExpectResult::kExpectFailure, "output_type must be");
}

TEST(PackUnpackTest, CpuKnownLayoutsAndTrimming) { KnownLayouts(false); }
TEST(PackUnpackTest, CpuAllBitsAndAxes) { BitsAndAxes(false); }
TEST(PackUnpackTest, CpuGroupedAndLargeIndices) { GroupedAndLargeIndices(false); }
TEST(PackUnpackTest, CpuEmptyAndLongRows) { EmptyAndLongRows(false); }
TEST(PackUnpackTest, CpuInvalidMetadata) { InvalidMetadata(false); }

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
TEST(PackUnpackTest, CudaKnownLayoutsAndTrimming) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  KnownLayouts(true);
}
TEST(PackUnpackTest, CudaAllBitsAndAxes) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  BitsAndAxes(true);
}
TEST(PackUnpackTest, CudaGroupedAndLargeIndices) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  GroupedAndLargeIndices(true);
}
TEST(PackUnpackTest, CudaEmptyAndLongRows) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  EmptyAndLongRows(true);
}
TEST(PackUnpackTest, CudaInvalidMetadata) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  InvalidMetadata(true);
}
#endif

struct AttributeInfo {
  Attrs attrs;
  Status GetAttr(const std::string& name, int64_t* out) const {
    const auto it = attrs.find(name);
    if (it == attrs.end()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Missing attribute ", name);
    *out = it->second;
    return Status::OK();
  }
  template <typename T>
  T GetAttrOrDefault(const std::string& name, T fallback) const {
    const auto it = attrs.find(name);
    return it == attrs.end() ? fallback : static_cast<T>(it->second);
  }
};

TEST(PackUnpackTest, DimensionArithmeticAndEmptyHugeShapes) {
  const int64_t maximum = std::numeric_limits<int64_t>::max();
  const packing::Attributes bytes(AttributeInfo{{{"bits", 8}}});
  EXPECT_EQ(bytes.WordCount(maximum, 8), maximum);
  const packing::Attributes five_bits(AttributeInfo{{{"bits", 5}}});
  EXPECT_EQ(five_bits.WordCount(maximum, 32), int64_t{1441151880758558720});
  const packing::Attributes ternary(AttributeInfo{{{"radix", 3}, {"group_size", 1}}});
  EXPECT_EQ(ternary.WordCount(maximum, 8), int64_t{2305843009213693952});
  const packing::Attributes expanding(AttributeInfo{{{"radix", 256}, {"group_size", 3}}});
  EXPECT_THROW(expanding.WordCount(maximum, 8), OnnxRuntimeException);
  auto allocator = std::make_shared<CPUAllocator>();
  Tensor input(DataTypeImpl::GetType<uint8_t>(), TensorShape({0, maximum}), allocator);
  TensorShapeVector output_shape;
  packing::Layout layout{};
  ASSERT_STATUS_OK(five_bits.Prepare(input, nullptr, false, output_shape, layout));
  EXPECT_EQ(output_shape, (TensorShapeVector{0, 5764607523034234880}));
  EXPECT_EQ(layout.inner, 0u);
  Tensor bad_type(DataTypeImpl::GetType<uint16_t>(), TensorShape({1}), allocator);
  EXPECT_FALSE(bytes.Prepare(bad_type, nullptr, false, output_shape, layout).IsOK());
  Tensor count(DataTypeImpl::GetType<int64_t>(), TensorShape({}), allocator);
  *count.MutableData<int64_t>() = 1;
  EXPECT_FALSE(bytes.Prepare(bad_type, &count, true, output_shape, layout).IsOK());
  Tensor word(DataTypeImpl::GetType<uint8_t>(), TensorShape({1}), allocator);
  Tensor bad_count(DataTypeImpl::GetType<int32_t>(), TensorShape({}), allocator);
  EXPECT_FALSE(bytes.Prepare(word, &bad_count, true, output_shape, layout).IsOK());
  Tensor empty_words(DataTypeImpl::GetType<uint8_t>(), TensorShape({0, 1}), allocator);
  *count.MutableData<int64_t>() = 6;
  const packing::Attributes grouped(AttributeInfo{{{"radix", 3}, {"group_size", 5}}});
  EXPECT_FALSE(grouped.Prepare(empty_words, &count, true, output_shape, layout).IsOK());
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
