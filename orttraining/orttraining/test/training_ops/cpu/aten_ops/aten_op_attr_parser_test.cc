#include "gtest/gtest.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_attr_parser.h"

namespace onnxruntime {
namespace test {

using namespace contrib::aten_ops;

TEST(AttributesJsonParserTest, NonArrayValue) {
  std::string json = "{\"a\":-1,\"b\":false,\"c\":0.5,\"d\":1e-5}";
  AttributesJsonParser parser(json);
  int int_value;
  ASSERT_TRUE(parser.TryGetValue<int>("a", int_value) && int_value == -1);
  bool bool_value;
  ASSERT_TRUE(parser.TryGetValue<bool>("b", bool_value) && !bool_value);
  float float_value;
  ASSERT_TRUE(parser.TryGetValue<float>("c", float_value) && float_value == 0.5f);
  ASSERT_TRUE(parser.TryGetValue<float>("d", float_value) && float_value == 1e-5f);

  ASSERT_FALSE(parser.TryGetValue<bool>("a", bool_value));
  ASSERT_FALSE(parser.TryGetValue<float>("b", float_value));
  ASSERT_FALSE(parser.TryGetValue<int>("c", int_value));
  std::vector<int> int_list;
  ASSERT_FALSE(parser.TryGetArrayValue<int>("a", int_list));
}

TEST(AttributesJsonParserTest, ArrayValue) {
  std::string json = "{\"a\":[-1,0,1],\"b\":[false,true],\"c\":[0.5],\"d\":[]}";
  AttributesJsonParser parser(json);
  std::vector<int> int_list;
  ASSERT_TRUE(parser.TryGetArrayValue<int>("a", int_list) && (int_list == std::vector<int>{-1, 0, 1}));
  std::vector<bool> bool_list;
  ASSERT_TRUE(parser.TryGetArrayValue<bool>("b", bool_list) && (bool_list == std::vector<bool>{false, true}));
  std::vector<float> float_list;
  ASSERT_TRUE(parser.TryGetArrayValue<float>("c", float_list) && (float_list == std::vector<float>{0.5}));
  float_list.clear();
  ASSERT_TRUE(parser.TryGetArrayValue<float>("d", float_list) && (float_list == std::vector<float>{}));

  bool_list.clear();
  ASSERT_FALSE(parser.TryGetArrayValue<bool>("a", bool_list));
  float_list.clear();
  ASSERT_FALSE(parser.TryGetArrayValue<float>("b", float_list));
  int_list.clear();
  ASSERT_FALSE(parser.TryGetArrayValue<int>("c", int_list));
  int int_value;
  ASSERT_FALSE(parser.TryGetValue<int>("a", int_value));
}

}  // namespace test
}  // namespace onnxruntime
