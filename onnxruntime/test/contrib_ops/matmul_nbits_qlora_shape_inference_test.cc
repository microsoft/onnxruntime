// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <tuple>

#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime::test {
namespace {

struct QLoRAShapeConfig {
  int64_t k = 64;
  int64_t n = 128;
  int64_t rank = 32;
  int64_t bits = 4;
  int64_t block_size = 32;
  int64_t lora_bits = 8;
  int64_t lora_block_size = 32;
};

template <typename T = float>
Status ResolveQLoRAShape(const QLoRAShapeConfig& c, const std::function<void(Node&)>& edit = nullptr) {
  Model model("qlora_shape", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  const auto k_blocks = (c.k + c.block_size - 1) / c.block_size;
  const auto lora_k_blocks = (c.k + c.lora_block_size - 1) / c.lora_block_size;
  const auto r_blocks = (c.rank + c.lora_block_size - 1) / c.lora_block_size;
  const auto lora_blob = c.lora_block_size * c.lora_bits / 8;
  auto* x = builder.MakeInput<T>(std::vector<int64_t>{2, 3, c.k});
  auto* w = builder.MakeInput<uint8_t>(std::vector<int64_t>{c.n, k_blocks, c.block_size * c.bits / 8});
  auto* s = builder.MakeInput<T>(std::vector<int64_t>{c.n, k_blocks});
  auto* a = builder.MakeInput<uint8_t>(std::vector<int64_t>{c.rank, lora_k_blocks, lora_blob});
  auto* as = builder.MakeInput<T>(std::vector<int64_t>{c.rank, lora_k_blocks});
  auto* b = builder.MakeInput<uint8_t>(std::vector<int64_t>{c.n, r_blocks, lora_blob});
  auto* bs = builder.MakeInput<T>(std::vector<int64_t>{c.n, r_blocks});
  auto* y = builder.MakeOutput();
  auto& node = builder.AddNode("MatMulNBitsQLoRA", {x, w, s, a, as, b, bs}, {y}, kMSDomain);
  node.AddAttribute("K", c.k);
  node.AddAttribute("N", c.n);
  node.AddAttribute("bits", c.bits);
  node.AddAttribute("block_size", c.block_size);
  node.AddAttribute("lora_bits", c.lora_bits);
  node.AddAttribute("lora_block_size", c.lora_block_size);
  if (edit) edit(node);
  builder.SetGraphOutputs();
  auto status = model.MainGraph().Resolve();
  if (status.IsOK()) {
    EXPECT_EQ(y->TypeAsProto()->tensor_type().elem_type(), utils::ToTensorProtoElementType<T>());
    auto expected = *x->Shape();
    expected.mutable_dim(expected.dim_size() - 1)->set_dim_value(c.n);
    CheckShapeEquality(y->Shape(), &expected);
  }
  return status;
}

void SetInputDimension(Node& node, size_t input, int axis, int64_t value) {
  auto* arg = node.MutableInputDefs()[input];
  auto shape = *arg->Shape();
  auto* dim = shape.mutable_dim(axis);
  dim->Clear();
  if (value >= 0) dim->set_dim_value(value);
  arg->SetShape(shape);
}

}  // namespace

TEST(MatMulNBitsQLoRASchema, GeneralQuantizationAndArbitraryRank) {
  for (const QLoRAShapeConfig& c : {
           QLoRAShapeConfig{63, 65, 7, 8, 64, 4, 16},
           {65, 33, 64, 2, 16, 8, 128},
           {256, 128, 129, 4, 128, 2, 64},
           {64, 128, 1},
           {64, 128, 16},
           {64, 128, 32}}) {
    SCOPED_TRACE(testing::Message() << "K=" << c.k << " N=" << c.n << " R=" << c.rank);
    ASSERT_STATUS_OK(ResolveQLoRAShape(c));
    ASSERT_STATUS_OK(ResolveQLoRAShape<MLFloat16>(c));
  }
}

TEST(MatMulNBitsQLoRASchema, EmptyAdaptersRetainScaleDefaults) {
  for (int mask : {1, 2, 3}) {
    ASSERT_STATUS_OK(ResolveQLoRAShape({63, 65, 7, 8, 64, 4, 16}, [mask](Node& node) {
      if (mask & 1) SetInputDimension(node, 3, 0, 0);
      if (mask & 2) SetInputDimension(node, 5, 1, 0);
    }));
  }
}

TEST(MatMulNBitsQLoRASchema, DynamicAdapterRankAndVectorInput) {
  ASSERT_STATUS_OK(ResolveQLoRAShape({}, [](Node& node) {
    SetInputDimension(node, 3, 0, -1);
    SetInputDimension(node, 4, 0, -1);
    SetInputDimension(node, 5, 1, -1);
    SetInputDimension(node, 6, 1, -1);
    ONNX_NAMESPACE::TensorShapeProto x_shape;
    x_shape.add_dim()->set_dim_value(64);
    node.MutableInputDefs()[0]->SetShape(x_shape);
  }));
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MatMulNBitsQLoRASchema, RejectsInconsistentAdapterShapes) {
  for (const auto& [input, axis, value] : {std::tuple<size_t, int, int64_t>{4, 0, 16},
                                           {3, 1, 3},
                                           {3, 2, 16},
                                           {5, 1, 2},
                                           {6, 1, 2}}) {
    const auto status = ResolveQLoRAShape({}, [=](Node& node) { SetInputDimension(node, input, axis, value); });
    EXPECT_FALSE(status.IsOK());
  }
}

TEST(MatMulNBitsQLoRASchema, RejectsInvalidQuantizationAttributes) {
  const std::pair<const char*, int64_t> invalid_attributes[] = {
      {"bits", 3},
      {"lora_bits", 16},
      {"block_size", 8},
      {"block_size", 48},
      {"lora_block_size", 0},
      {"lora_block_size", 24},
      {"K", 0},
      {"N", -1},
  };
  for (const auto& [name, value] : invalid_attributes) {
    SCOPED_TRACE(name);
    const auto status = ResolveQLoRAShape({}, [=](Node& node) { node.AddAttribute(name, value); });
    EXPECT_FALSE(status.IsOK());
  }
}
#endif  // ORT_NO_EXCEPTIONS

}  // namespace onnxruntime::test
