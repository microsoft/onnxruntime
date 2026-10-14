// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/pattern_rewriter.h"

#include <cmath>
#include <optional>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace rewrite {
namespace {
ONNX_NAMESPACE::AttributeProto FloatAttribute(const char* name, float value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr.set_f(value);
  return attr;
}

bool Float32(const Match& match) {
  for (const NodeArg* value : match.values) {
    const auto* type = value->TypeAsProto();
    if (!type || !type->has_tensor_type() ||
        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return false;
  }
  return true;
}

std::optional<float> Scalar(const Graph& graph, const NodeArg& value) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL || tensor->dims_size() != 0)
    return std::nullopt;
  Initializer initializer(graph, *tensor, graph.ModelPath());
  if (initializer.size() != 1) return std::nullopt;
  float scalar = *initializer.data<float>();
  if (!std::isfinite(scalar)) return std::nullopt;
  return scalar;
}

bool Integer(const Node& node, const char* name, int64_t value) {
  auto it = node.GetAttributes().find(name);
  return it != node.GetAttributes().end() && it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT &&
         it->second.i() == value;
}
}  // namespace

Rules MakeExampleRules() {
  Rules rules;
  {
    Pattern p(R"(
      (x) =>
        add = Add(x, 3.0)
        clip = Clip(add, 0.0, 6.0)
        mul = Mul(clip, x)
        return Div(mul, 6.0)
    )");
    auto guard = [](const Graph&, const Match& m) { return Float32(m); };
    Replacement replacement("(x) => HardSwish(x)");
    Rule rule{"HardSwish", p, {{kOnnxDomain, 14, 21}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
  {
    Pattern p(R"(
      (x) =>
        sigmoid = HardSigmoid(x, alpha=0.16666667, beta=0.5)
        return Mul(sigmoid, x)
    )");
    auto guard = [](const Graph&, const Match& m) { return Float32(m); };
    Replacement replacement("(x) => HardSwish(x)");
    Rule rule{"HardSigmoidMul", p, {{kOnnxDomain, 14, 21}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
#ifndef DISABLE_CONTRIB_OPS
  {
    Pattern p("(x, y, divisor) => mm = MatMul(x, y); return Div(mm, divisor)");
    auto guard = [](const Graph& graph, const Match& m) {
      auto scalar = Scalar(graph, *m.valueOf("divisor"));
      return Float32(m) && scalar && *scalar != 0.0f && std::isfinite(1.0f / *scalar);
    };
    Replacement plan("(x, y, alpha) => com.microsoft.FusedMatMul(x, y, alpha=alpha)");
    auto replacement = [plan](const Graph& graph, const Match& m) {
      m.update("alpha", FloatAttribute("alpha", 1.0f / *Scalar(graph, *m.valueOf("divisor"))));
      return plan;
    };
    Rule rule{"MatMulDiv", p, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
  {
    Pattern p("(x, y) => transpose = Transpose(y); return MatMul(x, transpose)");
    auto guard = [](const Graph&, const Match& m) {
      if (!Float32(m)) return false;
      const auto* shape = m.valueOf("y")->Shape();
      if (!shape || shape->dim_size() < 2) return false;
      auto it = m.nodes[0]->GetAttributes().find("perm");
      if (it == m.nodes[0]->GetAttributes().end()) return shape->dim_size() == 2;
      const auto& perm = it->second;
      if (perm.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
          perm.ints_size() != shape->dim_size())
        return false;
      int rank = shape->dim_size();
      for (int i = 0; i < rank; ++i) {
        int expected = i < rank - 2 ? i : (i == rank - 2 ? rank - 1 : rank - 2);
        if (perm.ints(i) != expected) return false;
      }
      return true;
    };
    Replacement replacement("(x,y) => com.microsoft.FusedMatMul(x,y,transB=1)");
    Rule rule{"TransposeMatMul", p, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
  {
    Pattern p("(x, y, divisor) => mm = com.microsoft::FusedMatMul(x, y); return Div(mm, divisor)");
    auto guard = [](const Graph& graph, const Match& m) {
      auto scalar = Scalar(graph, *m.valueOf("divisor"));
      auto it = m.nodes[0]->GetAttributes().find("alpha");
      float alpha = it == m.nodes[0]->GetAttributes().end() ? 1.0f : it->second.f();
      return Float32(m) && scalar && *scalar != 0.0f && std::isfinite(alpha / *scalar);
    };
    Replacement plan(R"(
      (x, y, alpha, transA, transB, transBatchA, transBatchB) =>
        com.microsoft.FusedMatMul(x, y, alpha=alpha, transA=transA, transB=transB,
                                 transBatchA=transBatchA, transBatchB=transBatchB)
    )");
    auto replacement = [plan](const Graph& graph, const Match& m) {
      const auto& node = *m.nodes[0];
      const auto& attributes = node.GetAttributes();
      auto it = attributes.find("alpha");
      float alpha = it == attributes.end() ? 1.0f : it->second.f();
      m.update("alpha", FloatAttribute("alpha", alpha / *Scalar(graph, *m.valueOf("divisor"))));
      for (const char* name : {"transA", "transB", "transBatchA", "transBatchB"}) {
        auto attribute = attributes.find(name);
        m.update(name, attribute == attributes.end() ? node.Op()->attributes().at(name).default_value
                                                     : attribute->second);
      }
      return plan;
    };
    Rule rule{"FusedMatMulDiv", p, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
  {
    Pattern p(R"(
      (x, scale, epsilon) =>
        square = Pow(x, 2.0)
        mean = ReduceMean(square)
        add = Add(mean, epsilon)
        sqrt = Sqrt(add)
        reciprocal = Reciprocal(sqrt)
        normalize = Mul(x, reciprocal)
        return Mul(normalize, scale)
    )");
    auto guard = [](const Graph& graph, const Match& m) {
      if (!Float32(m)) return false;
      auto eps = Scalar(graph, *m.valueOf("epsilon"));
      const auto* shape = m.valueOf("x")->Shape();
      const auto* scale_shape = m.valueOf("scale")->Shape();
      if (!eps || *eps <= 0 || !shape || shape->dim_size() < 1 || !scale_shape || scale_shape->dim_size() != 1)
        return false;
      const auto& dim = shape->dim(shape->dim_size() - 1);
      const auto& scale_dim = scale_shape->dim(0);
      if (!dim.has_dim_value() || !scale_dim.has_dim_value() || dim.dim_value() <= 0 ||
          dim.dim_value() != scale_dim.dim_value())
        return false;
      const auto& mean_node = *m.nodes[1];
      auto axes = mean_node.GetAttributes().find("axes");
      return Integer(mean_node, "keepdims", 1) && axes != mean_node.GetAttributes().end() &&
             axes->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS &&
             axes->second.ints_size() == 1 && axes->second.ints(0) == -1;
    };
    Replacement plan(R"(
      (x, scale, fused_epsilon) =>
        SimplifiedLayerNormalization(x, scale, axis=-1, epsilon=fused_epsilon, stash_type=1)
    )");
    auto replacement = [plan](const Graph& graph, const Match& m) {
      m.update("fused_epsilon", FloatAttribute("epsilon", *Scalar(graph, *m.valueOf("epsilon"))));
      return plan;
    };
    Rule rule{"RmsNormalization", p, {{kOnnxDomain, 7, 17}}, guard, replacement};
    rules.push_back(std::move(rule));
  }
#endif
  return rules;
}

}  // namespace rewrite
}  // namespace onnxruntime
