// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class WhereDummyDq

    Graph transformer that inserts a dummy DQ on Where node's initializer input
    to form Node Unit when Where node has one DQ and one scalar initializer input.

    If `Where` gets a float scalar `xf` and a `DequantizeLinear` as its two data inputs,
    `WhereDummyDq` inserts a dummy DQ so that `xf ≈ DQ(xq, scale, zp)`.

    The `xq`, `zp` are chosen per the table below (by the dtype of the existing DQ's zero-point),
    and `scale` is computed from them.

    We select these values in order to keep the `scale` non-negative:

      |                 |  uint8 | uint16 |  int8 |  int16 |
      |-----------------|--------|--------|-------|--------|
      | xf > 0          |        |        |       |        |
      |   xq            |   255  |  65535 |  127  |  32767 |
      |   zp            |   127  |  32767 |   0   |    0   |
      | xf < 0          |        |        |       |        |
      |   xq            |    0   |    0   | -128  | -32768 |
      |   zp            |   127  |  32767 |   0   |    0   |
      | xf = 0          |        |        |       |        |
      |   xq            |   127  |  32767 |   0   |    0   |
      |   zp            |   127  |  32767 |   0   |    0   |

    scale = xf / (xq - zp) if (xq != zp) else 1
*/
class WhereDummyDq : public GraphTransformer {
 public:
  WhereDummyDq() noexcept : GraphTransformer("WhereDummyDq") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool SatisfyCondition(const Graph& graph, const Node& node) const;
  Status InsertDummyDQ(Node& node, Graph& graph, bool& modified, const logging::Logger& logger) const;
};
}  // namespace onnxruntime
