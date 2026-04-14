// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/*
Fuse continuous Add without broadcasting into ScaledSum.

Here is the pattern to find and fuse:

 input_0  scale_0  input_1  scale_1
      \     /          \   /
        Div             Div
          \            /
             \      /
    input_2    Add
         \     /
           Add
            |

scale_0 and scale_1
> 1). MUST be scalar or single element 1D tensors,
> 2). and MUST be constant initializers.

==>

            input_0  input_1  input_2
                    \    |     /
                    ScaledSum
(attribute: scale_0=1/scale_0, scale_1=1/scale_1, scale_2=1)
                        |

**/
class ScaledSumFusion : public GraphTransformer {
 public:
  explicit ScaledSumFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ScaledSumFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
