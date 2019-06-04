// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace nuphar {

struct FuseCutConfig {
  const static int node_uses_valid_for_cut;
  const static int node_uses_cut_threshold;
};

}  // namespace nuphar
}  // namespace onnxruntime
