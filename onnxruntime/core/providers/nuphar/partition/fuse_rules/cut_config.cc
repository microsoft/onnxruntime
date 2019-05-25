// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cut_config.h"

namespace onnxruntime {
namespace nuphar {

const int FuseCutConfig::node_uses_valid_for_cut = 2;
const int FuseCutConfig::node_uses_cut_threshold = 180;

}  // namespace nuphar
}  // namespace onnxruntime
