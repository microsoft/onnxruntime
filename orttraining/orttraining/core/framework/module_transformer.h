// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {
namespace training {

class ModuleTransformer {
 public:
  std::string Transform(std::istream& model_istream,
                        const std::unordered_set<std::string>& weights_to_train,
                        const std::unordered_set<std::string>& output_names);
};

}  // namespace training
}  // namespace onnxruntime
