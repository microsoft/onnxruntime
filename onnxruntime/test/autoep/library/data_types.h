// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "example_plugin_ep_utils.h"

class MLDataTypes {
 public:
  static MLDataTypes& GetInstance();
  static OrtStatus* AllFixedSizeTensorTypesIRv9(/*out*/ std::vector<const OrtMLDataType*>& result);

 private:
  MLDataTypes();

  std::vector<const OrtMLDataType*> fixed_tensor_v9_;
};
