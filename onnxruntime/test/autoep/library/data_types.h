// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "example_plugin_ep_utils.h"

class MLDataTypes {
 public:
  static MLDataTypes& GetInstance();

  static OrtStatus* GetTensorType(ONNXTensorElementDataType elem_type, /*out*/ const OrtMLDataType*& tensor_type);
  static const OrtMLDataType* GetTensorType(ONNXTensorElementDataType elem_type);

  static OrtStatus* GetAllFixedSizeTensorTypesIRv9(/*out*/ std::vector<const OrtMLDataType*>& tensor_types);
  static std::vector<const OrtMLDataType*> GetAllFixedSizeTensorTypesIRv9();

 private:
  MLDataTypes();

  std::unordered_map<ONNXTensorElementDataType, const OrtMLDataType*> tensor_types_map_;
  std::vector<const OrtMLDataType*> fixed_tensor_v9_;
};
