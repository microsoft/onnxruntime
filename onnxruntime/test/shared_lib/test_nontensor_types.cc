// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/ml_value.h"
#include "core/framework/data_types.h"
#include "test_fixture.h"

using namespace onnxruntime;

TEST_F(CApiTest, GetNumValuesMap) {
  using T = std::map<int64_t, float>;
  auto m = std::make_unique<T>();
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(m.release(),
              DataTypeImpl::GetType<T>(),
              DataTypeImpl::GetType<T>()->GetDeleteFunc());
  auto ort_value = reinterpret_cast<OrtValue*>(value.release());
  int num_values;
  OrtStatus* st = OrtGetNumValues(ort_value, &num_values);
  int ec = OrtGetErrorCode(st);
  ASSERT_EQ(ec, ORT_OK);
  ASSERT_EQ(num_values, 2);
}

TEST_F(CApiTest, GetNumValuesVectorOfMaps) {
  using MapType = std::map<std::string, float>;
  using T = std::vector<MapType>;
  auto m = std::make_unique<T>();
  (*m).push_back(MapType());
  (*m).push_back(MapType());
  (*m).push_back(MapType());
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(m.release(),
              DataTypeImpl::GetType<T>(),
              DataTypeImpl::GetType<T>()->GetDeleteFunc());
  auto ort_value = reinterpret_cast<OrtValue*>(value.release());
  int num_values;
  OrtStatus* st = OrtGetNumValues(ort_value, &num_values);
  int ec = OrtGetErrorCode(st);
  ASSERT_EQ(ec, ORT_OK);
  ASSERT_EQ(num_values, 3);
}

// TEST_F(CApiTest, GetVectorOfMaps) {
// }
