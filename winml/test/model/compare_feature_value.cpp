#include "testPch.h"
#include "compare_feature_value.h"
#include "StringHelpers.h"
#include <core/framework/ort_value.h>
#include "ort_value_helper.h"

using namespace winrt::Windows::Foundation::Collections;
using namespace winml;

namespace CompareFeatureValuesHelper {

template <typename T>
bool IsResultCloselyMatch(const T& outvalue, const T& expected_value, const double diff, const double tol) {
  if (diff > tol)
    return false;
  if (std::isnan(diff) && !(std::isnan(outvalue) && std::isnan(expected_value)) && !(std::isinf(outvalue) && std::isinf(expected_value)))
    return false;
  return true;
}

bool CompareSequenceOfMapsStringToFloat(
  IVectorView<IMap<winrt::hstring, float>> featureValue,
  const Ort::Value& val,
  double perSampleTolerance,
  double relativePerSampleTolerance
) {
  if (val.GetCount() != featureValue.Size()) {
    printf(
      "Map lengths are not the same! Got %d, expected %d\n",
      static_cast<int>(featureValue.Size()),
      static_cast<int>(val.GetCount())
    );
  }

  int expectedValSequenceIndex = 0;
  Ort::AllocatorWithDefaultOptions allocator;
  for (IMap<winrt::hstring, float> mapVal : featureValue) {
    std::map<winrt::hstring, float> expectedKvp;
    std::vector<std::pair<winrt::hstring, float>> actualKvp;

    Ort::Value mapExpectedOutput(nullptr);
    Ort::Value mapExpectedOutputKeys(nullptr);
    Ort::Value mapExpectedOutputValues(nullptr);
    WINML_EXPECT_NO_THROW(mapExpectedOutput = val.GetValue(expectedValSequenceIndex, allocator));
    WINML_EXPECT_NO_THROW(mapExpectedOutputKeys = mapExpectedOutput.GetValue(0, allocator));
    WINML_EXPECT_NO_THROW(mapExpectedOutputValues = mapExpectedOutput.GetValue(1, allocator));

    auto expectedOutputKeys =
      OrtValueHelpers::LoadTensorFromOrtValue(mapExpectedOutputKeys).as<TensorString>().GetAsVectorView();
    auto expectedOutputValues =
      OrtValueHelpers::LoadTensorFromOrtValue(mapExpectedOutputValues).as<TensorFloat>().GetAsVectorView();
    for (uint32_t i = 0; i < expectedOutputKeys.Size(); i++) {
      expectedKvp[expectedOutputKeys.GetAt(i)] = expectedOutputValues.GetAt(i);
    }
    for (auto kvp : mapVal) {
      winrt::hstring actualKey = kvp.Key();
      float actualValue = kvp.Value();
      if (expectedKvp.find(actualKey) == expectedKvp.end()) {
        printf("Unexpected key in actual output: %ws", actualKey.c_str());
        return false;
      } else {
        // verify that the value is within tolerable ranges
        const double diff = std::fabs(expectedKvp[actualKey] - actualValue);
        const double tol = perSampleTolerance + relativePerSampleTolerance * std::fabs(expectedKvp[actualKey]);
        if (!IsResultCloselyMatch<double>(actualValue, expectedKvp[actualKey], diff, tol)) {
          printf("expected (%f), actual (%f), diff: %f, tol= %f .\n", expectedKvp[actualKey], actualValue, diff, tol);
          return false;
        }
      }
    }
    expectedValSequenceIndex++;
  }
  // If errors or discrepancies are not found, then return true
  return true;
}

}// namespace CompareFeatureValuesHelper
