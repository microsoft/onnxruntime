#pragma once
#include "testPch.h"
#include "onnxruntime_cxx_api.h"

namespace CompareFeatureValuesHelper {
bool CompareSequenceOfMapsStringToFloat(
  winrt::Windows::Foundation::Collections::IVectorView<
    winrt::Windows::Foundation::Collections::IMap<winrt::hstring, float>> featureValue,
  const Ort::Value& val,
  double per_sample_tolerance,
  double relative_per_sample_tolerance
);
}
