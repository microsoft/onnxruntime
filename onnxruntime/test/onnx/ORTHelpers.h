#pragma once

#include <algorithm>
#include <numeric>
#include <functional>
#include <span>
#include <wrl/client.h>
#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"

using namespace Microsoft::WRL;

Ort::Session CreateSession(const wchar_t* model_file_path);

Ort::Value Preprocess(Ort::Session& session,
    ComPtr<ID3D12Resource> currentBuffer);

winrt::com_array<float> Eval(Ort::Session& session, const Ort::Value& prev_input);

Ort::Value CreateTensorValueFromD3DResource(
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
);