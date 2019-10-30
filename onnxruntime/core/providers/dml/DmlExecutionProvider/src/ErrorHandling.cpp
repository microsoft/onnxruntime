// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

HRESULT MapLotusErrorToHRESULT(onnxruntime::common::Status status)
{
    switch (status.Code())
    {
    case onnxruntime::common::StatusCode::OK:
        return S_OK;
    case onnxruntime::common::StatusCode::FAIL:
        return E_FAIL;
    case onnxruntime::common::StatusCode::INVALID_ARGUMENT:
        return E_INVALIDARG;
    case onnxruntime::common::StatusCode::NO_SUCHFILE:
        return __HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case onnxruntime::common::StatusCode::NO_MODEL:
        return __HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case onnxruntime::common::StatusCode::ENGINE_ERROR:
        return E_FAIL;
    case onnxruntime::common::StatusCode::RUNTIME_EXCEPTION:
        return E_FAIL;
    case onnxruntime::common::StatusCode::INVALID_PROTOBUF:
        return __HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case onnxruntime::common::StatusCode::MODEL_LOADED:
        return __HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    case onnxruntime::common::StatusCode::NOT_IMPLEMENTED:
        return E_NOTIMPL;
    case onnxruntime::common::StatusCode::INVALID_GRAPH:
        return __HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case onnxruntime::common::StatusCode::EP_FAIL:
        return __HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    default:
        return E_FAIL;
    }
}

}