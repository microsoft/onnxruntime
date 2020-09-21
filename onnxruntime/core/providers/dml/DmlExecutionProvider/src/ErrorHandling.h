// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define THROW_IF_NOT_OK(status)                                                                                 \
    do {                                                                                                        \
        auto _status = status;                                                                                  \
        if (!_status.IsOK())                                                                                    \
        {                                                                                                       \
            THROW_HR(StatusCodeToHRESULT(static_cast<onnxruntime::common::StatusCode>(_status.Code())));                             \
        }                                                                                                       \
    } while (0)
