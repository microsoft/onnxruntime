// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>

#include <stddef.h>

// TODO: Move to api.h

extern "C" {
const void* EMSCRIPTEN_KEEPALIVE JsepOutput(void* context, int index, void* data);
};
