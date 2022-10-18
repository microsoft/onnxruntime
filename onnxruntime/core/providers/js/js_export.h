// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>

#include <stddef.h>

extern "C" {
void * EMSCRIPTEN_KEEPALIVE JSEP_Output(void * context, int index, void * data);
};
