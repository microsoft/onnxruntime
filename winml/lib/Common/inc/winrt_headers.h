// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "common.h"

#include "winrt/windows.media.h"
#include "winrt/windows.graphics.imaging.h"
#include "winrt/windows.foundation.h"
#include "winrt/windows.foundation.collections.h"
#include "winrt/windows.storage.streams.h"

#define STRINGIFY(x) #x
#define XSTRINGIFY(x) STRINGIFY(x)
#define CPPWINRT_HEADER(root_ns) comp_generated/winrt/##root_ns##.AI.MachineLearning.h
#define NATIVE_HEADER(root_ns) root_ns##.AI.MachineLearning.native.h
#define NATIVE_INTERNAL_HEADER(root_ns) root_ns##.AI.MachineLearning.native.internal.h
#define CREATE_CPPWINRT_COMPONENT_HEADER() XSTRINGIFY(CPPWINRT_HEADER(WINML_ROOT_NS))
#define CREATE_NATIVE_HEADER() XSTRINGIFY(NATIVE_HEADER(WINML_ROOT_NS))
#define CREATE_NATIVE_INTERNAL_HEADER() XSTRINGIFY(NATIVE_INTERNAL_HEADER(WINML_ROOT_NS))

#include CREATE_CPPWINRT_COMPONENT_HEADER()

// WinML Native Headers
#include CREATE_NATIVE_HEADER()
#include CREATE_NATIVE_INTERNAL_HEADER()

#include "Errors.h"
