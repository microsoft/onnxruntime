// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// IUnknown must be declared before winrt/base.h is included to light up support for native COM
// interfaces with C++/WinRT types (e.g. winrt::com_ptr<ITensorNative>).
#include <Unknwn.h>
#include <wil/cppwinrt.h>

#include <winrt/windows.media.h>
#include <winrt/windows.graphics.imaging.h>
#include <winrt/windows.foundation.h>
#include <winrt/windows.foundation.collections.h>
#include <winrt/windows.devices.enumeration.pnp.h>
#include <winrt/windows.graphics.directx.direct3d11.h>
#include <winrt/windows.media.capture.h>
#include <winrt/windows.media.h>
#include <winrt/windows.security.cryptography.core.h>
#include <winrt/windows.security.cryptography.h>
#include <winrt/windows.storage.h>
#include <winrt/windows.storage.streams.h>

// clang-format off
#define STRINGIFY(x) #x
#define XSTRINGIFY(x) STRINGIFY(x)
#define CPPWINRT_HEADER(root_ns) comp_generated/winrt/##root_ns##.AI.MachineLearning.h
#define CPPWINRT_EXPERIMENTAL_HEADER(root_ns) comp_generated/winrt/##root_ns##.AI.MachineLearning.Experimental.h
#define NATIVE_HEADER(root_ns) root_ns##.AI.MachineLearning.native.h
#define NATIVE_INTERNAL_HEADER(root_ns) root_ns##.AI.MachineLearning.native.internal.h
#define CREATE_CPPWINRT_COMPONENT_HEADER() XSTRINGIFY(CPPWINRT_HEADER(WINML_ROOT_NS))
#define CREATE_CPPWINRT_EXPERIMENTAL_COMPONENT_HEADER() XSTRINGIFY(CPPWINRT_EXPERIMENTAL_HEADER(WINML_ROOT_NS))
#define CREATE_NATIVE_HEADER() XSTRINGIFY(NATIVE_HEADER(WINML_ROOT_NS))
#define CREATE_NATIVE_INTERNAL_HEADER() XSTRINGIFY(NATIVE_INTERNAL_HEADER(WINML_ROOT_NS))
// clang-format on

#include CREATE_CPPWINRT_COMPONENT_HEADER()

#ifndef BUILD_INBOX
#include CREATE_CPPWINRT_EXPERIMENTAL_COMPONENT_HEADER()
#endif

// WinML Native Headers
#include CREATE_NATIVE_HEADER()
#include CREATE_NATIVE_INTERNAL_HEADER()

namespace winml = winrt::WINML_ROOT_NS::AI::MachineLearning;
namespace wf = winrt::Windows::Foundation;
namespace wfc = winrt::Windows::Foundation::Collections;
namespace wm = winrt::Windows::Media;
namespace wgi = winrt::Windows::Graphics::Imaging;
namespace wgdx = winrt::Windows::Graphics::DirectX;
namespace ws = winrt::Windows::Storage;
namespace wss = winrt::Windows::Storage::Streams;
