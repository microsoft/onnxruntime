// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "std.h"
#include "winrt_headers.h"

#include "winrt/Windows.Graphics.Imaging.h"

namespace FileHelpers {
std::wstring GetModulePath();
std::wstring GetWinMLPath();

wgi::SoftwareBitmap GetSoftwareBitmapFromFile(const std::wstring& filePath);
winml::ImageFeatureValue LoadImageFeatureValue(const std::wstring& imagePath);
}// namespace FileHelpers
