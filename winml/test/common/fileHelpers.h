// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "winrt/Windows.Graphics.Imaging.h"
#include "winrt/Windows.AI.MachineLearning.h"

namespace FileHelpers
{
    std::wstring GetModulePath();
    std::wstring GetWinMLPath();

    winrt::Windows::Graphics::Imaging::SoftwareBitmap GetSoftwareBitmapFromFile(const std::wstring& filePath);
    winrt::Windows::AI::MachineLearning::ImageFeatureValue LoadImageFeatureValue(const std::wstring& imagePath);
}
