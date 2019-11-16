//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <gtest/gtest.h>

class APITest : public ::testing::Test
{
protected:
    void LoadModel(const std::wstring& modelPath)
    {
        std::wstring fullPath = FileHelpers::GetModulePath() + modelPath;
        m_model = winrt::Windows::AI::MachineLearning::LearningModel::LoadFromFilePath(fullPath);
    }

    winrt::Windows::AI::MachineLearning::LearningModel m_model = nullptr;
    winrt::Windows::AI::MachineLearning::LearningModelDevice m_device = nullptr;
    winrt::Windows::AI::MachineLearning::LearningModelSession m_session = nullptr;

    uint64_t GetAdapterIdQuadPart()
    {
        LARGE_INTEGER id;
        id.LowPart = m_device.AdapterId().LowPart;
        id.HighPart = m_device.AdapterId().HighPart;
        return id.QuadPart;
    };

    _LUID GetAdapterIdAsLUID()
    {
        _LUID id;
        id.LowPart = m_device.AdapterId().LowPart;
        id.HighPart = m_device.AdapterId().HighPart;
        return id;
    }

    bool m_runGPUTests = true;
};
