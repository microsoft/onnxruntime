// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "std.h"

enum OutputBindingStrategy { Bound, Unbound, Empty };

namespace WinML::Engine::Test::ModelValidator
{
    void FnsCandy16(
        std::string instance,
        winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind,
        OutputBindingStrategy outputBindingStrategy,
        bool bindInputsAsIInspectable,
        float dataTolerance = false);

    void SqueezeNet(
        std::string instance,
        winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind,
        float dataTolerance,
        bool bindAsImage = false,
        OutputBindingStrategy outputBindingStrategy = OutputBindingStrategy::Bound,
        bool bindInputsAsIInspectable = false
    );
}
