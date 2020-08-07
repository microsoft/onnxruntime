#pragma once

#include "Dummy.g.h"

namespace winrt::Microsoft::AI::MachineLearning::Experimental::implementation
{
    struct Dummy : DummyT<Dummy>
    {
        Dummy() = default;

        void Test();
    };
}

namespace winrt::Microsoft::AI::MachineLearning::Experimental::factory_implementation
{
    struct Dummy : DummyT<Dummy, implementation::Dummy>
    {
    };
}
