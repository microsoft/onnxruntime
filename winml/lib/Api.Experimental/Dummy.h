#pragma once

#include "Dummy.g.h"

namespace WINML_EXPERIMENTALP {

struct Dummy : DummyT<Dummy>
{
    Dummy() = default;

    void Test();
};

}

namespace WINML_EXPERIMENTAL::factory_implementation {

struct Dummy : DummyT<Dummy, implementation::Dummy>
{
};

}
