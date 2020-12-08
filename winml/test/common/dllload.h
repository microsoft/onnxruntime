#pragma once
#include "Std.h"

int32_t __stdcall WINRT_RoGetActivationFactory(void* classId, winrt::guid const& iid, void** factory) noexcept;