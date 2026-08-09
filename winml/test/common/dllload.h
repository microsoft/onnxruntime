#pragma once
#include "Std.h"
#include "winrt/base.h"

int32_t __stdcall WINRT_RoGetActivationFactory(void* classId, winrt::guid const& iid, void** factory) noexcept;
