#pragma once
#include "TimerHelper.h"


enum WINML_RUNTIME_PERF
{
    LOAD_MODEL = 0,
    EVAL_MODEL,
    COUNT
};


extern Profiler<WINML_RUNTIME_PERF> g_Profiler;