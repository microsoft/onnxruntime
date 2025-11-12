#pragma once

// This file is inspired by a Stack Overflow thread.
// Reference:
// https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process/64166#64166

#include <windows.h>
#include <psapi.h>

struct MemoryUsage
{
    size_t peak_working_set_size;
    size_t peak_pagefile_usage;
};

struct CpuUsage
{
    double user_time;
    double kernel_time;

    double TotalTime() const
    {
        return user_time + kernel_time;
    }
};

MemoryUsage GetMemoryUsage();

class CpuUsageMonitor
{
public:
    CpuUsageMonitor();
    void Start();
    double GetCurrentValue();

private:
    bool m_started = false;
    ULARGE_INTEGER m_last_cpu = {};
    ULARGE_INTEGER m_last_sys_cpu = {};
    ULARGE_INTEGER m_last_user_cpu = {};
    int m_num_processors = 0;
    HANDLE m_process_handle = nullptr;
};
