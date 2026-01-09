#include "profiling_utils.h"
#include <iostream>

MemoryUsage GetMemoryUsage()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    MemoryUsage usage = {0, 0};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
    {
        usage.peak_working_set_size = pmc.PeakWorkingSetSize;
        usage.peak_pagefile_usage = pmc.PeakPagefileUsage;
    }
    return usage;
}

CpuUsageMonitor::CpuUsageMonitor() {}

void CpuUsageMonitor::Start()
{
    SYSTEM_INFO system_info;
    FILETIME current_time, kernel_mode_time, user_mode_time;

    ZeroMemory(&system_info, sizeof(SYSTEM_INFO));
    ZeroMemory(&current_time, sizeof(FILETIME));
    ZeroMemory(&kernel_mode_time, sizeof(FILETIME));
    ZeroMemory(&user_mode_time, sizeof(FILETIME));

    GetSystemInfo(&system_info);
    m_num_processors = system_info.dwNumberOfProcessors;

    GetSystemTimeAsFileTime(&current_time);
    memcpy(&m_last_cpu, &current_time, sizeof(FILETIME));

    m_process_handle = GetCurrentProcess();
    if (!GetProcessTimes(m_process_handle, &current_time, &current_time, &kernel_mode_time, &user_mode_time))
    {
        DWORD error = GetLastError();
        std::cout << "GetProcessTimes failed with error: " << error << std::endl;
        return;
    }
    memcpy(&m_last_sys_cpu, &kernel_mode_time, sizeof(FILETIME));
    memcpy(&m_last_user_cpu, &user_mode_time, sizeof(FILETIME));

    m_started = true;
}

double CpuUsageMonitor::GetCurrentValue()
{
    if (!m_started)
    {
        std::cerr << "Error: CpuUsageMonitor::Start() must be called before CpuUsageMonitor::GetCurrentValue()"
                  << std::endl;
        return -1.0;
    }

    FILETIME current_time, kernel_mode_time, user_mode_time;
    ULARGE_INTEGER now, sys, user;
    double percent;

    ZeroMemory(&current_time, sizeof(FILETIME));
    ZeroMemory(&kernel_mode_time, sizeof(FILETIME));
    ZeroMemory(&user_mode_time, sizeof(FILETIME));
    ZeroMemory(&now, sizeof(ULARGE_INTEGER));
    ZeroMemory(&sys, sizeof(ULARGE_INTEGER));
    ZeroMemory(&user, sizeof(ULARGE_INTEGER));

    GetSystemTimeAsFileTime(&current_time);
    memcpy(&now, &current_time, sizeof(FILETIME));

    GetProcessTimes(m_process_handle, &current_time, &current_time, &kernel_mode_time, &user_mode_time);
    memcpy(&sys, &kernel_mode_time, sizeof(FILETIME));
    memcpy(&user, &user_mode_time, sizeof(FILETIME));
    percent = (sys.QuadPart - m_last_sys_cpu.QuadPart) + (user.QuadPart - m_last_user_cpu.QuadPart);
    percent /= (now.QuadPart - m_last_cpu.QuadPart);
    percent /= m_num_processors;
    m_last_cpu = now;
    m_last_user_cpu = user;
    m_last_sys_cpu = sys;

    return percent * 100;
}
