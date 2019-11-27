// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <windows.h>
#include <cmath>
#ifndef DISABLE_GPU_COUNTERS
#include <Pdh.h>
#include <PdhMsg.h>
#endif
#include <psapi.h>
#include <string>
#include <array>
#include "core/common/common.h"

#define TIMER_SLOT_SIZE (128)
#define CONVERT_100NS_TO_SECOND(x) ((x)*0.0000001)
#define BYTE_TO_MB(x) ((x) / (1024.0 * 1024.0))

// A stopwatch to measure the time passed (in seconds) between current Stop call and the closest Start call that has been called before.
class Timer {
 public:
  void Start() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    start_time_ = static_cast<double>(t.QuadPart);
  }

  double Stop() {
    LARGE_INTEGER stop_time;
    QueryPerformanceCounter(&stop_time);
    double t = static_cast<double>(stop_time.QuadPart) - start_time_;
    LARGE_INTEGER tps;
    QueryPerformanceFrequency(&tps);
    return t / static_cast<double>(tps.QuadPart);
  }

 private:
  double start_time_;
};

typedef enum CounterType {
  TIMER = 0,
  CPU_USAGE,
  PAGE_FAULT_COUNT,
  PAGE_FILE_USAGE,
  PEAK_PAGE_FILE_USAGE,
  WORKING_SET_USAGE,
  PEAK_WORKING_SET_USAGE,
  // GPU specific counter starts here
  GPU_USAGE,
  GPU_DEDICATED_MEM_USAGE,
  GPU_SHARED_MEM_USAGE,
  TYPE_COUNT
} CounterType;

typedef enum ProfilerType {
  CPU,
  GPU,
  PROFILER_TYPE_COUNT
} ProfilerType;

class IPerfCounter {
 public:
  virtual void Reset() = 0;
  virtual void Stop() = 0;
  virtual void Start() = 0;
  virtual void GetValues(double (&values)[CounterType::TYPE_COUNT], double time) = 0;
};

class CpuPerfCounter : public IPerfCounter {
 public:
  CpuPerfCounter() {}

  ~CpuPerfCounter() {}

  void Reset() override {
    SYSTEM_INFO sysInfo = {0};
    GetSystemInfo(&sysInfo);

    m_startKernelTime = {0};
    m_startUserTime = {0};
    m_numProcessors = sysInfo.dwNumberOfProcessors;
    m_procHandle = GetCurrentProcess();
    ;
    m_pid = GetCurrentProcessId();
    ;
    m_previousStartCallFailed = true;
    m_processTime = 0;
    m_startPageFaultCount = 0;
    m_startPagefileUsage = 0;
    m_startPeakPagefileUsage = 0;
    m_startWorkingSetSize = 0;
    m_startPeakWorkingSetSize = 0;
    m_deltaPageFaultCount = 0;
    m_deltaPagefileUsage = 0;
    m_deltaPeakPagefileUsage = 0;
    m_deltaWorkingSetSize = 0;
    m_deltaPeakWorkingSetSize = 0;
  }

  void Start() override {
    FILETIME ftIgnore, ftKernel, ftUser;

    if (!GetProcessTimes(m_procHandle, &ftIgnore, &ftIgnore, &ftKernel, &ftUser) ||
        !GetProcessMemoryCounters(m_pid, m_startPageFaultCount, m_startPagefileUsage, m_startPeakPagefileUsage, m_startWorkingSetSize, m_startPeakWorkingSetSize)) {
      m_previousStartCallFailed = true;
    } else {
      memcpy(&m_startKernelTime, &ftKernel, sizeof(FILETIME));
      memcpy(&m_startUserTime, &ftUser, sizeof(FILETIME));
      m_previousStartCallFailed = false;
    }
  }

  void Stop() override {
    FILETIME ftIgnore, ftKernel, ftUser;
    ULARGE_INTEGER stopKernelTime, stopUserTime;
    ULONG stopPageFaultCount = 0;
    SIZE_T stopPagefileUsage = 0;
    SIZE_T stopPeakPagefileUsage = 0;
    SIZE_T stopWorkingSetSize = 0;
    SIZE_T stopPeakWorkingSetSize = 0;

    if (m_previousStartCallFailed ||
        m_numProcessors == 0 ||
        !GetProcessTimes(m_procHandle, &ftIgnore, &ftIgnore, &ftKernel, &ftUser) ||
        !GetProcessMemoryCounters(m_pid, stopPageFaultCount, stopPagefileUsage, stopPeakPagefileUsage, stopWorkingSetSize, stopPeakWorkingSetSize)) {
      return;
    }

    memcpy(&stopKernelTime, &ftKernel, sizeof(FILETIME));
    memcpy(&stopUserTime, &ftUser, sizeof(FILETIME));
    m_processTime = CONVERT_100NS_TO_SECOND((stopKernelTime.QuadPart - m_startKernelTime.QuadPart) + (stopUserTime.QuadPart - m_startUserTime.QuadPart)) / m_numProcessors;

    m_deltaPageFaultCount = stopPageFaultCount - m_startPageFaultCount;
    m_deltaPagefileUsage = (double)BYTE_TO_MB((double)stopPagefileUsage - (double)m_startPagefileUsage);
    m_deltaPeakPagefileUsage = (double)BYTE_TO_MB((double)stopPeakPagefileUsage - (double)m_startPeakPagefileUsage);
    m_deltaWorkingSetSize = (double)BYTE_TO_MB((double)stopWorkingSetSize - (double)m_startWorkingSetSize);
    m_deltaPeakWorkingSetSize = (double)BYTE_TO_MB((double)stopPeakWorkingSetSize - (double)m_startPeakWorkingSetSize);
  }

  void GetValues(double (&values)[CounterType::TYPE_COUNT], double time) override {
    values[CounterType::CPU_USAGE] = 100.0 * GetProcessTime() / time;
    values[CounterType::PAGE_FAULT_COUNT] = GetDeltaPageFaultCount();
    values[CounterType::PAGE_FILE_USAGE] = GetDeltaPageFileUsage();
    values[CounterType::PEAK_PAGE_FILE_USAGE] = GetDeltaPeakPageFileUsage();
    values[CounterType::WORKING_SET_USAGE] = GetDeltaWorkingSetUsage();
    values[CounterType::PEAK_WORKING_SET_USAGE] = GetDeltaPeakWorkingSetUsage();
  }

  double GetProcessTime() { return m_processTime; }
  ULONG GetDeltaPageFaultCount() { return m_deltaPageFaultCount; }
  double GetDeltaPageFileUsage() { return m_deltaPagefileUsage; }
  double GetDeltaPeakPageFileUsage() { return m_deltaPeakPagefileUsage; }
  double GetDeltaWorkingSetUsage() { return m_deltaWorkingSetSize; }
  double GetDeltaPeakWorkingSetUsage() { return m_deltaPeakWorkingSetSize; }

 private:
  bool GetProcessMemoryCounters(DWORD pid, ULONG& pageFaultCount, SIZE_T& pageFileUsage, SIZE_T& peakPageFileUsage, SIZE_T& workingSetSize, SIZE_T& peakWorkingSetSize) {
    HANDLE hProcess = NULL;

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (NULL == hProcess)
      return false;

    PROCESS_MEMORY_COUNTERS pmc = {0};

    bool result = GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc));
    if (result) {
      pageFaultCount = pmc.PageFaultCount;
      pageFileUsage = pmc.PagefileUsage;
      peakPageFileUsage = pmc.PeakPagefileUsage;
      workingSetSize = pmc.WorkingSetSize;
      peakWorkingSetSize = pmc.PeakWorkingSetSize;
    }

    CloseHandle(hProcess);

    return result;
  }

  ULARGE_INTEGER m_startKernelTime = {};
  ULARGE_INTEGER m_startUserTime = {};
  UINT m_numProcessors = 0;
  HANDLE m_procHandle;
  DWORD m_pid = 0;
  bool m_previousStartCallFailed;
  double m_processTime = 0;  // in second
  ULONG m_startPageFaultCount = 0;
  SIZE_T m_startPagefileUsage = 0;       // in byte
  SIZE_T m_startPeakPagefileUsage = 0;   // in byte
  SIZE_T m_startWorkingSetSize = 0;      // in byte
  SIZE_T m_startPeakWorkingSetSize = 0;  // in byte
  ULONG m_deltaPageFaultCount = 0;
  double m_deltaPagefileUsage = 0;       // in MByte
  double m_deltaPeakPagefileUsage = 0;   // in MByte
  double m_deltaWorkingSetSize = 0;      // in MByte
  double m_deltaPeakWorkingSetSize = 0;  // in MByte
};
#ifndef DISABLE_GPU_COUNTERS

class GpuPerfCounter : public IPerfCounter {
 public:
  GpuPerfCounter() : m_hPDH(NULL),
                     m_pfnPdhOpenQuery(NULL),
                     m_pfnPdhAddCounter(NULL),
                     m_pfnPdhCollectQueryData(NULL),
                     m_pfnPdhGetFormattedCounterArray(NULL),
                     m_pfnPdhGetFormattedCounterValue(NULL),
                     m_pfnPdhCloseQuery(NULL),
                     m_query(NULL) {
    //#ifdef DISABLE_LOADLIBRARY
    m_hPDH = LoadLibraryExW(L"pdh.dll", NULL, 0);
    //#endif
    if (m_hPDH != NULL) {
      m_pfnPdhOpenQuery = (PFNPdhOpenQuery)GetProcAddress(m_hPDH, "PdhOpenQueryW");
      m_pfnPdhAddCounter = (PFNPdhAddCounter)GetProcAddress(m_hPDH, "PdhAddCounterW");
      m_pfnPdhCollectQueryData = (PFNPdhCollectQueryData)GetProcAddress(m_hPDH, "PdhCollectQueryData");
      m_pfnPdhGetFormattedCounterArray = (PFNPdhGetFormattedCounterArray)GetProcAddress(m_hPDH, "PdhGetFormattedCounterArrayW");
      m_pfnPdhGetFormattedCounterValue = (PFNPdhGetFormattedCounterValue)GetProcAddress(m_hPDH, "PdhGetFormattedCounterValue");
      m_pfnPdhCloseQuery = (PFNPdhCloseQuery)GetProcAddress(m_hPDH, "PdhCloseQuery");
    }
  }
  ~GpuPerfCounter() {
    if (m_query) {
      CloseQuery(m_query);
      m_query = NULL;
    }

    if (m_hPDH) {
      FreeLibrary(m_hPDH);
      m_hPDH = NULL;
    }
  }

  // This function consumes a lot of memory
  // Avoid calling this function unless it's necessary
  void Reset() override {
    m_gpuUsage = 0;
    m_deltaGpuDedicatedMemory = 0;
    m_deltaGpuSharedMemory = 0;

    // Setup PDH performance query
    std::wstring pidStr = std::to_wstring(GetCurrentProcessId());
    std::wstring gpuUsageQueryStr = L"\\GPU Engine(pid_*_*)\\Utilization Percentage";
    std::wstring gpuDedicatedMemQueryStr = L"\\GPU Process Memory(pid_*_*)\\Dedicated Usage";
    std::wstring gpuSharedMemQueryStr = L"\\GPU Process Memory(pid_*_*)\\Shared Usage";
    gpuUsageQueryStr.replace(gpuUsageQueryStr.find('*'), 1, pidStr);
    gpuDedicatedMemQueryStr.replace(gpuDedicatedMemQueryStr.find('*'), 1, pidStr);
    gpuSharedMemQueryStr.replace(gpuSharedMemQueryStr.find('*'), 1, pidStr);

    // Open query
    if (m_query) CloseQuery(m_query);
    OpenQuery(NULL, NULL, &m_query);
    AddCounter(m_query, gpuUsageQueryStr.c_str(), NULL, &m_gpuUsageCounter);
    AddCounter(m_query, gpuDedicatedMemQueryStr.c_str(), NULL, &m_gpuDedicatedMemUsageCounter);
    AddCounter(m_query, gpuSharedMemQueryStr.c_str(), NULL, &m_gpuSharedMemUsageCounter);
  }

  void Start() override {
    PDH_FMT_COUNTERVALUE gpuDedicatedMemUsageCounterValue = {};
    PDH_FMT_COUNTERVALUE gpuSharedMemUsageCounterValue = {};
    PDH_STATUS status = S_OK;

    // Usage rate counter require two queries. Put first one at Start() and second on at Stop()
    CollectQueryData(m_query);

    // Gpu dedicated ememory
    status = GetFormattedCounterValue(m_gpuDedicatedMemUsageCounter, PDH_FMT_LARGE, NULL, &gpuDedicatedMemUsageCounterValue);
    m_startGpuDedicatedMemory = (ERROR_SUCCESS == status) ? (double)BYTE_TO_MB(gpuDedicatedMemUsageCounterValue.largeValue) : m_startGpuDedicatedMemory;

    // Gpu shared ememory
    status = GetFormattedCounterValue(m_gpuSharedMemUsageCounter, PDH_FMT_LARGE, NULL, &gpuSharedMemUsageCounterValue);
    m_startGpuSharedMemory = (ERROR_SUCCESS == status) ? (double)BYTE_TO_MB(gpuSharedMemUsageCounterValue.largeValue) : m_startGpuSharedMemory;
  }

  void Stop() override {
    PPDH_FMT_COUNTERVALUE_ITEM_W gpuUsageCounterValue = nullptr;
    PDH_FMT_COUNTERVALUE gpuDedicatedMemUsageCounterValue = {};
    PDH_FMT_COUNTERVALUE gpuSharedMemUsageCounterValue = {};
    DWORD bufferSize = 0;
    DWORD itemCount = 0;
    PDH_STATUS status = S_OK;

    // Query the gpu usage.
    // For different IHVs, compute shader usage could be counted as either 3D or compute engine usage.
    // Here we simply pick the max usage from all types of engines to see if bottleneck is from GPU.
    // The same concept has been used in task manager to display GPU usage.
    status = CollectQueryData(m_query);
    if (S_OK != status && PDH_NO_DATA != status)
      return;

    status = GetFormattedCounterArray(m_gpuUsageCounter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, gpuUsageCounterValue);
    if (PDH_MORE_DATA != status)
      return;

    gpuUsageCounterValue = (PPDH_FMT_COUNTERVALUE_ITEM_W)malloc(bufferSize);
    if (gpuUsageCounterValue != nullptr) {
      status = GetFormattedCounterArray(m_gpuUsageCounter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, gpuUsageCounterValue);
      if (ERROR_SUCCESS == status) {
        double maxValue = 0;
        for (size_t i = 0; i < itemCount; ++i) {
          maxValue = (gpuUsageCounterValue[i].FmtValue.doubleValue > maxValue) ? gpuUsageCounterValue[i].FmtValue.doubleValue : maxValue;
        }
        m_gpuUsage = maxValue;
      }
    }

    free(gpuUsageCounterValue);
    gpuUsageCounterValue = NULL;
    bufferSize = 0;
    itemCount = 0;

    double stopGpuDedicatedMemory;  // in MB
    double stopGpuSharedMemory;     // in MB

    // Gpu dedicated ememory delta. Don't update the value if counter doesn't get values correctly.
    status = GetFormattedCounterValue(m_gpuDedicatedMemUsageCounter, PDH_FMT_LARGE, NULL, &gpuDedicatedMemUsageCounterValue);
    if (ERROR_SUCCESS == status) {
      stopGpuDedicatedMemory = (double)BYTE_TO_MB(gpuDedicatedMemUsageCounterValue.largeValue);
      m_deltaGpuDedicatedMemory = stopGpuDedicatedMemory - m_startGpuDedicatedMemory;
    }

    // Gpu shared ememory. Don't update the value if counter doesn't get values correctly.
    status = GetFormattedCounterValue(m_gpuSharedMemUsageCounter, PDH_FMT_LARGE, NULL, &gpuSharedMemUsageCounterValue);
    if (ERROR_SUCCESS == status) {
      stopGpuSharedMemory = (double)BYTE_TO_MB(gpuSharedMemUsageCounterValue.largeValue);
      m_deltaGpuSharedMemory = stopGpuSharedMemory - m_startGpuSharedMemory;
    }
  }

  void GetValues(double (&values)[CounterType::TYPE_COUNT], double time) override {
    ORT_UNUSED_PARAMETER(time);
    values[CounterType::GPU_USAGE] = GetGpuUsage();
    values[CounterType::GPU_DEDICATED_MEM_USAGE] = GetDedicatedMemory();
    values[CounterType::GPU_SHARED_MEM_USAGE] = GetSharedMemory();
  }

  double GetGpuUsage() const { return m_gpuUsage; }
  double GetDedicatedMemory() const { return m_deltaGpuDedicatedMemory; }
  double GetSharedMemory() const { return m_deltaGpuSharedMemory; }

 private:
  // Pdh function prototypes
  typedef PDH_STATUS(WINAPI* PFNPdhOpenQuery)(_In_opt_ LPCWSTR szDataSource, _In_ DWORD_PTR dwUserData, _Out_ PDH_HQUERY* phQuery);
  typedef PDH_STATUS(WINAPI* PFNPdhAddCounter)(_In_ PDH_HQUERY hQuery, _In_ LPCWSTR szFullCounterPath, _In_ DWORD_PTR dwUserData, _Out_ PDH_HCOUNTER* phCounter);
  typedef PDH_STATUS(WINAPI* PFNPdhCollectQueryData)(_Inout_ PDH_HQUERY hQuery);
  typedef PDH_STATUS(WINAPI* PFNPdhGetFormattedCounterArray)(_In_ PDH_HCOUNTER hCounter, _In_ DWORD dwFormat, _Inout_ LPDWORD lpdwBufferSize, _Out_ LPDWORD lpdwItemCount, _Out_writes_bytes_opt_(*lpdwBufferSize) PPDH_FMT_COUNTERVALUE_ITEM_W ItemBuffer);
  typedef PDH_STATUS(WINAPI* PFNPdhGetFormattedCounterValue)(_In_ PDH_HCOUNTER hCounter, _In_ DWORD dwFormat, _Out_opt_ LPDWORD lpdwType, _Out_ PPDH_FMT_COUNTERVALUE pValue);
  typedef PDH_STATUS(WINAPI* PFNPdhCloseQuery)(_Inout_ PDH_HQUERY hQuery);

  PDH_STATUS OpenQuery(LPCWSTR szDataSource, DWORD_PTR dwUserData, PDH_HQUERY* phQuery) {
    return (m_pfnPdhOpenQuery) ? m_pfnPdhOpenQuery(szDataSource, dwUserData, phQuery) : ERROR_MOD_NOT_FOUND;
  }
  PDH_STATUS AddCounter(PDH_HQUERY hQuery, LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER* phCounter) {
    return (m_pfnPdhAddCounter) ? m_pfnPdhAddCounter(hQuery, szFullCounterPath, dwUserData, phCounter) : ERROR_MOD_NOT_FOUND;
  }
  PDH_STATUS CollectQueryData(PDH_HQUERY hQuery) {
    return (m_pfnPdhCollectQueryData) ? m_pfnPdhCollectQueryData(hQuery) : ERROR_MOD_NOT_FOUND;
  }
  PDH_STATUS GetFormattedCounterArray(PDH_HCOUNTER hCounter, DWORD dwFormat, LPDWORD lpdwBufferSize, LPDWORD lpdwItemCount, PPDH_FMT_COUNTERVALUE_ITEM_W ItemBuffer) {
    return (m_pfnPdhGetFormattedCounterArray) ? m_pfnPdhGetFormattedCounterArray(hCounter, dwFormat, lpdwBufferSize, lpdwItemCount, ItemBuffer) : ERROR_MOD_NOT_FOUND;
  }
  PDH_STATUS GetFormattedCounterValue(PDH_HCOUNTER hCounter, DWORD dwFormat, LPDWORD lpdwType, PPDH_FMT_COUNTERVALUE pValue) {
    return (m_pfnPdhGetFormattedCounterValue) ? m_pfnPdhGetFormattedCounterValue(hCounter, dwFormat, lpdwType, pValue) : ERROR_MOD_NOT_FOUND;
  }
  PDH_STATUS CloseQuery(PDH_HQUERY hQuery) {
    return (m_pfnPdhCloseQuery) ? m_pfnPdhCloseQuery(hQuery) : ERROR_MOD_NOT_FOUND;
  }

  // PDH Performance Query
  HMODULE m_hPDH;
  PFNPdhOpenQuery m_pfnPdhOpenQuery;
  PFNPdhAddCounter m_pfnPdhAddCounter;
  PFNPdhCollectQueryData m_pfnPdhCollectQueryData;
  PFNPdhGetFormattedCounterArray m_pfnPdhGetFormattedCounterArray;
  PFNPdhGetFormattedCounterValue m_pfnPdhGetFormattedCounterValue;
  PFNPdhCloseQuery m_pfnPdhCloseQuery;
  HQUERY m_query;
  HCOUNTER m_gpuUsageCounter = NULL;
  HCOUNTER m_gpuDedicatedMemUsageCounter = NULL;
  HCOUNTER m_gpuSharedMemUsageCounter = NULL;
  // Process info
  DWORD m_pid = 0;
  // Data
  double m_gpuUsage = 0;
  double m_startGpuDedicatedMemory = 0;  // in MB
  double m_startGpuSharedMemory = 0;     // in MB
  double m_deltaGpuDedicatedMemory = 0;  // in MB
  double m_deltaGpuSharedMemory = 0;     // in MB
};

#endif
;

// A statistics helper for Timer/CpuPerfCounter/GpuPerfCounter class.
// It keeps the latest "TIMER_SLOT_SIZE" measured data in a ring buffer.
// The statistic functions (e.g. GetVariance) assume data always starts from index 0 of the buffer.
class PerfCounterStatistics {
 public:
  void Enable(ProfilerType type) {
    if (type == ProfilerType::CPU) {
      m_perfCounter[type] = std::make_unique<CpuPerfCounter>();
    } else if (type == ProfilerType::GPU) {
      m_perfCounter[type] = std::make_unique<GpuPerfCounter>();
    }
  }

  void Disable(ProfilerType type) {
    m_perfCounter[type].reset();
  }

  void Reset(ProfilerType type) {
    m_pos = 0;
    m_bufferFull = false;
    for (int i = 0; i < CounterType::TYPE_COUNT; ++i) {
      if (!IsCounterTypeDisabled(static_cast<CounterType>(i))) {
        m_data[i].Reset();
      }
    }
    if (m_perfCounter[type]) {
      m_perfCounter[type]->Reset();
    }
  }

  void Start() {
    m_timer.Start();
    for (unsigned int i = 0; i < m_perfCounter.size(); ++i) {
      if (m_perfCounter[i]) {
        m_perfCounter[i]->Start();
      }
    }
  }

  void Stop() {
    double counterValue[CounterType::TYPE_COUNT] = {0.0f};
    // Query counters
    double time = m_timer.Stop();
    counterValue[CounterType::TIMER] = time;
    for (unsigned int i = 0; i < m_perfCounter.size(); ++i) {
      if (m_perfCounter[i]) {
        m_perfCounter[i]->Stop();
        m_perfCounter[i]->GetValues(counterValue, time);
      }
    }

    // Update data blocks
    for (int i = 0; i < CounterType::TYPE_COUNT; ++i) {
      m_data[i].total = m_data[i].total - m_data[i].measured[m_pos] + counterValue[i];
      m_data[i].measured[m_pos] = counterValue[i];
      m_data[i].max = (counterValue[i] > m_data[i].max) ? counterValue[i] : m_data[i].max;
      m_data[i].min = (counterValue[i] < m_data[i].min) ? counterValue[i] : m_data[i].min;
    }

    // Update buffer index
    if (m_pos + 1 >= TIMER_SLOT_SIZE) {
      m_pos = 0;
      m_bufferFull = true;
    } else {
      ++m_pos;
    }
  }

  int GetCount() const { return (m_bufferFull) ? TIMER_SLOT_SIZE : m_pos; }

  double GetAverage(CounterType t) const { return IsCounterTypeDisabled(t) ? 0 : m_data[t].total / GetCount(); }
  double GetMin(CounterType t) const { return IsCounterTypeDisabled(t) ? 0 : m_data[t].min; }
  double GetMax(CounterType t) const { return IsCounterTypeDisabled(t) ? 0 : m_data[t].max; }
  double GetValues(CounterType t, int index) const { return IsCounterTypeDisabled(t) ? 0 : m_data[t].measured[index]; }
  double GetStdev(CounterType t) const { return IsCounterTypeDisabled(t) ? 0 : sqrt(GetVariance(t)); }
  double GetVariance(CounterType t) const {
    if (IsCounterTypeDisabled(t))
      return 0;

    int count = GetCount();
    double average = m_data[t].total / count;
    double var = 0;
    for (int i = 0; i < count; ++i) {
      var += (m_data[t].measured[i] - average) * (m_data[t].measured[i] - average);
    }
    return var / count;
  }

 private:
  bool IsCounterTypeDisabled(CounterType t) const {
    switch (t) {
      case CPU_USAGE:
      case PAGE_FAULT_COUNT:
      case PAGE_FILE_USAGE:
      case PEAK_PAGE_FILE_USAGE:
      case WORKING_SET_USAGE:
      case PEAK_WORKING_SET_USAGE:
        return m_perfCounter[ProfilerType::CPU] == nullptr;
      case GPU_USAGE:
      case GPU_DEDICATED_MEM_USAGE:
      case GPU_SHARED_MEM_USAGE:
        return m_perfCounter[ProfilerType::GPU] == nullptr;
      case TIMER:
        return false;
      default:
        return true;
    }
  }

  struct DataBlock {
    void Reset() {
      max = 0;
      min = DBL_MAX;
      total = 0;
      memset(measured, 0, sizeof(double) * TIMER_SLOT_SIZE);
    }

    double max;
    double min;
    double total;
    double measured[TIMER_SLOT_SIZE];
  };

  int m_pos;
  bool m_bufferFull;
  std::array<std::unique_ptr<IPerfCounter>, ProfilerType::PROFILER_TYPE_COUNT> m_perfCounter;

  Timer m_timer;
  DataBlock m_data[CounterType::TYPE_COUNT];
};

// A class to wrap up multiple PerfCounterStatistics objects.
// To create a profiler, define intervals in an enum and use it to create the profiler object.
// See an example in engine/test/Model/ModelTest.cpp
template <typename T>
class Profiler {
 public:
  void Reset(ProfilerType type) {
    for (int i = 0; i < T::kCount; ++i) {
      m_perfCounterStat[i].Reset(type);
    }
  }

  PerfCounterStatistics& GetCounter(int t) {
    return m_perfCounterStat[t];
  }

  PerfCounterStatistics& operator[](int t) {
    return m_perfCounterStat[t];
  }

  void Enable(ProfilerType type) {
    for (int i = 0; i < T::kCount; ++i) {
      m_perfCounterStat[i].Enable(type);
    }
  }

  void Disable(ProfilerType type) {
    for (int i = 0; i < T::kCount; ++i) {
      m_perfCounterStat[i].Disable(type);
    }
  }

  //Checks if Profiler is still reseted
  bool IsStillReset() {
    for (int i = 0; i < T::kCount; ++i) {
      if (m_perfCounterStat[i].GetCount() > 0) {
        return false;
      }
    }
    return true;
  }

 private:
  PerfCounterStatistics m_perfCounterStat[T::kCount];
};

#define WINML_PROFILING

#ifdef WINML_PROFILING
#define WINML_PROFILING_START(profiler, interval) profiler[interval].Start()
#define WINML_PROFILING_STOP(profiler, interval) profiler[interval].Stop()
#else
#define WINML_PROFILING_START(profiler, interval) \
  do {                                            \
  } while (0)
#define WINML_PROFILING_STOP(profiler, interval) \
  do {                                           \
  } while (0)
#endif
