#include "mem.h"

#include <stdlib.h>

#ifdef _WIN32

#include "windows.h"
#include "psapi.h"

void PrintMemoryUsage(const char* message) {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
  SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
  SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

  printf("%s VirtualMemory=%d, PhysicalMemory=%d\n", message, virtualMemUsedByMe, physMemUsedByMe);
}

#else

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line) {
  // This assumes that a digit will be found and the line ends in " Kb".
  int i = strlen(line);
  const char* p = line;
  while (*p < '0' || *p > '9') p++;
  line[i - 3] = '\0';
  i = atoi(p);
  return i;
}

int getCurrentProcessVirtualMemSize() {  // Note: this value is in KB!
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "VmSize:", 7) == 0) {
      result = parseLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}

int getCurrentProcessMemorySize() {  // Note: this value is in KB!
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      result = parseLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}

void PrintMemoryUsage(const char* message) {
  printf("%s VirtualMemory=%d, PhysicalMemory=%d\n", message,
         getCurrentProcessVirtualMemSize(),
         getCurrentProcessMemorySize());
}

#endif
