// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#define _GNU_SOURCE
#include <pthread.h>

// The following functions are defined in Open Enclave's standard
// library headers but lack a corresponding implementation.

// Note: Not provided by liboehostfs as of Open Enclave 0.6.
ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
  (void)fd;
  (void)buf;
  (void)count;
  (void)offset;
  puts("FATAL: Open Enclave pread() stub called");
  abort();
}

// Note: Not provided by liboehostfs as of Open Enclave 0.6.
int fstat(int fd, struct stat *buf)
{
  (void)fd;
  (void)buf;
  puts("FATAL: Open Enclave fstat() stub called");
  abort();
}

int lstat(const char *path, struct stat *buf)
{
  (void)path;
  (void)buf;
  puts("FATAL: Open Enclave lstat() stub called");
  abort();
}

char *realpath(const char *path, char *resolved_path)
{
  (void)path;
  (void)resolved_path;
  puts("FATAL: Open Enclave realpath() stub called");
  abort();
}

char *dlerror(void)
{
  puts("FATAL: Open Enclave dlerror() stub called");
  abort();
}

void *dlopen(const char *filename, int flags)
{
  (void)filename;
  (void)flags;
  puts("FATAL: Open Enclave dlopen() stub called");
  abort();
}

int dlclose(void *handle)
{
  (void)handle;
  puts("FATAL: Open Enclave dlclose() stub called");
  abort();
}

void *dlsym(void *handle, const char *symbol)
{
  (void)handle;
  (void)symbol;
  puts("FATAL: Open Enclave dlsym() stub called");
  abort();
}

struct tm *localtime_r(const time_t *timep, struct tm *result)
{
  (void)timep;
  return result;
}

double difftime(time_t end, time_t beginning)
{
  (void)end;
  (void)beginning;
  return 0.0;
}

time_t mktime(struct tm *tm)
{
  (void)tm;
  return -1;
}

int pthread_setaffinity_np(pthread_t thread, size_t cpusetsize, const cpu_set_t *cpuset)
{
  (void)thread;
  (void)cpusetsize;
  (void)cpuset;
  return 0;
}

int pthread_attr_init(pthread_attr_t *attr)
{
  (void)attr;
  return 0;
}

int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize)
{
  (void)attr;
  (void)stacksize;
  return 0;
}

int pthread_setcancelstate(int state, int *oldstate)
{
  (void)state;
  (void)oldstate;
  return 0;
}

 int sched_getcpu(void)
 {
   return 0;
 }
 