// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/ort_mutex.h"
#include <assert.h>
#include <stdexcept>
#include <system_error>

namespace onnxruntime {
#ifndef USE_NSYNC
OrtMutex::~OrtMutex() {
  pthread_mutex_destroy(&data_);
}
#endif

void
OrtMutex::lock() {
#ifndef USE_NSYNC
  int ec = pthread_mutex_lock(&data_);
  if (ec)
    throw std::system_error(ec, std::system_category(), "mutex lock failed");
#else
  nsync::nsync_mu_lock(&data_);
#endif
}

bool
OrtMutex::try_lock()
noexcept {
#ifndef USE_NSYNC
  return pthread_mutex_trylock(&data_) == 0;
#else
  return nsync::nsync_mu_trylock(&data_) == 0;
#endif
}

void
OrtMutex::unlock()
noexcept
{
#ifdef USE_NSYNC
  nsync::nsync_mu_unlock(&data_);
#else
int ec = pthread_mutex_unlock(&data_);
(void) ec;
//Don't throw!
assert(ec== 0);
#endif
}
}