// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <map>
#include <functional>
#include <thread>
#include <mutex>

#include <openenclave/enclave.h>

#include "test/openenclave/session_enclave/shared/atomic_flag_lock.h"
#include "session_t.h" // generated from session.edl
#include "threading.h"

// Copied from https://github.com/microsoft/openenclave/blob/master/tests/libcxx/enc/enc.cpp.

typedef struct _oe_pthread_hooks {
  int (*create)(
      pthread_t* thread,
      const pthread_attr_t* attr,
      void* (*start_routine)(void*),
      void* arg);

  int (*join)(pthread_t thread, void** retval);

  int (*detach)(pthread_t thread);
} oe_pthread_hooks_t;

extern "C" void oe_register_pthread_hooks(oe_pthread_hooks_t* pthread_hooks);

static std::vector<std::function<void*()>> _thread_functions;
static uint64_t _next_enc_thread_id = 0;
static uint64_t _enc_key = 0;  // Monotonically increasing enclave key

// Map of enc_key to thread_id returned by pthread_self()
static std::map<uint64_t, pthread_t> _key_to_thread_id_map;

static atomic_flag_lock _enc_lock;

struct thread_args {
  uint64_t enc_key;
  int join_ret;
  int detach_ret;
};
// Each new thread will point to memory created by the host after thread
// creation
thread_args _thread_args[MAX_ENC_KEYS];

static int _pthread_create_hook(
    pthread_t* enc_thread,
    const pthread_attr_t*,
    void* (*start_routine)(void*),
    void* arg) {
  *enc_thread = 0;
  uint64_t enc_key;
  {
    atomic_lock lock(_enc_lock);
    _thread_functions.push_back(
        [start_routine, arg]() { return start_routine(arg); });
    enc_key = _enc_key = ++_next_enc_thread_id;
    // printf("_pthread_create_hook(): enc_key is %lu\n", enc_key);
    // Populate the enclave key to thread id map in advance
    _key_to_thread_id_map.insert(std::make_pair(enc_key, *enc_thread));

    if (_next_enc_thread_id > (MAX_ENC_KEYS - 1)) {
      printf("Exceeded max number of enclave threads supported %lu\n", MAX_ENC_KEYS - 1);
    }
  }

  // Send the enclave id so that host can maintain the map between
  // enclave and host id
  if (OE_OK != host_create_thread(enc_key, oe_get_enclave())) {
    printf("_pthread_create_hook(): Error in call to host_create_pthread for enc_key=%lu\n", enc_key);
    oe_abort();
  }

  // Block until the enclave pthread_id becomes available in the map
  while (*enc_thread == 0) {
    {
      atomic_lock lock(_enc_lock);
      *enc_thread = _key_to_thread_id_map[enc_key];
    }
    if (*enc_thread == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  printf("_pthread_create_hook(): pthread_create success for enc_key=%lu; thread id=%#10lx\n", _enc_key, *enc_thread);

  return 0;
}

static int _pthread_join_hook(pthread_t enc_thread, void**) {
  // Find the enc_key from the enc_thread
  uint64_t join_enc_key;
  {
    atomic_lock lock(_enc_lock);
    std::map<uint64_t, pthread_t>::iterator it = std::find_if(
        _key_to_thread_id_map.begin(),
        _key_to_thread_id_map.end(),
        [&enc_thread](const std::pair<uint64_t, pthread_t>& p) {
          return p.second == enc_thread;
        });
    if (it == _key_to_thread_id_map.end()) {
      printf("_pthread_join_hook(): Error: enc_key for thread ID %#10lx not found\n", enc_thread);
      oe_abort();
    }

    join_enc_key = it->first;
    _thread_args[join_enc_key - 1].enc_key = join_enc_key;
  }

  printf("_pthread_join_hook(): enc_key for thread ID %#10lx is %ld\n", enc_thread, join_enc_key);

  int join_ret = 0;
  if (host_join_thread(&join_ret, join_enc_key) != OE_OK) {
    printf("_pthread_join_hook(): Error in call to host host_join_pthread for enc_key=%ld\n", join_enc_key);
    oe_abort();
  }

  {
    atomic_lock lock(_enc_lock);
    _thread_args[join_enc_key - 1].join_ret = join_ret;

    // Since join succeeded, delete the _key_to_thread_id_map
    if (!join_ret) {
      _key_to_thread_id_map.erase(join_enc_key);
    }
  }

  return join_ret;
}

static int _pthread_detach_hook(pthread_t enc_thread) {
  // Find the enc_key from the enc_thread
  uint64_t det_enc_key;
  {
    atomic_lock lock(_enc_lock);
    std::map<uint64_t, pthread_t>::iterator it = std::find_if(
        _key_to_thread_id_map.begin(),
        _key_to_thread_id_map.end(),
        [&enc_thread](const std::pair<uint64_t, pthread_t>& p) {
          return p.second == enc_thread;
        });
    if (it == _key_to_thread_id_map.end()) {
      printf("_pthread_detach_hook(): Error: enc_key for thread ID %#10lx not found\n", enc_thread);
      oe_abort();
    }

    det_enc_key = it->first;
    _thread_args[det_enc_key - 1].enc_key = det_enc_key;
  }

  printf("_pthread_detach_hook(): enc_key for thread ID %#10lx is %ld\n", enc_thread, det_enc_key);

  int det_ret = 0;
  if (host_detach_thread(&det_ret, det_enc_key) != OE_OK) {
    printf("_pthread_detach_hook(): Error in call to host host_detach_thread for enc_key=%ld\n", det_enc_key);
    oe_abort();
  }

  // Since detach succeeded, delete the _key_to_thread_id_map
  if (0 == det_ret) {
    atomic_lock lock(_enc_lock);
    _key_to_thread_id_map.erase(det_enc_key);
  }

  return det_ret;
}

// Launches the new thread in the enclave
void EnclaveThreadFun(uint64_t enc_key) {
  _thread_args[_enc_key - 1].enc_key = enc_key;
  _thread_args[_enc_key - 1].join_ret = -1;
  _thread_args[_enc_key - 1].detach_ret = -1;

  std::function<void()> f;

  {
    atomic_lock lock(_enc_lock);
    _key_to_thread_id_map[enc_key] = pthread_self();
  }

  std::this_thread::yield();

  {
    atomic_lock lock(_enc_lock);
    f = _thread_functions.back();
    _thread_functions.pop_back();
  }

  printf("enc_key=%ld tid=%lx starting...\n", enc_key, pthread_self());
  f();
  printf("enc_key=%ld tid=%lx returning...\n", enc_key, pthread_self());
}

namespace onnxruntime {
namespace openenclave {
void InitializeOpenEnclavePThreads() {
  static oe_pthread_hooks_t pthread_hooks = {.create = _pthread_create_hook,
                                             .join = _pthread_join_hook,
                                             .detach = _pthread_detach_hook};

  oe_register_pthread_hooks(&pthread_hooks);
}
}  // namespace openenclave
}  // namespace onnxruntime