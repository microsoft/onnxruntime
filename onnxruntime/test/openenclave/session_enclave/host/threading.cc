// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <thread>
#include <mutex>
#include <map>

#include <openenclave/host.h>

#include "test/openenclave/session_enclave/shared/atomic_flag_lock.h"
#include "session_u.h" // generated from session.edl

// Copied from https://github.com/microsoft/openenclave/blob/master/tests/libcxx/host/host.cpp.

static std::map<uint64_t, pthread_t> _enclave_host_id_map;
static atomic_flag_lock _host_lock;

extern "C" {
void host_exit(int arg) {
  exit(arg);
}

struct thread_args_t {
  oe_enclave_t* enclave;
  uint64_t enc_key;
};

void* host_enclave_thread(void* args) {
  thread_args_t* thread_args = reinterpret_cast<thread_args_t*>(args);
  // need to cache the values for enc_key and enclave now before _host_lock
  // is unlocked after assigning the thread_id to the _enclave_host_id_map
  // because args is a local variable in the calling method which may exit at
  // any time after _host_lock is unlocked which may cause a segfault
  uint64_t enc_key = thread_args->enc_key;
  oe_enclave_t* enclave = thread_args->enclave;
  pthread_t thread_id = pthread_self();

  {
    atomic_lock lock(_host_lock);
    // Populate the enclave_host_id map with the host thread id
    _enclave_host_id_map[enc_key] = thread_id;
  }

  printf("host_enclave_thread(): enc_key=%lu has host thread_id of %#10lx\n", enc_key, thread_id);

  // Launch the thread
  oe_result_t result = EnclaveThreadFun(enclave, enc_key);
  if (result != OE_OK) {
    printf("EnclaveThreadFun failed.\n");
    return (void*)1;
  }

  return nullptr;
}

void host_create_thread(uint64_t enc_key, oe_enclave_t* enclave) {
  thread_args_t thread_args = {enclave, enc_key};
  pthread_t thread_id = 0;

  {
    // Using atomic locks to protect the enclave_host_id_map
    atomic_lock lock(_host_lock);
    _enclave_host_id_map.insert(std::make_pair(enc_key, thread_id));
  }

  // New Thread is created and executes host_enclave_thread
  int ret = pthread_create(&thread_id, nullptr, host_enclave_thread, &thread_args);
  if (ret != 0) {
    printf("host_create_thread(): pthread_create error %d\n", ret);
    abort();
  }

  // Main host thread waits for the enclave id to host id mapping to be
  // updated
  pthread_t mapped_thread_id = 0;
  while (0 == mapped_thread_id) {
    {
      atomic_lock lock(_host_lock);
      mapped_thread_id = _enclave_host_id_map[enc_key];
    }
    if (0 == mapped_thread_id) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Sanity check
  if (thread_id != mapped_thread_id) {
    printf("Host thread id incorrect in the enclave_host_id_map\n");
    abort();
  }
}

int host_join_thread(uint64_t enc_key) {
  int ret_val = 0;
  pthread_t thread_id = 0;

  // Find the thread_id from the enclave_host_id_map using the enc_key
  {
    // Using atomic locks to protect the enclave_host_id_map
    atomic_lock lock(_host_lock);
    std::map<uint64_t, pthread_t>::iterator it =
        _enclave_host_id_map.find(enc_key);
    if (it != _enclave_host_id_map.end()) {
      thread_id = it->second;
      lock.unlock();

      void* value_ptr = nullptr;
      ret_val = pthread_join(thread_id, &value_ptr);

      // Update the shared memory only after pthread_join returns
      if (0 == ret_val) {
        // Delete the enclave_host_id mapping as thread_id may be reused
        // in future
        lock.lock();
        _enclave_host_id_map.erase(enc_key);
        printf("host_join_thread() succeeded for enclave id=%lu, host id=%#10lx\n", enc_key, thread_id);
      }
    } else {
      printf("host_join_thread() failed to find enclave id=%lu in host map\n", enc_key);
      abort();
    }
  }

  return ret_val;
}

int host_detach_thread(uint64_t enc_key) {
  int ret_val = 0;
  printf("host_detach_thread():enclave key=%lu\n", enc_key);

  // Find the thread_id from the enclave_host_id_map using the enc_key

  // Using atomic locks to protect the enclave_host_id_map
  atomic_lock lock(_host_lock);
  std::map<uint64_t, pthread_t>::iterator it =
      _enclave_host_id_map.find(enc_key);
  if (it != _enclave_host_id_map.end()) {
    pthread_t thread_id = it->second;
    lock.unlock();

    ret_val = pthread_detach(thread_id);
    if (0 == ret_val) {
      // Delete the _enclave_host_id mapping as thread_id may be reused
      // in future
      lock.lock();
      _enclave_host_id_map.erase(enc_key);
    }
    printf(
        "host_detach_thread() returned=%d for enclave id=%lu, host "
        "thread id=%#10lx\n",
        ret_val,
        enc_key,
        thread_id);
  } else {
    printf(
        "host_detach_thread() failed to find enclave key=%lu in host "
        "map\n",
        enc_key);
    abort();
  }
  return ret_val;
}
}