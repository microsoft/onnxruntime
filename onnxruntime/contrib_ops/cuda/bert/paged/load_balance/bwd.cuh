// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::paged {

template <int N, typename T>
class BrokerWorkDistributor {
  using Ticket = uint32_t;
  using HT_t = uint32_t;
  using HT = uint64_t;

  volatile Ticket tickets[N];
  T ring_buffer[N];

  HT head_tail;
  int count;

  template <typename L>
  __device__ __forceinline__ L atomicLoad(L* l) {
    return *l;
  }

  __device__ HT_t* head(HT* head_tail) {
    return reinterpret_cast<HT_t*>(head_tail) + 1;
  }

  __device__ HT_t* tail(HT* head_tail) {
    return reinterpret_cast<HT_t*>(head_tail);
  }

  __forceinline__ __device__ void waitForTicket(const unsigned int P, const Ticket number) {
    while (tickets[P] != number) {
      backoff();  // back off
    }
  }

  __forceinline__ __device__ bool ensureDequeue() {
    int Num = atomicLoad(&count);
    bool ensurance = false;

    while (!ensurance && Num > 0) {
      if (atomicSub(&count, 1) > 0) {
        ensurance = true;
      } else {
        Num = atomicAdd(&count, 1) + 1;
      }
    }
    return ensurance;
  }

  __forceinline__ __device__ bool ensureEnqueue() {
    int Num = atomicLoad(&count);
    bool ensurance = false;

    while (!ensurance && Num < N) {
      if (atomicAdd(&count, 1) < N) {
        ensurance = true;
      } else {
        Num = atomicSub(&count, 1) - 1;
      }
    }
    return ensurance;
  }

  __forceinline__ __device__ void readData(T& val) {
    const unsigned int Pos = atomicAdd(head(const_cast<HT*>(&head_tail)), 1);
    const unsigned int P = Pos % N;

    waitForTicket(P, 2 * (Pos / N) + 1);
    val = ring_buffer[P];
    __threadfence();
    tickets[P] = 2 * ((Pos + N) / N);
  }

  __forceinline__ __device__ void putData(const T data) {
    const unsigned int Pos = atomicAdd(tail(const_cast<HT*>(&head_tail)), 1);
    const unsigned int P = Pos % N;
    const unsigned int B = 2 * (Pos / N);

    waitForTicket(P, B);
    ring_buffer[P] = data;
    __threadfence();
    tickets[P] = B + 1;
  }

public:
  __device__ void init() {
    const int lid = threadIdx.x + blockIdx.x * blockDim.x;

    if (lid == 0) {
      count = 0;
      head_tail = 0x0ULL;
    }

    for (int v = lid; v < N; v += blockDim.x * gridDim.x) {
      ring_buffer[v] = T(0x0);
      tickets[v] = 0x0;
    }
  }

  __forceinline__ __device__ bool enqueue(const T& data) {
    bool writeData = ensureEnqueue();
    if (writeData) {
      putData(data);
    }
    return writeData;
  }

  __forceinline__ __device__ bool dequeue(T& data) {
    bool hasData = ensureDequeue();
    if (hasData) {
      readData(data);
    }
    return hasData;
  }
};

}  // namespace onnxruntime::contrib::paged
