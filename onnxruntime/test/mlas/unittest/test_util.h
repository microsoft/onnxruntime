// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "mlas.h"
#include "gtest/gtest.h"

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
#include "core/platform/threadpool.h"
#endif


#if !defined(UNUSED_VARIABLE)
#if defined(__GNUC__)
# define UNUSED_VARIABLE __attribute__((unused))
#else
# define UNUSED_VARIABLE
#endif
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

MLAS_THREADPOOL* GetMlasThreadPool(void);

template <typename T>
class MatrixGuardBuffer {
 public:
  MatrixGuardBuffer() {
    _BaseBuffer = nullptr;
    _BaseBufferSize = 0;
    _ElementsAllocated = 0;
  }

  ~MatrixGuardBuffer(void) {
    ReleaseBuffer();
  }

  T* GetBuffer(size_t Elements, bool ZeroFill = false) {
    //
    // Check if the internal buffer needs to be reallocated.
    //

    if (Elements > _ElementsAllocated) {
      ReleaseBuffer();

      //
      // Reserve a virtual address range for the allocation plus an unmapped
      // guard region.
      //

      constexpr size_t BufferAlignment = 64 * 1024;
      constexpr size_t GuardPadding = 256 * 1024;

      size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

      _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
      _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
      _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

      if (_BaseBuffer == nullptr) {
        abort();
      }

      //
      // Commit the number of bytes for the allocation leaving the upper
      // guard region as unmapped.
      //

#if defined(_WIN32)
      if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        ORT_THROW_EX(std::bad_alloc);
      }
#else
      if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
        abort();
      }
#endif

      _ElementsAllocated = BytesToAllocate / sizeof(T);
      _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
    }

    //
    //
    //

    T* GuardAddress = _GuardAddress;
    T* buffer = GuardAddress - Elements;

    if (ZeroFill) {
      std::fill_n(buffer, Elements, T(0));

    } else {
      const int MinimumFillValue = -23;
      const int MaximumFillValue = 23;

      int FillValue = MinimumFillValue;
      T* FillAddress = buffer;

      while (FillAddress < GuardAddress) {
        *FillAddress++ = (T)FillValue;

        FillValue++;

        if (FillValue > MaximumFillValue) {
          FillValue = MinimumFillValue;
        }
      }
    }

    return buffer;
  }

  void ReleaseBuffer(void) {
    if (_BaseBuffer != nullptr) {
#if defined(_WIN32)
      VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
      munmap(_BaseBuffer, _BaseBufferSize);
#endif

      _BaseBuffer = nullptr;
      _BaseBufferSize = 0;
    }

    _ElementsAllocated = 0;
  }

 private:
  size_t _ElementsAllocated;
  void* _BaseBuffer;
  size_t _BaseBufferSize;
  T* _GuardAddress;
};

class MlasTestBase {
 public:
  virtual ~MlasTestBase(void) {}

  virtual void ExecuteShort(void) {}

  virtual void ExecuteLong(void) {}
};

typedef std::function<size_t(bool is_short_execute)> TestRegister;

bool AddTestRegister(TestRegister test_register);

//
// Base Test Fixture which setup/teardown MlasTest in one test suite.
//
template <typename TMlasTester>
class MlasTestFixture : public testing::Test {
 public:
  static void SetUpTestSuite() {
    mlas_tester = new TMlasTester();
  };

  static void TearDownTestSuite() {
    if (nullptr != mlas_tester) {
      delete mlas_tester;
    }
    mlas_tester = nullptr;
  };

  // Do not forgot to define this static member element when upon usage.
  static TMlasTester* mlas_tester;
};

// Long Execute test. It is too heavy register each single test, treat long execute big groups.
template <typename TMlasTester>
class MlasLongExecuteTests : public MlasTestFixture<TMlasTester> {
 public:
  void TestBody() override {
    MlasTestFixture<TMlasTester>::mlas_tester->ExecuteLong();
  }

  static size_t RegisterLongExecute() {
    testing::RegisterTest(
        TMlasTester::GetTestSuiteName(),
        "LongExecute",
        nullptr,
        "LongExecute",
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<TMlasTester>* {
          return new MlasLongExecuteTests<TMlasTester>();
        });
    return 1;
  }
};

// Some short Execute may not need to distinguish each parameters,
// because they finish quickly, and may disturb others by inject too many small tests.
// Register it as whole using following helper.
template <typename TMlasTester>
class MlasDirectShortExecuteTests : public MlasTestFixture<TMlasTester> {
 public:
  void TestBody() override {
    MlasTestFixture<TMlasTester>::mlas_tester->ExecuteShort();
  }

  static size_t RegisterShortExecute() {
    testing::RegisterTest(
        TMlasTester::GetTestSuiteName(),
        "ShortExecute",
        nullptr,
        nullptr,
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<TMlasTester>* {
          return new MlasDirectShortExecuteTests<TMlasTester>();
        });
    return 1;
  }
};

inline
void ReorderInputNchw(const int64_t* input_shape, const float* S, float* D) {
  const int64_t nchwc_block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  int64_t batch_count = input_shape[0];
  int64_t channel_count = input_shape[1];
  int64_t nchwc_channel_count = (channel_count + nchwc_block_size - 1) & ~(nchwc_block_size - 1);
  int64_t spatial_count = input_shape[2] * input_shape[3];
  for (int64_t n = 0; n < batch_count; n++) {
    MlasReorderInputNchw(S, D, static_cast<size_t>(channel_count), static_cast<size_t>(spatial_count));
    S += spatial_count * channel_count;
    D += spatial_count * nchwc_channel_count;
  }
}
