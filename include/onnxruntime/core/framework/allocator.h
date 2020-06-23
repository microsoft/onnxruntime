// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <map>
#include <string>
#include <cstring>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/fence.h"
#include "core/session/onnxruntime_c_api.h"

// Struct to represent a physical device.
struct OrtDevice {
  using DeviceType = int8_t;
  using MemoryType = int8_t;
  using DeviceId = int16_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;  //CUDA or HIP
  static const DeviceType FPGA = 2;

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType CUDA_PINNED = 1;
    static const MemoryType HIP_PINNED = 2;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_)
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_) {}

  constexpr OrtDevice() : OrtDevice(CPU, MemType::DEFAULT, 0) {}

  DeviceType Type() const {
    return device_type;
  }

  MemoryType MemType() const {
    return memory_type;
  }

  DeviceId Id() const {
    return device_id;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "Device:["
         << "DeviceType:" << static_cast<int>(device_type)
         << " MemoryType:" << static_cast<int>(memory_type)
         << " DeviceId:" << device_id
         << "]";
    return ostr.str();
  }

 private:
  // Device type.
  DeviceType device_type;

  // Memory type.
  MemoryType memory_type;

  // Device index.
  DeviceId device_id;
};

inline bool operator==(const OrtDevice& left, const OrtDevice& other) {
  return left.Id() == other.Id() && left.MemType() == other.MemType() && left.Type() == other.Type();
}

inline bool operator!=(const OrtDevice& left, const OrtDevice& other) {
  return !(left == other);
}

struct OrtMemoryInfo {
  OrtMemoryInfo() = default;  // to allow default construction of Tensor

  // use string for name, so we could have customized allocator in execution provider.
  const char* name = nullptr;
  int id = -1;
  OrtMemType mem_type = OrtMemTypeDefault;
  OrtAllocatorType alloc_type = Invalid;
  OrtDevice device;

  constexpr OrtMemoryInfo(const char* name_, OrtAllocatorType type_, OrtDevice device_ = OrtDevice(), int id_ = 0,
                          OrtMemType mem_type_ = OrtMemTypeDefault)
#if ((defined(__GNUC__) && __GNUC__ > 4) || defined(__clang__))
      // this causes a spurious error in CentOS gcc 4.8 build so disable if GCC version < 5
      __attribute__((nonnull))
#endif
      : name(name_),
        id(id_),
        mem_type(mem_type_),
        alloc_type(type_),
        device(device_) {
  }

  // To make OrtMemoryInfo become a valid key in std map
  bool operator<(const OrtMemoryInfo& other) const {
    if (alloc_type != other.alloc_type)
      return alloc_type < other.alloc_type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (id != other.id)
      return id < other.id;

    return strcmp(name, other.name) < 0;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "OrtMemoryInfo:["
         << "name:" << name
         << " id:" << id
         << " OrtMemType:" << mem_type
         << " OrtAllocatorType:" << alloc_type
         << " " << device.ToString()
         << "]";
    return ostr.str();
  }
};

inline bool operator==(const OrtMemoryInfo& left, const OrtMemoryInfo& other) {
  return left.mem_type == other.mem_type &&
         left.alloc_type == other.alloc_type &&
         left.id == other.id &&
         strcmp(left.name, other.name) == 0;
}

inline bool operator!=(const OrtMemoryInfo& lhs, const OrtMemoryInfo& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info);

namespace onnxruntime {
constexpr const char* CPU = "Cpu";
constexpr const char* CUDA = "Cuda";
constexpr const char* CUDA_PINNED = "CudaPinned";
constexpr const char* MIGRAPHX = "MIGraphX";
constexpr const char* MIGRAPHX_PINNED = "MIGraphXPinned";
constexpr const char* TRT = "Tensorrt";
constexpr const char* TRT_PINNED = "TensorrtPinned";

// forward declaration
class SessionState;

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

class IAllocator {
 public:
  IAllocator(const OrtMemoryInfo& info) : memory_info_(info) {}
  virtual ~IAllocator() = default;
  /**
  @remarks Use SafeInt when calculating the size of memory to allocate using Alloc.
  */
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  const OrtMemoryInfo& Info() const { return memory_info_; };

  /**
     optional CreateFence interface, as provider like DML has its own fence
  */
  virtual FencePtr CreateFence(const SessionState* /*unused*/) { return nullptr; }

  static bool CalcMemSizeForArray(size_t nmemb, size_t size, size_t* out) noexcept {
    return CalcMemSizeForArrayWithAlignment(nmemb, size, 0, out);
  }

  /**
  * Calculate the memory size for an array. The size is bounds checked using SafeInt. 
   * \tparam alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   */
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept ORT_MUST_USE_RESULT;

  /**
   * https://cwe.mitre.org/data/definitions/190.html
   * \param alignment must be power of 2
   * \param nmemb Number of members or elements in the array
   * \param size Size of each element
   * \param out Total size required after any alignment is applied
   * \return true, successful. false, overflow
   * \remarks This was the original API and was implemented in the header. Replaced with the above version 
   *          implemented in the .cc file so that the SafeInt dependency is internal.
   */
  template <size_t alignment>
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept ORT_MUST_USE_RESULT;

  /**
   * allocate memory for an array which has nmemb items of data, each size bytes long
   */
  void* AllocArray(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArray(nmemb, size, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
 * allocate memory for an array which has nmemb items of data, each size bytes long
 */
  template <size_t alignment>
  void* AllocArrayWithAlignment(size_t nmemb, size_t size) {
    size_t len;
    if (!CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, &len))
      return nullptr;
    return Alloc(len);
  }

  /**
     Create a std::unique_ptr that is allocated and freed by the provided IAllocator.
     @param allocator The allocator.
     @param count_or_bytes The exact bytes to allocate if T is void, otherwise the number of elements to allocate.
     @returns std::unique_ptr with allocated memory and deleter.
  */
  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes) {
    if (allocator == nullptr) return nullptr;
    // for now limit to fundamental types. we could support others, but to do so either we or the caller
    // needs to call the dtor for the objects, for buffers allocated on device we don't have destructor
    //static_assert(std::is_fundamental<T>::value, "Fundamental type required as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      if (!CalcMemSizeForArray(count_or_bytes,
                               sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type),
                               &alloc_size)) return nullptr;
    }

    return IAllocatorUniquePtr<T>{
        static_cast<T*>(allocator->Alloc(alloc_size)),  // allocate
        [=](T* ptr) {                                   // capture 'allocator' by value so it's always valid
          allocator->Free(ptr);
        }};
  }

 private:
  OrtMemoryInfo memory_info_;
};

template <size_t alignment>
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept {
  return CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out);
}

/**
   The resource allocator on a physical device.
   This allocator will directly allocate resource from system call
*/
class IDeviceAllocator : public IAllocator {
 public:
  IDeviceAllocator(const OrtMemoryInfo& info) : IAllocator(info) {}
  ~IDeviceAllocator() override = default;
  void* Alloc(size_t size) override = 0;
  void Free(void* p) override = 0;
};

class CPUAllocator : public IDeviceAllocator {
 public:
  explicit CPUAllocator(const OrtMemoryInfo& memory_info) : IDeviceAllocator(memory_info) {}

  CPUAllocator() : IDeviceAllocator(OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
class MiMallocAllocator : public IDeviceAllocator {
 public:
  explicit MiMallocAllocator(const OrtMemoryInfo& memory_info) : IDeviceAllocator(memory_info) {}
  MiMallocAllocator() : IDeviceAllocator(OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

#endif

#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
using TAllocator = MiMallocAllocator;
#else
using TAllocator = CPUAllocator;
#endif

using AllocatorPtr = std::shared_ptr<IAllocator>;

}  // namespace onnxruntime
