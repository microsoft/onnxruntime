#pragma once

#include <gsl/gsl>
#include <vector>

#include "onnxruntime_c_api.h"

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

#define RETURN_FALSE_AND_PRINT_IF_ERROR(fn, ort_api)   \
  do {                                                 \
    OrtStatus* status = (fn);                          \
    if (status != nullptr) {                           \
      std::cerr << (ort_api).GetErrorMessage(status) << std::endl ;  \
      return false;                                    \
    }                                                  \
  } while (0)    

struct OrtArrayOfConstObjects {
  OrtArrayOfConstObjects() = default;
  explicit OrtArrayOfConstObjects(OrtTypeTag object_type) : object_type(object_type) {}
  OrtArrayOfConstObjects(OrtTypeTag object_type, size_t size, const void* initial_val = nullptr)
      : object_type(object_type), storage(size, initial_val) {}

  OrtTypeTag object_type = OrtTypeTag::ORT_TYPE_TAG_Void;
  std::vector<const void*> storage;
};

// Convert an OrtArrayOfConstObjects into a span of Ort___ pointers.
template <typename T>
static void GetSpanFromArrayOfConstObjects(const OrtArrayOfConstObjects* ort_array,
                                           /*out*/ gsl::span<const T* const>& span) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t size = 0;
  ASSERT_ORTSTATUS_OK(ort_api.ArrayOfConstObjects_GetSize(ort_array, &size));

  const void* const* raw_data = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.ArrayOfConstObjects_GetData(ort_array, &raw_data));

  auto data = reinterpret_cast<const T* const*>(raw_data);
  span = gsl::span<const T* const>(data, size);
}

// Helper to release a C API Ort object at the end of its scope.
// Useful when not using the public C++ API.
//    Example:
//      {
//        OrtTensorTypeAndShapeInfo* info = nullptr;
//        DeferOrtRelease<OrtTensorTypeAndShapeInfo> defer_release(&info, c_api.ReleaseTensorTypeAndShapeInfo);
//        ...
//      } /* Release is called at end of scope*/
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** obj_ptr, std::function<void(T*)> release_func) : obj_ptr_(obj_ptr), release_func_(release_func) {}
  ~DeferOrtRelease() {
    if (obj_ptr_ != nullptr && *obj_ptr_ != nullptr) {
      release_func_(*obj_ptr_);
      *obj_ptr_ = nullptr;
    }
  }
  T** obj_ptr_ = nullptr;
  std::function<void(T*)> release_func_ = nullptr;
};