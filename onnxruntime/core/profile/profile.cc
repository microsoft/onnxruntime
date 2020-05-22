// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/profile/profile.h"
#include "core/common/common.h"
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_Win32)
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#endif

namespace onnxruntime {
namespace profile {

void NvtxRangeCreator::BeginImpl() {
// enable only for debug builds because this function is for profiling only.
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  range_id_ = nvtxRangeStartEx(&eventAttrib);
#endif
}

void NvtxRangeCreator::EndImpl() {
// enable only for debug builds because this function is for profiling only.
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  nvtxRangeEnd(range_id_);
#endif
}

void NvtxNestedRangeCreator::BeginImpl() {
// enable only for debug builds because this function is for profiling only.
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();
  
  nvtxRangePushEx(&eventAttrib);
#endif
}

void NvtxNestedRangeCreator::EndImpl() {
// enable only for debug builds because this function is for profiling only.
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  nvtxRangePop();
#endif
}

void NvtxMarkerCreator::Mark() {
// enable only for debug builds because this function is for profiling only.
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  nvtxEventAttributes_t eventAttrib; 
  eventAttrib.version = NVTX_VERSION; 
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  nvtxMarkEx(&eventAttrib); 
#endif
}


}  // namespace contrib
}  // namespace onnxruntime
