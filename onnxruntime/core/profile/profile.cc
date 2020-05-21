// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/profile/profile.h"
#include "core/common/common.h"
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>

namespace onnxruntime {
namespace profile {

void NvtxRangeCreator::BeginImpl() {
// enable only for debug builds because this function is for profiling only.
#ifndef NDEBUG 
#else
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color_;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  range_id_ = nvtxRangeStartEx(&eventAttrib);
#endif
}

void NvtxRangeCreator::EndImpl() {
  nvtxRangeEnd(range_id_);
}

void NvtxNestedRangeCreator::BeginImpl() {
// enable only for debug builds because this function is for profiling only.
#ifndef NDEBUG 
#else
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color_;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();
  
  nvtxRangePushEx(&eventAttrib);
#endif
}

void NvtxNestedRangeCreator::EndImpl() {
#ifndef NDEBUG 
#else
  nvtxRangePop();
#endif
}

void NvtxMarkerCreator::Mark() {
#ifndef NDEBUG 
#else
  nvtxEventAttributes_t eventAttrib; 
  eventAttrib.version = NVTX_VERSION; 
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color_;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  nvtxMarkEx(&eventAttrib); 
#endif
}


}  // namespace contrib
}  // namespace onnxruntime
