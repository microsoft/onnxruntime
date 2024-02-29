// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_NVTX_PROFILE
#include "nvtx_profile.h"
#include "core/common/common.h"
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

namespace onnxruntime {
namespace profile {

void NvtxRangeCreator::BeginImpl() {
  // enable only for debug builds because this function is for profiling only.
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  range_id_ = nvtxRangeStartEx(&eventAttrib);
}

void NvtxRangeCreator::EndImpl() {
  // enable only for debug builds because this function is for profiling only.
  nvtxRangeEnd(range_id_);
}

void NvtxNestedRangeCreator::BeginImpl() {
  // enable only for debug builds because this function is for profiling only.
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  nvtxRangePushEx(&eventAttrib);
}

void NvtxNestedRangeCreator::EndImpl() {
  // enable only for debug builds because this function is for profiling only.
  nvtxRangePop();
}

void NvtxMarkerCreator::Mark() {
  // enable only for debug builds because this function is for profiling only.
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color_);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message_.c_str();

  nvtxMarkEx(&eventAttrib);
}

}  // namespace profile
}  // namespace onnxruntime

#endif
