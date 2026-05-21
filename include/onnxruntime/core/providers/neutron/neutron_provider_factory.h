// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "onnxruntime_c_api.h"

typedef enum {
  NEUTRON_FLAG_USE_NONE = 0x000,
  NEUTRON_FLAG_USE_ARENA = 0x001,

  NEUTRON_FLAG_LAST = NEUTRON_FLAG_USE_ARENA,
} NeutronFlags;

typedef struct {
  uint32_t flags;
  bool offline_packed;
  bool neutron_op_only;
} NeutronProviderOptions;

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Neutron,
                          _In_ OrtSessionOptions* options, NeutronProviderOptions neutron_options);

#ifdef __cplusplus
}
#endif
