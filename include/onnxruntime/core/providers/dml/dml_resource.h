// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/resource.h"

#define ORT_DML_RESOUCE_VERSION 1

enum DmlResource : int {
  dml_device_t = dml_resource_offset,
  d3d12_device_t,
  cmd_list_t,
  cmd_recorder_t
};
