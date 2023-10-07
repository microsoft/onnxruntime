// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct OrtDmlProviderOptions {
  int device_id = 0;
  bool skip_software_device_check = false;
  bool disable_metacommands = false;
  bool enable_dynamic_graph_fusion = false;
};
