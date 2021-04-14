// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"
#ifdef __cplusplus
#include <vector>
#include <string>
struct ProviderInfo_OpenVINO {
  virtual std::vector<std::string> GetAvailableDevices() const = 0;
};
#endif
