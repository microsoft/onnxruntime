// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include "QnnTypes.h"

#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace htp_usr_drv {

// Search and get the path of HtpUsrDrv.dll.
std::string GetHtpUsrDrvPath();

// Get the version of HtpUsrDrv.dll.
Qnn_Version_t GetHtpUsrDrvVersion();

// Get whether QNN backend is switched to usr driver path.
Ort::Status IsHtpUsrDrvEnabled(const std::string& backend_lib_dir, const uint32_t htp_arch, bool& enabled);

}  // namespace htp_usr_drv
}  // namespace qnn
}  // namespace onnxruntime
