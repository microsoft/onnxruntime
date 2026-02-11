// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <string>
#include <type_traits>

#include "QnnCommon.h"
#include "System/QnnSystemContext.h"

#include "core/providers/qnn/builder/qnn_cache_compatibility_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/htp_usr_drv_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

int RedirectStdout() {
  // Save the current stdout file descriptor.
#ifdef _WIN32
  int saved_stdout_fd = _dup(_fileno(stdout));
#else
  int saved_stdout_fd = dup(fileno(stdout));
#endif
  if (saved_stdout_fd == -1) {
    return saved_stdout_fd;
  }

  // Redirect stdout to NUL.
#ifdef _WIN32
  const char* filename = "NUL";
#else
  const char* filename = "/dev/null";
#endif
#ifdef _WIN32
  FILE* redirected = nullptr;
  if (freopen_s(&redirected, filename, "w", stdout) != 0 || redirected == nullptr) {
    _close(saved_stdout_fd);
#else
  if (freopen(filename, "w", stdout) == nullptr) {
    close(saved_stdout_fd);
#endif
    return saved_stdout_fd;
  }

  return saved_stdout_fd;
}

void RestoreStdout(int saved_stdout_fd) {
  if (saved_stdout_fd == -1) {
    return;
  }

  fflush(stdout);
#ifdef _WIN32
  _dup2(saved_stdout_fd, _fileno(stdout));
  _close(saved_stdout_fd);
#else
  dup2(saved_stdout_fd, fileno(stdout));
  close(saved_stdout_fd);
#endif
}

}  // namespace

Ort::Status QnnCacheCompatibilityManager::CreateFakeContextBinary(std::unique_ptr<unsigned char[]>& context_buffer,
                                                                  uint64_t& context_buffer_size) {
  const auto& qnn_interface = qnn_backend_manager_->GetQnnInterface();
  const auto& backend_handle = qnn_backend_manager_->GetQnnBackendHandle();
  const auto& device_handle = qnn_backend_manager_->GetQnnDeviceHandle();

  Qnn_ErrorHandle_t result = 0;

  // Create context.
  Qnn_ContextHandle_t context_raw = nullptr;
  result = qnn_interface.contextCreate(backend_handle, device_handle, nullptr, &context_raw);
  RETURN_IF(result != QNN_CONTEXT_NO_ERROR,
            ("Failed to create context. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  std::unique_ptr<std::remove_pointer_t<Qnn_ContextHandle_t>, std::function<void(Qnn_ContextHandle_t)>>
      context(context_raw, [&qnn_interface](Qnn_ContextHandle_t context) { qnn_interface.contextFree(context, nullptr); });

  // Create graph.
  Qnn_GraphHandle_t graph = nullptr;
  result = qnn_interface.graphCreate(context.get(), "fake_context_bin", nullptr, &graph);
  RETURN_IF(result != QNN_GRAPH_NO_ERROR,
            ("Failed to create graph. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Create input tensor.
  QnnTensorWrapper input_tensor_wrapper("input",
                                        QNN_TENSOR_TYPE_APP_WRITE,
                                        QNN_DATATYPE_FLOAT_16,
                                        QnnQuantParamsWrapper(),
                                        {1, 32, 32, 3});
  Qnn_Tensor_t& input_tensor = input_tensor_wrapper.GetQnnTensor();
  result = qnn_interface.tensorCreateGraphTensor(graph, &input_tensor);
  RETURN_IF(result != QNN_TENSOR_NO_ERROR,
            ("Failed to create tensor. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Create output tensor.
  QnnTensorWrapper output_tensor_wrapper("output",
                                         QNN_TENSOR_TYPE_APP_READ,
                                         QNN_DATATYPE_FLOAT_16,
                                         QnnQuantParamsWrapper(),
                                         {1, 32, 32, 3});
  Qnn_Tensor_t& output_tensor = output_tensor_wrapper.GetQnnTensor();
  result = qnn_interface.tensorCreateGraphTensor(graph, &output_tensor);
  RETURN_IF(result != QNN_TENSOR_NO_ERROR,
            ("Failed to create tensor. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Create node.
  QnnOpConfigWrapper op_config_wrapper("node",
                                       QNN_OP_PACKAGE_NAME_QTI_AISW,
                                       QNN_OP_RELU,
                                       {input_tensor},
                                       {output_tensor},
                                       {});
  const Qnn_OpConfig_t& op_config = op_config_wrapper.GetQnnOpConfig();
  result = qnn_interface.backendValidateOpConfig(backend_handle, op_config);
  RETURN_IF(result != QNN_SUCCESS,
            ("Failed to validate node. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());
  result = qnn_interface.graphAddNode(graph, op_config);
  RETURN_IF(result != QNN_GRAPH_NO_ERROR,
            ("Failed to add node. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Finalize graph.
  int saved_stdout_fd = RedirectStdout();  // Redirect stdout to avoid confusing user.
  result = qnn_interface.graphFinalize(graph, nullptr, nullptr);
  RestoreStdout(saved_stdout_fd);
  RETURN_IF(result != QNN_GRAPH_NO_ERROR,
            ("Failed to finalize graph. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Get binary size.
  uint64_t required_buffer_size = 0;
  result = qnn_interface.contextGetBinarySize(context.get(), &required_buffer_size);
  RETURN_IF(result != QNN_CONTEXT_NO_ERROR,
            ("Failed to get context binary size. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());

  // Allocate buffer.
  context_buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  RETURN_IF(context_buffer == nullptr, "Failed to allocate buffer for context binary.");

  // Get binary.
  result = qnn_interface.contextGetBinary(context.get(),
                                          reinterpret_cast<void*>(context_buffer.get()),
                                          required_buffer_size,
                                          &context_buffer_size);
  RETURN_IF(result != QNN_CONTEXT_NO_ERROR,
            ("Failed to get context binary. Error: " + utils::GetQnnErrorMessage(qnn_interface, result)).c_str());
  RETURN_IF(required_buffer_size < context_buffer_size, "Written buffer size exceeded allocated buffer size.");

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::GetCompatibilityInfo(unsigned char* context_buffer,
                                                               uint64_t context_buffer_size,
                                                               QnnCompatibilityInfo& info) {
  const auto& qnn_sys_interface = qnn_backend_manager_->GetQnnSystemInterface();
  RETURN_IF(qnn_sys_interface.systemContextCreate == nullptr ||
                qnn_sys_interface.systemContextGetBinaryInfo == nullptr ||
                qnn_sys_interface.systemContextFree == nullptr,
            "Failed to get valid QnnSystem function pointers.");

  QnnSystemContext_Handle_t sys_ctx_raw_handle = nullptr;
  RETURN_IF(qnn_sys_interface.systemContextCreate(&sys_ctx_raw_handle) != QNN_SUCCESS,
            "Failed to create system handle.");

  std::unique_ptr<std::remove_pointer_t<QnnSystemContext_Handle_t>, std::function<void(QnnSystemContext_Handle_t)>>
      sys_ctx_handle(
          sys_ctx_raw_handle,
          [&qnn_sys_interface](QnnSystemContext_Handle_t sys_ctx_handle) {
            qnn_sys_interface.systemContextFree(sys_ctx_handle);
          });

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  RETURN_IF(qnn_sys_interface.systemContextGetBinaryInfo(sys_ctx_handle.get(),
                                                         static_cast<void*>(context_buffer),
                                                         context_buffer_size,
                                                         &binary_info,
                                                         &binary_info_size) != QNN_SUCCESS,
            "Failed to get context binary info.");
  RETURN_IF(binary_info == nullptr, "Failed to get context binary info.");

  info.backend_id = qnn_backend_manager_->GetBackendId();
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    info.backend_api_version = binary_info->contextBinaryInfoV3.backendApiVersion;
    info.context_blob_version = binary_info->contextBinaryInfoV3.contextBlobVersion;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    info.backend_api_version = binary_info->contextBinaryInfoV2.backendApiVersion;
    info.context_blob_version = binary_info->contextBinaryInfoV2.contextBlobVersion;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    info.backend_api_version = binary_info->contextBinaryInfoV1.backendApiVersion;
    info.context_blob_version = binary_info->contextBinaryInfoV1.contextBlobVersion;
  } else {
    return MAKE_FAIL("Unknown context binary info version.");
  }

  // Although HTP arch can be extracted from V3 context binary info, unify to query through backend manager again.
  QnnHtpDevice_Arch_t htp_arch;
  RETURN_IF_ERROR(qnn_backend_manager_->GetHtpArch(htp_arch));
  info.htp_arch = static_cast<uint32_t>(htp_arch);

  RETURN_IF_ERROR(htp_usr_drv::IsHtpUsrDrvEnabled(qnn_backend_manager_->GetBackendLibDir(),
                                                  info.htp_arch,
                                                  info.is_htp_usr_drv));

  // There is no way to query HNRD's backend API version with current APIs. Fortunately, since backend API versions are
  // bumped along with SDK verions, adopt SDK versions in HNRD scenarios, which can be extracted from driver's file
  // version. Note that backend API versions are not entirely discarded to align with QNN behavior.
  if (info.is_htp_usr_drv) {
    info.sdk_version = htp_usr_drv::GetHtpUsrDrvVersion();
    RETURN_IF(info.sdk_version.major == 0 && info.sdk_version.minor == 0 && info.sdk_version.patch == 0,
              "Failed to get HtpUsrDrv file version.");
  } else {
    const std::string& sdk_version = qnn_backend_manager_->GetSdkVersion();
    auto split_versions = utils::SplitString(sdk_version, ".");
    RETURN_IF(split_versions.size() != 4 || split_versions[0].substr(0, 1) != "v",
              "Expected SDK version in vMajor.Minor.Patch.BuildId format.");
    info.sdk_version = QnnVersion{static_cast<uint32_t>(std::stoi(std::string(split_versions[0].substr(1)))),
                                  static_cast<uint32_t>(std::stoi(std::string(split_versions[1]))),
                                  static_cast<uint32_t>(std::stoi(std::string(split_versions[2])))};
  }

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::ValidateCompatibilityInfo(const QnnCompatibilityInfo& info,
                                                                    OrtCompiledModelCompatibility& compatibility) {
  compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  // A fake context binary is created here to acquire context blob version which is not querable through current APIs
  // but can only be extracted from context binary info.
  // Remove this context binary if context blob version could be directly queried through new APIs.
  std::unique_ptr<unsigned char[]> context_buffer;
  uint64_t context_buffer_size;
  RETURN_IF_ERROR(CreateFakeContextBinary(context_buffer, context_buffer_size));

  // Get runtime info to be compared with the given one.
  QnnCompatibilityInfo runtime_info;
  RETURN_IF_ERROR(GetCompatibilityInfo(context_buffer.get(), context_buffer_size, runtime_info));

  auto is_htp_arch_compatible = [cb_htp_arch = info.htp_arch, rt_htp_arch = runtime_info.htp_arch] {
    if (cb_htp_arch < rt_htp_arch) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    } else if (cb_htp_arch == rt_htp_arch) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
    } else {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  };

  // The comparison order is as below:
  //   1. backend ID
  //   2. backend API version (if no HNRD involved) / SDK version (if HNRD involved)
  //   3. context blob bersion
  //   4. HTP arch
  //
  // Note that since the fake context binary approach is adopted, there is no need to compare context blob version with
  // SDK and HNRD separately in HNRD paths. The fake context binary has context blob version tagged with the smaller
  // one between SDK and HNRD, and thus comparing to it directly reflect comparing to both SDK and HNRD.

  if (info.backend_id != runtime_info.backend_id) {
    compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return Ort::Status();
  }

  // Deliberately leave all branches without reduction for readibility and maintainability.
  if (!info.is_htp_usr_drv && !runtime_info.is_htp_usr_drv) {
    if (info.backend_api_version <= runtime_info.backend_api_version) {
      if (info.context_blob_version <= runtime_info.context_blob_version) {
        compatibility = is_htp_arch_compatible();
      } else {
        compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      }
    } else {
      compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  } else if (!info.is_htp_usr_drv && runtime_info.is_htp_usr_drv) {
    if (info.sdk_version <= runtime_info.sdk_version) {
      if (info.context_blob_version <= runtime_info.context_blob_version) {
        compatibility = is_htp_arch_compatible();
      } else {
        compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      }
    } else {
      compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  } else if (info.is_htp_usr_drv && !runtime_info.is_htp_usr_drv) {
    // Unexpected usage of context binary generated by user driver path but executing on traditional path.
    compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  } else {  // (info.is_htp_usr_drv && runtime_info.is_htp_usr_drv)
    if (info.sdk_version <= runtime_info.sdk_version) {
      if (info.context_blob_version <= runtime_info.context_blob_version) {
        compatibility = is_htp_arch_compatible();
      } else {
        compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      }
    } else {
      compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
