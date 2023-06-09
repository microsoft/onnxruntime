// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_backend_manager.h"
#include "qnn_model.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "QnnOpDef.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "DSP/QnnDspCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "core/common/gsl.h"
#include "core/framework/endian_utils.h"
#include "core/common/logging/capture.h"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace onnxruntime {
namespace qnn {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList,
                                                                uint32_t* numProviders);

constexpr const char* QNN_PROVIDER = "ORTQNNEP";

template <typename F, class T>
Status QnnBackendManager::GetQnnInterfaceProviders(const char* lib_path,
                                                   const char* interface_provider_name,
                                                   void** backend_lib_handle,
                                                   T*** interface_providers,
                                                   uint32_t& num_providers) {
  std::string error_msg;
  *backend_lib_handle = LoadLib(lib_path,
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_LOCAL),
                                error_msg);
  ORT_RETURN_IF(nullptr == *backend_lib_handle, "Unable to load backend, error: ", error_msg, " ", DlError());

  // Get QNN Interface providers
  F GetInterfaceProviders{nullptr};
  GetInterfaceProviders = ResolveSymbol<F>(*backend_lib_handle, interface_provider_name, *logger_);
  ORT_RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");

  auto result = GetInterfaceProviders((const T***)interface_providers, &num_providers);
  ORT_RETURN_IF((QNN_SUCCESS != result || nullptr == *interface_providers || 0 == num_providers),
                "Failed to get QNN providers.");

  return Status::OK();
}

Status QnnBackendManager::LoadBackend() {
  QnnInterface_t** interface_providers{nullptr};
  uint32_t num_providers{0};
  auto rt = GetQnnInterfaceProviders<QnnInterfaceGetProvidersFn_t,
                                     QnnInterface_t>(backend_path_.c_str(),
                                                     "QnnInterface_getProviders",
                                                     &backend_lib_handle_,
                                                     &interface_providers,
                                                     num_providers);
  ORT_RETURN_IF_ERROR(rt);

  bool found_valid_interface{false};
  LOGS_DEFAULT(VERBOSE) << "QNN_API_VERSION_MAJOR: " << QNN_API_VERSION_MAJOR
                        << " QNN_API_VERSION_MINOR: " << QNN_API_VERSION_MINOR;
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    LOGS_DEFAULT(VERBOSE) << "interface_providers major: " << interface_providers[pIdx]->apiVersion.coreApiVersion.major
                          << " interface_providers minor: " << interface_providers[pIdx]->apiVersion.coreApiVersion.minor;
    if (QNN_API_VERSION_MAJOR == interface_providers[pIdx]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= interface_providers[pIdx]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_interface_ = interface_providers[pIdx]->QNN_INTERFACE_VER_NAME;
      auto backend_id = interface_providers[pIdx]->backendId;
      if (QNN_BACKEND_ID_DSP == backend_id || QNN_BACKEND_ID_HTP == backend_id) {
        is_npu_backend_ = true;
      }
      LOGS_DEFAULT(INFO) << "Found valid interface, version: " << QNN_API_VERSION_MAJOR
                         << "." << QNN_API_VERSION_MINOR
                         << " backend provider name: " << interface_providers[pIdx]->providerName
                         << " backend id: " << backend_id;
      break;
    }
  }

  ORT_RETURN_IF_NOT(found_valid_interface, "Unable to find a valid interface.");

  return Status::OK();
}

Status QnnBackendManager::LoadQnnSystemLib() {
#ifdef _WIN32
  std::string system_lib_file = "QnnSystem.dll";
#else
  std::string system_lib_file = "libQnnSystem.so";
#endif  // #ifdef _WIN32
  std::filesystem::path lib_file_path(backend_path_.c_str());
  std::string sys_file_path(lib_file_path.remove_filename().string() + system_lib_file);
  QnnSystemInterface_t** system_interface_providers{nullptr};
  uint32_t num_providers = 0;
  auto rt = GetQnnInterfaceProviders<QnnSystemInterfaceGetProvidersFn_t,
                                     QnnSystemInterface_t>(sys_file_path.c_str(),
                                                           "QnnSystemInterface_getProviders",
                                                           &system_lib_handle_,
                                                           &system_interface_providers,
                                                           num_providers);
  ORT_RETURN_IF_ERROR(rt);

  bool found_valid_interface{false};
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    LOGS_DEFAULT(VERBOSE) << "system_interface_providers major: " << system_interface_providers[pIdx]->systemApiVersion.major
                          << " system_interface_providers minor: " << system_interface_providers[pIdx]->systemApiVersion.minor;
    int64_t systems_version_major = static_cast<int64_t>(system_interface_providers[pIdx]->systemApiVersion.major);
    int64_t systems_version_minor = static_cast<int64_t>(system_interface_providers[pIdx]->systemApiVersion.minor);
    if (systems_version_major == QNN_SYSTEM_API_VERSION_MAJOR &&
        systems_version_minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      found_valid_interface = true;
      qnn_sys_interface_ = system_interface_providers[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
      LOGS_DEFAULT(INFO) << "Found valid system interface, version: " << QNN_API_VERSION_MAJOR
                         << "." << QNN_API_VERSION_MINOR
                         << " backend provider name: " << system_interface_providers[pIdx]->providerName;
      break;
    }
  }

  ORT_RETURN_IF_NOT(found_valid_interface, "Unable to find a valid system interface.");

  return Status::OK();
}

void QnnLogging(const char* format,
                QnnLog_Level_t level,
                uint64_t timestamp,
                va_list argument_parameter) {
  ORT_UNUSED_PARAMETER(level);
  ORT_UNUSED_PARAMETER(timestamp);

  // Always output Qnn log as Ort verbose log
  const auto& logger = ::onnxruntime::logging::LoggingManager::DefaultLogger();
  const auto severity = ::onnxruntime::logging::Severity::kVERBOSE;
  const auto data_type = ::onnxruntime::logging::DataType::SYSTEM;
  if (logger.OutputIsEnabled(severity, data_type)) {
    ::onnxruntime::logging::Capture(logger,
                                    severity,
                                    ::onnxruntime::logging::Category::onnxruntime,
                                    data_type,
                                    ORT_WHERE)
        .ProcessPrintf(format, argument_parameter);
  }
}

void QnnBackendManager::InitializeQnnLog() {
  // Set Qnn log level align with Ort log level
  QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_WARN;
  auto ort_log_level = logger_->GetSeverity();
  switch (ort_log_level) {
    case logging::Severity::kVERBOSE:
      qnn_log_level = QNN_LOG_LEVEL_DEBUG;
      break;
    case logging::Severity::kINFO:
      qnn_log_level = QNN_LOG_LEVEL_INFO;
      break;
    case logging::Severity::kWARNING:
      qnn_log_level = QNN_LOG_LEVEL_WARN;
      break;
    case logging::Severity::kERROR:
      qnn_log_level = QNN_LOG_LEVEL_ERROR;
      break;
    default:
      break;
  }
  LOGS(*logger_, VERBOSE) << "Set Qnn log level: " << qnn_log_level;

  if (QNN_SUCCESS != qnn_interface_.logCreate(QnnLogging, qnn_log_level, &log_handle_)) {
    LOGS(*logger_, WARNING) << "Unable to initialize logging in the QNN backend.";
  }
}

Status QnnBackendManager::InitializeBackend() {
  if (true == backend_initialized_) {
    LOGS_DEFAULT(INFO) << "Backend intialized already.";
    return Status::OK();
  }

  auto result = qnn_interface_.backendCreate(log_handle_, (const QnnBackend_Config_t**)backend_config_, &backend_handle_);
  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result, "Failed to initialize backend");

  backend_initialized_ = true;
  return Status::OK();
}

Status QnnBackendManager::ShutdownBackend() {
  if (false == backend_initialized_) {
    return Status::OK();
  }

  if (nullptr != qnn_interface_.backendFree) {
    ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != qnn_interface_.backendFree(backend_handle_),
                  "Failed to shutdown backend!");
  }

  backend_initialized_ = false;

  return Status::OK();
}

bool QnnBackendManager::IsDevicePropertySupported() {
  if (nullptr != qnn_interface_.propertyHasCapability) {
    auto rt = qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == rt || QNN_PROPERTY_ERROR_UNKNOWN_KEY == rt) {
      LOGS_DEFAULT(INFO) << "Device property not supported or unknown to backend.";
      return false;
    }
  }

  return true;
}

Status QnnBackendManager::CreateDevice() {
  if (true == device_created_) {
    LOGS_DEFAULT(INFO) << "Device intialized already.";
    return Status::OK();
  }

  // Create device if its property supported
  if (!IsDevicePropertySupported()) {
    LOGS_DEFAULT(INFO) << "Skip to create device.";
    return Status::OK();
  }

  LOGS_DEFAULT(INFO) << "Create device.";
  if (nullptr != qnn_interface_.deviceCreate) {
    auto result = qnn_interface_.deviceCreate(log_handle_, nullptr, &device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create device. Error: ", result);
    }
  }
  device_created_ = true;

  return Status::OK();
}

Status QnnBackendManager::ReleaseDevice() {
  if (false == device_created_) {
    return Status::OK();
  }

  if (nullptr != qnn_interface_.deviceFree) {
    auto result = qnn_interface_.deviceFree(device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to release device. Error: ", result);
    }
  }

  device_created_ = false;

  return Status::OK();
}

Status QnnBackendManager::InitializeProfiling() {
  if (ProfilingLevel::OFF == profiling_level_ || ProfilingLevel::INVALID == profiling_level_) {
    LOGS_DEFAULT(INFO) << "Profiling turned off.";
    return Status::OK();
  }

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  if (ProfilingLevel::BASIC == profiling_level_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  } else if (ProfilingLevel::DETAILED == profiling_level_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
  }
  auto result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_backend_handle_);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to create QNN profile!");

  return Status::OK();
}

Status QnnBackendManager::ReleaseProfilehandle() {
  // Free Profiling object if it was created
  if (nullptr != profile_backend_handle_) {
    ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_backend_handle_),
                  "Could not free backend profile handle!");
  }
  profile_backend_handle_ = nullptr;

  return Status::OK();
}

Status QnnBackendManager::CreateContext() {
  if (true == context_created_) {
    LOGS_DEFAULT(INFO) << "Context created already.";
    return Status::OK();
  }

  auto result = qnn_interface_.contextCreate(backend_handle_,
                                             device_handle_,
                                             (const QnnContext_Config_t**)&context_config_,
                                             &context_);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context.");

  context_created_ = true;
  return Status::OK();
}

Status QnnBackendManager::ReleaseContext() {
  if (false == context_created_) {
    return Status::OK();
  }

  auto result = qnn_interface_.contextFree(context_, nullptr);
  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to release context.");

  context_created_ = false;
  return Status::OK();
}

bool QnnBackendManager::IsContextCacheFileExists(const std::string& customer_context_cache_path,
                                                 const std::string& model_description,
                                                 const onnxruntime::PathString& model_pathstring) {
  // Avoid duplicate work
  if (!context_cache_path_.empty()) {
    return ctx_file_exists_;
  }
  model_description_ = model_description;
  // Use user provided context cache file path if exist, otherwise try model_file.onnx.bin by default
  if (customer_context_cache_path.empty()) {
    context_cache_path_ = PathToUTF8String(model_pathstring) + ".bin";
  } else {
    context_cache_path_ = customer_context_cache_path;
  }

  ctx_file_exists_ = std::filesystem::exists(context_cache_path_);

  return ctx_file_exists_;
}

Status WriteInt16ToBinaryFile(std::ofstream& of_stream, uint16_t value) {
  const std::vector<uint16_t> data{value};
  std::vector<unsigned char> data_bytes(sizeof(uint16_t) / sizeof(unsigned char));
  ORT_RETURN_IF_ERROR(onnxruntime::utils::WriteLittleEndian(gsl::make_span(data), gsl::make_span(data_bytes)));
  of_stream.write(reinterpret_cast<char*>(data_bytes.data()), data_bytes.size());
  return Status::OK();
}

Status QnnBackendManager::DumpQnnContext(const std::string& model_name, const std::string& graph_name) {
  if (nullptr == qnn_interface_.contextGetBinarySize ||
      nullptr == qnn_interface_.contextGetBinary) {
    LOGS(*logger_, ERROR) << "Failed to get valid function pointer.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get valid function pointer.");
  }

  uint64_t required_buffer_size(0);
  Qnn_ErrorHandle_t rt = qnn_interface_.contextGetBinarySize(context_, &required_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get QNN context binary size. Error code: " << rt;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get QNN context binary size.");
  }

  std::unique_ptr<unsigned char[]> context_buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  if (nullptr == context_buffer) {
    LOGS(*logger_, ERROR) << "Failed to allocate buffer for context cache.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate buffer for context cache.");
  }

  uint64_t written_buffer_size(0);
  rt = qnn_interface_.contextGetBinary(context_,
                                       reinterpret_cast<void*>(context_buffer.get()),
                                       required_buffer_size,
                                       &written_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get context binary.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get context binary.");
  }

  if (required_buffer_size < written_buffer_size) {
    LOGS(*logger_, ERROR) << "Context written buffer size: " << written_buffer_size
                          << " exceeds allocated buffer size: " << required_buffer_size;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Context written buffer exceeds allocated buffer size.");
  }

  std::ofstream of_stream(context_cache_path_.c_str(), std::ofstream::binary);
  if (!of_stream) {
    LOGS(*logger_, ERROR) << "Failed to open cached context file.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to open context cache file.");
  }

  // Write Ort metadata into context binary file
  uint16_t model_name_length = static_cast<uint16_t>(model_name.length());
  uint16_t graph_name_length = static_cast<uint16_t>(graph_name.length());
  uint16_t model_description_length = static_cast<uint16_t>(model_description_.length());

  // Header: uint16_t(totale_length)|uint16_t(model_name_length)|model_name|uint16_t(graph_name_length)|graph_name|uint16_t(model_description_length)|model_description
  uint16_t header_length = 4 * sizeof(uint16_t) + model_name_length + graph_name_length + model_description_length;
  uint16_t totale_length = header_length + static_cast<uint16_t>(strlen(QNN_PROVIDER));
  of_stream.write(QNN_PROVIDER, strlen(QNN_PROVIDER));

  ORT_RETURN_IF_ERROR(WriteInt16ToBinaryFile(of_stream, header_length));

  ORT_RETURN_IF_ERROR(WriteInt16ToBinaryFile(of_stream, model_name_length));
  of_stream.write(model_name.c_str(), model_name_length);

  ORT_RETURN_IF_ERROR(WriteInt16ToBinaryFile(of_stream, graph_name_length));
  of_stream.write(graph_name.c_str(), graph_name_length);

  ORT_RETURN_IF_ERROR(WriteInt16ToBinaryFile(of_stream, model_description_length));
  of_stream.write(model_description_.c_str(), model_description_length);
  model_description_.clear();

  LOGS(*logger_, VERBOSE) << "Dump metadata with length: " << totale_length;

  of_stream.write(reinterpret_cast<char*>(context_buffer.get()), written_buffer_size);

  LOGS(*logger_, VERBOSE) << "Dump QNN Context completed.";
  return Status::OK();
}

Status QnnBackendManager::LoadCachedQnnContext(QnnModel& qnn_model) {
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  ORT_RETURN_IF(result, "Failed to get valid function pointer.");

  ORT_RETURN_IF(!ctx_file_exists_, "Qnn context binary file not exist for some reason!");

  uint64_t buffer_size{0};
  std::ifstream cache_file(context_cache_path_.c_str(), std::ifstream::binary);
  ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to open cache file.");
  cache_file.seekg(0, cache_file.end);
  buffer_size = cache_file.tellg();
  ORT_RETURN_IF(0 == buffer_size, "Empty cache file encountered.");
  cache_file.seekg(0, cache_file.beg);
  // Skip Ort generated metadata
  if (ort_generated_ctx_cache_) {
    cache_file.seekg(ort_ctx_metadata_length_);
    buffer_size -= ort_ctx_metadata_length_;
  }

  std::unique_ptr<unsigned char[]> buffer = std::make_unique<unsigned char[]>(buffer_size);
  ORT_RETURN_IF(nullptr == buffer, "Failed to allocate memory for cache file.");

  // Load file into buffer
  const auto& read_result = cache_file.read(reinterpret_cast<char*>(buffer.get()), buffer_size);
  cache_file.close();
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context file.");

  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create system handle.");

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size{0};
  rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                     static_cast<void*>(buffer.get()),
                                                     buffer_size,
                                                     &binary_info,
                                                     &binary_info_size);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");

  // binary_info life cycle is here
  // Binary info to graph info
  // retrieve Qnn graph infor from binary info
  ORT_RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV1.graphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV2.graphs;
  }

  ORT_RETURN_IF(graph_count > 1, "Load from Qnn cached context only support 1 sub-graph.");
  ORT_RETURN_IF(graphs_info == nullptr, "Failed to get graph info from Qnn cached context.");

  ORT_RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinary,
                "Invalid function pointer for contextCreateFromBinary.");
  rt = qnn_interface_.contextCreateFromBinary(backend_handle_,
                                              device_handle_,
                                              (const QnnContext_Config_t**)&context_config_,
                                              static_cast<void*>(buffer.get()),
                                              buffer_size,
                                              &context_,
                                              profile_backend_handle_);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create context from binary.");

  ORT_RETURN_IF_ERROR(qnn_model.DeserializeGraphInfoFromBinaryInfo(graphs_info[0]));

  qnn_sys_interface_.systemContextFree(sys_ctx_handle);
  sys_ctx_handle = nullptr;

  ORT_RETURN_IF_ERROR(ExtractBackendProfilingInfo());
  context_created_ = true;

  model_description_.clear();
  model_description_from_ctx_cache_.clear();
  LOGS(*logger_, VERBOSE) << "Load from cached QNN Context completed.";
  return Status::OK();
}

/* \brief: Read string data from binary file with given length
 * \param[in] binary_file - file stream of the binary file
 * \param[out] result_str - string read from binary file
 * \param[out] length - length to read
 */
Status ReadStringFromBinaryFile(std::ifstream& binary_file, std::string& result_str, size_t length) {
  result_str.resize(length);
  const auto& read_result = binary_file.read(result_str.data(), length);
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context binary file.");

  return Status::OK();
}

/* \brief: Read a uint16_t from binary file
 * \param[in] binary_file - file stream of the binary file
 * \param[out] value - uint16_t value
 */
Status ReadInt16FromBinaryFile(std::ifstream& binary_file, uint16_t& value) {
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(sizeof(uint16_t));
  ORT_RETURN_IF(nullptr == buffer, "Failed to allocate memory for buffer.");
  const auto& read_result = binary_file.read(buffer.get(), sizeof(uint16_t));
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context binary file.");

  auto src = gsl::make_span<const unsigned char>(reinterpret_cast<unsigned char*>(buffer.get()), sizeof(uint16_t));
  std::vector<uint16_t> dst(1);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(src, gsl::make_span(dst)));
  value = dst[0];

  return Status::OK();
}

/* \brief: Try to get metadata from Ort generated context cache binary file.
 *  Cached context binary file generated by Ort has some metadata which can be used for validation with the model
 *  to avoid user choose a wrong context binary file which is not for this model
 *  It is treated as Qnn generated context binary file if no metadata found from the file
 */
Status QnnBackendManager::GetMetadataFromOrtContextFile() {
  // Only try parse meta data once
  if (ctx_metadata_tried_) {
    return Status::OK();
  }
  ctx_metadata_tried_ = true;

  uint64_t buffer_size = 0;
  std::ifstream cache_file(context_cache_path_.c_str(), std::ifstream::binary);
  ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to open context cache file.");
  cache_file.seekg(0, cache_file.end);
  buffer_size = cache_file.tellg();
  ORT_RETURN_IF(0 == buffer_size, "Empty cache file encountered.");
  cache_file.seekg(0, cache_file.beg);

  // Read ort flag
  std::string ort_flag("");
  size_t ort_flag_length = strlen(QNN_PROVIDER);
  ORT_RETURN_IF_ERROR(ReadStringFromBinaryFile(cache_file, ort_flag, ort_flag_length));

  // It's not Ort generated context binary file
  if (strncmp(ort_flag.c_str(), QNN_PROVIDER, ort_flag_length) != 0) {
    return Status::OK();
  }
  ort_generated_ctx_cache_ = true;

  uint16_t str_length = 0;
  ORT_RETURN_IF_ERROR(ReadInt16FromBinaryFile(cache_file, str_length));
  ort_ctx_metadata_length_ = str_length + static_cast<uint16_t>(ort_flag_length);

  ORT_RETURN_IF_ERROR(ReadInt16FromBinaryFile(cache_file, str_length));
  ORT_RETURN_IF_ERROR(ReadStringFromBinaryFile(cache_file, model_name_from_ctx_cache_, static_cast<size_t>(str_length)));

  ORT_RETURN_IF_ERROR(ReadInt16FromBinaryFile(cache_file, str_length));
  ORT_RETURN_IF_ERROR(ReadStringFromBinaryFile(cache_file, graph_name_from_ctx_cache_, static_cast<size_t>(str_length)));

  ORT_RETURN_IF_ERROR(ReadInt16FromBinaryFile(cache_file, str_length));
  ORT_RETURN_IF_ERROR(ReadStringFromBinaryFile(cache_file, model_description_from_ctx_cache_, static_cast<size_t>(str_length)));

  return Status::OK();
}

/* \brief: Validate the model file name and graph name with Ort generated context cache metadata
 * \param[in] model_name - model file name
 * \param[in] graph_name - graph name, e.g Ort_QNN_[hash_id]_[id]. Since GetCapability is called twice,
 *                         [hash_id]_[id] changes even for same graph,
 *                          so only validate the graph name for 2nd call
 */
Status QnnBackendManager::ValidateWithContextFile(const std::string& model_name, const std::string& graph_name) {
  ORT_RETURN_IF(!ctx_file_exists_, "Qnn context binary file not exist for some reason!");

  // Get metadata from cached context binary file
  ORT_RETURN_IF_ERROR(GetMetadataFromOrtContextFile());

  // The context binary file doesn't have ORT metadata, so it is generated from QNN toolchain not from ORT
  if (!ort_generated_ctx_cache_) {
    return Status::OK();
  }

  ORT_RETURN_IF(model_name != model_name_from_ctx_cache_,
                "Model file name from context cache metadata: " + model_name_from_ctx_cache_ +
                    " is different with target: " + model_name +
                    ". Please make sure the context binary file matches the model.");

  ORT_RETURN_IF(model_description_ != model_description_from_ctx_cache_,
                "Model description from context cache metadata: " + model_description_from_ctx_cache_ +
                    " is different with target: " + model_description_ +
                    ". Please make sure the context binary file matches the model.");

  ORT_RETURN_IF(graph_name != graph_name_from_ctx_cache_ && get_capability_round_2_,
                "Graph name from context cache metadata: " + graph_name_from_ctx_cache_ +
                    " is different with target: " + graph_name +
                    ". You may need to re-generate the context binary file.");

  get_capability_round_2_ = true;
  return Status::OK();
}

Status QnnBackendManager::SetupBackend(const logging::Logger& logger, bool load_from_cached_context) {
  if (backend_setup_completed_) {
    LOGS(logger, VERBOSE) << "Backend setup already!";
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(LoadBackend());
  LOGS(logger, VERBOSE) << "LoadBackend succeed.";

  if (load_from_cached_context) {
    ORT_RETURN_IF_ERROR(LoadQnnSystemLib());
  }

  LOGS(logger, VERBOSE) << "Backend build version: "
                        << GetBackendBuildId();

  SetLogger(&logger);
  LOGS(logger, VERBOSE) << "SetLogger succeed.";

  ORT_RETURN_IF_ERROR(InitializeBackend());
  LOGS(logger, VERBOSE) << "InitializeBackend succeed.";

  ORT_RETURN_IF_ERROR(CreateDevice());
  LOGS(logger, VERBOSE) << "CreateDevice succeed.";

  ORT_RETURN_IF_ERROR(InitializeProfiling());
  LOGS(logger, VERBOSE) << "InitializeProfiling succeed.";

  if (!load_from_cached_context) {
    ORT_RETURN_IF_ERROR(CreateContext());
    LOGS(logger, VERBOSE) << "CreateContext succeed.";
  }

  if (htp_performance_mode_ != HtpPerformanceMode::kHtpDefault) {
    ORT_RETURN_IF_ERROR(SetHtpPowerConfig());
    LOGS(logger, VERBOSE) << "SetHtpPowerConfig succeed.";
  }

  LOGS(logger, VERBOSE) << "QNN SetupBackend succeed";

  backend_setup_completed_ = true;

  return Status::OK();
}

Status QnnBackendManager::SetHtpPowerConfig() {
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;
  // Get power client id
  uint32_t powerconfig_client_id = 0;
  status = htp_perf_infra.createPowerConfigId(/*device_id=*/0, /*core_id=*/0, &powerconfig_client_id);
  ORT_RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  constexpr const int kNumConfigs = 1;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs(
      kNumConfigs);
  QnnHtpPerfInfrastructure_PowerConfig_t& dcvs_config = power_configs[0];
  dcvs_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = dcvs_config.dcvsV3Config;
  dcvs_v3.contextId = powerconfig_client_id;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.dcvsEnable = kDcvsDisable;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  // choose performance mode
  switch (htp_performance_mode_) {
    case HtpPerformanceMode::kHtpBurst:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMinLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case HtpPerformanceMode::kHtpSustainedHighPerformance:
    case HtpPerformanceMode::kHtpHighPerformance:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepLowLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      break;
    case HtpPerformanceMode::kHtpPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case HtpPerformanceMode::kHtpLowPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case HtpPerformanceMode::kHtpHighPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    case HtpPerformanceMode::kHtpLowBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case HtpPerformanceMode::kHtpBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    default:
      ORT_THROW("Invalid performance profile %d", static_cast<int>(htp_performance_mode_));
      break;
  }
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr_ = ObtainNullTermPtrVector(power_configs);
  status = htp_perf_infra.setPowerConfig(powerconfig_client_id, perf_power_configs_ptr_.data());
  ORT_RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for HTP performance mode.");

  // Set rpc control latency here, but note that v68 doesn't support rpc polling mode.
  if (rpc_control_latency_ != 0) {
    constexpr int kNumRpcPollingPowerConfigs = 1;
    std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> rpc_power_configs(kNumRpcPollingPowerConfigs);
    QnnHtpPerfInfrastructure_PowerConfig_t& rpc_control_latency = rpc_power_configs[0];
    // v68 doesn't support this.
    QnnHtpPerfInfrastructure_PowerConfig_t& rpc_polling_time = rpc_power_configs[1];
    rpc_control_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_polling_time.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_control_latency.rpcControlLatencyConfig = rpc_control_latency_;
    perf_power_configs_ptr_ = ObtainNullTermPtrVector(rpc_power_configs);
    status = htp_perf_infra.setPowerConfig(powerconfig_client_id, perf_power_configs_ptr_.data());
    ORT_RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for RPC control latency.");
  }

  return Status::OK();
}

void QnnBackendManager::Split(std::vector<std::string>& split_string,
                              const std::string& tokenized_string,
                              const char separator) {
  split_string.clear();
  std::istringstream tokenized_string_stream(tokenized_string);
  while (!tokenized_string_stream.eof()) {
    std::string value;
    getline(tokenized_string_stream, value, separator);
    if (!value.empty()) {
      split_string.push_back(value);
    }
  }
}

void QnnBackendManager::ReleaseResources() {
  if (!backend_setup_completed_) {
    return;
  }

  auto result = ReleaseContext();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseContext.");
  }

  result = ReleaseProfilehandle();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseProfilehandle.");
  }

  result = ReleaseDevice();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseDevice.");
  }

  result = ShutdownBackend();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ShutdownBackend.");
  }

  result = TerminateQnnLog();
  if (Status::OK() != result) {
    ORT_THROW("Failed to TerminateQnnLog.");
  }

  if (backend_lib_handle_) {
    result = UnloadLib(backend_lib_handle_);
    if (Status::OK() != result) {
      ORT_THROW("Failed to unload backend library.");
    }
  }

  backend_setup_completed_ = false;

  return;
}

Status QnnBackendManager::ExtractBackendProfilingInfo() {
  if (ProfilingLevel::OFF == profiling_level_ || ProfilingLevel::INVALID == profiling_level_) {
    return Status::OK();
  }
  ORT_RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  auto result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile events.");

  if (num_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_events: " << profile_events << " num_events: " << num_events;
  }

  for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
    ORT_RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_events + event_idx)));
    ORT_RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_events + event_idx)));
  }
  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  auto result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile sub events.");

  if (num_sub_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_sub_events: " << profile_sub_events << " num_sub_events: " << num_sub_events;
  }

  for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; sub_event_idx++) {
    ORT_RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_sub_events + sub_event_idx)));
    ORT_RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_sub_events + sub_event_idx)));
  }
  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id) {
  QnnProfile_EventData_t event_data;
  auto result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get provile event data.");

  LOGS(*logger_, VERBOSE) << "Profiling Event Info - Event Type: " << event_data.type
                          << ", Event Value: " << event_data.value
                          << ", Event Identifier: " << event_data.identifier
                          << ", Event Unit: " << event_data.unit;

  return Status::OK();
}

QnnBackendManager::~QnnBackendManager() {
  ReleaseResources();
}

void* QnnBackendManager::LoadLib(const char* file_name, int flags, std::string& error_msg) {
#ifdef _WIN32
  DWORD as_is, to_be;
  bool loaded_before = false;

  if (!file_name || ::strlen(file_name) == 0) {
    error_msg = "filename is null or empty";
    return nullptr;
  }

  // POSIX asks one of symbol resolving approaches:
  // NOW or LAZY must be specified
  if (!(flags & static_cast<int>(DlOpenFlag::DL_NOW))) {
    error_msg = "flags must include DL_NOW";
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &as_is) == 0) {
    error_msg = "enumerate modules failed before loading module";
    return nullptr;
  }

  // search from system lib path first
  HMODULE mod = LoadLibraryExA(file_name, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!mod) {
    error_msg = "load library failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, nullptr, 0, &to_be) == 0) {
    error_msg = "enumerate modules failed after loading module";
    FreeLibrary(mod);
    return nullptr;
  }

  if (as_is == to_be) {
    loaded_before = true;
  }

  // (not loaded_before) and DL_LOCAL means this lib was not loaded yet
  // add it into the local set
  //
  // If loaded_before and DL_LOCAL, means this lib was already loaded
  // 2 cases here for how it was loaded before:
  // a. with DL_LOCAL, just ignore since it was already in local set
  // b. with DL_GLOBAL, POSIX asks it in global, ignore it, too
  if ((!loaded_before) && (flags & static_cast<int>(DlOpenFlag::DL_LOCAL))) {
    mod_handles_.insert(mod);
  }

  // once callers ask for global, needs to be in global thereafter
  // so the lib should be removed from local set
  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    mod_handles_.erase(mod);
  }

  return static_cast<void*>(mod);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  int real_flags = 0;

  if (flags & static_cast<int>(DlOpenFlag::DL_NOW)) {
    real_flags |= RTLD_NOW;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_LOCAL)) {
    real_flags |= RTLD_LOCAL;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    real_flags |= RTLD_GLOBAL;
  }

  return ::dlopen(file_name, real_flags);
#endif
}

Status QnnBackendManager::UnloadLib(void* handle) {
  if (!handle) {
    return Status::OK();
  }

#ifdef _WIN32
  HMODULE mod = static_cast<HMODULE>(handle);
  if (FreeLibrary(mod) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
  mod_handles_.erase(mod);
#else
  auto rt = ::dlclose(handle);
  if (rt != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
#endif

  return Status::OK();
}

void* QnnBackendManager::LibFunction(void* handle, const char* symbol, std::string& error_msg) {
#ifdef _WIN32
  FARPROC sym_addr = nullptr;
  DWORD size, size_needed;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &size) == 0) {
    error_msg = "enumerate modules failed before memory allocation";
    return nullptr;
  }

  HMODULE* mod_list = static_cast<HMODULE*>(malloc(size));
  if (!mod_list) {
    error_msg = "malloc failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    error_msg = "enumerate modules failed after memory allocation";
    free(mod_list);
    return nullptr;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles_.find(mod_list[i]);
      if (iter != mod_handles_.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void**)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    error_msg = "can't resolve symbol";
    return NULL;
  }

  return *(void**)(&sym_addr);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  if (handle == DL_DEFAULT) {
    return ::dlsym(RTLD_DEFAULT, symbol);
  }

  return ::dlsym(handle, symbol);
#endif
}

}  // namespace qnn
}  // namespace onnxruntime
