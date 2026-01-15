// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#include <winver.h>
#pragma comment(lib, "Version.lib")
#endif

#include "HTP/QnnHtpDevice.h"
#include "QnnTypes.h"

#include "core/providers/qnn-abi/htp_usr_drv_utils.h"

namespace onnxruntime {
namespace qnn {
namespace htp_usr_drv {

namespace {

// All implementations are directly copied from QNN SDK with few modifications.
// QNN-EP COPY START

std::string getLibName() {
#ifdef _WIN32
  return "HtpUsrDrv.dll";
#else
  return "libHtpUsrDrv.so";
#endif
}

#ifdef _WIN32
std::string WideToUtf8(const std::wstring& wstr) {
  if (wstr.empty()) return std::string();
  int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
  std::string strTo(sizeNeeded - 1, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &strTo[0], sizeNeeded - 1, NULL, NULL);
  return strTo;
}

std::string getServiceBinaryPath(std::wstring const& service_name) {
  const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();

  // In this part, we use wide characters here because they are all Windows specific operations,
  // for some string operations, we process them with standard C++ string APIs instead of using
  // PAL string libraries since it does not support wide characters currently.

  // Get a handle to the SCM database
  SC_HANDLE handle_sc_manager = OpenSCManagerW(NULL,                   // local computer
                                               NULL,                   // ServicesActive database
                                               STANDARD_RIGHTS_READ);  // standard read access
  if (nullptr == handle_sc_manager) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to open SCManager which is required to access service configuration on Windows. Error: " +
                 std::to_string(GetLastError()))
                    .c_str());
    return std::string();
  }

  // Get a handle to the service
  SC_HANDLE handle_service = OpenServiceW(handle_sc_manager,      // SCM database
                                          service_name.c_str(),   // name of service
                                          SERVICE_QUERY_CONFIG);  // need query config access

  if (NULL == handle_service) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to open service " + WideToUtf8(service_name) +
                 " which is required to query service information. " +
                 "Error: " + std::to_string(GetLastError()))
                    .c_str());
    CloseServiceHandle(handle_sc_manager);
    return std::string();
  }

  // Query the buffer size required by service configuration
  // When first calling it with null pointer and zero buffer size,
  // this function acts as a query function to return how many bytes it requires
  // and set error to ERROR_INSUFFICIENT_BUFFER.

  DWORD bufferSize;  // Store the size of buffer used as an output
  if (!QueryServiceConfigW(handle_service, NULL, 0, &bufferSize) &&
      (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to query service configuration to get size of config object. Error: " +
                 std::to_string(GetLastError()))
                    .c_str());
    CloseServiceHandle(handle_service);
    CloseServiceHandle(handle_sc_manager);
    return std::string();
  }
  // Get the configuration of the specified service
  LPQUERY_SERVICE_CONFIGW service_config =
      static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
  if (!QueryServiceConfigW(handle_service, service_config, bufferSize, &bufferSize)) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to query service configuration. Error: " + std::to_string(GetLastError())).c_str());
    LocalFree(service_config);
    CloseServiceHandle(handle_service);
    CloseServiceHandle(handle_sc_manager);
    return std::string();
  }

  // Read the driver file path
  std::wstring driver_path = std::wstring(service_config->lpBinaryPathName);
  // Get the parent directory of the driver file
  driver_path = driver_path.substr(0, driver_path.find_last_of(L"\\"));

  // Clean up resources
  LocalFree(service_config);
  CloseServiceHandle(handle_service);
  CloseServiceHandle(handle_sc_manager);

  // Driver path would contain invalid path string, like:
  // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
  // "\SystemRoot" should be replace with a correct one (e.g. C:\windows)
  const std::wstring system_root_placeholder = L"\\SystemRoot";
  if (0 != driver_path.compare(0, system_root_placeholder.length(), system_root_placeholder)) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("The string pattern does not match. We expect that we can find " +
                 WideToUtf8(system_root_placeholder) +
                 "in the beginning of the queried path " +
                 WideToUtf8(driver_path))
                    .c_str());
    return std::string();
  }

  // Replace \SystemRoot with an absolute path which is got from system ENV windir
  // Here, we don't use PAL Text because the operated strings are wide characters
  // PAL does not provide operations for this type
  // ENV name used to get the root path of the system
  const std::wstring system_root_env = L"windir";

  // Query the number of wide charactors this variable requires
  DWORD num_words = GetEnvironmentVariableW(system_root_env.c_str(), NULL, 0);
  if (num_words == 0) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                "Failed to query the buffer size when calling GetEnvironmentVariableW().");
    return std::string();
  }

  // Query the actual system root name from environment variable
  std::vector<wchar_t> system_root(num_words + 1);
  num_words = GetEnvironmentVariableW(system_root_env.c_str(), system_root.data(), num_words + 1);
  if (num_words == 0) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Failed to read value from environment variables.");
    return std::string();
  }
  driver_path.replace(0, system_root_placeholder.length(), std::wstring(system_root.data()));

  // driver_path is wide char string, we need to convert it to std::string
  // Assume to use UTF-8 wide string for conversion
  return WideToUtf8(driver_path);
}

bool isMcdmRegEnabled() {
  const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();

  // Fastrpc team uses reg key to switch between fastrpc and mcdm as an interim approach.
  // This block shall be removed once the reg key mechanisms is removed from the meta.
  // How to check if MCDM is available? Explain in Powershell command as a quick example:
  // 1. Query info for the reg key of MCDM switch
  //    PS> reg query HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\qcnspmcdm\Enum /v 0
  //    <<<  0    REG_SZ    ACPI\QCOM0D0A\2&daba3ff&0
  // 2. Query the switch value with the info from Step 1
  //    PS> reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\`
  //        ACPI\QCOM0D0A\2&daba3ff&0\Device Parameters\NSPM\INFO" /v FastRPCEnabledDSPInfo
  //    <<< FastRPCEnabledDSPInfo    REG_BINARY    08
  // 3. Check the value from Step2. 08 indicates MCDM is enabled
  HKEY hkey = 0;
  LONG res = ERROR_SUCCESS;
  void* buf = NULL;
  DWORD vsize = 0;
  // Step 1. Query info for the reg key of MCDM switch
  char const* reg_key_mcdm_switch_info = "SYSTEM\\CurrentControlSet\\Services\\qcnspmcdm\\Enum";
  res = RegOpenKeyExA(HKEY_LOCAL_MACHINE, reg_key_mcdm_switch_info, 0, KEY_READ, &hkey);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to open reg key handle: HKEY_LOCAL_MACHINE\\" + std::string(reg_key_mcdm_switch_info))
                    .c_str());
    return false;
  }
  res = RegGetValueA(hkey, NULL, "0", RRF_RT_REG_SZ, NULL, NULL, &vsize);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to read the size of reg key: HKEY_LOCAL_MACHINE\\" +
                 std::string(reg_key_mcdm_switch_info) + "\\0")
                    .c_str());
    RegCloseKey(hkey);
    return false;
  }
  buf = malloc(vsize);
  if (NULL == buf) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Failed to allocate buffer for reg key retrieval");
    RegCloseKey(hkey);
    return false;
  }
  res = RegGetValueA(hkey, NULL, "0", RRF_RT_REG_SZ, NULL, buf, &vsize);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to read the reg key: HKEY_LOCAL_MACHINE\\" + std::string(reg_key_mcdm_switch_info) + "\\0")
                    .c_str());
    RegCloseKey(hkey);
    free(buf);
    return false;
  }
  std::string reg_key_mcdm_switch = "SYSTEM\\CurrentControlSet\\Enum\\";
  reg_key_mcdm_switch.append(static_cast<char*>(buf)).append("\\Device Parameters\\NSPM\\INFO");
  free(buf);
  RegCloseKey(hkey);
  // Step 2. Query the switch value with the info from Step 1
  hkey = 0;
  res = ERROR_SUCCESS;
  buf = NULL;
  vsize = 0;
  res = RegOpenKeyExA(HKEY_LOCAL_MACHINE, reg_key_mcdm_switch.c_str(), 0, KEY_READ, &hkey);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to open reg key handle: HKEY_LOCAL_MACHINE\\" + std::string(reg_key_mcdm_switch_info))
                    .c_str());
    return false;
  }
  res = RegGetValueA(hkey, NULL, "FastRPCEnabledDSPInfo", RRF_RT_REG_BINARY, NULL, NULL, &vsize);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to read the size of reg key: HKEY_LOCAL_MACHINE\\" +
                 std::string(reg_key_mcdm_switch_info) + "\\FastRPCEnabledDSPInfo")
                    .c_str());
    RegCloseKey(hkey);
    return false;
  }
  if (vsize != 1) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Size of value from " + reg_key_mcdm_switch + " unexpected. " +
                 "Expected 1 but get " + std::to_string(vsize) + " bytes")
                    .c_str());
    RegCloseKey(hkey);
    return false;
  }
  unsigned char mcdm_switch = 0;
  res = RegGetValueA(
      hkey, NULL, "FastRPCEnabledDSPInfo", RRF_RT_REG_BINARY, NULL, &mcdm_switch, &vsize);
  if (ERROR_SUCCESS != res) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to read the reg key: HKEY_LOCAL_MACHINE\\" + reg_key_mcdm_switch + "\\FastRPCEnabledDSPInfo")
                    .c_str());
    RegCloseKey(hkey);
    return false;
  }
  RegCloseKey(hkey);
  // Step 3. Check the value from Step2. 08 indicates MCDM is enabled
  return (mcdm_switch == 0x8) ? true : false;
}

bool getFileVersion(const std::string& path, int& major, int& minor, int& teeny, int& build) {
  const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();

  LPCSTR filePath = path.c_str();
  DWORD handle = 0;

  // First, confirm the file exists
  if (!std::filesystem::exists(path)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("The file does not exist. File: " + path).c_str());
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, ("The file does not exist. File: " + path).c_str());
    return false;
  }

  // Query version info size first
  DWORD infoSize = GetFileVersionInfoSizeA(filePath, &handle);
  if (!infoSize) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to get file version info size. Error code: " + std::to_string(GetLastError()) +
                 ". This might be caused due to absent of version info of the file, " +
                 "ignore this error and set all version parts to zero.")
                    .c_str());
    major = 0;
    minor = 0;
    teeny = 0;
    build = 0;
    return true;
  }

  // If the size is valid, query the version info data
  std::vector<char> versionInfo(infoSize);
  if (!GetFileVersionInfoA(filePath, handle, infoSize, versionInfo.data())) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to get file version info buffer. Error code: " + std::to_string(GetLastError())).c_str());
    return false;
  }

  LPBYTE buffer = NULL;
  UINT bufSize = 0;
  // Retrieve the file info from version info data to parse the version numbers
  if (!VerQueryValueA(versionInfo.data(), "\\", reinterpret_cast<LPVOID*>(&buffer), &bufSize)) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Failed to query file info from the buffer. Error code: " + std::to_string(GetLastError())).c_str());
    return false;
  }

  // Parse version data
  VS_FIXEDFILEINFO* fileInfo = reinterpret_cast<VS_FIXEDFILEINFO*>(buffer);
  major = HIWORD(fileInfo->dwFileVersionMS);
  minor = LOWORD(fileInfo->dwFileVersionMS);
  teeny = HIWORD(fileInfo->dwFileVersionLS);
  build = LOWORD(fileInfo->dwFileVersionLS);
  return true;
}

std::string getDspDriverPath() {
  // Check if mcdm driver avialable and fall back to fastrpc if not
  static std::string dspDriverPath = [] {
    std::string mcdmDriverPath = getServiceBinaryPath(L"qcnspmcdm");
    return (!mcdmDriverPath.empty() && isMcdmRegEnabled()) ? mcdmDriverPath
                                                           : getServiceBinaryPath(L"qcadsprpc");
  }();
  return dspDriverPath;
}
#endif
// QNN-EP COPY END

}  // namespace

std::string GetHtpUsrDrvPath() {
#ifdef _WIN32
  return getDspDriverPath() + "/HTP/" + getLibName();
#else
  return getLibName();
#endif
}

Qnn_Version_t GetHtpUsrDrvVersion() {
  Qnn_Version_t version = QNN_VERSION_INIT;

#ifdef _WIN32
  std::string path = GetHtpUsrDrvPath();
  int major, minor, teeny, build;
  if (!getFileVersion(path, major, minor, teeny, build)) {
    // Only drivers after QNN 2.24 are tagged with file version.
    return version;
  }

  version.major = major;
  version.minor = minor;
  version.patch = teeny;
#endif

  return version;
}

Ort::Status IsHtpUsrDrvEnabled(const std::string& backend_lib_dir, const uint32_t htp_arch, bool& enabled) {
  enabled = false;

  std::string htp_arch_string;
  switch (htp_arch) {
    case QNN_HTP_DEVICE_ARCH_V68:
      htp_arch_string = "V68";
      break;
    case QNN_HTP_DEVICE_ARCH_V69:
      htp_arch_string = "V69";
      break;
    case QNN_HTP_DEVICE_ARCH_V73:
      htp_arch_string = "V73";
      break;
    case QNN_HTP_DEVICE_ARCH_V75:
      htp_arch_string = "V75";
      break;
    case QNN_HTP_DEVICE_ARCH_V81:
      htp_arch_string = "V81";
      break;
    default:
      return MAKE_FAIL(("Unknown HTP arch " + std::to_string(htp_arch)).c_str());
  }

  const std::filesystem::path backend_lib_dir_path(backend_lib_dir);

#ifdef _WIN32
  const std::string prepare_lib_path = "QnnHtpPrepare.dll";
  const std::string stub_lib_path = "QnnHtp" + htp_arch_string + "Stub.dll";
  const std::string skel_lib_path = "libQnnHtp" + htp_arch_string + "Skel.so";
#else
  const std::string prepare_lib_path = "libQnnHtpPrepare.so";
  const std::string stub_lib_path = "libQnnHtp" + htp_arch_string + "Stub.so";
  const std::string skel_lib_path = "libQnnHtp" + htp_arch_string + "Skel.so";
#endif

  if (!std::filesystem::exists(backend_lib_dir_path / prepare_lib_path) ||
      !std::filesystem::exists(backend_lib_dir_path / stub_lib_path) ||
      !std::filesystem::exists(backend_lib_dir_path / skel_lib_path)) {
    enabled = true;
  }

  return Ort::Status();
}

}  // namespace htp_usr_drv
}  // namespace qnn
}  // namespace onnxruntime
