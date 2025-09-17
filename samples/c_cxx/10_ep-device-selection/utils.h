#pragma once

#include <filesystem>
#include <format>
#include <iostream>

#ifndef __cpp_lib_format
#error                                                                         \
    "__cpp_lib_format is not defined! This samples requires a C++ 20 compiler"
#endif

std::filesystem::path get_executable_path();

#define LOG(...) std::cout << std::format(__VA_ARGS__) << "\n"
#define THROW_ERROR(...)                                                       \
  LOG(__VA_ARGS__);                                                            \
  throw std::runtime_error(std::format(__VA_ARGS__));

#define DEFER(resource, x)                                                     \
  std::shared_ptr<void> resource##_finalizer(nullptr, [&](...) { x; })

#define CHECK_ORT(call)                                                        \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != nullptr) {                                                   \
      THROW_ERROR("{}", Ort::GetApi().GetErrorMessage(status));                \
    }                                                                          \
  }

inline static std::string to_uppercase(const std::string &s) {
  std::string rtn;
  rtn.resize(s.size());
  std::transform(s.begin(), s.end(), rtn.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return rtn;
}

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#if _WIN32
#define DLL_PREFIX ""
#define DLL_SUFFIX ".dll"
#else
#define DLL_PREFIX "lib"
#define DLL_SUFFIX ".so"
#endif
#define PROVIDER_DLL_NAME(X) DLL_NAME("onnxruntime_providers_" X))
