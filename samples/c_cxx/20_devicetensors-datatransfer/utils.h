#pragma once
#include <filesystem>
#include <format>
#include <iostream>
void loadInputImage(void *pData, char *imageFileName, bool fp16);
void saveOutputImage(void *pData, char *imageFileName, bool fp16);

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#if _WIN32
#define DLL_PREFIX ""
#define DLL_SUFFIX ".dll"
#else
#define DLL_PREFIX "lib"
#define DLL_SUFFIX ".so"
#endif
#define LOG(...) std::cout << std::format(__VA_ARGS__) << "\n"
#define THROW_ERROR(...)                                                       \
  LOG(__VA_ARGS__);                                                            \
  throw std::runtime_error(std::format(__VA_ARGS__));
#define CHECK_ORT(call)                                                        \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != nullptr) {                                                   \
      THROW_ERROR("{}", Ort::GetApi().GetErrorMessage(status));                \
    }                                                                          \
  }

std::filesystem::path get_executable_path();
std::filesystem::path get_executable_parent_path();
