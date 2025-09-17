#include "utils.h"
#include <filesystem>
#include <vector>

#ifdef _WIN32
#include <windows.h> // For GetModuleFileNameW
#elif __APPLE__
#include <limits.h>      // For PATH_MAX or similar
#include <mach-o/dyld.h> // For _NSGetExecutablePath
#elif __linux__
#include <limits.h> // For PATH_MAX
#include <unistd.h> // For readlink
#endif

std::filesystem::path get_executable_path() {
#ifdef _WIN32
  // Windows: Use GetModuleFileNameW for wide characters
  std::vector<wchar_t> pathBuf(MAX_PATH);
  DWORD length = GetModuleFileNameW(NULL, pathBuf.data(), pathBuf.size());

  while (length == pathBuf.size()) {
    pathBuf.resize(pathBuf.size() * 2);
    length = GetModuleFileNameW(NULL, pathBuf.data(), pathBuf.size());
  }

  if (length == 0) {
    std::cerr << "Error: GetModuleFileNameW failed with error "
              << GetLastError() << std::endl;
    return {};
  }
  return std::filesystem::path(pathBuf.data());

#elif __APPLE__
  // macOS: Use _NSGetExecutablePath
  std::vector<char> pathBuf(PATH_MAX);
  uint32_t length = pathBuf.size();
  if (_NSGetExecutablePath(pathBuf.data(), &length) != 0) {
    // Buffer was too small, resize and try again
    pathBuf.resize(length + 1); // +1 for null terminator
    _NSGetExecutablePath(pathBuf.data(), &length);
  }
  return std::filesystem::canonical(
      pathBuf.data()); // canonical to resolve symlinks

#elif __linux__
  // Linux: Use /proc/self/exe symlink
  return std::filesystem::canonical(
      std::filesystem::read_symlink("/proc/self/exe"));
#endif
}
