#include "utils.h"
#include "half.hpp"
#include "lodepng/lodepng.h"

#ifdef _WIN32
#include <windows.h>  // For GetModuleFileNameW
#elif __APPLE__
#include <limits.h>       // For PATH_MAX or similar
#include <mach-o/dyld.h>  // For _NSGetExecutablePath
#elif __linux__
#include <limits.h>  // For PATH_MAX
#include <unistd.h>  // For readlink
#endif

std::filesystem::path get_executable_parent_path() { return get_executable_path().parent_path(); }

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
    pathBuf.resize(length + 1);  // +1 for null terminator
    _NSGetExecutablePath(pathBuf.data(), &length);
  }
  return std::filesystem::canonical(
      pathBuf.data());  // canonical to resolve symlinks

#elif __linux__
  // Linux: Use /proc/self/exe symlink
  return std::filesystem::canonical(
      std::filesystem::read_symlink("/proc/self/exe"));
#endif
}

using half_float::half;
constexpr int image_dim = 1080;

void loadInputImage(void* pData, char* imageFileName, bool fp16) {
  half* hData = (half*)pData;
  float* fData = (float*)pData;

  unsigned char* image;
  unsigned int width, height;
  unsigned int error =
      lodepng_decode32_file(&image, &width, &height, imageFileName);
  if (error) {
    printf("\nFailed to load the input image. Exiting\n");
    exit(0);
  }

  if (width != image_dim || height != image_dim) {
    printf("\nImage not of right size. Exiting\n");
    exit(0);
  }

  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++) {
      unsigned char r = image[(y * width + x) * 4 + 0];
      unsigned char g = image[(y * width + x) * 4 + 1];
      unsigned char b = image[(y * width + x) * 4 + 2];

      if (fp16) {
        hData[0 * width * height + y * width + x] = (half)b;
        hData[1 * width * height + y * width + x] = (half)g;
        hData[2 * width * height + y * width + x] = (half)r;
      } else {
        fData[0 * width * height + y * width + x] = (float)b;
        fData[1 * width * height + y * width + x] = (float)g;
        fData[2 * width * height + y * width + x] = (float)r;
      }
    }

  free(image);
}

unsigned char clampAndConvert(float val) {
  if (val < 0)
    val = 0;
  if (val > 255)
    val = 255;
  return (unsigned char)val;
}

void saveOutputImage(void* pData, char* imageFileName, bool fp16) {
  half* hData = (half*)pData;
  float* fData = (float*)pData;

  unsigned int width = image_dim, height = image_dim;  // hardcoded in the model

  std::vector<unsigned char> image(width * height * 4);
  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++) {
      float b, g, r;
      if (fp16) {
        b = (float)hData[0 * width * height + y * width + x];
        g = (float)hData[1 * width * height + y * width + x];
        r = (float)hData[2 * width * height + y * width + x];
      } else {
        b = fData[0 * width * height + y * width + x];
        g = fData[1 * width * height + y * width + x];
        r = fData[2 * width * height + y * width + x];
      }

      image[(y * width + x) * 4 + 0] = clampAndConvert(r);
      image[(y * width + x) * 4 + 1] = clampAndConvert(g);
      image[(y * width + x) * 4 + 2] = clampAndConvert(b);
      image[(y * width + x) * 4 + 3] = 255;
    }

  lodepng_encode32_file(imageFileName, &image[0], width, height);
}
