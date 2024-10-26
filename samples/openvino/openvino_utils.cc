#include <windows.h>
#include "openvino_utils.h"

namespace onnxruntime {
    std::string GetEnvironmentVar(const std::string& var_name) {
// TODO(leca): #ifdef _WIN32
//#endif
        constexpr DWORD kBufferSize = 32767;

        // Create buffer to hold the result
        std::string buffer(kBufferSize, '\0');

        // The last argument is the size of the buffer pointed to by the lpBuffer parameter, including the null-terminating character, in characters.
        // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer, not including the terminating null character.
        // Therefore, If the function succeeds, kBufferSize should be larger than char_count.
        auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);

        if (kBufferSize > char_count) {
            buffer.resize(char_count);
            return buffer;
        }

        return std::string();
    }
}
