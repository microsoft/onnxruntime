// CMake's CHECK_INCLUDE_FILE_CXX macro can't be used because it doesn't check the machine's SDK folder
#if not __has_include("dxcore.h")
#error
#endif

int main() {
    return 0;
}
