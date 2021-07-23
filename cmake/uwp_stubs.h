#pragma once
// Part of the standard C runtime and the STL are missing when targeting UWP, and some applications that rely on standard,
// cross-platform headers fail to build.
// Here we provide stubs for functions required by some onnxruntime dependencies.
#ifdef __cplusplus
// Extending the std namespace is undefined behavior
// NOLINTNEXTLINE
namespace std {
    inline char *getenv(const char*) { return nullptr; }
}
#endif
