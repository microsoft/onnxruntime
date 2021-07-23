#pragma once

#ifdef __cplusplus
namespace std {
    inline char *getenv(const char*) { return nullptr; }
}
#endif
