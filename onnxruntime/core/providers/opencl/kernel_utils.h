#if !defined(__OPENCL_C_VERSION__)
#define CXX_MODE 1
#endif

#if CXX_MODE
#include <CL/cl_platform.h>
#include "stdint.h"

#if defined(__GNUC__)
#define CL_PACKED __attribute__((packed))
#endif
#else  // CL_MODE
#if !defined(__ENDIAN_LITTLE__) || !__ENDIAN_LITTLE__
#define CL_PACKED __attribute__((packed))
#define CL_ALIGNED __attribute__((aligned(4)))
#endif
#endif

#ifndef CL_PACKED
#error CL_PACKED not defined
#endif

enum ACT_TYPE {
  ActivationType_None = 0,
  ActivationType_ReLU = 1,
  ActivationType_ReLU6 = 2,
};

typedef struct CL_PACKED CL_ALIGNED(4) {
  union {
    int32_t x;
    int32_t w;  // width
  };
  union {
    int32_t y;
    int32_t h;  // height
  };
} Dim2;

typedef struct CL_PACKED CL_ALIGNED(4) {
  union {
    int32_t x;
    int32_t w;  // width
  };
  union {
    int32_t y;
    int32_t h;  // height
  };
  union {
    int32_t z;
    int32_t d;  // depth
  };
} Dim3;

#if CXX_MODE
Dim2 dim2(int width, int height) {
  return Dim2{{width}, {height}};
}

Dim3 dim3(int width, int height, int depth) {
  return Dim3{{width}, {height}, {depth}};
}
#endif
