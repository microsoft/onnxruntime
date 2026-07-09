# Cross-compile ONNX Runtime for Windows x64 from a non-Windows host (e.g. Linux)
# using the LLVM/clang toolchain. This is what lets downstream consumers such as
# the Edge browser build Windows-x64 ONNX Runtime on faster/cheaper Linux
# builders. Today it is exercised to build the onnxruntime_mlas static library
# (see .github/workflows/linux_mlas_win_cross.yml), which contains the MLAS x64
# MASM assembly kernels.
#
# Toolchain pieces:
#   * clang-cl / lld-link  - C/C++ compile + link, targeting x86_64-pc-windows-msvc
#   * llvm-ml              - the LLVM MASM assembler for onnxruntime/core/mlas/lib/amd64/*.asm
#   * Windows SDK + MSVC CRT headers/import libs staged by `xwin`
#     (https://github.com/Jake-Shadle/xwin): run `xwin splat --output <dir>` and
#     pass that directory below.
#
# Required (cache var or environment variable):
#   ORT_WIN_XWIN_ROOT  - the `xwin splat` output directory (contains crt/ and sdk/)
# Optional:
#   ORT_LLVM_BIN       - directory holding clang-cl/lld-link/llvm-ml/... (else PATH)

if(NOT ORT_WIN_XWIN_ROOT)
  set(ORT_WIN_XWIN_ROOT "$ENV{ORT_WIN_XWIN_ROOT}")
endif()
if(NOT ORT_WIN_XWIN_ROOT)
  message(FATAL_ERROR
    "win x64 cross toolchain: set ORT_WIN_XWIN_ROOT to the `xwin splat` output directory")
endif()
if(NOT ORT_LLVM_BIN)
  set(ORT_LLVM_BIN "$ENV{ORT_LLVM_BIN}")
endif()

# CMake re-parses this toolchain file inside its try_compile subprojects, where
# the outer -D cache entries are not visible. Forward the settings this file needs.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ORT_WIN_XWIN_ROOT ORT_LLVM_BIN)

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

macro(_ort_find_llvm_tool _var _name _required)
  if(ORT_LLVM_BIN)
    find_program(${_var} NAMES ${_name} PATHS "${ORT_LLVM_BIN}" NO_DEFAULT_PATH)
  endif()
  if(NOT ${_var})
    find_program(${_var} NAMES ${_name})
  endif()
  if(${_required} AND NOT ${_var})
    message(FATAL_ERROR "win x64 cross toolchain: could not find '${_name}'")
  endif()
endmacro()

_ort_find_llvm_tool(_ort_clang_cl clang-cl TRUE)
_ort_find_llvm_tool(_ort_lld_link lld-link TRUE)
_ort_find_llvm_tool(_ort_llvm_ml  llvm-ml  TRUE)
_ort_find_llvm_tool(_ort_llvm_lib llvm-lib TRUE)
# mt (manifest) and rc (resource compiler) are only needed for DLL/exe links, not
# for the onnxruntime_mlas static library, so they are optional.
_ort_find_llvm_tool(_ort_llvm_mt  llvm-mt  FALSE)
_ort_find_llvm_tool(_ort_llvm_rc  llvm-rc  FALSE)

set(CMAKE_C_COMPILER        "${_ort_clang_cl}")
set(CMAKE_CXX_COMPILER      "${_ort_clang_cl}")
set(CMAKE_ASM_MASM_COMPILER "${_ort_llvm_ml}")
set(CMAKE_LINKER            "${_ort_lld_link}")
set(CMAKE_AR                "${_ort_llvm_lib}")
if(_ort_llvm_mt)
  set(CMAKE_MT              "${_ort_llvm_mt}")
endif()
if(_ort_llvm_rc)
  set(CMAKE_RC_COMPILER     "${_ort_llvm_rc}")
endif()

set(CMAKE_C_COMPILER_TARGET   x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)

set(_ort_crt "${ORT_WIN_XWIN_ROOT}/crt")
set(_ort_sdk "${ORT_WIN_XWIN_ROOT}/sdk")

# Windows SDK / CRT header search. Passed via /imsvc (system-include semantics) so
# ORT's -WX and warning flags do not apply to SDK headers. C/C++ only; the .asm
# files must not receive these. -Wno-unused-command-line-argument silences the
# MSVC-only flags ORT passes that clang-cl accepts-but-ignores (e.g.
# /experimental:external), which -WX would otherwise turn into errors.
string(JOIN " " _ort_win_include_flags
  "/imsvc${_ort_crt}/include"
  "/imsvc${_ort_sdk}/include/ucrt"
  "/imsvc${_ort_sdk}/include/um"
  "/imsvc${_ort_sdk}/include/shared"
  "/imsvc${_ort_sdk}/include/winrt")
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:${_ort_win_include_flags}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-command-line-argument>")

# The xwin CRT's <intrin.h> transitively pulls clang's <mmintrin.h>, whose legacy
# MMX intrinsics use the 64-bit integer vector typedefs (e.g. __v4hi). Some clang
# x86_64 builds (notably the apt.llvm.org snapshot used in CI) only treat those as
# real vectors when 'mmx' is in the *global* target features; without it the
# vector_size attribute is dropped and the header fails to compile
# ("__v4hi (aka 'short')"). Enable mmx globally so the header's typedefs stay
# vectors. ORT itself uses no MMX; this only affects the transitively-included
# header, not codegen for ORT's own SIMD kernels.
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-mmmx>")

add_link_options(
  "/libpath:${_ort_crt}/lib/x86_64"
  "/libpath:${_ort_sdk}/lib/ucrt/x86_64"
  "/libpath:${_ort_sdk}/lib/um/x86_64")

# llvm-rc (resource compiler, used for the DLL's version resource) does not honor
# /imsvc, so give it the Windows SDK headers on its own include path.
set(CMAKE_RC_FLAGS_INIT
  "/I \"${_ort_crt}/include\" /I \"${_ort_sdk}/include/um\" /I \"${_ort_sdk}/include/shared\" /I \"${_ort_sdk}/include/ucrt\"")

# MLAS amd64 .asm via llvm-ml: unlike ml64.exe, llvm-ml is not implicitly 64-bit,
# does not search the source file's own directory for includes, and needs the
# LLVM_ML define that selects the dual-assembler-compatible spellings. ml64.exe
# needs none of these. CMAKE_CURRENT_LIST_DIR is this file's directory (cmake/).
set(CMAKE_ASM_MASM_FLAGS_INIT
  "-m64 /DLLVM_ML=1 -I${CMAKE_CURRENT_LIST_DIR}/../onnxruntime/core/mlas/lib/amd64")

# onnxruntime_mlas is a static library, so skip CMake's link-based compiler check
# (manifest/rc/link), which is irrelevant when cross-building a static archive.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
