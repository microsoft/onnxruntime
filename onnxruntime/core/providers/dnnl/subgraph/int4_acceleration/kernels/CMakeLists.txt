cmake_minimum_required(VERSION 3.12)

project(bestla LANGUAGES CXX VERSION 0.1.0)
file(GLOB headers ${PROJECT_NAME}/*.h ${PROJECT_NAME}/*.hpp)
file(GLOB xbyak_headers ${PROJECT_NAME}/xbyak/*.h ${PROJECT_NAME}/xbyak/*.hpp)

option(BTLA_USE_OPENMP "Enable OpenMP thread pool" ON)

option(BTLA_UT_ALL "Enable all unit tests" OFF)
option(BTLA_UT_DEBUG "Enable debug unit tests" OFF)
option(BTLA_UT_EPILOGUE "Enable unit test for epilogue" OFF)
option(BTLA_UT_PROLOGUE_A "Enable unit test for activation prologue" OFF)
option(BTLA_UT_PROLOGUE_B "Enable unit test for weight prologue" OFF)
option(BTLA_UT_GEMM "Enable unit test for micro gemm kernels" OFF)
option(BTLA_UT_WRAPPER "Enable unit test for parallel gemms" OFF)
option(BTLA_UT_PARALLEL "Enable unit test for parallel set" OFF)
option(BTLA_UT_KERNEL_JIT "Enable unit test for jit kernels" OFF)
option(BTLA_UT_KERNEL_INTRIN "Enable unit test for intrinsic kernels" OFF)
option(BTLA_UT_KERNEL_WRAPPER "Enable unit test for runtime ISA kernels" OFF)
option(BTLA_UT_NOASAN "Disable sanitize" OFF)
option(BTLA_UT_BENCHMARK "Benchmark ON may take a long time to finish all tests" OFF)
option(BTLA_UT_OPENMP "Use OpenMP" ON)

add_library(${PROJECT_NAME} INTERFACE)
add_library(neural_speed::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(
	${PROJECT_NAME} INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)


if(BTLA_USE_OPENMP)
  message(STATUS "BesTLA using OpenMP")
  target_compile_definitions(${PROJECT_NAME} INTERFACE BTLA_USE_OPENMP)
endif(BTLA_USE_OPENMP)

if(WIN32)
	target_compile_definitions(${PROJECT_NAME} INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif(WIN32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if(BTLA_UT_ALL)
  set(BTLA_UT_EPILOGUE ON)
  set(BTLA_UT_PROLOGUE_A ON)
  set(BTLA_UT_PROLOGUE_B ON)
  set(BTLA_UT_GEMM ON)
  set(BTLA_UT_WRAPPER ON)
  set(BTLA_UT_PARALLEL ON)
  set(BTLA_UT_KERNEL_JIT ON)
  set(BTLA_UT_KERNEL_INTRIN ON)
  set(BTLA_UT_KERNEL_WRAPPER ON)
endif(BTLA_UT_ALL)

set(UT_BUILD FALSE)
if(BTLA_UT_DEBUG OR BTLA_UT_PROLOGUE_A OR BTLA_UT_PROLOGUE_B OR BTLA_UT_EPILOGUE OR BTLA_UT_GEMM
OR BTLA_UT_WRAPPER OR BTLA_UT_PARALLEL OR BTLA_UT_KERNEL_JIT OR BTLA_UT_KERNEL_INTRIN
OR BTLA_UT_KERNEL_WRAPPER)
  set(UT_BUILD TRUE)
endif()

function(add_ut_flag UT_OPTION)
	if(${${UT_OPTION}})
	  target_compile_definitions(${PROJECT_NAME}_ut PRIVATE ${UT_OPTION})
	endif()
endfunction()

if(UT_BUILD)
	file(GLOB srcs ${PROJECT_NAME}/ut/*.cc ${PROJECT_NAME}/ut/*.cpp) #compile everthing even run parts of UTs
	file(GLOB ut_headers ${PROJECT_NAME}/ut/*.h)
  include_directories(${PROJECT_NAME})
	add_executable(${PROJECT_NAME}_ut ${srcs} ${headers} ${ut_headers})
  if(BTLA_UT_OPENMP)
    include(FindOpenMP)
    target_link_libraries(${PROJECT_NAME}_ut PRIVATE OpenMP::OpenMP_CXX)
  endif()
	if(NOT WIN32)
		if(NOT BTLA_UT_NOASAN)
		  target_compile_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		  target_link_options(${PROJECT_NAME}_ut PRIVATE -fsanitize=address)
		endif()
		target_link_options(${PROJECT_NAME}_ut PRIVATE -lpthread)
	endif()

	add_ut_flag(BTLA_UT_DEBUG)
	add_ut_flag(BTLA_UT_EPILOGUE)
	add_ut_flag(BTLA_UT_PROLOGUE_A)
	add_ut_flag(BTLA_UT_PROLOGUE_B)
	add_ut_flag(BTLA_UT_GEMM)
	add_ut_flag(BTLA_UT_PARALLEL)
	add_ut_flag(BTLA_UT_WRAPPER)
	add_ut_flag(BTLA_UT_KERNEL_INTRIN)
	add_ut_flag(BTLA_UT_KERNEL_JIT)
	add_ut_flag(BTLA_UT_KERNEL_WRAPPER)
	add_ut_flag(BTLA_UT_BENCHMARK)

	target_link_libraries(${PROJECT_NAME}_ut PRIVATE ${PROJECT_NAME})
endif(UT_BUILD)

