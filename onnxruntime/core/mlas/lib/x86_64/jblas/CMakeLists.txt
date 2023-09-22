cmake_minimum_required(VERSION 2.6...3.0.2)

project(jblas LANGUAGES CXX VERSION 0.1.0)

file(GLOB headers ${PROJECT_NAME}/*.h ${PROJECT_NAME}/*.hpp)
file(GLOB xbyak_headers ${PROJECT_NAME}/xbyak/*.h ${PROJECT_NAME}/xbyak/*.hpp)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(
	${PROJECT_NAME} INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if(WIN32)
	target_compile_definitions(${PROJECT_NAME} INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
	target_compile_options(${PROJECT_NAME} INTERFACE /GR /wd4068 /wd4849 /wd6262) 
	#4068 ignore unroll and GCC flags
	#4849 ignore collapse
	#6262 ignore stack too large

	target_link_options(${PROJECT_NAME} INTERFACE /STACK:3145728) #Stack require up to L2 cache size
else()
	target_compile_options(${PROJECT_NAME} INTERFACE -frtti) 
endif(WIN32)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindOpenMP)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
