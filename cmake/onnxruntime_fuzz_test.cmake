# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Check that the options are properly set for
# the fuzzing project
if (onnxruntime_FUZZ_ENABLED)
	message(STATUS "Building dependency protobuf-mutator and libfuzzer")
	
	# set the options used to control the protobuf-mutator build
	set(PROTOBUF_LIBRARIES "$<TARGET_FILE:libprotobuf>")
	set(PROTOBUF_INCLUDE_DIRS "$<TARGET_PROPERTY:libprotobuf,INCLUDE_DIRECTORIES>")
	set(LIB_PROTO_MUTATOR_TESTING OFF)
	
	# include the protobuf-mutator CMakeLists.txt rather than the projects CMakeLists.txt to avoid target clashes
	# with google test
	add_subdirectory("external/libprotobuf-mutator/src")
	
	# add the appropriate include directory and compilation flags
	# needed by the protobuf-mutator target and the libfuzzer
	set(PROTOBUF_MUT_INCLUDE_DIRS "external/libprotobuf-mutator")
	target_include_directories(protobuf-mutator PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${PROTOBUF_MUT_INCLUDE_DIRS})
	target_include_directories(protobuf-mutator-libfuzzer PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${PROTOBUF_MUT_INCLUDE_DIRS})
	target_compile_options(protobuf-mutator PRIVATE "/wd4244" "/wd4245" "/wd4267" "/wd4100" "/wd4456")
	target_compile_options(protobuf-mutator-libfuzzer PRIVATE "/wd4146" "/wd4267")
	
	# add Fuzzing Engine Build Configuration 
	message(STATUS "Building Fuzzing engine")
	
	# set Fuzz root directory
	set(SEC_FUZZ_ROOT ${TEST_SRC_DIR}/fuzzing)
	
	# Security fuzzing engine src file reference 
	set(SEC_FUZ_SRC "${SEC_FUZZ_ROOT}/src/BetaDistribution.cpp" 
					"${SEC_FUZZ_ROOT}/src/OnnxPrediction" 
					"${SEC_FUZZ_ROOT}/src/testlog.cpp" 
					"${SEC_FUZZ_ROOT}/src/test.cpp")
					
	# compile the executables
	onnxruntime_add_executable(onnxruntime_security_fuzz ${SEC_FUZ_SRC})
	
	# compile with c++17
	target_compile_features(onnxruntime_security_fuzz PUBLIC cxx_std_17)
	
	# Security fuzzing engine header file reference
	onnxruntime_add_include_to_target(onnxruntime_security_fuzz libprotobuf onnx onnxruntime)
	
	# Assign all include to one variable
	set(SEC_FUZ_INC "${SEC_FUZZ_ROOT}/include")
	set(INCLUDE_FILES ${SEC_FUZ_INC} "$<TARGET_PROPERTY:protobuf-mutator,INCLUDE_DIRECTORIES>")
	
	# add all these include directory to the Fuzzing engine
	target_include_directories(onnxruntime_security_fuzz PRIVATE ${INCLUDE_FILES})
	
	# add link libraries the project
	target_link_libraries(onnxruntime_security_fuzz libprotobuf onnx_proto onnxruntime protobuf-mutator)
	
	# add the dependencies
	add_dependencies(onnxruntime_security_fuzz libprotobuf onnx_proto onnxruntime protobuf-mutator)
	
	# copy the dlls to the execution directory
	add_custom_command(TARGET onnxruntime_security_fuzz POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime>  $<TARGET_FILE_DIR:onnxruntime_security_fuzz>
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:libprotobuf>  $<TARGET_FILE_DIR:onnxruntime_security_fuzz>)
endif()