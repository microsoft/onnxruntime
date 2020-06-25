
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
	
	
endif()