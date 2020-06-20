
# Check that the options are propely set for
# the fuzzing project
if (onnxruntime_FUZZ_ENABLED)
	message(STATUS "Building dependency protobuf-mutator")
	
	# Ensure that the protobuf-mutator library
	# uses the protobuf library in the onnxruntime project
	# by creating and copying into CMAKE_MODULE_PATH an empty FindProtobuf.cmake
	# file and configuring the required protobuf options for the project here
	file(WRITE "external/libprotobuf-mutator/cmake/external/FindProtobuf.cmake" "")
	set(Protobuf_LIBRARIES "$<TARGET_FILE:libprotobuf>")
	set(Protobuf_SRC_ROOT_FOLDER ${protobuf_SOURCE_DIR})
	set(Protobuf_BINARY_DIR ${protobuf_BINARY_DIR})
	set(PROTOBUF_LIBRARIES ${Protobuf_LIBRARIES})
	set(PROTOBUF_INCLUDE_DIRS "$<TARGET_PROPERTY:libprotobuf,INCLUDE_DIRECTORIES>")
	add_subdirectory("external/libprotobuf-mutator")
	
endif()