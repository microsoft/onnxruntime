include(ExternalProject)

if (WIN32)

  function(get_vcpkg)
    ExternalProject_Add(vcpkg
      GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
      PREFIX vcpkg
      CONFIGURE_COMMAND ""
      INSTALL_COMMAND ""
      UPDATE_COMMAND ""
      BUILD_COMMAND "<SOURCE_DIR>/bootstrap-vcpkg.bat")

    ExternalProject_Get_Property(vcpkg SOURCE_DIR)
    set(VCPKG_SRC ${SOURCE_DIR} PARENT_SCOPE)
    set(VCPKG_DEPENDENCIES "vcpkg" PARENT_SCOPE)
  endfunction()

  function(vcpkg_install PACKAGE_NAME)
    add_custom_command(
      OUTPUT ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${onnxruntime_target_platform}-windows/BUILD_INFO
      COMMAND ${VCPKG_SRC}/vcpkg install ${PACKAGE_NAME}:${onnxruntime_target_platform}-windows
      WORKING_DIRECTORY ${VCPKG_SRC}
      DEPENDS vcpkg)

    add_custom_target(get${PACKAGE_NAME}
      ALL
      DEPENDS ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${onnxruntime_target_platform}-windows/BUILD_INFO)

    list(APPEND VCPKG_DEPENDENCIES "get${PACKAGE_NAME}")
    set(VCPKG_DEPENDENCIES ${VCPKG_DEPENDENCIES} PARENT_SCOPE)
  endfunction()

  get_vcpkg()
  vcpkg_install(openssl)
  vcpkg_install(openssl-windows)
  vcpkg_install(rapidjson)
  vcpkg_install(re2)
  vcpkg_install(boost-interprocess)
  vcpkg_install(boost-stacktrace)
  vcpkg_install(zlib)
  vcpkg_install(pthread)
  vcpkg_install(b64)

  add_dependencies(getb64 getpthread)
  add_dependencies(getpthread getzlib)
  add_dependencies(getzlib getboost-stacktrace)
  add_dependencies(getboost-stacktrace getboost-interprocess)
  add_dependencies(getboost-interprocess getre2)
  add_dependencies(getre2 getrapidjson)
  add_dependencies(getrapidjson getopenssl-windows)
  add_dependencies(getopenssl-windows getopenssl)

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r22.12
                      PREFIX triton
                      CMAKE_ARGS -DVCPKG_TARGET_TRIPLET=${onnxruntime_target_platform}-windows -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton ${VCPKG_DEPENDENCIES})

else()

  set(OPENSSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openssl-src) # default path by CMake
  set(OPENSSL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/openssl-install)
  set(OPENSSL_INCLUDE_DIR ${OPENSSL_INSTALL_DIR}/include)
  set(OPENSSL_CONFIGURE_COMMAND ${OPENSSL_SOURCE_DIR}/config)

  ExternalProject_Add(openssl
                      GIT_REPOSITORY https://github.com/openssl/openssl.git
                      GIT_TAG openssl-3.0.6
                      SOURCE_DIR ${OPENSSL_SOURCE_DIR}
                      USES_TERMINAL_DOWNLOAD TRUE
                      CONFIGURE_COMMAND ${OPENSSL_CONFIGURE_COMMAND} --prefix=${OPENSSL_INSTALL_DIR} --openssldir=${OPENSSL_INSTALL_DIR}
                      BUILD_COMMAND make
                      TEST_COMMAND ""
                      INSTALL_COMMAND make install
                      INSTALL_DIR ${OPENSSL_INSTALL_DIR})
	      
  file(MAKE_DIRECTORY ${OPENSSL_INCLUDE_DIR})

  add_library(OpenSSL::SSL STATIC IMPORTED GLOBAL)
  set_property(TARGET OpenSSL::SSL PROPERTY IMPORTED_LOCATION ${OPENSSL_INSTALL_DIR}/lib64/libssl.so)
  set_property(TARGET OpenSSL::SSL PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE_DIR})
  add_dependencies(OpenSSL::SSL OpenSSL)

  add_library(OpenSSL::Crypto STATIC IMPORTED GLOBAL)
  set_property(TARGET OpenSSL::Crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_INSTALL_DIR}/lib64/libcrypto.so)
  set_property(TARGET OpenSSL::Crypto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OPENSSL_INCLUDE_DIR})
  add_dependencies(OpenSSL::Crypto OpenSSL)

  ExternalProject_Add(rapidjson
	              GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
	              GIT_TAG f54b0e47a08782a6131cc3d60f94d038fa6e0a51
		      PREFIX rapidjson
	              CMAKE_ARGS -DRAPIDJSON_BUILD_TESTS=OFF -DRAPIDJSON_BUILD_DOC=OFF -DRAPIDJSON_BUILD_EXAMPLES=OFF)
	 
  ExternalProject_Get_Property(rapidjson source_dir)
  set(RAPIDJSON_INCLUDE_DIR ${source_dir}/include)
  include_directories(${RAPIDJSON_INCLUDE_DIR})
  message(WARNING "rapidjson inc: " ${RAPIDJSON_INCLUDE_DIR})
  
  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r22.12
                      PREFIX triton
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton rapidjson openssl)
 
endif() #if (WIN32)

ExternalProject_Get_Property(triton SOURCE_DIR)
set(TRITON_SRC ${SOURCE_DIR})

ExternalProject_Get_Property(triton BINARY_DIR)
set(TRITON_BIN ${BINARY_DIR}/binary)
set(TRITON_THIRD_PARTY ${BINARY_DIR}/third-party)
