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
                      CMAKE_ARGS -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton ${VCPKG_DEPENDENCIES})

else()

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r22.12
                      PREFIX triton
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

endif() #if (WIN32)

ExternalProject_Get_Property(triton SOURCE_DIR)
set(TRITON_SRC ${SOURCE_DIR})

ExternalProject_Get_Property(triton BINARY_DIR)
set(TRITON_BIN ${BINARY_DIR}/binary)
set(TRITON_THIRD_PARTY ${BINARY_DIR}/third-party)
