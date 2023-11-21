# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# There are effectively three ways to consume DirectML in this repo:
#
# 1) Public = the build points at a pre-built copy of DirectML distributed as a NuGet package.
# 2) Custom = the build points at a local copy of DirectML (bin/, include/, lib/). The dml_INCLUDE_DIR and
#    dml_LIB_DIR variables are also expected to be set to the custom build location.
# 3) Internal = the build points at the DirectML source repo and builds it as part of the main project.
#
# Build Type | onnxruntime_USE_CUSTOM_DIRECTML | dml_EXTERNAL_PROJECT
# -----------|---------------------------------|---------------------
# Public     | OFF                             | OFF
# Custom     | ON                              | OFF
# Internal   | ON                              | ON
#
# The "Public" build type is the default, and any mainline branches (e.g. master, rel-*) subject to CI
# should use the public build configuration. Topic branches can use the internal build type for testing,
# but they must be buildable with a public NuGet package before merging with a mainline branch.

set(onnxruntime_USE_CUSTOM_DIRECTML OFF CACHE BOOL "Depend on a custom/internal build of DirectML.")
set(dml_EXTERNAL_PROJECT OFF CACHE BOOL "Build DirectML as a source dependency.")
set(DML_SHARED_LIB DirectML.dll)

if (NOT onnxruntime_USE_CUSTOM_DIRECTML)
  if (NOT(MSVC) OR NOT(WIN32))
    message(FATAL_ERROR "NuGet packages are only supported for MSVC on Windows.")
  endif()

  # Retrieve the latest version of nuget
  include(ExternalProject)
  ExternalProject_Add(nuget
    PREFIX nuget
    URL "https://dist.nuget.org/win-x86-commandline/v5.3.0/nuget.exe"
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")

  set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/../NuGet.config)
  set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/../packages.config)
  get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/../packages ABSOLUTE)
  set(DML_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.AI.DirectML.Preview.1.13.0-devb7c99852e0b25080ea3388fea784008631dfd4c4)

  # Restore nuget packages, which will pull down the DirectML redist package.
  add_custom_command(
    OUTPUT
      ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib
    DEPENDS
      ${PACKAGES_CONFIG}
      ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  include_directories(BEFORE "${DML_PACKAGE_DIR}/include")
  add_custom_target(
    RESTORE_PACKAGES ALL
    DEPENDS
      ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib
      ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib
  )

  add_dependencies(RESTORE_PACKAGES nuget)
else()
  if (dml_EXTERNAL_PROJECT)
    set(dml_preset_config $<IF:$<CONFIG:Debug>,debug,release>)
    set(dml_preset_name ${onnxruntime_target_platform}-win-redist-${dml_preset_config})
    target_compile_definitions(DirectML INTERFACE DML_TARGET_VERSION_USE_LATEST=1)
    include(ExternalProject)
    ExternalProject_Add(
        directml_repo
        GIT_REPOSITORY https://dev.azure.com/microsoft/WindowsAI/_git/DirectML
        GIT_TAG d460f0f46967bea878786f1bed69487692c779bf
        GIT_SHALLOW OFF # not allowed when GIT_TAG is a commit SHA, which is preferred (it's stable, unlike branches)
        GIT_PROGRESS ON
        BUILD_IN_SOURCE ON
        CONFIGURE_COMMAND ${CMAKE_COMMAND} --preset ${dml_preset_name} -DDML_BUILD_TESTS=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build --preset ${dml_preset_name}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install build/${dml_preset_name}
        STEP_TARGETS install
    )

    # Target that consumers can use to link with the internal build of DirectML.
    set(directml_install_path ${CMAKE_BINARY_DIR}/directml_repo-prefix/src/directml_repo/build/${dml_preset_name}/install)
    set(DML_PACKAGE_DIR ${directml_install_path})
    add_library(DirectML INTERFACE)
    target_link_libraries(DirectML INTERFACE ${directml_install_path}/lib/DirectML.lib)
    add_dependencies(DirectML directml_repo-install)
    include_directories(BEFORE ${directml_install_path}/include)
  else()
    include_directories(BEFORE ${dml_INCLUDE_DIR})
    set(DML_PACKAGE_DIR ${dml_INCLUDE_DIR}/..)
  endif()
endif()
