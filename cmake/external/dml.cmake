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
  set(DML_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.AI.DirectML.Preview.1.9.0-dev2b57b4f738b1d0dcc2dd31ecd502e36f4e3ea5a0)
  set(DML_SHARED_LIB DirectML.dll)

  # If using the preview package, extract the SHA-1 from the path so we can unmangle the filenames later.
  # e.g. "Microsoft.AI.DirectML.Preview.1.9.0-dev2b57b4f738b1d0dcc2dd31ecd502e36f4e3ea5a0"
  if(DML_PACKAGE_DIR MATCHES ".*Preview.*-dev(.*)")
    set(DML_PREVIEW_FILENAME_SUFFIX ".${CMAKE_MATCH_1}")
  endif()

  # Restore nuget packages, which will pull down the DirectML redist package.
  add_custom_command(
    OUTPUT
      ${DML_PACKAGE_DIR}/bin/x64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
      ${DML_PACKAGE_DIR}/bin/x86-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
      ${DML_PACKAGE_DIR}/bin/arm-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
      ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
    DEPENDS
      ${PACKAGES_CONFIG}
      ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  # If using a preview package, unmangle the filenames from the nuget so they're useable.
  # e.g. Map DirectML.2b57b4f738b1d0dcc2dd31ecd502e36f4e3ea5a0.dll -> DirectML.dll
  if(DEFINED DML_PREVIEW_FILENAME_SUFFIX)
    add_custom_command(
      OUTPUT
        ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib
        ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib
        ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib
        ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib
      DEPENDS
        ${DML_PACKAGE_DIR}/bin/x64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
        ${DML_PACKAGE_DIR}/bin/x86-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
        ${DML_PACKAGE_DIR}/bin/arm-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib
        ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib

      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib          ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.dll          ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.pdb          ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.pdb
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.dll    ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.Debug.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.pdb    ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.Debug.pdb

      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x86-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib          ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x86-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.dll          ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x86-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.pdb          ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.pdb
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.dll    ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.Debug.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.pdb    ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.Debug.pdb

      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib          ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.dll          ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.pdb          ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.pdb
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.dll    ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.Debug.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.pdb    ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.Debug.pdb

      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.lib        ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.dll        ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML${DML_PREVIEW_FILENAME_SUFFIX}.pdb        ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.pdb
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.dll  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.Debug.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.Debug${DML_PREVIEW_FILENAME_SUFFIX}.pdb  ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.Debug.pdb

      VERBATIM
    )
  endif()

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
    
    include(ExternalProject)
    ExternalProject_Add(
        directml_repo
        GIT_REPOSITORY https://dev.azure.com/microsoft/WindowsAI/_git/DirectML
        GIT_TAG 2290bd6495fdf8c35822816213516d13f3742cc9
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
    add_library(DirectML INTERFACE)
    target_link_libraries(DirectML INTERFACE ${directml_install_path}/lib/DirectML.lib)
    add_dependencies(DirectML directml_repo-install)
    include_directories(BEFORE ${directml_install_path}/include)
  else()
    include_directories(${dml_INCLUDE_DIR})
  endif()
endif()
