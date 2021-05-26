# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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
  set(DML_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.AI.DirectML.1.5.1)
  set(DML_SHARED_LIB DirectML.dll)

  # Restore nuget packages, which will pull down the DirectML redist package
  add_custom_command(
    OUTPUT ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib
    DEPENDS ${PACKAGES_CONFIG} ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM)

  include_directories(BEFORE "${DML_PACKAGE_DIR}/include")
  add_custom_target(RESTORE_PACKAGES ALL DEPENDS ${DML_PACKAGE_DIR}/bin/x64-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/x86-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/arm-win/DirectML.lib ${DML_PACKAGE_DIR}/bin/arm64-win/DirectML.lib)
  add_dependencies(RESTORE_PACKAGES nuget)
else()
  include_directories(${dml_INCLUDE_DIR})
endif()
