# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.0)

# Determines the version of a native nuget package from the root packages.config.
#
# id  : package id
# out : name of variable to set result
function(package_version id out packages_config)
    file(READ ${packages_config} packages_config_contents)
    string(REGEX MATCH "package[ ]*id[ ]*=[ ]*\"${id}\"" found_package_id ${packages_config_contents})
    if (NOT(found_package_id))
        message(FATAL_ERROR "Could not find '${id}' in packages.config!")
    endif()

    set(pattern ".*id[ ]*=[ ]*\"${id}\"[ ]+version=\"([0-9a-zA-Z\\.-]+)\"[ ]+targetFramework.*")
    string(REGEX REPLACE ${pattern} "\\1" version ${packages_config_contents})
    set(${out} ${version} PARENT_SCOPE)
endfunction()

# Downloads the nuget packages based on packages.config
function(
    add_fetch_nuget_target
    nuget_target # Target to be written to
    target_dependency # The file in the nuget package that is needed
)
    # Pull down the nuget packages
    if (NOT(MSVC) OR NOT(WIN32))
    message(FATAL_ERROR "NuGet packages are only supported for MSVC on Windows.")
    endif()

    # Retrieve the latest version of nuget
    include(ExternalProject)
    ExternalProject_Add(nuget_exe
                        PREFIX nuget_exe
                        URL "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
                        DOWNLOAD_NO_EXTRACT 1
                        CONFIGURE_COMMAND ""
                        BUILD_COMMAND ""
                        UPDATE_COMMAND ""
                        INSTALL_COMMAND "")

    set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/../NuGet.config)
    set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/../packages.config)
    get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/../packages ABSOLUTE)

    # Restore nuget packages
    add_custom_command(
    OUTPUT ${target_dependency}
    DEPENDS ${PACKAGES_CONFIG} ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget_exe/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM)

    add_custom_target(${nuget_target} DEPENDS ${target_dependency})
    add_dependencies(${nuget_target} nuget_exe)
endfunction()
