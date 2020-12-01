# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.0)

# Determines the version of a native nuget package from the root packages.config.
#
# id  : package id
# out : name of variable to set result
function(pkg_version id out packages_config)
    file(READ ${packages_config} packages_config_contents)
    string(REGEX MATCH "package[ ]*id[ ]*=[ ]*\"${id}\"" found_package_id ${packages_config_contents})
    if (NOT(found_package_id))
        message(FATAL_ERROR "Could not find '${id}' in packages.config!")
    endif()

    set(pattern ".*id[ ]*=[ ]*\"${id}\"[ ]+version=\"([0-9a-zA-Z\\.-]+)\"[ ]+targetFramework.*")
    string(REGEX REPLACE ${pattern} "\\1" version ${packages_config_contents})
    set(${out} ${version} PARENT_SCOPE)
endfunction()