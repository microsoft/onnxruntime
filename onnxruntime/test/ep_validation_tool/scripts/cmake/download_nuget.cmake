# Forked from https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?path=/scripts/cmake/getdependency.cmake&version=GBmain
# at 0bc1df4d293a75b776ea0ecbe217a22724f77837.

# Creates a custom target to obtain the output file(s) via NuGet.
# PKG_NAME - NuGet package name (without version string)
# PKG_VERSION - NuGet package version
# CONFIG_FILE - NuGet configuration to use
# OUT_DIR - Where to save and unpack the NuGet file
function(nugetDL PKG_NAME PKG_VERSION CONFIG_FILE OUT_DIR)
    # Find nuget.exe.
    find_program(NUGET_EXE NAMES nuget.exe)
    if (NUGET_EXE STREQUAL "NUGET_EXE-NOTFOUND")
        message(FATAL "nuget.exe not found. Please add it to your PATH.")
    endif()

    # Create output dir.
    file(MAKE_DIRECTORY ${OUT_DIR})

    # Run nuget.exe to download and install the package to OUT_DIR.
    # NOTE: Version is required. If not specified, nuget.exe will fail reporting it cannot find the package.
    execute_process(COMMAND ${NUGET_EXE} install ${PKG_NAME} -o ${OUT_DIR}/ -DependencyVersion Ignore -ConfigFile ${CONFIG_FILE} -Verbosity detailed -Version ${PKG_VERSION}
        RESULT_VARIABLE NUGET_RETURN_VALUE)
    if(NOT "${NUGET_RETURN_VALUE}" EQUAL "0")
        message(FATAL_ERROR "NuGet restore for package ${PKG_NAME} with version '${PKG_VERSION}' to directory ${OUT_DIR} failed with error code ${NUGET_RETURN_VALUE}.")
    endif()

    # Get package version without metadata.
    string(REPLACE "+" ";" PKG_VERSION_MOD ${PKG_VERSION})
    list(GET PKG_VERSION_MOD 0 PKG_VERSION_NO_METADATA)

    # Import package cmake file if it exists.
    if(EXISTS ${OUT_DIR}/${PKG_NAME}.${PKG_VERSION_NO_METADATA}/build/native/${PKG_NAME}.cmake)
        include(${OUT_DIR}/${PKG_NAME}.${PKG_VERSION_NO_METADATA}/build/native/${PKG_NAME}.cmake)
    elseif(EXISTS ${OUT_DIR}/${PKG_NAME}.${PKG_VERSION_NO_METADATA}/build/${PKG_NAME}.cmake)
        include(${OUT_DIR}/${PKG_NAME}.${PKG_VERSION_NO_METADATA}/build/${PKG_NAME}.cmake)
    endif()

    # Return NuGet dir.
    set(NUGET_DIR "${OUT_DIR}/${PKG_NAME}.${PKG_VERSION_NO_METADATA}" PARENT_SCOPE)
endfunction()