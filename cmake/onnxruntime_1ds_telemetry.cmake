# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file handles telemetry integration for non-Windows platforms
# (macOS, Linux, Android, iOS) using the 1DS SDK (cpp_client_telemetry).
# The SDK is fetched via FetchContent in onnxruntime_external_deps.cmake.

if(onnxruntime_USE_TELEMETRY AND NOT WIN32)
    if(NOT TARGET mat)
        message(FATAL_ERROR "Telemetry enabled for non-Windows but 'mat' target not found. "
                            "Ensure cpp_client_telemetry is fetched in onnxruntime_external_deps.cmake.")
    endif()

    message(STATUS "Enabling 1DS telemetry for non-Windows platforms")

    # Add compile definition so C++ code can detect 1DS telemetry at compile time
    add_compile_definitions(USE_1DS_TELEMETRY)

    # Platform-specific status messages
    if(APPLE)
        if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
            message(STATUS "  Platform: iOS")
        else()
            message(STATUS "  Platform: macOS")
        endif()
    elseif(ANDROID)
        message(STATUS "  Platform: Android")
    elseif(UNIX)
        message(STATUS "  Platform: Linux")
    endif()
else()
    if(NOT onnxruntime_USE_TELEMETRY)
        message(STATUS "Telemetry is disabled (use -Donnxruntime_USE_TELEMETRY=ON to enable)")
    endif()
endif()
