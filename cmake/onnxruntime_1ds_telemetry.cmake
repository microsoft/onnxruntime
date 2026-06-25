# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file handles telemetry integration for non-Windows platforms
# (macOS, Linux, Android, iOS) using the 1DS SDK (cpp_client_telemetry).
# The SDK is provided either by the vcpkg port "cpp-client-telemetry" (target
# MSTelemetry::mat) or fetched via FetchContent (target mat) in
# onnxruntime_external_deps.cmake.

if(onnxruntime_USE_TELEMETRY AND NOT WIN32)
    if(NOT TARGET mat AND NOT TARGET MSTelemetry::mat)
        message(FATAL_ERROR "Telemetry enabled for non-Windows but no 1DS SDK target "
                            "('mat' or 'MSTelemetry::mat') was found. Ensure cpp_client_telemetry "
                            "is provided via the vcpkg port or fetched in onnxruntime_external_deps.cmake.")
    endif()

    message(STATUS "Enabling 1DS telemetry for non-Windows platforms")

    # USE_1DS_TELEMETRY is defined on the onnxruntime_common target in onnxruntime_common.cmake
    # (its only consumer is core/platform/posix/env.cc), so it is not added globally here.

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
endif()
