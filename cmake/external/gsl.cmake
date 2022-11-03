# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

FetchContent_Declare(
    GSL
    GIT_REPOSITORY https://github.com/microsoft/gsl
    GIT_TAG a3534567187d2edc428efd3f13466ff75fe5805c  # v4.0.0
    GIT_SHALLOW ON
    )

FetchContent_MakeAvailable(GSL)

set(GSL_TARGET "Microsoft.GSL::GSL")
set(GSL_INCLUDE_DIR "$<TARGET_PROPERTY:${GSL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
