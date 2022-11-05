# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

FetchContent_Declare(
    GSL
    URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
    URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
    )

FetchContent_MakeAvailable(GSL)

set(GSL_TARGET "Microsoft.GSL::GSL")
set(GSL_INCLUDE_DIR "$<TARGET_PROPERTY:${GSL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
