# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Add delayloaded libraries to a target
function(target_delayload target_name)
    if(NOT MSVC)
        message(SEND_ERROR "Delayloading is only supported in MSVC")
    endif()
    foreach(lib ${ARGN})
        target_link_options(${target_name} PRIVATE /DELAYLOAD:"${lib}")
    endforeach()

    target_link_libraries(${target_name} PRIVATE delayimp.lib)
endfunction()
