#
# Setup running flake8 on python scripts to enforce PEP8 
# NOTE: Currently skips the check if flake8 is not installed. PRs editing python scripts are rare
#       so don't want to add a hard dependency on flake8 to all builds.
#

find_program(flake8_BIN NAMES flake8)

if(flake8_BIN)
    exec_program(${flake8_BIN} ARGS "--version" OUTPUT_VARIABLE FLAKE8_VERSION)
else()
    # see if we can run the python module instead if there's no executable
    set(FLAKE8_NOT_FOUND false)
    exec_program("${PYTHON_EXECUTABLE}"
                 ARGS "-m flake8 --version"
                 OUTPUT_VARIABLE FLAKE8_VERSION
                 RETURN_VALUE FLAKE8_NOT_FOUND)
    if(${FLAKE8_NOT_FOUND})
        message(WARNING "Could not find 'flake8' to check python scripts. Please install flake8 using pip.")
    else()
        set(flake8_BIN ${PYTHON_EXECUTABLE} "-m" "flake8")
    endif(${FLAKE8_NOT_FOUND})
endif()

if(flake8_BIN)
    # check flake8 version
    string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" _flake8_ver_check "${FLAKE8_VERSION}")
    if(NOT "${_flake8_ver_check}" STREQUAL "")
        set(FLAKE8_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(FLAKE8_VERSION_MINOR ${CMAKE_MATCH_2})
    else()
        set(FLAKE8_VERSION_MAJOR 0)
        set(FLAKE8_VERSION_MINOR 0)
    endif()

    math(EXPR FLAKE8_VERSION_DECIMAL "(${FLAKE8_VERSION_MAJOR} * 100) + ${FLAKE8_VERSION_MINOR}")

    # require minimum version that we've tested with (3.8)
    if (${FLAKE8_VERSION_DECIMAL} GREATER_EQUAL 308)
        # need to exclude a subset of scripts from ${ONNXRUNTIME_ROOT} so create a complete
        # list and then filter it
        file(GLOB_RECURSE python_scripts CONFIGURE_DEPENDS
            "${ONNXRUNTIME_ROOT}/*.py"
        )

        # generated flatbuffer schema files
        list(FILTER python_scripts EXCLUDE REGEX "onnxruntime/core/flatbuffers/ort_flatbuffers_py")

        # scripts in these directories still need updating
        list(FILTER python_scripts EXCLUDE REGEX "onnxruntime/core/providers/nuphar")
        list(FILTER python_scripts EXCLUDE REGEX "onnxruntime/python/tools")
        list(FILTER python_scripts EXCLUDE REGEX "onnxruntime/test")

        # can just add the 'tools' directory and flake8 will recurse into it
        list(APPEND python_scripts ${REPO_ROOT}/tools/)

        # Training scripts need updating to make PEP8 compliant.
        # file(GLOB_RECURSE training_scripts CONFIGURE_DEPENDS
        #     "${ORTTRAINING_ROOT}/*.py"
        # )

        source_group(TREE ${REPO_ROOT} FILES ${python_scripts})

        add_custom_target(pep8_check 
            ALL 
            DEPENDS ${python_scripts}
            WORKING_DIRECTORY 
            COMMAND echo "Checking python scripts for PEP8 conformance using flake8"
            #MESSAGE(${python_scripts})
            COMMAND ${flake8_BIN} "--config" "${REPO_ROOT}/.flake8" ${python_scripts}
            VERBATIM
        )
    else()
        message(WARNING "'flake8' version is too old. Requires 3.8 or later. Found ${FLAKE8_VERSION}")
    endif()
endif()
