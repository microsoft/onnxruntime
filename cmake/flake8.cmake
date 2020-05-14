#
# Setup running flake8 on python scripts to enforce PEP8 
# NOTE: Currently skips the check if flake8 is not installed. PRs editing python scripts are rare
#       so don't want to add a hard dependency on flake8 to all builds.
#

find_program(flake8_BIN NAMES flake8)

if(NOT flake8_BIN)
    # see if we can run the python module instead if there's no executable
    set(FLAKE8_NOT_FOUND false)
    exec_program("${PYTHON_EXECUTABLE}"
                 ARGS "-m flake8 --version"
                 RETURN_VALUE FLAKE8_NOT_FOUND)
    if(${FLAKE8_NOT_FOUND})
        message(WARNING "Could not find 'flake8' to check python scripts. Please install flake8 using pip.")
    else()
        set(flake8_BIN ${PYTHON_EXECUTABLE} "-m" "flake8")
    endif(${FLAKE8_NOT_FOUND})
endif()

if(flake8_BIN)
    # need to exclude a subset of scripts from ${ONNXRUNTIME_ROOT} so create a complete
    # list and then filter it
    file(GLOB_RECURSE python_scripts CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/*.py"
    )

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
endif()
