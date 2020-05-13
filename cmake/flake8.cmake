#
# Setup running flake8 on python scripts to enforce PEP8 
# NOTE: Currently skips the check if flake8 is not installed. PRs editing python scripts are rare
#       so don't want to add a hard dependency on flake8 to all builds.
#
find_program(flake8_BIN NAMES flake8)
if(NOT flake8_BIN)
    message(WARNING "Could not find 'flake8' to check python scripts. Please install flake8 using pip.")
else()    
    set (FLAKE8_CONFIG ${REPO_ROOT}/.flake8)
endif()

if(flake8_BIN)
    # ideally we could use GLOB_RECURSE here but the scripts in onnxruntime/python/tools 
    # need a lot of work to be PEP8 compliant so deferring that for now and excluding them
    # by just using GLOB_RECURSE for a subset of the onnxruntime/python subdirectories
    file(GLOB python_scripts CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/__init__.py"
        "${ONNXRUNTIME_ROOT}/python/*.py"
    )
    
    file(GLOB_RECURSE python_scripts_2 CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/python/backend/*.py"
        "${ONNXRUNTIME_ROOT}/python/datasets/*.py"
        "${ONNXRUNTIME_ROOT}/python/training/*.py"
    )

    file(GLOB_RECURSE tools_scripts CONFIGURE_DEPENDS
        "${REPO_ROOT}/tools/*.py"
    )

    # These also need a lot of work to make PEP8 compliant
    # file(GLOB_RECURSE training_scripts CONFIGURE_DEPENDS
    #     "${ORTTRAINING_ROOT}/tools/*.py"
    # )

    list(APPEND python_scripts ${python_scripts_2} ${tools_scripts})

    source_group(TREE ${REPO_ROOT} FILES ${python_scripts})

    add_custom_target(pep8_check 
        ALL 
        DEPENDS ${python_scripts}
        WORKING_DIRECTORY 
        COMMAND echo "Checking python scripts for PEP8 conformance using flake8 with config ${FLAKE8_CONFIG}"
        MESSAGE(${python_scripts})
        COMMAND ${flake8_BIN} --config ${FLAKE8_CONFIG} ${python_scripts}
    )
endif()
