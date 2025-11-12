# Forked from https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?path=/scripts/cmake/parse_encryption_key_packages.cmake&version=GC2e88aa051dc10c8a8434a6240e25be4ca6963c9c

# Parses encryption key packages JSON file and sets the packages and versions
# input: ENCRYPTION_KEY_PACKAGES_PATH -- path to the JSON file containing encryption key packages
#        ENC_KEY_DIR -- directory where the encryption keys will be stored
function(parse_encryption_key_packages ENCRYPTION_KEY_PACKAGES_PATH ENC_KEY_DIR SYMBOL_ENCRYPTION_KEY_MANAGER_INCLUDES SYMBOL_KEY_ID_TO_KEY_MAPPING)

    file(MAKE_DIRECTORY ${ENC_KEY_DIR})
    file(READ ${ENCRYPTION_KEY_PACKAGES_PATH} ENCRYPTION_KEY_PACKAGES_JSON_STRING)
    string(JSON NUM_OF_ENCRYPTION_KEY_PACKAGES LENGTH ${ENCRYPTION_KEY_PACKAGES_JSON_STRING})
    math(EXPR LAST_ENCRYPTION_KEY_PACKAGE_INDEX "${NUM_OF_ENCRYPTION_KEY_PACKAGES} - 1")
    # This is used to generate include statements automatically for each encryption key
    set(ENCRYPTION_KEY_INCLUDES "")
    # This is used to generate mapping from key ID to key (e.g., {"long", actual_key})
    set(KEY_ID_TO_KEY_MAPPING "")
    # This is used to generate mapping from key package name to key ID,
    set(KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING "")
    foreach(IDX RANGE ${LAST_ENCRYPTION_KEY_PACKAGE_INDEX})
        string(JSON CURR_ENCRYPTION_KEY_PACKAGE_NAME GET ${ENCRYPTION_KEY_PACKAGES_JSON_STRING} ${IDX} "encryption_key_package_name")
        string(JSON CURR_ENCRYPTION_KEY_PACKAGE_VERSION GET ${ENCRYPTION_KEY_PACKAGES_JSON_STRING} ${IDX} "encrption_key_package_version")
        string(JSON CURR_ENCRYPTION_KEY_ID GET ${ENCRYPTION_KEY_PACKAGES_JSON_STRING} ${IDX} "encryption_key_id")
        
        nugetDL(${CURR_ENCRYPTION_KEY_PACKAGE_NAME}
                ${CURR_ENCRYPTION_KEY_PACKAGE_VERSION}
                ${CMAKE_CURRENT_LIST_DIR}/Nuget.config
                ${CMAKE_CURRENT_LIST_DIR}/packages)
        file(COPY ${${CURR_ENCRYPTION_KEY_PACKAGE_NAME}_include_path}/key/ DESTINATION ${ENC_KEY_DIR}/${CURR_ENCRYPTION_KEY_ID})

        string(TOUPPER ${CURR_ENCRYPTION_KEY_ID} CURR_ENCRYPTION_KEY_ID_UPPER)
        set(INCLUDE_LINE "#include \"${CURR_ENCRYPTION_KEY_ID}/key.h\"\n")
        set(DECLARATION_LINE "const char ${CURR_ENCRYPTION_KEY_ID_UPPER}[] = MODELKEY;\n")
        set(UNDEF_LINE "#undef MODELKEY")

        # Handling visual look of statements when embedded in source file
        set(KEY_ID_KEY_PAIR "{\"${CURR_ENCRYPTION_KEY_ID}\", ${CURR_ENCRYPTION_KEY_ID_UPPER}}")
        set(KEY_PACKAGE_NAME_KEY_ID_PAIR "{\"${CURR_ENCRYPTION_KEY_PACKAGE_NAME}.${CURR_ENCRYPTION_KEY_PACKAGE_VERSION}\", ${CURR_ENCRYPTION_KEY_ID_UPPER}}")
        if (${IDX} EQUAL 0)
            set(KEY_ID_TO_KEY_MAPPING "${KEY_ID_KEY_PAIR}")
            set(ENCRYPTION_KEY_INCLUDES "${INCLUDE_LINE}${DECLARATION_LINE}${UNDEF_LINE}")
            set(KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING "${KEY_PACKAGE_NAME_KEY_ID_PAIR}")
        else()
            set(KEY_ID_TO_KEY_MAPPING "${KEY_ID_TO_KEY_MAPPING}\t${KEY_ID_KEY_PAIR}")
            set(ENCRYPTION_KEY_INCLUDES "${ENCRYPTION_KEY_INCLUDES}\n${INCLUDE_LINE}${DECLARATION_LINE}${UNDEF_LINE}")
            set(KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING "${KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING}\t${KEY_PACKAGE_NAME_KEY_ID_PAIR}")
        endif()

        if(NOT ${IDX} EQUAL ${LAST_ENCRYPTION_KEY_PACKAGE_INDEX})
            set(KEY_ID_TO_KEY_MAPPING "${KEY_ID_TO_KEY_MAPPING},\n")
            set(KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING "${KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING},\n")
        endif()
        set(KEY_ID_TO_KEY_MAPPING "${KEY_ID_TO_KEY_MAPPING}")
        set(KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING, "${KEY_PACKAGE_NAME_TO_KEY_ID_MAPPING}")
    endforeach()

    # Define symbols to be used in main CMakeLists.txt
    set(${SYMBOL_ENCRYPTION_KEY_MANAGER_INCLUDES} ${ENCRYPTION_KEY_INCLUDES} PARENT_SCOPE)
    set(${SYMBOL_KEY_ID_TO_KEY_MAPPING} ${KEY_ID_TO_KEY_MAPPING} PARENT_SCOPE)
    
endfunction()
