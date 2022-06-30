if (onnxruntime_USE_TVM)
    message(STATUS "onnxruntime_USE_TVM: Fetch ipp-crypto for TVM EP")

    FetchContent_Declare(
            ipp_crypto
            GIT_REPOSITORY https://github.com/intel/ipp-crypto.git
            GIT_TAG        ippcp_2021.5
    )

    FetchContent_GetProperties(ipp_crypto)
    if(NOT ipp_crypto_POPULATED)
        FetchContent_Populate(ipp_crypto)
    endif()

    set(ipp_crypto_INCLUDE_DIRS ${ipp_crypto_SOURCE_DIR}/include)

endif()
