# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The embedded curl transport is supported only on Linux.")
endif()

if(TARGET CURL::libcurl)
  message(FATAL_ERROR
    "A CURL::libcurl target already exists. Linux telemetry requires its pinned static curl target "
    "so packaged binaries never depend on a system libcurl.")
endif()

function(onnxruntime_telemetry_save_cache_variable name)
  get_property(is_set CACHE "${name}" PROPERTY TYPE SET)
  set("_onnxruntime_telemetry_cache_${name}_is_set" "${is_set}" PARENT_SCOPE)
  if(is_set)
    get_property(type CACHE "${name}" PROPERTY TYPE)
    get_property(help CACHE "${name}" PROPERTY HELPSTRING)
    get_property(value CACHE "${name}" PROPERTY VALUE)
    set("_onnxruntime_telemetry_cache_${name}_type" "${type}" PARENT_SCOPE)
    set("_onnxruntime_telemetry_cache_${name}_help" "${help}" PARENT_SCOPE)
    set("_onnxruntime_telemetry_cache_${name}_value" "${value}" PARENT_SCOPE)
  endif()
endfunction()

function(onnxruntime_telemetry_restore_cache_variable name)
  if(_onnxruntime_telemetry_cache_${name}_is_set)
    set(cache_type "${_onnxruntime_telemetry_cache_${name}_type}")
    if(cache_type STREQUAL "UNINITIALIZED")
      set(cache_type STRING)
    endif()
    set(${name}
      "${_onnxruntime_telemetry_cache_${name}_value}"
      CACHE "${cache_type}"
      "${_onnxruntime_telemetry_cache_${name}_help}"
      FORCE)
  else()
    unset(${name} CACHE)
  endif()
endfunction()

# Isolate all dependency options from parent normal variables and cache entries. Keep only the
# HTTP/HTTPS feature surface required by 1DS and restore the caller's cache after creating targets.
block(SCOPE_FOR VARIABLES POLICIES)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

  set(_onnxruntime_telemetry_off_options
    BUILD_SHARED_LIBS
    BUILD_STATIC_CURL
    BUILD_TESTING
    SHARE_LIB_OBJECT
    ENABLE_PROGRAMS
    ENABLE_TESTING
    ENABLE_DEBUG
    ENABLE_UNICODE
    CURL_BUILD_EVERYTHING
    CURL_CLANG_TIDY
    CURL_CODE_COVERAGE
    CURL_DEBUG_GLOBAL_MEM
    CURL_DROP_UNUSED
    CURL_GCC_ANALYZER
    CURL_LINT
    CURL_LTO
    CURL_STATIC_CRT
    CURL_WERROR
    CURL_WINDOWS_SSPI
    PICKY_COMPILER
    MSVC_STATIC_RUNTIME
    USE_APPLE_IDN
    USE_APPLE_SECTRUST
    USE_WIN32_IDN
    USE_WIN32_LDAP
    GEN_FILES
    UNSAFE_BUILD
    INSTALL_MBEDTLS_HEADERS
    MBEDTLS_FATAL_WARNINGS
    USE_SHARED_MBEDTLS_LIBRARY
    LINK_WITH_TRUSTED_STORAGE
    BUILD_CURL_EXE
    BUILD_EXAMPLES
    BUILD_LIBCURL_DOCS
    BUILD_MISC_DOCS
    ENABLE_CURL_MANUAL
    CURL_ENABLE_EXPORT_TARGET
    CURL_ENABLE_NTLM
    CURL_ENABLE_SMB
    CURL_DISABLE_HTTP
    CURL_DISABLE_BASIC_AUTH
    CURL_DISABLE_BINDLOCAL
    CURL_DISABLE_FORM_API
    CURL_DISABLE_GETOPTIONS
    CURL_DISABLE_HEADERS_API
    CURL_DISABLE_HTTP_AUTH
    CURL_DISABLE_LIBCURL_OPTION
    CURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG
    CURL_DISABLE_PARSEDATE
    CURL_DISABLE_PROGRESS_METER
    CURL_DISABLE_PROXY
    CURL_DISABLE_SHA512_256
    CURL_DISABLE_SHUFFLE_DNS
    CURL_DISABLE_SOCKETPAIR
    CURL_DISABLE_TYPECHECK
    CURL_DISABLE_VERBOSE_STRINGS
    CURL_USE_OPENSSL
    CURL_USE_SCHANNEL
    CURL_USE_WOLFSSL
    CURL_USE_GNUTLS
    CURL_USE_RUSTLS
    CURL_USE_PKGCONFIG
    CURL_USE_CMAKECONFIG
    CURL_CA_FALLBACK
    CURL_CA_NATIVE
    CURL_CA_SEARCH_SAFE
    USE_LIBIDN2
    CURL_USE_LIBPSL
    CURL_USE_LIBSSH2
    CURL_USE_LIBSSH
    CURL_USE_GSSAPI
    CURL_USE_GSASL
    CURL_USE_LIBBACKTRACE
    CURL_USE_LIBUV
    USE_NGHTTP2
    USE_NGTCP2
    USE_QUICHE
    USE_HTTPSRR
    USE_ECH
    USE_SSLS_EXPORT
    USE_PROXY_HTTP3
    ENABLE_ARES
    ENABLE_UNIX_SOCKETS)

  set(_onnxruntime_telemetry_on_options
    BUILD_STATIC_LIBS
    USE_STATIC_MBEDTLS_LIBRARY
    LINK_WITH_PTHREAD
    DISABLE_PACKAGE_CONFIG_AND_INSTALL
    CURL_DISABLE_INSTALL
    CURL_ENABLE_SSL
    CURL_USE_MBEDTLS
    ENABLE_THREADED_RESOLVER
    ENABLE_IPV6
    HTTP_ONLY
    HAVE_MBEDTLS_DES_CRYPT_ECB
    CURL_DISABLE_ALTSVC
    CURL_DISABLE_CA_SEARCH
    CURL_DISABLE_SRP
    CURL_DISABLE_HSTS
    CURL_DISABLE_COOKIES
    CURL_DISABLE_DICT
    CURL_DISABLE_FILE
    CURL_DISABLE_FTP
    CURL_DISABLE_GOPHER
    CURL_DISABLE_IMAP
    CURL_DISABLE_LDAP
    CURL_DISABLE_LDAPS
    CURL_DISABLE_MQTT
    CURL_DISABLE_NETRC
    CURL_DISABLE_MIME
    CURL_DISABLE_POP3
    CURL_DISABLE_RTSP
    CURL_DISABLE_SMTP
    CURL_DISABLE_TELNET
    CURL_DISABLE_TFTP
    CURL_DISABLE_WEBSOCKETS
    CURL_DISABLE_IPFS
    CURL_DISABLE_DOH
    CURL_DISABLE_AWS
    CURL_DISABLE_BEARER_AUTH
    CURL_DISABLE_DIGEST_AUTH
    CURL_DISABLE_KERBEROS_AUTH
    CURL_DISABLE_NEGOTIATE_AUTH)

  set(_onnxruntime_telemetry_string_options
    MBEDTLS_CONFIG_FILE
    MBEDTLS_USER_CONFIG_FILE
    MBEDTLS_TARGET_PREFIX
    CURL_CA_BUNDLE
    CURL_CA_PATH
    CURL_CA_EMBED
    CURL_DEFAULT_SSL_BACKEND
    CURL_ZLIB
    CURL_BROTLI
    CURL_ZSTD)

  set(_onnxruntime_telemetry_all_options
    ${_onnxruntime_telemetry_off_options}
    ${_onnxruntime_telemetry_on_options}
    ${_onnxruntime_telemetry_string_options})
  list(REMOVE_DUPLICATES _onnxruntime_telemetry_all_options)

  foreach(option IN LISTS _onnxruntime_telemetry_all_options)
    onnxruntime_telemetry_save_cache_variable(${option})
  endforeach()
  foreach(option IN LISTS _onnxruntime_telemetry_off_options)
    set(${option} OFF)
    set(${option} OFF CACHE BOOL "Disable optional embedded telemetry dependency feature" FORCE)
  endforeach()
  foreach(option IN LISTS _onnxruntime_telemetry_on_options)
    set(${option} ON)
    set(${option} ON CACHE BOOL "Enable required embedded telemetry dependency feature" FORCE)
  endforeach()

  set(MBEDTLS_CONFIG_FILE "")
  set(MBEDTLS_USER_CONFIG_FILE "")
  set(MBEDTLS_TARGET_PREFIX "")
  set(CURL_CA_BUNDLE none)
  set(CURL_CA_PATH none)
  set(CURL_CA_EMBED "")
  set(CURL_DEFAULT_SSL_BACKEND mbedtls)
  set(CURL_ZLIB OFF)
  set(CURL_BROTLI OFF)
  set(CURL_ZSTD OFF)
  set(MBEDTLS_CONFIG_FILE "" CACHE FILEPATH "Use the default mbedTLS configuration" FORCE)
  set(MBEDTLS_USER_CONFIG_FILE "" CACHE FILEPATH "Do not append a caller mbedTLS configuration" FORCE)
  set(MBEDTLS_TARGET_PREFIX "" CACHE STRING "Use the expected embedded mbedTLS target names" FORCE)
  set(CURL_CA_BUNDLE none CACHE STRING "Select the target Linux CA bundle at runtime" FORCE)
  set(CURL_CA_PATH none CACHE STRING "Select the target Linux CA path at runtime" FORCE)
  set(CURL_CA_EMBED "" CACHE STRING "Do not embed a build-host CA bundle" FORCE)
  set(CURL_DEFAULT_SSL_BACKEND mbedtls CACHE STRING "Use the pinned mbedTLS backend" FORCE)
  set(CURL_ZLIB OFF CACHE BOOL "Disable zlib" FORCE)
  set(CURL_BROTLI OFF CACHE BOOL "Disable brotli" FORCE)
  set(CURL_ZSTD OFF CACHE BOOL "Disable zstd" FORCE)

  onnxruntime_fetchcontent_declare(
    onnxruntime_mbedtls
    URL ${DEP_URL_mbedtls}
    URL_HASH SHA1=${DEP_SHA1_mbedtls}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  onnxruntime_fetchcontent_makeavailable(onnxruntime_mbedtls)

  foreach(target mbedtls mbedx509 mbedcrypto)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Embedded telemetry dependency target not found: ${target}")
    endif()
  endforeach()

  # curl's FindMbedTLS module accepts target names through these variables, avoiding host discovery.
  set(MBEDTLS_INCLUDE_DIR "${onnxruntime_mbedtls_SOURCE_DIR}/include")
  set(MBEDTLS_LIBRARY MbedTLS::mbedtls)
  set(MBEDX509_LIBRARY MbedTLS::mbedx509)
  set(MBEDCRYPTO_LIBRARY MbedTLS::mbedcrypto)
  set(MBEDTLS_USE_STATIC_LIBS ON)

  onnxruntime_fetchcontent_declare(
    onnxruntime_curl
    URL ${DEP_URL_curl}
    URL_HASH SHA1=${DEP_SHA1_curl}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  onnxruntime_fetchcontent_makeavailable(onnxruntime_curl)

  if(NOT TARGET CURL::libcurl OR NOT TARGET libcurl_static)
    message(FATAL_ERROR "The pinned static CURL::libcurl target was not created.")
  endif()

  # The sentinel defaults above mask inherited variables and disable build-host CA detection.
  # Remove them from curl's generated config so 1DS can set the target host's CA bundle at runtime.
  set(_onnxruntime_telemetry_curl_config
    "${onnxruntime_curl_BINARY_DIR}/lib/curl_config.h")
  file(READ "${_onnxruntime_telemetry_curl_config}"
    _onnxruntime_telemetry_curl_config_contents)
  foreach(definition CURL_CA_BUNDLE CURL_CA_PATH)
    string(REGEX REPLACE
      "#define ${definition} \"[^\"]*\""
      "/* #undef ${definition} */"
      _onnxruntime_telemetry_curl_config_contents
      "${_onnxruntime_telemetry_curl_config_contents}")
  endforeach()
  file(WRITE "${_onnxruntime_telemetry_curl_config}"
    "${_onnxruntime_telemetry_curl_config_contents}")

  # Replace curl's non-exportable imported helper with the underlying target so ORT's static package
  # export can rewrite it into the onnxruntime namespace.
  get_target_property(_onnxruntime_telemetry_curl_link_libraries
    libcurl_static INTERFACE_LINK_LIBRARIES)
  list(FILTER _onnxruntime_telemetry_curl_link_libraries EXCLUDE REGEX "CURL::mbedtls")
  set_target_properties(libcurl_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_onnxruntime_telemetry_curl_link_libraries}")
  target_link_libraries(libcurl_static PRIVATE mbedtls)

  foreach(target mbedtls mbedx509 mbedcrypto libcurl_static)
    set_target_properties(${target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-ffunction-sections;-fdata-sections>)
    target_compile_definitions(${target} PRIVATE
      MBEDTLS_THREADING_C
      MBEDTLS_THREADING_PTHREAD)
  endforeach()

  foreach(option IN LISTS _onnxruntime_telemetry_all_options)
    onnxruntime_telemetry_restore_cache_variable(${option})
  endforeach()
endblock()
