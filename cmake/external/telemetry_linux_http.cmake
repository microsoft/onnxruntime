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
    set(${name}
      "${_onnxruntime_telemetry_cache_${name}_value}"
      CACHE "${_onnxruntime_telemetry_cache_${name}_type}"
      "${_onnxruntime_telemetry_cache_${name}_help}"
      FORCE)
  else()
    unset(${name} CACHE)
  endif()
endfunction()

set(_onnxruntime_telemetry_http_cache_variables
  BUILD_SHARED_LIBS
  BUILD_STATIC_LIBS
  BUILD_STATIC_CURL
  BUILD_TESTING
  SHARE_LIB_OBJECT
  ENABLE_PROGRAMS
  ENABLE_TESTING
  ENABLE_DEBUG
  ENABLE_UNICODE
  ENABLE_THREADED_RESOLVER
  ENABLE_IPV6
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
  MBEDTLS_CONFIG_FILE
  MBEDTLS_USER_CONFIG_FILE
  MBEDTLS_TARGET_PREFIX
  USE_STATIC_MBEDTLS_LIBRARY
  USE_SHARED_MBEDTLS_LIBRARY
  LINK_WITH_PTHREAD
  LINK_WITH_TRUSTED_STORAGE
  DISABLE_PACKAGE_CONFIG_AND_INSTALL
  BUILD_CURL_EXE
  BUILD_EXAMPLES
  BUILD_LIBCURL_DOCS
  BUILD_MISC_DOCS
  ENABLE_CURL_MANUAL
  CURL_DISABLE_INSTALL
  CURL_ENABLE_EXPORT_TARGET
  CURL_ENABLE_SSL
  CURL_ENABLE_NTLM
  CURL_ENABLE_SMB
  CURL_DEFAULT_SSL_BACKEND
  CURL_USE_MBEDTLS
  CURL_USE_OPENSSL
  CURL_USE_SCHANNEL
  CURL_USE_WOLFSSL
  CURL_USE_GNUTLS
  CURL_USE_RUSTLS
  CURL_USE_PKGCONFIG
  CURL_USE_CMAKECONFIG
  CURL_CA_BUNDLE
  CURL_CA_PATH
  CURL_CA_EMBED
  CURL_CA_FALLBACK
  CURL_CA_NATIVE
  CURL_CA_SEARCH_SAFE
  HTTP_ONLY
  CURL_ZLIB
  CURL_BROTLI
  CURL_ZSTD
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
  HAVE_MBEDTLS_DES_CRYPT_ECB
  ENABLE_ARES
  ENABLE_UNIX_SOCKETS
  CURL_DISABLE_ALTSVC
  CURL_DISABLE_CA_SEARCH
  CURL_DISABLE_SRP
  CURL_DISABLE_HSTS
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

block(SCOPE_FOR VARIABLES POLICIES)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

foreach(_onnxruntime_telemetry_cache_variable IN LISTS _onnxruntime_telemetry_http_cache_variables)
  onnxruntime_telemetry_save_cache_variable(${_onnxruntime_telemetry_cache_variable})
endforeach()

# Normal variables take precedence over cache entries. Set both so caller variables cannot select
# shared libraries, a different TLS backend, or optional host dependencies in the nested projects.
foreach(_onnxruntime_telemetry_off_option IN ITEMS
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
  set(${_onnxruntime_telemetry_off_option} OFF)
  set(${_onnxruntime_telemetry_off_option}
    OFF CACHE BOOL "Disable optional embedded telemetry dependency feature" FORCE)
endforeach()
foreach(_onnxruntime_telemetry_on_option IN ITEMS
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
  set(${_onnxruntime_telemetry_on_option} ON)
  set(${_onnxruntime_telemetry_on_option}
    ON CACHE BOOL "Enable required embedded telemetry dependency feature" FORCE)
endforeach()
set(CURL_ZLIB OFF)
set(CURL_BROTLI OFF)
set(CURL_ZSTD OFF)
set(MBEDTLS_CONFIG_FILE "")
set(MBEDTLS_USER_CONFIG_FILE "")
set(CURL_CA_BUNDLE none)
set(CURL_CA_PATH none)
set(CURL_CA_EMBED "")
set(CURL_DEFAULT_SSL_BACKEND mbedtls)
set(MBEDTLS_TARGET_PREFIX "")
set(MBEDTLS_TARGET_PREFIX "" CACHE STRING "Use the expected embedded mbedTLS target names" FORCE)
set(CURL_DEFAULT_SSL_BACKEND mbedtls CACHE STRING "Use the pinned mbedTLS backend" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build embedded telemetry dependencies statically" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libcurl" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable dependency tests" FORCE)

set(ENABLE_PROGRAMS OFF CACHE BOOL "Disable mbedTLS programs" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "Disable mbedTLS tests" FORCE)
set(GEN_FILES OFF CACHE BOOL "Use generated files from the mbedTLS release archive" FORCE)
set(UNSAFE_BUILD OFF CACHE BOOL "Require secure mbedTLS configuration" FORCE)
set(INSTALL_MBEDTLS_HEADERS OFF CACHE BOOL "Keep mbedTLS headers internal" FORCE)
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "Do not inherit dependency warnings as errors" FORCE)
set(MBEDTLS_CONFIG_FILE "" CACHE FILEPATH "Use the pinned mbedTLS configuration" FORCE)
set(MBEDTLS_USER_CONFIG_FILE "" CACHE FILEPATH "Do not append a host mbedTLS configuration" FORCE)
set(USE_STATIC_MBEDTLS_LIBRARY ON CACHE BOOL "Build static mbedTLS libraries" FORCE)
set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "Disable shared mbedTLS libraries" FORCE)
set(LINK_WITH_PTHREAD ON CACHE BOOL "Protect mbedTLS global state with pthread mutexes" FORCE)
set(DISABLE_PACKAGE_CONFIG_AND_INSTALL ON CACHE BOOL "Keep mbedTLS internal" FORCE)

onnxruntime_fetchcontent_declare(
  onnxruntime_mbedtls
  URL ${DEP_URL_mbedtls}
  URL_HASH SHA1=${DEP_SHA1_mbedtls}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL)
onnxruntime_fetchcontent_makeavailable(onnxruntime_mbedtls)

foreach(_onnxruntime_telemetry_mbedtls_target mbedtls mbedx509 mbedcrypto)
  if(NOT TARGET ${_onnxruntime_telemetry_mbedtls_target})
    message(FATAL_ERROR
      "Embedded telemetry dependency target not found: ${_onnxruntime_telemetry_mbedtls_target}")
  endif()
endforeach()

# curl's FindMbedTLS module accepts target names through these variables. Supplying them directly
# prevents discovery of a host installation and preserves the static target dependency graph.
set(MBEDTLS_INCLUDE_DIR "${onnxruntime_mbedtls_SOURCE_DIR}/include")
set(MBEDTLS_LIBRARY MbedTLS::mbedtls)
set(MBEDX509_LIBRARY MbedTLS::mbedx509)
set(MBEDCRYPTO_LIBRARY MbedTLS::mbedcrypto)
set(MBEDTLS_USE_STATIC_LIBS ON)

set(BUILD_CURL_EXE OFF CACHE BOOL "Disable the curl executable" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable curl examples" FORCE)
set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "Disable libcurl documentation" FORCE)
set(BUILD_MISC_DOCS OFF CACHE BOOL "Disable curl documentation" FORCE)
set(ENABLE_CURL_MANUAL OFF CACHE BOOL "Disable the curl manual" FORCE)
set(CURL_DISABLE_INSTALL ON CACHE BOOL "Keep curl internal" FORCE)
set(CURL_ENABLE_EXPORT_TARGET OFF CACHE BOOL "Do not export the internal curl target" FORCE)

set(CURL_ENABLE_SSL ON CACHE BOOL "Enable HTTPS" FORCE)
set(CURL_USE_MBEDTLS ON CACHE BOOL "Use mbedTLS for HTTPS" FORCE)
set(CURL_USE_OPENSSL OFF CACHE BOOL "Do not use OpenSSL" FORCE)
set(CURL_USE_SCHANNEL OFF CACHE BOOL "Do not use Schannel" FORCE)
set(CURL_USE_WOLFSSL OFF CACHE BOOL "Do not use wolfSSL" FORCE)
set(CURL_USE_GNUTLS OFF CACHE BOOL "Do not use GnuTLS" FORCE)
set(CURL_USE_RUSTLS OFF CACHE BOOL "Do not use Rustls" FORCE)
set(CURL_USE_PKGCONFIG OFF CACHE BOOL "Do not discover host dependencies with pkg-config" FORCE)
set(CURL_USE_CMAKECONFIG OFF CACHE BOOL "Do not discover host dependency packages" FORCE)
set(CURL_CA_BUNDLE none CACHE STRING "Select the Linux CA bundle at runtime" FORCE)
set(CURL_CA_PATH none CACHE STRING "Select the Linux CA bundle at runtime" FORCE)
set(CURL_CA_EMBED "" CACHE STRING "Do not embed a build-host CA bundle" FORCE)
set(CURL_CA_FALLBACK OFF CACHE BOOL "Do not use an alternate TLS CA store" FORCE)
set(CURL_CA_NATIVE OFF CACHE BOOL "mbedTLS has no native Linux CA store" FORCE)
set(CURL_CA_SEARCH_SAFE OFF CACHE BOOL "Disable Windows CA search" FORCE)

set(HTTP_ONLY ON CACHE BOOL "Build only HTTP and HTTPS support" FORCE)
set(CURL_ZLIB OFF CACHE STRING "Disable zlib" FORCE)
set(CURL_BROTLI OFF CACHE STRING "Disable brotli" FORCE)
set(CURL_ZSTD OFF CACHE STRING "Disable zstd" FORCE)
set(USE_LIBIDN2 OFF CACHE BOOL "Disable libidn2" FORCE)
set(CURL_USE_LIBPSL OFF CACHE BOOL "Disable libpsl" FORCE)
set(CURL_USE_LIBSSH2 OFF CACHE BOOL "Disable libssh2" FORCE)
set(CURL_USE_LIBSSH OFF CACHE BOOL "Disable libssh" FORCE)
set(CURL_USE_GSSAPI OFF CACHE BOOL "Disable GSSAPI" FORCE)
set(CURL_USE_GSASL OFF CACHE BOOL "Disable GSASL" FORCE)
set(USE_NGHTTP2 OFF CACHE BOOL "Disable HTTP/2" FORCE)
set(USE_NGTCP2 OFF CACHE BOOL "Disable ngtcp2" FORCE)
set(USE_QUICHE OFF CACHE BOOL "Disable quiche" FORCE)
# curl otherwise checks this with try_compile(), whose isolated project cannot see in-tree targets.
set(HAVE_MBEDTLS_DES_CRYPT_ECB ON CACHE BOOL "mbedTLS 3.6.7 provides mbedtls_des_crypt_ecb" FORCE)
set(ENABLE_ARES OFF CACHE BOOL "Use the threaded resolver instead of c-ares" FORCE)
set(ENABLE_UNIX_SOCKETS OFF CACHE BOOL "Disable Unix domain sockets" FORCE)

set(CURL_DISABLE_ALTSVC ON CACHE BOOL "Disable alt-svc" FORCE)
set(CURL_DISABLE_HSTS ON CACHE BOOL "Disable HSTS caching" FORCE)
set(CURL_DISABLE_COOKIES ON CACHE BOOL "Disable cookies" FORCE)
set(CURL_DISABLE_NETRC ON CACHE BOOL "Disable netrc" FORCE)
set(CURL_DISABLE_MIME ON CACHE BOOL "Disable MIME" FORCE)
set(CURL_DISABLE_DOH ON CACHE BOOL "Disable DNS-over-HTTPS" FORCE)
set(CURL_DISABLE_AWS ON CACHE BOOL "Disable AWS request signing" FORCE)
set(CURL_DISABLE_BEARER_AUTH ON CACHE BOOL "Disable bearer authentication" FORCE)
set(CURL_DISABLE_DIGEST_AUTH ON CACHE BOOL "Disable digest authentication" FORCE)
set(CURL_DISABLE_KERBEROS_AUTH ON CACHE BOOL "Disable Kerberos authentication" FORCE)
set(CURL_DISABLE_NEGOTIATE_AUTH ON CACHE BOOL "Disable negotiate authentication" FORCE)

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

# CURL_CA_BUNDLE/PATH are set to "none" above to mask inherited normal variables and disable
# build-host auto-detection. Remove those sentinel defaults from curl's generated config so 1DS can
# supply the target host's CA bundle through CURLOPT_CAINFO at runtime.
set(_onnxruntime_telemetry_curl_config
  "${onnxruntime_curl_BINARY_DIR}/lib/curl_config.h")
file(READ "${_onnxruntime_telemetry_curl_config}"
  _onnxruntime_telemetry_curl_config_contents)
foreach(_onnxruntime_telemetry_ca_definition CURL_CA_BUNDLE CURL_CA_PATH)
  string(REGEX REPLACE
    "#define ${_onnxruntime_telemetry_ca_definition} \"[^\"]*\""
    "/* #undef ${_onnxruntime_telemetry_ca_definition} */"
    _onnxruntime_telemetry_curl_config_contents
    "${_onnxruntime_telemetry_curl_config_contents}")
endforeach()
file(WRITE "${_onnxruntime_telemetry_curl_config}"
  "${_onnxruntime_telemetry_curl_config_contents}")

# curl's in-tree target records the non-exportable imported CURL::mbedtls helper. Replace it with
# the underlying target so ORT's static package export can rewrite it into the onnxruntime namespace.
get_target_property(_onnxruntime_telemetry_curl_link_libraries
  libcurl_static INTERFACE_LINK_LIBRARIES)
list(FILTER _onnxruntime_telemetry_curl_link_libraries EXCLUDE REGEX "CURL::mbedtls")
set_target_properties(libcurl_static PROPERTIES
  INTERFACE_LINK_LIBRARIES "${_onnxruntime_telemetry_curl_link_libraries}")
target_link_libraries(libcurl_static PRIVATE mbedtls)

foreach(_onnxruntime_telemetry_http_target mbedtls mbedx509 mbedcrypto libcurl_static)
  set_target_properties(${_onnxruntime_telemetry_http_target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden)
  target_compile_options(${_onnxruntime_telemetry_http_target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-ffunction-sections;-fdata-sections>)
  target_compile_definitions(${_onnxruntime_telemetry_http_target} PRIVATE
    MBEDTLS_THREADING_C
    MBEDTLS_THREADING_PTHREAD)
endforeach()

foreach(_onnxruntime_telemetry_cache_variable IN LISTS _onnxruntime_telemetry_http_cache_variables)
  onnxruntime_telemetry_restore_cache_variable(${_onnxruntime_telemetry_cache_variable})
endforeach()
endblock()
