set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE BOOL "" FORCE)

set(PATCH_EIGEN_S390X_WERROR ${PROJECT_SOURCE_DIR}/patches/eigen/s390x-build-werror.patch)
# Backport of https://gitlab.com/libeigen/eigen/-/merge_requests/2831
set(PATCH_EIGEN_PCMP_INT64 ${PROJECT_SOURCE_DIR}/patches/eigen/sse2-pcmp-lt-int64-sign.patch)

onnxruntime_fetchcontent_declare(
    Eigen3
    URL ${DEP_URL_eigen}
    URL_HASH SHA1=${DEP_SHA1_eigen}
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PATCH_EIGEN_S390X_WERROR} &&
                  ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PATCH_EIGEN_PCMP_INT64}
    EXCLUDE_FROM_ALL
)
onnxruntime_fetchcontent_makeavailable(Eigen3)
