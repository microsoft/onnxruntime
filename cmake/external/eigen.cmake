set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE BOOL "" FORCE)

set(PATCH_EIGEN_S390X ${PROJECT_SOURCE_DIR}/patches/eigen/s390x-build.patch)
set(PATCH_EIGEN_S390X_WERROR ${PROJECT_SOURCE_DIR}/patches/eigen/s390x-build-werror.patch)

onnxruntime_fetchcontent_declare(
    Eigen3
    URL ${DEP_URL_eigen}
    URL_HASH SHA1=${DEP_SHA1_eigen}
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PATCH_EIGEN_S390X} &&
                  ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PATCH_EIGEN_S390X_WERROR}
    EXCLUDE_FROM_ALL
)
onnxruntime_fetchcontent_makeavailable(Eigen3)
