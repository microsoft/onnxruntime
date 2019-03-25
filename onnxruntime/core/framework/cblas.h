// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the exact cblas.h header file, placed here purely in order to get
// the enums.

#ifndef CBLAS_H

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER { CblasRowMajor = 101,
                   CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};
enum CBLAS_UPLO { CblasUpper = 121,
                  CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131,
                  CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141,
                  CblasRight = 142 };
#endif

#ifndef CBLAS_ENUM_ONLY
#define CBLAS_H
#define CBLAS_INDEX int

int cblas_errprn(int ierr, int info, char* form, ...);
void cblas_xerbla(int p, const char* rout, const char* form, ...);

/*
* ===========================================================================
* Prototypes for level 1 BLAS functions (complex are recast as routines)
* ===========================================================================
*/
float cblas_sdsdot(int N, float alpha, const float* X,
                   int incX, const float* Y, int incY);
double cblas_dsdot(int N, const float* X, int incX, const float* Y,
                   int incY);
float cblas_sdot(int N, const float* X, int incX,
                 const float* Y, int incY);
double cblas_ddot(int N, const double* X, int incX,
                  const double* Y, int incY);
/*
* Functions having prefixes Z and C only
*/
void cblas_cdotu_sub(int N, const void* X, int incX,
                     const void* Y, int incY, void* dotu);
void cblas_cdotc_sub(int N, const void* X, int incX,
                     const void* Y, int incY, void* dotc);

void cblas_zdotu_sub(int N, const void* X, int incX,
                     const void* Y, int incY, void* dotu);
void cblas_zdotc_sub(int N, const void* X, int incX,
                     const void* Y, int incY, void* dotc);

/*
* Functions having prefixes S D SC DZ
*/
float cblas_snrm2(int N, const float* X, int incX);
float cblas_sasum(int N, const float* X, int incX);

double cblas_dnrm2(int N, const double* X, int incX);
double cblas_dasum(int N, const double* X, int incX);

float cblas_scnrm2(int N, const void* X, int incX);
float cblas_scasum(int N, const void* X, int incX);

double cblas_dznrm2(int N, const void* X, int incX);
double cblas_dzasum(int N, const void* X, int incX);

/*
* Functions having standard 4 prefixes (S D C Z)
*/
CBLAS_INDEX cblas_isamax(int N, const float* X, int incX);
CBLAS_INDEX cblas_idamax(int N, const double* X, int incX);
CBLAS_INDEX cblas_icamax(int N, const void* X, int incX);
CBLAS_INDEX cblas_izamax(int N, const void* X, int incX);

/*
* ===========================================================================
* Prototypes for level 1 BLAS routines
* ===========================================================================
*/

/*
* Routines with standard 4 prefixes (s, d, c, z)
*/
void cblas_sswap(int N, float* X, int incX,
                 float* Y, int incY);
void cblas_scopy(int N, const float* X, int incX,
                 float* Y, int incY);
void cblas_saxpy(int N, float alpha, const float* X,
                 int incX, float* Y, int incY);
void catlas_saxpby(int N, float alpha, const float* X,
                   int incX, float beta, float* Y, int incY);
void catlas_sset(int N, float alpha, float* X, int incX);

void cblas_dswap(int N, double* X, int incX,
                 double* Y, int incY);
void cblas_dcopy(int N, const double* X, int incX,
                 double* Y, int incY);
void cblas_daxpy(int N, double alpha, const double* X,
                 int incX, double* Y, int incY);
void catlas_daxpby(int N, double alpha, const double* X,
                   int incX, double beta, double* Y, int incY);
void catlas_dset(int N, double alpha, double* X, int incX);

void cblas_cswap(int N, void* X, int incX,
                 void* Y, int incY);
void cblas_ccopy(int N, const void* X, int incX,
                 void* Y, int incY);
void cblas_caxpy(int N, const void* alpha, const void* X,
                 int incX, void* Y, int incY);
void catlas_caxpby(int N, const void* alpha, const void* X,
                   int incX, const void* beta, void* Y, int incY);
void catlas_cset(int N, const void* alpha, void* X, int incX);

void cblas_zswap(int N, void* X, int incX,
                 void* Y, int incY);
void cblas_zcopy(int N, const void* X, int incX,
                 void* Y, int incY);
void cblas_zaxpy(int N, const void* alpha, const void* X,
                 int incX, void* Y, int incY);
void catlas_zaxpby(int N, const void* alpha, const void* X,
                   int incX, const void* beta, void* Y, int incY);
void catlas_zset(int N, const void* alpha, void* X, int incX);

/*
* Routines with S and D prefix only
*/
void cblas_srotg(float* a, float* b, float* c, float* s);
void cblas_srotmg(float* d1, float* d2, float* b1, float b2, float* P);
void cblas_srot(int N, float* X, int incX,
                float* Y, int incY, float c, float s);
void cblas_srotm(int N, float* X, int incX,
                 float* Y, int incY, const float* P);

void cblas_drotg(double* a, double* b, double* c, double* s);
void cblas_drotmg(double* d1, double* d2, double* b1, double b2, double* P);
void cblas_drot(int N, double* X, int incX,
                double* Y, int incY, double c, double s);
void cblas_drotm(int N, double* X, int incX,
                 double* Y, int incY, const double* P);

/*
* Routines with S D C Z CS and ZD prefixes
*/
void cblas_sscal(int N, float alpha, float* X, int incX);
void cblas_dscal(int N, double alpha, double* X, int incX);
void cblas_cscal(int N, const void* alpha, void* X, int incX);
void cblas_zscal(int N, const void* alpha, void* X, int incX);
void cblas_csscal(int N, float alpha, void* X, int incX);
void cblas_zdscal(int N, double alpha, void* X, int incX);

/*
* Extra reference routines provided by ATLAS, but not mandated by the standard
*/
void cblas_crotg(void* a, void* b, void* c, void* s);
void cblas_zrotg(void* a, void* b, void* c, void* s);
void cblas_csrot(int N, void* X, int incX, void* Y, int incY,
                 float c, float s);
void cblas_zdrot(int N, void* X, int incX, void* Y, int incY,
                 double c, double s);

/*
* ===========================================================================
* Prototypes for level 2 BLAS
* ===========================================================================
*/

/*
* Routines with standard 4 prefixes (S, D, C, Z)
*/
void cblas_sgemv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 float alpha, const float* A, int lda,
                 const float* X, int incX, float beta,
                 float* Y, int incY);
void cblas_sgbmv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, float alpha,
                 const float* A, int lda, const float* X,
                 int incX, float beta, float* Y, int incY);
void cblas_strmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const float* A, int lda,
                 float* X, int incX);
void cblas_stbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const float* A, int lda,
                 float* X, int incX);
void cblas_stpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const float* Ap, float* X, int incX);
void cblas_strsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const float* A, int lda, float* X,
                 int incX);
void cblas_stbsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const float* A, int lda,
                 float* X, int incX);
void cblas_stpsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const float* Ap, float* X, int incX);

void cblas_dgemv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 double alpha, const double* A, int lda,
                 const double* X, int incX, double beta,
                 double* Y, int incY);
void cblas_dgbmv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, double alpha,
                 const double* A, int lda, const double* X,
                 int incX, double beta, double* Y, int incY);
void cblas_dtrmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const double* A, int lda,
                 double* X, int incX);
void cblas_dtbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const double* A, int lda,
                 double* X, int incX);
void cblas_dtpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const double* Ap, double* X, int incX);
void cblas_dtrsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const double* A, int lda, double* X,
                 int incX);
void cblas_dtbsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const double* A, int lda,
                 double* X, int incX);
void cblas_dtpsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const double* Ap, double* X, int incX);

void cblas_cgemv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* X, int incX, const void* beta,
                 void* Y, int incY);
void cblas_cgbmv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const void* alpha,
                 const void* A, int lda, const void* X,
                 int incX, const void* beta, void* Y, int incY);
void cblas_ctrmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* A, int lda,
                 void* X, int incX);
void cblas_ctbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const void* A, int lda,
                 void* X, int incX);
void cblas_ctpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* Ap, void* X, int incX);
void cblas_ctrsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* A, int lda, void* X,
                 int incX);
void cblas_ctbsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const void* A, int lda,
                 void* X, int incX);
void cblas_ctpsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* Ap, void* X, int incX);

void cblas_zgemv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* X, int incX, const void* beta,
                 void* Y, int incY);
void cblas_zgbmv(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const void* alpha,
                 const void* A, int lda, const void* X,
                 int incX, const void* beta, void* Y, int incY);
void cblas_ztrmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* A, int lda,
                 void* X, int incX);
void cblas_ztbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const void* A, int lda,
                 void* X, int incX);
void cblas_ztpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* Ap, void* X, int incX);
void cblas_ztrsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* A, int lda, void* X,
                 int incX);
void cblas_ztbsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, int K, const void* A, int lda,
                 void* X, int incX);
void cblas_ztpsv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 int N, const void* Ap, void* X, int incX);

/*
* Routines with S and D prefixes only
*/
void cblas_ssymv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, float alpha, const float* A,
                 int lda, const float* X, int incX,
                 float beta, float* Y, int incY);
void cblas_ssbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, int K, float alpha, const float* A,
                 int lda, const float* X, int incX,
                 float beta, float* Y, int incY);
void cblas_sspmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, float alpha, const float* Ap,
                 const float* X, int incX,
                 float beta, float* Y, int incY);
void cblas_sger(enum CBLAS_ORDER Order, int M, int N,
                float alpha, const float* X, int incX,
                const float* Y, int incY, float* A, int lda);
void cblas_ssyr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, float alpha, const float* X,
                int incX, float* A, int lda);
void cblas_sspr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, float alpha, const float* X,
                int incX, float* Ap);
void cblas_ssyr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, float alpha, const float* X,
                 int incX, const float* Y, int incY, float* A,
                 int lda);
void cblas_sspr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, float alpha, const float* X,
                 int incX, const float* Y, int incY, float* A);

void cblas_dsymv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, double alpha, const double* A,
                 int lda, const double* X, int incX,
                 double beta, double* Y, int incY);
void cblas_dsbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, int K, double alpha, const double* A,
                 int lda, const double* X, int incX,
                 double beta, double* Y, int incY);
void cblas_dspmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, double alpha, const double* Ap,
                 const double* X, int incX,
                 double beta, double* Y, int incY);
void cblas_dger(enum CBLAS_ORDER Order, int M, int N,
                double alpha, const double* X, int incX,
                const double* Y, int incY, double* A, int lda);
void cblas_dsyr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, double alpha, const double* X,
                int incX, double* A, int lda);
void cblas_dspr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, double alpha, const double* X,
                int incX, double* Ap);
void cblas_dsyr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, double alpha, const double* X,
                 int incX, const double* Y, int incY, double* A,
                 int lda);
void cblas_dspr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, double alpha, const double* X,
                 int incX, const double* Y, int incY, double* A);

/*
* Routines with C and Z prefixes only
*/
void cblas_chemv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, const void* alpha, const void* A,
                 int lda, const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_chbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, int K, const void* alpha, const void* A,
                 int lda, const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_chpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, const void* alpha, const void* Ap,
                 const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_cgeru(enum CBLAS_ORDER Order, int M, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_cgerc(enum CBLAS_ORDER Order, int M, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_cher(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, float alpha, const void* X, int incX,
                void* A, int lda);
void cblas_chpr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, float alpha, const void* X,
                int incX, void* A);
void cblas_cher2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_chpr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* Ap);

void cblas_zhemv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, const void* alpha, const void* A,
                 int lda, const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_zhbmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, int K, const void* alpha, const void* A,
                 int lda, const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_zhpmv(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 int N, const void* alpha, const void* Ap,
                 const void* X, int incX,
                 const void* beta, void* Y, int incY);
void cblas_zgeru(enum CBLAS_ORDER Order, int M, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_zgerc(enum CBLAS_ORDER Order, int M, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_zher(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, double alpha, const void* X, int incX,
                void* A, int lda);
void cblas_zhpr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                int N, double alpha, const void* X,
                int incX, void* A);
void cblas_zher2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* A, int lda);
void cblas_zhpr2(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo, int N,
                 const void* alpha, const void* X, int incX,
                 const void* Y, int incY, void* Ap);

/*
* ===========================================================================
* Prototypes for level 3 BLAS
* ===========================================================================
*/

/*
* Routines with standard 4 prefixes (S, D, C, Z)
*/
void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, float alpha, const float* A,
                 int lda, const float* B, int ldb,
                 float beta, float* C, int ldc);
void cblas_ssymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 float alpha, const float* A, int lda,
                 const float* B, int ldb, float beta,
                 float* C, int ldc);
void cblas_ssyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 float alpha, const float* A, int lda,
                 float beta, float* C, int ldc);
void cblas_ssyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  float alpha, const float* A, int lda,
                  const float* B, int ldb, float beta,
                  float* C, int ldc);
void cblas_strmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 float alpha, const float* A, int lda,
                 float* B, int ldb);
void cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 float alpha, const float* A, int lda,
                 float* B, int ldb);

void cblas_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, const double* A,
                 int lda, const double* B, int ldb,
                 double beta, double* C, int ldc);
void cblas_dsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 double alpha, const double* A, int lda,
                 const double* B, int ldb, double beta,
                 double* C, int ldc);
void cblas_dsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 double alpha, const double* A, int lda,
                 double beta, double* C, int ldc);
void cblas_dsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  double alpha, const double* A, int lda,
                  const double* B, int ldb, double beta,
                  double* C, int ldc);
void cblas_dtrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 double alpha, const double* A, int lda,
                 double* B, int ldb);
void cblas_dtrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 double alpha, const double* A, int lda,
                 double* B, int ldb);

void cblas_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const void* alpha, const void* A,
                 int lda, const void* B, int ldb,
                 const void* beta, void* C, int ldc);
void cblas_csymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* B, int ldb, const void* beta,
                 void* C, int ldc);
void cblas_csyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const void* alpha, const void* A, int lda,
                 const void* beta, void* C, int ldc);
void cblas_csyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void* alpha, const void* A, int lda,
                  const void* B, int ldb, const void* beta,
                  void* C, int ldc);
void cblas_ctrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 const void* alpha, const void* A, int lda,
                 void* B, int ldb);
void cblas_ctrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 const void* alpha, const void* A, int lda,
                 void* B, int ldb);

void cblas_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const void* alpha, const void* A,
                 int lda, const void* B, int ldb,
                 const void* beta, void* C, int ldc);
void cblas_zsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* B, int ldb, const void* beta,
                 void* C, int ldc);
void cblas_zsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const void* alpha, const void* A, int lda,
                 const void* beta, void* C, int ldc);
void cblas_zsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void* alpha, const void* A, int lda,
                  const void* B, int ldb, const void* beta,
                  void* C, int ldc);
void cblas_ztrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 const void* alpha, const void* A, int lda,
                 void* B, int ldb);
void cblas_ztrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N,
                 const void* alpha, const void* A, int lda,
                 void* B, int ldb);

/*
* Routines with prefixes C and Z only
*/
void cblas_chemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* B, int ldb, const void* beta,
                 void* C, int ldc);
void cblas_cherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 float alpha, const void* A, int lda,
                 float beta, void* C, int ldc);
void cblas_cher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void* alpha, const void* A, int lda,
                  const void* B, int ldb, float beta,
                  void* C, int ldc);
void cblas_zhemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, int M, int N,
                 const void* alpha, const void* A, int lda,
                 const void* B, int ldb, const void* beta,
                 void* C, int ldc);
void cblas_zherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, int N, int K,
                 double alpha, const void* A, int lda,
                 double beta, void* C, int ldc);
void cblas_zher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void* alpha, const void* A, int lda,
                  const void* B, int ldb, double beta,
                  void* C, int ldc);

#endif /* end #ifdef CBLAS_ENUM_ONLY */
#endif
