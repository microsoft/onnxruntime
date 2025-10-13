// TODO: fill out abstract for this file
// Base off of qnbitgemm.h

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"


/**
 * @brief Parameters for TMAC kernel
 */
struct MlasTMACKernelParams {
    size_t g;
    size_t ngroups_per_elem;
    size_t q_group_size;
    size_t act_group_size;

    size_t kfactor;
    size_t bits;
    size_t actk;
    size_t bm;
    size_t simd_n_in;
    size_t simd_n_out;
    size_t chunk_n;
};

const MlasTMACKernelParams& GetTMACKernelParams(size_t M, size_t N, size_t nbits, size_t block_size);

typedef
void(MLAS_QNBIT_GEMM_LUT_GEN)(
	int32_t group_size,
	int8_t* lut,
	const float* b,
	float* scales,
	float* biases,
	int K
);

typedef
void(MLAS_QNBIT_LUT_GEMM_COMPUTE)(
	const void* A,
	const void* a_scales,
	const void* LUT,
	const void* LUT_Scales,
	const void* LUT_Biases,
	void* C,
	int bm,
	int K,
	int M,                // batch size (number of rows in activation)
	int N,
	size_t BlkLen
);

//
// Kernel dispatch structure.
//
// NOTE: This name must match the forward declaration in mlasi.h:
//   struct MLAS_QNBIT_LUT_GEMM_DISPATCH;
// Keep it minimal for now; extend with function pointers as kernels are added.
struct MLAS_QNBIT_LUT_GEMM_DISPATCH {
	// Intentionally empty placeholder; add members as needed.
	MLAS_QNBIT_GEMM_LUT_GEN* GenerateLUT = nullptr;

    MLAS_QNBIT_LUT_GEMM_COMPUTE* ComputeGemm = nullptr;

};
