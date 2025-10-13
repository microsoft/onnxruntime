// TODO: fill out abstract for this file
// Base off of qnbitgemm.h

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

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