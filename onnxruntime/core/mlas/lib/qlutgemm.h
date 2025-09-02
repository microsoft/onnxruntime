// TODO: fill out abstract for this file
// Base off of qnbitgemm.h

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

typedef
void(MLAS_QNBIT_GEMM_LUT_GEN)(
	int32_t group_size,
	int8_t* lut,
	onnxruntime::MLFloat16* b,
	onnxruntime::MLFloat16* scales,
	onnxruntime::MLFloat16* biases
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

};