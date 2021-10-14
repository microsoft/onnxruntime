/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Tools
 **/

#pragma once
#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include <mutex>

namespace fastertransformer{

// for cublasMM_cublasLtMM_wrapper to prevent that 
// multiple threads uses the handler in the same time.
static std::mutex mu_;

//for int8 cublasLtMM with algo
//ATransform should be m*n, CUBLASLT_ORDER_COL32
//kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//res is m*n, CUBLASLT_ORDER_COL32
template <typename T>
void cublasLtMM_withAlgo(int *res, int batchCount, int m, int n, int k,
                         int64_t stridea, int64_t strideb, int64_t stridec,
                         const int8_t *ATransform, const T *kernel, cublasLtHandle_t cublasLt_handle,
                         cudaStream_t stream, std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
                         bool use_ORDER_COL32_2R_4R4)
{
  cublasOperation_t opTranspose = CUBLAS_OP_T;
#ifdef CUDA11_MODE
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#ifdef CUDA11_MODE
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif



  int ldaTransform = 32 * m;
  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
  int ldcTransform = 32 * m;

  // create matmulDesc
#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  if (batchCount > 1)
  {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
  }

  int alphaI = 1;
  int betaI = 0;

  //get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, INT8_DATATYPE);
  std::string markStr(mark);
  int findAlgo = 0;
  if (cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() && cublasLtAlgoMap[markStr].workspaceSize == 0)
  {
    //printf("find algo %s\n", markStr.c_str());
    findAlgo = 1;

    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, cublasLtAlgoMap[markStr].algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasLtAlgoMap[markStr].customOption), sizeof(cublasLtAlgoMap[markStr].customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasLtAlgoMap[markStr].tile), sizeof(cublasLtAlgoMap[markStr].tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasLtAlgoMap[markStr].splitK_val), sizeof(cublasLtAlgoMap[markStr].splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasLtAlgoMap[markStr].swizzle), sizeof(cublasLtAlgoMap[markStr].swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasLtAlgoMap[markStr].stages), sizeof(cublasLtAlgoMap[markStr].stages));
#endif
  }
  else
  {
    findAlgo = 1;
    int algoId;
    if (use_ORDER_COL32_2R_4R4)
    {
      algoId = 7;
    }
    else
    {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    int stages;
    if (use_ORDER_COL32_2R_4R4)
      stages = 15;
    else
      stages = 13;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
  }

  cublasLtMatmul(cublasLt_handle,
                 matmulDesc,
                 &alphaI,
                 ATransform,
                 AtransformDesc,
                 kernel,
                 BtransformDesc,
                 &betaI,
                 res,
                 CtransformDesc,
                 res,
                 CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

//for int8 IO cublasLtMM with algo
//ATransform should be m*k CUBLASLT_ORDER_COL32
//kernel should be n*k CUBLASLT_ORDER_COL4_4R2_8C
//res is m*n CUBLASLT_ORDER_COL32
template <typename T>
void cublasLtMM_withAlgo_int8IO(int8_t *res, int batchCount, int m, int n, int k,
                                int64_t stridea, int64_t strideb, int64_t stridec,
                                const float alpha, const int8_t *ATransform, const T *kernel,
                                cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                                std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
                                bool use_ORDER_COL32_2R_4R4)
{
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  //int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
  //cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  cudaDataType_t scaleType = CUDA_R_32F;
#ifdef CUDA11_MODE
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#ifdef CUDA11_MODE
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif


  int ldaTransform = 32 * m;

  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;


  int ldcTransform = 32 * m;

  // create matmulDesc
#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType, sizeof(scaleType));
  //cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(cublasLtPointerMode_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));  
  if (batchCount > 1)
  {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
  }
  //get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, INT8_DATATYPE);
  std::string markStr(mark);
  int findAlgo = 0;
  if (cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() && cublasLtAlgoMap[markStr].workspaceSize == 0)
  {
    findAlgo = 1;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, cublasLtAlgoMap[markStr].algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasLtAlgoMap[markStr].customOption), sizeof(cublasLtAlgoMap[markStr].customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasLtAlgoMap[markStr].tile), sizeof(cublasLtAlgoMap[markStr].tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasLtAlgoMap[markStr].splitK_val), sizeof(cublasLtAlgoMap[markStr].splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasLtAlgoMap[markStr].swizzle), sizeof(cublasLtAlgoMap[markStr].swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasLtAlgoMap[markStr].stages), sizeof(cublasLtAlgoMap[markStr].stages));
#endif
  }
  else
  {
    findAlgo = 1;
    int algoId;
    if (use_ORDER_COL32_2R_4R4)
    {
      algoId = 7;
    }
    else
    {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    int stages;
    if (use_ORDER_COL32_2R_4R4)
      stages = 15;
    else
      stages = 13;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
  }

  float beta = 0.0f;
  cublasLtMatmul(cublasLt_handle,
                 matmulDesc,
                 &alpha,
                 ATransform,
                 AtransformDesc,
                 kernel,
                 BtransformDesc,
                 &beta,
                 res,
                 CtransformDesc,
                 res,
                 CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

template <typename T>
void cublasMM_cublasLtMM_wrapper(cublasLtHandle_t ltHandle, cublasHandle_t handle, cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k, const void *alpha,
                                 const void *A, cudaDataType_t Atype, int lda,
                                 const void *B, cudaDataType_t Btype, int ldb,
                                 const void *beta, T *C, cudaDataType_t Ctype, int ldc,
                                 cudaStream_t stream,
                                 std::map<std::string, cublasLtMatmulAlgo_info>& cublasAlgoMap,
                                 int sm, void* cublas_workspace){
  mu_.lock();
  int is_fp16 = Atype == CUDA_R_16F ? 1 : 0;
  int batchCount = 1;
  //fp32 use cublas as default
  //fp16 use cublasLt as default
  bool using_cublasLt = is_fp16 ? true : false;

  int findAlgo = 0;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, is_fp16 ? HALF_DATATYPE : FLOAT_DATATYPE);
  if(cublasAlgoMap.find(mark) != cublasAlgoMap.end())
  {
    findAlgo = 1;
    if (cublasAlgoMap[mark].stages != -1)
      using_cublasLt = true;
    else
      using_cublasLt = false;
  }

  if(using_cublasLt){
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t scaleType;
#ifdef CUDA11_MODE
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if(is_fp16){
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_16F;
#else
      computeType = CUDA_R_16F;
#endif
      scaleType = CUDA_R_16F;
    }else{
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_32F;
#else
      computeType = CUDA_R_32F;
#endif
      scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    //Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);
#ifdef CUDA11_MODE
      cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
      cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    cublasLtMatmulAlgo_t algo;
    void * workSpace = cublas_workspace;
    int workspaceSize = cublas_workspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if(findAlgo){
      if (cublasAlgoMap[mark].workspaceSize > workspaceSize)
        findAlgo = 0;
      else
      {
        cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, cublasAlgoMap[mark].algoId, &algo);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasAlgoMap[mark].customOption), sizeof(cublasAlgoMap[mark].customOption));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasAlgoMap[mark].tile), sizeof(cublasAlgoMap[mark].tile));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasAlgoMap[mark].splitK_val), sizeof(cublasAlgoMap[mark].splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasAlgoMap[mark].swizzle), sizeof(cublasAlgoMap[mark].swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasAlgoMap[mark].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasAlgoMap[mark].stages), sizeof(cublasAlgoMap[mark].stages));
#endif
      }
    }


    cublasLtMatmul(ltHandle,
                    operationDesc,
                    alpha,
                    A,
                    Adesc,
                    B,
                    Bdesc,
                    beta,
                    C,
                    Cdesc,
                    C,
                    Cdesc,
                    (findAlgo == 1 ? (&algo) : NULL), workSpace, workspaceSize, stream);

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
  }
  else{
    int cublasAlgo = is_fp16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if (findAlgo)
      cublasAlgo = cublasAlgoMap[mark].algoId;

    cudaDataType_t computeType = is_fp16 ? CUDA_R_16F : CUDA_R_32F;

    check_cuda_error(cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                                  A, Atype, lda,
                                  B, Btype, ldb,
                                  beta, C, Ctype, ldc,
                                  computeType, static_cast<cublasGemmAlgo_t>(cublasAlgo)));
  }
  mu_.unlock();
}

//used in encoder
template <typename T1, typename T2>
void readAlgoFromConfig(int int8_mode, T1& cublasAlgoMap, T2& parameterMap, bool use_parameterMap = true)
{
  cublasAlgoMap.clear();
  if (use_parameterMap)
    parameterMap.clear();
  FILE* fd;
  if (int8_mode == 0)
    fd = fopen(GEMM_CONFIG, "r");
  else
    fd = fopen(IGEMM_CONFIG, "r");
  if (fd == NULL)
    return;
  int batchCount2, m2, n2, k2, algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize, stages;
  float exec_time;
  int batch_size, seq_len, head_num, size_per_head, dataType;
  char tmp[1024];
  if(!fgets(tmp, 1024, fd))
  {
    printf("[ERROR] fgets fail at %s:%d \n", __FILE__, __LINE__);
    exit(-1);
  }
  while(fscanf(fd,"%d %d %d %d %d ### %d %d %d %d %d %d %d %d %d %d %d %d %f\n", &batch_size, &seq_len, &head_num, &size_per_head, &dataType, &batchCount2, &m2, &n2, &k2, &algoId, &customOption, &tile, &splitK_val, &swizzle, &reductionScheme, &workspaceSize, &stages, &exec_time)!=EOF)
  {
    if (dataType != FLOAT_DATATYPE && dataType != HALF_DATATYPE && dataType != INT8_DATATYPE)
    {
      printf("[WARNING][readAlgoFromConfig] wrong dataType %d!\n", dataType);
      continue;
    }
    char mark[256];
    sprintf(mark, "%d_%d_%d_%d_%d", batch_size, seq_len, head_num, size_per_head, dataType);
    std::string markStr0(mark);
    sprintf(mark, "%d_%d_%d_%d_%d", batchCount2, m2, n2, k2, dataType);
    std::string markStr(mark);
    //workspaceSize should be zero
    if (cublasAlgoMap.find(markStr) == cublasAlgoMap.end())
    {
      if (use_parameterMap)
        parameterMap[markStr0] = 1;
      cublasAlgoMap[markStr].algoId = algoId;
      cublasAlgoMap[markStr].customOption = customOption;
      cublasAlgoMap[markStr].tile = tile;
      cublasAlgoMap[markStr].splitK_val = splitK_val;
      cublasAlgoMap[markStr].swizzle = swizzle;
      cublasAlgoMap[markStr].reductionScheme = reductionScheme;
      cublasAlgoMap[markStr].workspaceSize = workspaceSize;
      cublasAlgoMap[markStr].stages = stages;
      cublasAlgoMap[markStr].exec_time = exec_time;
    }
  }
  fclose(fd);
}

//used in decoder
template <typename T>
void readAlgoFromConfig(T& cublasAlgoMap, int num=-1)
{
  cublasAlgoMap.clear();
  FILE* fd;
  fd = fopen("decoding_gemm_config.in", "r");
  if (fd == NULL)
    return;
  int batchCount2, m2, n2, k2, algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize, stages;
  float exec_time;
  int dataType;
  int readInAlgo = 0;
  char tmp[1024];
  if(!fgets(tmp, 1024, fd))
  {
    printf("[ERROR] fgets fail at %s:%d \n", __FILE__, __LINE__);
    exit(-1);
  }
  while(fscanf(fd,"%d %d %d %d %d %d %d %d %d %d %d %d %d %f\n", &dataType, &batchCount2, &m2, &n2, &k2, &algoId, &customOption, &tile, &splitK_val, &swizzle, &reductionScheme, &workspaceSize, &stages, &exec_time)!=EOF)
  {
    if (dataType != FLOAT_DATATYPE && dataType != HALF_DATATYPE && dataType != INT8_DATATYPE)
    {
      printf("[WARNING][readAlgoFromConfig] wrong dataType %d!\n", dataType);
      continue;
    }
    char mark[256];
    sprintf(mark, "%d_%d_%d_%d_%d", batchCount2, m2, n2, k2, dataType);
    std::string markStr(mark);
    //workspaceSize should be zero
    if (cublasAlgoMap.find(markStr) == cublasAlgoMap.end())
    {
      cublasAlgoMap[markStr].algoId = algoId;
      cublasAlgoMap[markStr].customOption = customOption;
      cublasAlgoMap[markStr].tile = tile;
      cublasAlgoMap[markStr].splitK_val = splitK_val;
      cublasAlgoMap[markStr].swizzle = swizzle;
      cublasAlgoMap[markStr].reductionScheme = reductionScheme;
      cublasAlgoMap[markStr].workspaceSize = workspaceSize;
      cublasAlgoMap[markStr].stages = stages;
      cublasAlgoMap[markStr].exec_time = exec_time;
    }
    readInAlgo++;
    if (num != -1 && readInAlgo == num)
      break;
  }
  fclose(fd);
}

//used in decoder
template <typename T>
void cublasMM_cublasLtMM_wrapper_decoder(cublasLtHandle_t ltHandle, cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k, const void *alpha,
                                         const void *A, cudaDataType_t Atype, int lda,
                                         const void *B, cudaDataType_t Btype, int ldb,
                                         const void *beta, T *C, cudaDataType_t Ctype, int ldc,
                                         cudaStream_t stream, 
                                         std::map<std::string, cublasLtMatmulAlgo_info>& cublasAlgoMap,
                                         void* cublas_workspace){
  // TODO
  // Disable the mutex because it affects the performance of multi-thread gpt significantly
  // And we don't need to lock because each instance in gpt has its own cublas handler.
  // mu_.lock();
  int is_fp16 = Atype == CUDA_R_16F ? 1 : 0;
  int batchCount = 1;
  //use cublas as default
  bool using_cublasLt = false;

  int findAlgo = 0;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, is_fp16 ? HALF_DATATYPE : FLOAT_DATATYPE);
  if(cublasAlgoMap.find(mark) != cublasAlgoMap.end())
  {
    findAlgo = 1;
    if (cublasAlgoMap[mark].stages != -1)
      using_cublasLt = true;
    else
      using_cublasLt = false;
  }

#ifndef NDEBUG
  if(findAlgo == 0)
  {
    static int not_find_count = 0;
    not_find_count++;
    if(not_find_count < 50)
      printf("[WARNING] %s No find Algo, using default algo. \n", mark);
  }
#endif

  if(using_cublasLt){
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t scaleType;
#ifdef CUDA11_MODE
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if(is_fp16){
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_16F;
#else
      computeType = CUDA_R_16F;
#endif
      scaleType = CUDA_R_16F;
    }else{
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_32F;
#else
      computeType = CUDA_R_32F;
#endif
      scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    //Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);
#ifdef CUDA11_MODE
      cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
      cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    cublasLtMatmulAlgo_t algo;
    void * workSpace = cublas_workspace;
    int workspaceSize = cublas_workspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if(findAlgo){
      if (cublasAlgoMap[mark].workspaceSize > workspaceSize)
        findAlgo = 0;
      else
      {
        cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, cublasAlgoMap[mark].algoId, &algo);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasAlgoMap[mark].customOption), sizeof(cublasAlgoMap[mark].customOption));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasAlgoMap[mark].tile), sizeof(cublasAlgoMap[mark].tile));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasAlgoMap[mark].splitK_val), sizeof(cublasAlgoMap[mark].splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasAlgoMap[mark].swizzle), sizeof(cublasAlgoMap[mark].swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasAlgoMap[mark].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasAlgoMap[mark].stages), sizeof(cublasAlgoMap[mark].stages));
#endif
      }
    }


    cublasLtMatmul(ltHandle,
                    operationDesc,
                    alpha,
                    A,
                    Adesc,
                    B,
                    Bdesc,
                    beta,
                    C,
                    Cdesc,
                    C,
                    Cdesc,
                    (findAlgo == 1 ? (&algo) : NULL), workSpace, workspaceSize, stream);

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
  }
  else{
    int cublasAlgo = is_fp16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if (findAlgo)
      cublasAlgo = cublasAlgoMap[mark].algoId;

    cudaDataType_t computeType = is_fp16 ? CUDA_R_16F : CUDA_R_32F;

    check_cuda_error(cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                                  A, Atype, lda,
                                  B, Btype, ldb,
                                  beta, C, Ctype, ldc,
                                  computeType, static_cast<cublasGemmAlgo_t>(cublasAlgo)));
  }
  // mu_.unlock();
}

template <typename T>
int getAlgoIdFromMap(T& cublasAlgoMap, int batchCount, int m, int n, int k, int dataType)
{
  char mark[256];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, dataType);
  if (cublasAlgoMap.find(mark) != cublasAlgoMap.end())
    return cublasAlgoMap[mark].algoId;
  else
    return dataType == FLOAT_DATATYPE ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}

}
