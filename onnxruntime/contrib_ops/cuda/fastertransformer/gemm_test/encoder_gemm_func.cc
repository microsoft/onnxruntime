/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "encoder_gemm_func.h"
#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include <vector>
#include <chrono>
#include <stdio.h>
#include <string.h>

// Let try a fixed number of combinations
constexpr int ALGO_COMBINATIONS = 5000;

namespace fastertransformer{

// Utility function to print customMatmulPerf_t structure
int printPerfStructure(int batch_size, int seq_len, int head_num, int size_per_head, int m, int n, int k, const customMatmulPerf_t &perf, FILE* fout, int is_fp16, int hasPrint) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;
    
    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
#else
    stages=0;
#endif

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d stages=%d} status %d "
        "time %fms workspace=%d mathMode=%d waves=%f\n",       
        algoId, tile, matmulTileName[tile],
        numSplitsK, reductionScheme,
        swizzle, customOption, stages,
        perf.status,
        perf.time,
        (int)perf.workspaceSize,
        (int)perf.mathMode,
        perf.wavesCount);
    if (hasPrint == 0){
      fprintf(fout, "%d %d %d %d %d ### %d %d %d %d %d %d %d %d %d %d %d %d %f\n", batch_size, seq_len, head_num, size_per_head, is_fp16 ? HALF_DATATYPE:FLOAT_DATATYPE,
                1, m, n, k, algoId, customOption, tile, numSplitsK, swizzle, reductionScheme, (int)perf.workspaceSize, stages, perf.time);
      return 1;
    }
    else{
      return hasPrint;
    }

}

static inline bool
time_compare(const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b) {
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}


static cublasStatus_t 
customMatmulRun(cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
                 cublasLtMatmulDesc_t operationDesc,
                 const void *alpha, /* host or device pointer */
                 const void *A,
                 cublasLtMatrixLayout_t Adesc,
                 const void *B,
                 cublasLtMatrixLayout_t Bdesc,
                 const void *beta, /* host or device pointer */
                 const void *C,
                 cublasLtMatrixLayout_t Cdesc,
                 void *D,
                 cublasLtMatrixLayout_t Ddesc,
                 const cublasLtMatmulAlgo_t &algo,
                 int kernelRepeats,  
                 void *workSpace,
                 size_t workSpaceSizeInBytes,                 
                 customMatmulPerf_t &perfResults,                 
                 cudaStream_t stream,
                 cudaEvent_t &startEvent,
                 cudaEvent_t &stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int repeats = kernelRepeats;    
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck( ltHandle,
                                                         operationDesc,
                                                         Adesc,
                                                         Bdesc,
                                                         Cdesc,
                                                         Ddesc,
                                                         &algo, 
                                                         &heurResult);     
                                                                                 
    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cudaError_t err, err1, err2, err3;
            err  = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < repeats; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul( ltHandle,
                                                              operationDesc,
                                                              alpha,
                                                              A, Adesc,
                                                              B, Bdesc,
                                                              beta,
                                                              C, Cdesc,
                                                              D, Ddesc,
                                                              &algo,
                                                              workSpace,
                                                              workSpaceSizeInBytes,
                                                              stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }                                     
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                perfResults.algo = algo;  
                perfResults.time = time/repeats;  
                perfResults.workspaceSize = heurResult.workspaceSize; 
                perfResults.wavesCount = heurResult.wavesCount;                                                                       
            }
        }
        else {
            //printf("not enough workspace! %ld\n", heurResult.workspaceSize);
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; //Not enough workspace
        }        
    }
    
    return algoStatus;
}

template<typename T>
int LtHgemmCustomFind(cublasLtHandle_t ltHandle,
                  int batch_size, 
                  int seq_len, 
                  int head_num, 
                  int size_per_head,
                  int m,
                  int n,
                  int k,
                  const T *alpha, /* host pointer */
                  const T *A,
                  const T *B,
                  const T *beta, /* host pointer */
                  T *C,
                  void *workSpace,
                  size_t workSpaceSize,
                  FILE* fout,
                  customMatmulPerf_t perfResults[ALGO_COMBINATIONS])
{
    const int AlgoCombinations = ALGO_COMBINATIONS;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaEvent_t startEvent = NULL;
    cudaEvent_t stopEvent = NULL;
    int is_fp16 = (sizeof(T) == sizeof(half) ? 1 : 0);

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    cudaStream_t stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
     // Let try a fixed number of combinations
    int AlgoCount = 0;
    int AlgoCountRestrict = 0;  // workspace == 0
    constexpr int maxNumTraversal = 50;  // max number of traversal
    cublasLtMatmulAlgo_t algos[ALGO_COMBINATIONS];  // 0 <= workspace <= 32MB
    cublasLtMatmulAlgo_t algosRestrict[ALGO_COMBINATIONS];  // workspace == 0
    int kernelRepeats = 100; //number of time the CUDA kernels will be run back to back
    int nbAlgoIds = 0;  // Number of algorithms actually returned by cublasLtMatmulAlgoGetIds function.
    #define ALGO_IDS 100  // Number of algorithms requested.
    int algoIdA[ALGO_IDS];  // 	Array containing the algorithm IDs returned by cublasLtMatmulAlgoGetIds function.
    cudaDataType_t Atype, Btype, Ctype, scaleType;
#ifdef CUDA11_MODE
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if(sizeof(T) == sizeof(float)){
      scaleType = CUDA_R_32F, Atype = CUDA_R_32F, Btype = CUDA_R_32F, Ctype = CUDA_R_32F;
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_32F;
#else
      computeType = CUDA_R_32F;
#endif
    }else{
      scaleType = CUDA_R_16F, Atype = CUDA_R_16F, Btype = CUDA_R_16F, Ctype = CUDA_R_16F;
#ifdef CUDA11_MODE
      computeType = CUBLAS_COMPUTE_16F;
#else
      computeType = CUDA_R_16F;
#endif
    }

	// Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
#ifdef CUDA11_MODE
    status = cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType); //  creates a matrix multiply descriptor 
#else
    status = cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    // Create matrix descriptors. We are good with the details here so no need to set any extra attributes
    status = cublasLtMatrixLayoutCreate(
        &Adesc, Atype, m, k, m);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(
        &Bdesc, Btype, k, n, k);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    
    // Create CUDA event to time the execution time of each algo    
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {       
        goto CLEANUP;
    } 

    // Request the 100 first AlgoId available 
    status = cublasLtMatmulAlgoGetIds( ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP; 
    
    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {   
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute( &algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int nbTiles = int(sizeWritten/sizeof(int));
        int *tileA = new int[ nbTiles == 0 ? 1:nbTiles];
        if(nbTiles == 0){
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }
#ifdef CUDA11_MODE
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int nbStages = int(sizeWritten/sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages = 1;
        } else {
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int)*nbStages, &sizeWritten);
        }
#endif
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int)*nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);        
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        
        /* Loop over the different tiles */        
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
#ifdef CUDA11_MODE
            /* Loop over different stages count */
          for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
#endif
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++) {
               cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
               /* Loop over the CTAs swizzling support */
               for (int k = 0; k <= swizzlingMax; k++) {
                    int splitK_trial = 0;
                    if (splitkSupport) {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in addtion to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                        /* Setup attribute of the algo to run */                                                
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                       int splitK_val = 0;
                       int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));  
                                                                        
                        if (l > 0) { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1 ; redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations); redScheme = redScheme << 1) {
                                if (redScheme & redMask) {
                                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme));
                                    
                                    cublasLtMatmulHeuristicResult_t heurResult;
                                    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck( ltHandle,
                                                                                        operationDesc,
                                                                                        Adesc,
                                                                                        Bdesc,
                                                                                        Cdesc,
                                                                                        Cdesc,
                                                                                        &algo, 
                                                                                        &heurResult);
                                    if (heurResult.workspaceSize > workSpaceSize) {
                                      // printf("not enough workspace! %ld\n", heurResult.workspaceSize);
                                      algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; //Not enough workspace
                                    }else if(heurResult.workspaceSize == 0){
                                      if(algoStatus == CUBLAS_STATUS_SUCCESS){
                                        algosRestrict[AlgoCountRestrict++] = algo;
                                      }
                                    }
                                    if(algoStatus == CUBLAS_STATUS_SUCCESS){
                                      algos[AlgoCount++] = algo;
                                    }                      
                                } // end if
                            } // end for
                        } else { // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (AlgoCount < AlgoCombinations) {       
                                cublasLtMatmulHeuristicResult_t heurResult;
                                cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck( ltHandle,
                                                                                    operationDesc,
                                                                                    Adesc,
                                                                                    Bdesc,
                                                                                    Cdesc,
                                                                                    Cdesc,
                                                                                    &algo, 
                                                                                    &heurResult);
                                if (heurResult.workspaceSize > workSpaceSize) {
                                  // printf("not enough workspace! %ld\n", heurResult.workspaceSize);
                                  algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; //Not enough workspace
                                }else if(heurResult.workspaceSize == 0){
                                  if(algoStatus == CUBLAS_STATUS_SUCCESS){
                                    algosRestrict[AlgoCountRestrict++] = algo;
                                  }
                                }
                                if(algoStatus == CUBLAS_STATUS_SUCCESS){
                                  algos[AlgoCount++] = algo;
                                }
                            }
                        }
                    }  // end l
                }  // end k
            } //end customOption
#ifdef CUDA11_MODE
          } // end stagesIdx       
#endif                  
        } // end tileIdx
        delete [] tileA;
    } // end idx

    printf("AlgoCount: %d\n", AlgoCount);
    if(AlgoCount < maxNumTraversal){
      // 0 <= workspacesize <= 32MB
      for(int i=0;i<AlgoCount;i++){
        status = customMatmulRun( ltHandle,
                                  operationDesc,
                                  alpha, /* host or device pointer */
                                  A, Adesc,
                                  B, Bdesc,
                                  beta, /* host or device pointer */
                                  C, Cdesc,
                                  C, Cdesc,
                                  algos[i],
                                  kernelRepeats,  
                                  workSpace,
                                  workSpaceSize,                 
                                  perfResults[i],
                                  stream,
                                  startEvent, stopEvent);
        perfResults[i].status = status;
        // if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
      }
    }else{
      // Heuristic + workspacesize==0
      AlgoCount = 0;
      nbAlgoIds = 0;
      cublasLtMatmulPreference_t pref;
      cublasLtMatmulPreferenceCreate(&pref);
      uint64_t maxWorkSpaceSize = workSpaceSize; //(32MB)
      cublasLtMatmulPreferenceSetAttribute(
        pref, 
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maxWorkSpaceSize,
        sizeof(maxWorkSpaceSize));
      cublasLtMatmulHeuristicResult_t heuristicResultsArray[maxNumTraversal];

      cublasLtMatmulAlgoGetHeuristic(
            ltHandle,
            operationDesc,
            Adesc,
            Bdesc,
            Cdesc,
            Cdesc,
            pref,
            maxNumTraversal,
            heuristicResultsArray,
            &nbAlgoIds);
      cublasLtMatmulPreferenceDestroy(pref);
      printf("return %d and run heuristic algo\n", nbAlgoIds);
      for(int i = 0; i < nbAlgoIds; i++){
        if(heuristicResultsArray[i].state == CUBLAS_STATUS_SUCCESS){
          status = customMatmulRun( ltHandle,
                                  operationDesc,
                                  alpha, /* host or device pointer */
                                  A, Adesc,
                                  B, Bdesc,
                                  beta, /* host or device pointer */
                                  C, Cdesc,
                                  C, Cdesc,
                                  heuristicResultsArray[i].algo,
                                  kernelRepeats,  
                                  workSpace,
                                  workSpaceSize,                 
                                  perfResults[AlgoCount],
                                  stream,
                                  startEvent, stopEvent);
          perfResults[AlgoCount].status = status;
          if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
        }
      }

      // workspacesize==0
      printf("workspacesize==0, run %d algos\n", AlgoCountRestrict);
      for(int i=0;i<AlgoCountRestrict && i<(maxNumTraversal - nbAlgoIds);i++){
        status = customMatmulRun( ltHandle,
                                  operationDesc,
                                  alpha, /* host or device pointer */
                                  A, Adesc,
                                  B, Bdesc,
                                  beta, /* host or device pointer */
                                  C, Cdesc,
                                  C, Cdesc,
                                  algosRestrict[i],
                                  kernelRepeats,  
                                  NULL,
                                  0,                 
                                  perfResults[AlgoCount],
                                  stream,
                                  startEvent, stopEvent);
        perfResults[AlgoCount].status = status;
        if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
      }
    }
    

    // Sort the results per run duration 
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details 
    for (int i = 0, hasPrint = 1; i < AlgoCount; i++) {                
        printf( "result %03d : ", i);
        hasPrint = printPerfStructure(batch_size, seq_len, head_num, size_per_head, m, n, k, perfResults[i], fout, is_fp16, hasPrint);                          
    }


CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}



template<typename T>
void generate_encoder_gemm_config(int batch_size,
                                    int seq_len,
                                    int head_num,
                                    int size_per_head,
                                    void *buffer_in, 
                                    bool isAppend)
{

  void *cublas_workspace;
  void *buffer;
  int workSpaceSize;
  if (std::is_same<T, half>::value)
  {
    //cublas_workspace_ should be the start pointer of cudaMalloc()
    //to ensure 16B alignemnet
    cublas_workspace = buffer_in;
    buffer = (void*)((char*)cublas_workspace + CUBLAS_WORKSPACE_SIZE);
    workSpaceSize = CUBLAS_WORKSPACE_SIZE;
  }
  else
  {
    cublas_workspace = nullptr;
    buffer = buffer_in;
    workSpaceSize = 0;
  }
  
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);
  
  //check config 
  FILE *fd;
  int line_count = 0;
  if (!isAppend)
  {
    fopen_s(&fd, GEMM_CONFIG, "w+");
  }
  else
  {
    fopen_s(&fd, GEMM_CONFIG, "a+");
    std::vector<std::string> config;
    char line[1024];
    while (fgets(line, 1024, fd) != NULL)
    {
      config.push_back(std::string(line));
    }
    line_count = static_cast<int>(config.size());
    if (config.size() >= (MAX_CONFIG_NUM*GEMM_NUM + 1)) // 6 cublas/cublasLt, first row is not included
    {
      int startIdx = static_cast<int>(config.size()) - ((MAX_CONFIG_NUM - 1)*GEMM_NUM);
      fclose(fd);
      fopen_s(&fd, GEMM_CONFIG, "w+");
      fprintf(fd, "%s", config[0].c_str());
      for (size_t i = startIdx ; i < config.size() ; i++)
      {
        fprintf(fd, "%s", config[i].c_str());
      }
      line_count = static_cast<int>(config.size()) - (GEMM_NUM + 3);
    }
  }

  const int gemm_num = 6;
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  int batchCount[gemm_num] = {1,1,1,1,1,1};
  constexpr int MESSAGE_BUFFER_SIZE = 256;
  char mess[gemm_num][MESSAGE_BUFFER_SIZE];

  //gemm1 
  M[0] = batch_size * seq_len;
  K[0] = head_num * size_per_head;
  N[0] = K[0];
  strcpy_s(mess[0], MESSAGE_BUFFER_SIZE, "from_tensor * weightQ/K/V, attr * output_kernel");

  //gemm2
  M[1] = M[0];
  K[1] = K[0];
  N[1] = 4 * N[0];
  strcpy_s(mess[1], MESSAGE_BUFFER_SIZE, "attr_output * inter_kernel");

  //gemm3
  M[2] = M[0];
  K[2] = 4 * K[0];
  N[2] = N[0];
  strcpy_s(mess[2], MESSAGE_BUFFER_SIZE, "inter_matmul * output_kernel");

  M[3] = seq_len;
  N[3] = seq_len;
  K[3] = size_per_head;
  batchCount[3] = batch_size * head_num;
  strcpy_s(mess[3], MESSAGE_BUFFER_SIZE, "attention batched Gemm1");

  M[4] = seq_len;
  N[4] = size_per_head; 
  K[4] = seq_len;
  batchCount[4] = batch_size * head_num;
  strcpy_s(mess[4], MESSAGE_BUFFER_SIZE, "attention batched Gemm2");

  M[5] = batch_size * seq_len;
  N[5] = head_num * size_per_head; 
  K[5] = N[5];
  batchCount[5] = 3;
  strcpy_s(mess[5], MESSAGE_BUFFER_SIZE, "from_tensor * weight_QKV in BatchGemm");

  cublasHandle_t cublas_handle;
  check_cuda_error(cublasCreate(&cublas_handle));
  cublasLtHandle_t ltHandle;
  check_cuda_error(cublasLtCreate(&ltHandle));

  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  cudaDataType_t computeType;
  int startAlgo, endAlgo;
  const int ites = 100;
  //struct timeval start, end;

  if(sizeof(T) == sizeof(float)){
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
    computeType = CUDA_R_32F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT;
    endAlgo = (int)CUBLAS_GEMM_ALGO23;
  }
  else{
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
    computeType = CUDA_R_16F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  }

  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  printf("***Encoder Gemm Testing Begin***\n");
  printf("***Cublas Gemm Testing Begin***\n");
  if (line_count == 0){
    fprintf(fd, "batch_size, seq_len, head_num, size_per_head dataType ### batchCount, n, m, k, algoId, "\
    "customOption, tile, numSplitsK, swizzle, reductionScheme, workspaceSize, stages, exec_time\n");
  }
  for(int i = 0; i < gemm_num; ++i)
  {
    // if(i != 0 && i != 5) continue; 

    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A = (T*)buffer;
    T* d_B = d_A + m * k * batchCount[i];
    T* d_C = d_B + k * n * batchCount[i];

    // array of pointer for batchedGemm
    T* harray[9];
    harray[0] = (T*)buffer;
    harray[1] = (T*)((char*)buffer + sizeof(T) * m * k);
    harray[2] = (T*)((char*)buffer + 2 * sizeof(T) * m * k);
    harray[3] = (T*)((char*)buffer + 3 * sizeof(T) * m * k);
    harray[4] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + sizeof(T) * k * n);
    harray[5] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 2 * sizeof(T) * k * n);
    harray[6] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n);
    harray[7] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + sizeof(T) * m * n);
    harray[8] = (T*)((char*)buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + 2 * sizeof(T) * m * n);

    T** darray = 0;
    check_cuda_error(cudaMalloc((void**)&darray, sizeof(T*) * 9));
    cudaMemcpy((void*)darray, (void*)harray, sizeof(T*) * 9, cudaMemcpyHostToDevice);
    T** dAarray = darray;
    T** dBarray = darray + 3;
    T** dCarray = darray + 6;

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for(int algo = startAlgo; algo <= endAlgo; algo++)
    {
      cublasStatus_t status;
      cudaDeviceSynchronize();

      //gettimeofday(&start, NULL);
      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

      for(int ite = 0; ite < ites; ++ite)
      {
        if(i < 3)
        {
          status = cublasGemmEx(cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, 
                &alpha, 
                d_B, BType, n, 
                d_A, AType, k, 
                &beta, 
                d_C, CType, n, 
                computeType, 
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else if(i == 3)
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, size_per_head,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, size_per_head, seq_len * size_per_head,
                &beta,
                d_C, CType, seq_len, seq_len * seq_len,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else if(i == 4)
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                size_per_head, seq_len, seq_len,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, seq_len, seq_len * seq_len,
                &beta,
                d_C, CType, size_per_head, seq_len * size_per_head,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else if(i == 5)
        {
          status = cublasGemmBatchedEx(cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N, 
                            n, m, k, 
                            &alpha, 
                            (const void* const*) dBarray, BType, n,
                            (const void* const*) dAarray, AType, k,
                            &beta,
                            (void* const*)dCarray, CType, n,
                            3, 
                            computeType,
                            static_cast<cublasGemmAlgo_t>(algo));
        }
        if(status != CUBLAS_STATUS_SUCCESS) break;
      }
      cudaDeviceSynchronize();

      //gettimeofday(&end, NULL);
      std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

      if(status == CUBLAS_STATUS_SUCCESS)
      {
        float avg_ms = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / ites;
        printf("algo_%d costs %.3fms \n", algo, avg_ms);
        if (avg_ms < exec_time)
        {
          exec_time = avg_ms;
          fast_algo = algo;
        }
      }
    }
    printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
    int is_fp16 = 0;
    if (sizeof(T) == sizeof(half))
        is_fp16 = 1;
 
    //for fp16, we compare cublasLt
    if(i < 3 && is_fp16 == 1){
      printf("***cublasLt Gemm Testing Beign***\n");
      customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
      
      LtHgemmCustomFind<T>(ltHandle, batch_size, seq_len, head_num, size_per_head, n, m, k, &alpha, d_B, d_A, 
                      &beta, d_C, cublas_workspace, workSpaceSize, fd, perfResults);
      if(perfResults[0].time < exec_time){
        printPerfStructure(batch_size, seq_len, head_num, size_per_head, n, m, k, perfResults[0], fd, is_fp16, 0);
      }else{
        fprintf(fd, "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n", batch_size, seq_len, head_num, size_per_head, is_fp16 ? HALF_DATATYPE:FLOAT_DATATYPE,
              batchCount[i], n, m, k, fast_algo, exec_time);
      }
      printf("***cublasLt Gemm Testing End***\n");
    }
    else
    {
      fprintf(fd, "%d %d %d %d %d ### %d %d %d %d %d -1 -1 -1 -1 -1 -1 -1 %f\n", batch_size, seq_len, head_num, size_per_head, is_fp16 ? HALF_DATATYPE:FLOAT_DATATYPE,
              batchCount[i], n, m, k, fast_algo, exec_time);
    }
    cudaFree(darray);
  }
  printf("***cublas Gemm Testing End***\n\n");
  fclose(fd); 
  printf("***Encoder Gemm Testing End***\n");
  return;
}

template void generate_encoder_gemm_config<float>(int batch_size, int seq_len, int head_num, int size_per_head, void *buffer, bool isAppend);
template void generate_encoder_gemm_config<half>(int batch_size, int seq_len, int head_num, int size_per_head, void *buffer, bool isAppend);


size_t calGemmTestBufSizeInByte(int batch_size, int seq_len, int head_num, int size_per_head, int int8_mode, int is_fp16)
{
    size_t buf_size_in_byte;
    if (int8_mode > 0)
    {	
      int m = batch_size*seq_len;
      int n = head_num*size_per_head;
      int k = n;

      size_t size1 = 3*(m*k*sizeof(int8_t) + k*n*sizeof(int8_t) + m*n*sizeof(int));
      size_t size2 = batch_size*head_num*(seq_len*size_per_head*sizeof(int8_t) + size_per_head*seq_len*sizeof(int8_t) + seq_len*seq_len*sizeof(int));
      size_t size3 = batch_size*head_num*(seq_len*seq_len*sizeof(int8_t) + seq_len*size_per_head*sizeof(int8_t) + seq_len*size_per_head*sizeof(int));
      size_t size4 = m*k*sizeof(int8_t) + k*4*n*sizeof(int8_t) + 4*m*n*sizeof(int);
      buf_size_in_byte = size1 > size2 ? size1 : size2;
      buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
      buf_size_in_byte = buf_size_in_byte > size4 ? buf_size_in_byte : size4;
    }
    else
    {
      int m = batch_size*seq_len;
      int n = head_num*size_per_head;
      int k = n;
      int wordSize = (is_fp16 == 1 ? sizeof(half) : sizeof(float));
      size_t size1 = 3*(m*k + k*n + m*n)*wordSize;
      size_t size2 = batch_size*head_num*(seq_len*seq_len + seq_len*size_per_head + seq_len*size_per_head)*wordSize;
      size_t size3 = (m*k + k*4*n + m*4*n)*wordSize;
      buf_size_in_byte = size1 > size2 ? size1 : size2;
      buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
      buf_size_in_byte += ((is_fp16 == 1) ? CUBLAS_WORKSPACE_SIZE : 0);
    }
    return buf_size_in_byte;
}

}

