#include <algorithm>
#include <string>
#include <vector>

#include <cuda.h>
#include <cublasLt.h>

#include "cublaslt_util.h"


constexpr size_t maxWorkspaceBytes = 4 * 1024 * 1024; // 4MB

// Utility function to print MatmulPerf_t structure
static void printPerfStructure(const MatmulPerf_t &perf, int const &m, int const &n, int const &k) {
  AlgoProps p;
  p.populate(perf.algo);
  // Calculate GFLOPS
  double timeAvg = (perf.time * 1e-3) / NUM_ITERATIONS; // Convert to seconds, then divide by loops
  double gflop = (2 * static_cast<unsigned long long int>(m * n) * k) * 1e-9; // Real

  std::cout << "AlgoID=" << p.algoId << " TileID=" << p.tile << " ("
            << matmulTileName[p.tile] << ") K=" << p.numSplitsK
            << " ReductionScheme=" << p.reductionScheme << " Swizzle=" << p.swizzle
            << " CustomOption=" << p.customOption << " Status=" << perf.status
            << " Time=" << perf.time << " WorkspaceSize=" << perf.workspaceSize
            << " MathMode=" << p.mathMode << " WavesCount=" << perf.wavesCount
            << " GFlops=" << (gflop / timeAvg) << std::endl;
}

static inline bool time_compare(const MatmulPerf_t &perf_a, const MatmulPerf_t &perf_b) {
  return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t handle,
                                    cublasLtMatmulDesc_t operationDesc,
                                    void const *alpha,
                                    void const *A, cublasLtMatrixLayout_t Adesc,
                                    void const *B, cublasLtMatrixLayout_t Bdesc,
                                    void const *beta,
                                    void const *C, cublasLtMatrixLayout_t Cdesc,
                                    void *D, cublasLtMatrixLayout_t Ddesc,
                                    cublasLtMatmulAlgo_t const &algo,
                                    void *workSpace, size_t workSpaceSizeInBytes,
                                    MatmulPerf_t &perfResults,
                                    cudaStream_t stream, cudaEvent_t &start, cudaEvent_t &stop) {
  cublasLtMatmulHeuristicResult_t result;
  cublasStatus_t status = cublasLtMatmulAlgoCheck(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &result);

  if (status == CUBLAS_STATUS_SUCCESS) {
    if (result.workspaceSize <= workSpaceSizeInBytes) {
      cudaError_t err, err1, err2, err3;
      err = cudaEventRecord(start, stream);
      for (int loop = 0; loop < NUM_ITERATIONS; loop++) {
        cublasStatus_t oneRunStatus = cublasLtMatmul(handle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, &algo, workSpace, workSpaceSizeInBytes, stream);
        if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
          status = oneRunStatus;
          break;
        }
      }
      err1 = cudaEventRecord(stop, stream);
      err2 = cudaEventSynchronize(stop);

      float time;
      err3 = cudaEventElapsedTime(&time, start, stop);
      if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
        status = CUBLAS_STATUS_INTERNAL_ERROR;
      }
      // For the moment only add successful findings
      if (status == CUBLAS_STATUS_SUCCESS) {
        perfResults.algo = algo;
        perfResults.time = time / NUM_ITERATIONS; // Average time
        perfResults.workspaceSize = result.workspaceSize;
        perfResults.wavesCount = result.wavesCount;
      }
    } else {
      status = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
    }
  }
  return status;
}

// Sample wrapper running through multiple algo and config attributes
// combination for single precision gemm using cublasLt low-level API
void LtGemmSearch(cublasLtHandle_t handle,
                  cublasOperation_t transA,
                  cublasOperation_t transB,
                  int const &m, int const &n, int const &k,
                  void const *alpha, /* host pointer */
                  void const *A, int const &lda,
                  void const *B, int const &ldb,
                  void const *beta, /* host pointer */
                  void *C, int const &ldc,
                  void *workSpace, size_t workSpaceSize,
                  cublasComputeType_t computeType,
                  cudaDataType_t scaleType,
                  cudaDataType_t Atype,
                  cudaDataType_t Btype,
                  cudaDataType_t Ctype,
                  std::vector<MatmulPerf_t> &perfResults) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  // Let try a fixed number of combinations
  int algoCount = 0;
  int numAlgoIds = 0;
  int algoIds[MAX_NUM_ALGO_IDS];
  // MatmulPerf_t perfResults[algoCombinations];

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

  const int mathMode = Ctype == CUDA_R_16F ? 1 : 0;
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MATH_MODE_MASK, &mathMode, sizeof(mathMode));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, Atype, transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, lda));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, Btype, transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldb));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoGetIds(handle, computeType, scaleType, Atype, Btype, Ctype, Ctype, MAX_NUM_ALGO_IDS, algoIds, &numAlgoIds));

  std::cout << "Number of algos: " << numAlgoIds << std::endl;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  cudaStream_t stream = nullptr;

  // Loop over the Algo IDs
  for (int idx = 0; (idx < numAlgoIds) && (algoCount < algoCombinations); idx++) {
    cublasLtMatmulAlgo_t algo;
    status = cublasLtMatmulAlgoInit(handle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIds[idx], &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }

    int mathMode = -1;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof(mathMode), nullptr);

    size_t sizeWritten = 0;
    // Query the tiles enums supported by that algo
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
    int numTiles = int(sizeWritten / sizeof(int));
    int *tiles = new int[numTiles == 0 ? 1 : numTiles];
    if (numTiles == 0) {
      tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
      numTiles = 1;
    }

    int splitkSupport, reductionMask, swizzlingMax, customOptionMax, epilogueMask;
    // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(int) * numTiles, &sizeWritten));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reductionMask, sizeof(reductionMask), &sizeWritten));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKs[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Loop over the different tiles
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
      // only check tile > 64x64
      if (tiles[tileIdx] < 15) {
          continue;
      }
      // Loop over the different custom option if any
      for (int customOption = 0; customOption <= customOptionMax; customOption++) {
        CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
        // Loop over the CTAs swizzling support
        for (int k = 0; k <= swizzlingMax; k++) {
          int splitK_trial = 0;
        //   if (splitkSupport) {
        //     splitK_trial += sizeof(splitKs) / sizeof(splitKs[0]);
        //   }
          // Loop over the splitK value over a fixed sequence splitKs in addition to the case where splitK is not enabled
          for (int l = 0; (l < (1 + splitK_trial)) && (algoCount < algoCombinations); l++) {
            // Setup attribute of the algo to run
            CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[tileIdx], sizeof(tiles[tileIdx])));
            int splitK_val = 0;
            int reductionScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
            CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
            CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
            CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(int)));

            if (l > 0) { // Split-K case
              splitK_val = splitKs[l - 1];
              CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKs[l - 1], sizeof(splitKs[l - 1])));
              // Going over all the reduction scheme
              for (reductionScheme = 1; reductionScheme < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) && (algoCount < algoCombinations); reductionScheme = reductionScheme << 1) {
                if (reductionScheme & reductionMask) {
                  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme)));

                  status = customMatmulRun(handle,
                                           operationDesc,
                                           alpha,
                                           A, Adesc, B, Bdesc,
                                           beta,
                                           C, Cdesc, C, Cdesc,
                                           algo,
                                           workSpace, workSpaceSize,
                                           perfResults[algoCount],
                                           stream, start, stop);
                  perfResults[algoCount].status = status;
                  if (status == CUBLAS_STATUS_SUCCESS) {
                    algoCount++;
                  }
                }
              }
            } else { // Non-splitK case
              // if user preference is ok with workspace
              if (algoCount < algoCombinations) {
                status = customMatmulRun(handle,
                                         operationDesc,
                                         alpha,
                                         A, Adesc, B, Bdesc,
                                         beta,
                                         C, Cdesc, C, Cdesc,
                                         algo,
                                         workSpace, workSpaceSize,
                                         perfResults[algoCount],
                                         stream, start, stop);
                perfResults[algoCount].status = status;
                if (status == CUBLAS_STATUS_SUCCESS)
                  algoCount++;
              }
            }
          }
        }
      }
    }
    delete[] tiles;
  }

  // Sort the results per run duration
  std::sort(perfResults.begin(), perfResults.end(), time_compare);
  // Print timing and perf details of the fastest combinations
//   for (int i = 0; i < perfResults.size(); i++){
  for (int i = 0; i < 8; i++) {
    if (perfResults[i].time == 1000000.f)
      break;
    printPerfStructure(perfResults[i], m, n, k);
  }

  // Descriptors are no longer needed as all GPU work was already enqueued
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

template <typename T>
void call_cuBLASLt(TGemm<T> &gemm, T *d_Bias, const T *h_C_cublas) {
  std::cout << "\nRunning with cuBLASLt "
            << (std::is_same<T, half>::value ? "FP16..." : "FP32...")
            << std::endl;
  CUDA_CHECK(cudaMemset(gemm.C, 0, gemm.elemC * sizeof(T)));

  cublasLtHandle_t handle;
  CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&handle));

  size_t workspaceSize = 4 * 1024 * 1024;
  void *workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

  std::vector<MatmulPerf_t> perfResults(algoCombinations);
  LtGemmSearch(handle, gemm, workspace, workspaceSize, perfResults);
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &gemm.opA, sizeof(gemm.opA)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &gemm.opB, sizeof(gemm.opB)));

  // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
  // CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  // CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc,
  // CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_Bias, sizeof(d_Bias)));

  // create matrix descriptors
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Adesc, TGemm<T>::Types::cudaTypeI,
                                          gemm.rA, gemm.cA, gemm.ldA));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, TGemm<T>::Types::cudaTypeI,
                                          gemm.rB, gemm.cB, gemm.ldB));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, TGemm<T>::Types::cudaTypeO,
                                          gemm.m, gemm.n, gemm.ldC));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(handle,
                                operationDesc,
                                &gemm.alpha,
                                gemm.A, Adesc,
                                gemm.B, Bdesc,
                                &gemm.beta,
                                gemm.C, Cdesc,
                                gemm.C, Cdesc,
                                &perfResults[0].algo,
                                workspace, perfResults[0].workspaceSize,
                                0));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float time = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
  printf("cuBLASLt time:  %3.4f ms \n", time / NUM_ITERATIONS);
  // printf("     TFLOPS:  %.2f\n\n", (((double)M * N * K * 2)/(time/1000.)) /
  // 1e12);

  T *h_C_cublaslt = (T *)malloc(gemm.elemC * sizeof(T));
  if (h_C_cublaslt == nullptr) {
    printf("h_C_cublaslt is nullptr. \n");
  } else {
    memset(h_C_cublaslt, 0, gemm.elemC * sizeof(T));
  }
  CUDA_CHECK(cudaMemcpy(h_C_cublaslt, gemm.C, gemm.elemC * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  verify(h_C_cublas, h_C_cublaslt, gemm.elemC,
         std::is_same<T, half>::value ? "cuBLASLt FP16" : "cuBLASLt FP32");
  free(h_C_cublaslt);

  // descriptors are no longer needed as all GPU work was already enqueued
  if (Cdesc)
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(workspace));
  CUBLAS_RETURN_IF_ERROR(cublasLtDestroy(handle));
}

#define INSTANTIATE_CUBLASLT(T)                                                \
  template void call_cuBLASLt<T>(TGemm<T> & gemm, T * d_Bias,                  \
                                 const T *h_C_cublas);
INSTANTIATE_CUBLASLT(float)
INSTANTIATE_CUBLASLT(half)