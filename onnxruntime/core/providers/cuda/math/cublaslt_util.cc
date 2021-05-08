#include <algorithm>
#include <string>
#include <vector>

#include <cuda.h>
#include <cublasLt.h>

#include "cublaslt_util.h"

namespace onnxruntime {
namespace cuda {

constexpr size_t maxWorkspaceBytes = 4 * 1024 * 1024; // 4MB

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
  (cublasLtMatmulPreferenceCreate(&preference));
  (cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

  const int mathMode = Ctype == CUDA_R_16F ? 1 : 0;
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MATH_MODE_MASK, &mathMode, sizeof(mathMode));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  (cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
  (cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  (cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  (cublasLtMatrixLayoutCreate(&Adesc, Atype, transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, lda));
//   (cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   (cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

  (cublasLtMatrixLayoutCreate(&Bdesc, Btype, transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldb));
//   (cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   (cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

  (cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));
//   (cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//   (cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

  (cublasLtMatmulAlgoGetIds(handle, computeType, scaleType, Atype, Btype, Ctype, Ctype, MAX_NUM_ALGO_IDS, algoIds, &numAlgoIds));

  // std::cout << "Number of algos: " << numAlgoIds << std::endl;

  cudaEvent_t start, stop;
  (cudaEventCreate(&start));
  (cudaEventCreate(&stop));

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
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
    int numTiles = int(sizeWritten / sizeof(int));
    int *tiles = new int[numTiles == 0 ? 1 : numTiles];
    if (numTiles == 0) {
      tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
      numTiles = 1;
    }

    int splitkSupport, reductionMask, swizzlingMax, customOptionMax, epilogueMask;
    // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(int) * numTiles, &sizeWritten));
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reductionMask, sizeof(reductionMask), &sizeWritten));
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));
    (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

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
        (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
        // Loop over the CTAs swizzling support
        for (int k = 0; k <= swizzlingMax; k++) {
          int splitK_trial = 0;
        //   if (splitkSupport) {
        //     splitK_trial += sizeof(splitKs) / sizeof(splitKs[0]);
        //   }
          // Loop over the splitK value over a fixed sequence splitKs in addition to the case where splitK is not enabled
          for (int l = 0; (l < (1 + splitK_trial)) && (algoCount < algoCombinations); l++) {
            // Setup attribute of the algo to run
            (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[tileIdx], sizeof(tiles[tileIdx])));
            int splitK_val = 0;
            int reductionScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
            (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
            (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
            (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(int)));

            if (l > 0) { // Split-K case
              splitK_val = splitKs[l - 1];
              (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKs[l - 1], sizeof(splitKs[l - 1])));
              // Going over all the reduction scheme
              for (reductionScheme = 1; reductionScheme < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) && (algoCount < algoCombinations); reductionScheme = reductionScheme << 1) {
                if (reductionScheme & reductionMask) {
                  (cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme)));

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

  // Descriptors are no longer needed as all GPU work was already enqueued
  (cublasLtMatmulPreferenceDestroy(preference));
  (cublasLtMatrixLayoutDestroy(Cdesc));
  (cublasLtMatrixLayoutDestroy(Bdesc));
  (cublasLtMatrixLayoutDestroy(Adesc));
  (cublasLtMatmulDescDestroy(operationDesc));
  (cudaEventDestroy(start));
  (cudaEventDestroy(stop));
}

}  // namespace cuda
}  // namespace onnxruntime