// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cudnn_loader.h"

#ifndef USE_CUDA_MINIMAL

#define ORT_CUDNN_FORWARD_STATUS(name, ...)                            \
  using Fn = decltype(&name);                                          \
  auto fn = onnxruntime::cuda::CudnnLibrary::Get().Resolve<Fn>(#name); \
  return fn != nullptr ? fn(__VA_ARGS__) : CUDNN_STATUS_NOT_INITIALIZED

extern "C" {

size_t CUDNNWINAPI cudnnGetVersion(void) {
  using Fn = decltype(&cudnnGetVersion);
  auto fn = onnxruntime::cuda::CudnnLibrary::Get().Resolve<Fn>("cudnnGetVersion");
  return fn != nullptr ? fn() : 0;
}

const char* CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
  using Fn = decltype(&cudnnGetErrorString);
  auto fn = onnxruntime::cuda::CudnnLibrary::Get().Resolve<Fn>("cudnnGetErrorString");
  return fn != nullptr ? fn(status) : onnxruntime::cuda::CudnnUnavailableErrorString();
}

cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t* handle) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreate, handle);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroy, handle);
}

cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetStream, handle, streamId);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateTensorDescriptor, tensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyTensorDescriptor, tensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
                                                     int nbDims, const int dimA[], const int strideA[]) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetTensorNdDescriptor, tensorDesc, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
                                                     cudnnDataType_t dataType, int n, int c, int h, int w) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetTensor4dDescriptor, tensorDesc, format, dataType, n, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
                                                     cudnnDataType_t* dataType, int* nbDims, int dimA[], int strideA[]) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetTensorNdDescriptor, tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateFilterDescriptor, filterDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyFilterDescriptor, filterDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                                                     cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetFilterNdDescriptor, filterDesc, dataType, format, nbDims, filterDimA);
}

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                                                     cudnnTensorFormat_t format, int k, int c, int h, int w) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetFilter4dDescriptor, filterDesc, dataType, format, k, c, h, w);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateConvolutionDescriptor, convDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyConvolutionDescriptor, convDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength,
                                                          const int padA[], const int filterStrideA[],
                                                          const int dilationA[], cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t computeType) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetConvolutionNdDescriptor, convDesc, arrayLength, padA, filterStrideA, dilationA, mode,
                           computeType);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetConvolutionGroupCount, convDesc, groupCount);
}

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnMathType_t mathType) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetConvolutionMathType, convDesc, mathType);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateActivationDescriptor, activationDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyActivationDescriptor, activationDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                       cudnnActivationMode_t mode,
                                                       cudnnNanPropagation_t reluNanOpt, double coef) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetActivationDescriptor, activationDesc, mode, reluNanOpt, coef);
}

cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreatePoolingDescriptor, poolingDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyPoolingDescriptor, poolingDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                      const cudnnPoolingMode_t mode,
                                                      const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
                                                      const int windowDimA[], const int paddingA[],
                                                      const int strideA[]) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetPoolingNdDescriptor, poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA,
                           paddingA, strideA);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* normDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateLRNDescriptor, normDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyLRNDescriptor, lrnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha,
                                                double lrnBeta, double lrnK) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetLRNDescriptor, normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* reduceTensorDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateReduceTensorDescriptor, reduceTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyReduceTensorDescriptor, reduceTensorDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         cudnnReduceTensorOp_t reduceTensorOp,
                                                         cudnnDataType_t reduceTensorCompType,
                                                         cudnnNanPropagation_t reduceTensorNanOpt,
                                                         cudnnReduceTensorIndices_t reduceTensorIndices,
                                                         cudnnIndicesType_t reduceTensorIndicesType) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetReduceTensorDescriptor, reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
                           reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateRNNDescriptor, rnnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyRNNDescriptor, rnnDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo,
                                                   cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode,
                                                   cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
                                                   cudnnDataType_t dataType, cudnnDataType_t mathPrec,
                                                   cudnnMathType_t mathType, int32_t inputSize, int32_t hiddenSize,
                                                   int32_t projSize, int32_t numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc, uint32_t auxFlags) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetRNNDescriptor_v8, rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType,
                           mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t* rnnDataDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateRNNDataDescriptor, rnnDataDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyRNNDataDescriptor, rnnDataDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
                                                    cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize,
                                                    int vectorSize, const int seqLengthArray[], void* paddingFill) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetRNNDataDescriptor, rnnDataDesc, dataType, layout, maxSeqLength, batchSize,
                           vectorSize, seqLengthArray, paddingFill);
}

cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnCreateDropoutDescriptor, dropoutDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDestroyDropoutDescriptor, dropoutDesc);
}

cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle,
                                                    float dropout, void* states, size_t stateSizeInBytes,
                                                    unsigned long long seed) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSetDropoutDescriptor, dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDropoutGetStatesSize, handle, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                                        const cudnnTensorDescriptor_t xDesc,
                                                        cudnnBatchNormMode_t mode) {
  ORT_CUDNN_FORWARD_STATUS(cudnnDeriveBNTensorDescriptor, derivedBnDesc, xDesc, mode);
}

cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle, const void* alpha,
                                         const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta,
                                         const cudnnTensorDescriptor_t cDesc, void* C) {
  ORT_CUDNN_FORWARD_STATUS(cudnnAddTensor, handle, alpha, aDesc, A, beta, cDesc, C);
}

cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                                                 const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
                                                 const void* beta, const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnActivationForward, handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                              const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
                                              const void* beta, const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnPoolingForward, handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
                                                      cudnnLRNMode_t lrnMode, const void* alpha,
                                                      const cudnnTensorDescriptor_t xDesc, const void* x,
                                                      const void* beta, const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnLRNCrossChannelForward, handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode, const void* alpha,
                                              const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta,
                                              const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSoftmaxForward, handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                               cudnnSoftmaxMode_t mode, const void* alpha,
                                               const cudnnTensorDescriptor_t yDesc, const void* y,
                                               const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta,
                                               const cudnnTensorDescriptor_t dxDesc, void* dx) {
  ORT_CUDNN_FORWARD_STATUS(cudnnSoftmaxBackward, handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias,
    const void* estimatedMean, const void* estimatedVariance, double epsilon) {
  ORT_CUDNN_FORWARD_STATUS(cudnnBatchNormalizationForwardInference, handle, mode, alpha, beta, xDesc, x, yDesc, y,
                           bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
}

cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias,
    double exponentialAverageFactor, void* resultRunningMean, void* resultRunningVariance, double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance) {
  ORT_CUDNN_FORWARD_STATUS(cudnnBatchNormalizationForwardTraining, handle, mode, alpha, beta, xDesc, x, yDesc, y,
                           bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
                           resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta,
    const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnConvolutionForward, handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
                           workSpaceSizeInBytes, beta, yDesc, y);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void* alpha1, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* alpha2,
    const cudnnTensorDescriptor_t zDesc, const void* z, const cudnnTensorDescriptor_t biasDesc, const void* bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void* y) {
  ORT_CUDNN_FORWARD_STATUS(cudnnConvolutionBiasActivationForward, handle, alpha1, xDesc, x, wDesc, w, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc,
                           y);
}

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t wDesc, const void* w,
    const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta,
    const cudnnTensorDescriptor_t dxDesc, void* dx) {
  ORT_CUDNN_FORWARD_STATUS(cudnnConvolutionBackwardData, handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc,
    const void* w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void* y,
    const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnFindConvolutionForwardAlgorithmEx, handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
                           requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc,
    const void* dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void* dx,
    const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnFindConvolutionBackwardDataAlgorithmEx, handle, wDesc, w, dyDesc, dy, convDesc, dxDesc,
                           dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetConvolutionForwardAlgorithm_v7, handle, srcDesc, filterDesc, convDesc, destDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
}

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t* sizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetConvolutionForwardWorkspaceSize, handle, xDesc, wDesc, convDesc, yDesc, algo,
                           sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                       const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                       const cudnnTensorDescriptor_t aDesc,
                                                       const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetReductionIndicesSize, handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                         const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                         const cudnnTensorDescriptor_t aDesc,
                                                         const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetReductionWorkspaceSize, handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                            void* indices, size_t indicesSizeInBytes, void* workspace,
                                            size_t workspaceSizeInBytes, const void* alpha,
                                            const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta,
                                            const cudnnTensorDescriptor_t cDesc, void* C) {
  ORT_CUDNN_FORWARD_STATUS(cudnnReduceTensor, handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
                           workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnForwardMode_t fwdMode, cudnnRNNDataDescriptor_t xDesc,
                                                    size_t* workSpaceSize, size_t* reserveSpaceSize) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetRNNTempSpaceSizes, handle, rnnDesc, fwdMode, xDesc, workSpaceSize,
                           reserveSpaceSize);
}

cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t pseudoLayer, size_t weightSpaceSize,
    const void* weightSpace, int32_t linLayerID, cudnnTensorDescriptor_t mDesc, void** mAddr,
    cudnnTensorDescriptor_t bDesc, void** bAddr) {
  ORT_CUDNN_FORWARD_STATUS(cudnnGetRNNWeightParams, handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace,
                           linLayerID, mDesc, mAddr, bDesc, bAddr);
}

cudnnStatus_t CUDNNWINAPI cudnnRNNForward(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t xDesc, const void* x, cudnnRNNDataDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t hDesc, const void* hx, void* hy, cudnnTensorDescriptor_t cDesc, const void* cx, void* cy,
    size_t weightSpaceSize, const void* weightSpace, size_t workSpaceSize, void* workSpace, size_t reserveSpaceSize,
    void* reserveSpace) {
  ORT_CUDNN_FORWARD_STATUS(cudnnRNNForward, handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx,
                           hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace,
                           reserveSpaceSize, reserveSpace);
}

}  // extern "C"

#undef ORT_CUDNN_FORWARD_STATUS

#endif  // USE_CUDA_MINIMAL
