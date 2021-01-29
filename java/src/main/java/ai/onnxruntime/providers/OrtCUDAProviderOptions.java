/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Options for configuring the CUDA provider. */
public final class OrtCUDAProviderOptions {

  public final int deviceID;
  public final OrtCudnnConvAlgoSearch convAlgo;
  public final long cudaMemLimit;
  public final int arenaExtendStrategy;
  public final boolean copyInDefaultStream;

  public OrtCUDAProviderOptions(
      int deviceID,
      OrtCudnnConvAlgoSearch convAlgo,
      long cudaMemLimit,
      int arenaExtendStrategy,
      boolean copyInDefaultStream) {
    if (deviceID < 0) {
      throw new IllegalArgumentException("Device id must be non-negative, recieved " + deviceID);
    }
    if (cudaMemLimit < 1) {
      throw new IllegalArgumentException("cudaMemLimit must be positive, recieved " + cudaMemLimit);
    }
    this.deviceID = deviceID;
    this.convAlgo = convAlgo;
    this.cudaMemLimit = cudaMemLimit;
    this.arenaExtendStrategy = arenaExtendStrategy;
    this.copyInDefaultStream = copyInDefaultStream;
  }

  /** Convolution algorithms in cuDNN. */
  public enum OrtCudnnConvAlgoSearch {
    EXHAUSTIVE(2), // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
    HEURISTIC(1), // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
    DEFAULT(0); // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

    public final int value;

    OrtCudnnConvAlgoSearch(int value) {
      this.value = value;
    }
  }
}
