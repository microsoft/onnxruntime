/*
 * Copyright (c) 2020, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.HashMap;
import java.util.Map;

/** The execution providers available through the Java API. */
public enum OrtProvider {
  /** The CPU execution provider. */
  CPU("CPUExecutionProvider"),
  /** CUDA execution provider for Nvidia GPUs. */
  CUDA("CUDAExecutionProvider"),
  /** The Intel Deep Neural Network Library execution provider. */
  DNNL("DnnlExecutionProvider"),
  /** The OpenVINO execution provider. */
  OPEN_VINO("OpenVINOExecutionProvider"),
  /** The AMD/Xilinx VitisAI execution provider. */
  VITIS_AI("VitisAIExecutionProvider"),
  /** The TensorRT execution provider for Nvidia GPUs. */
  TENSOR_RT("TensorrtExecutionProvider"),
  /** The Android NNAPI execution provider. */
  NNAPI("NnapiExecutionProvider"),
  /** The RockChip NPU execution provider. */
  RK_NPU("RknpuExecutionProvider"),
  /** The Windows DirectML execution provider. */
  DIRECT_ML("DmlExecutionProvider"),
  /** The AMD MIGraphX execution provider. */
  MI_GRAPH_X("MIGraphXExecutionProvider"),
  /** The ARM Compute Library execution provider. */
  ACL("ACLExecutionProvider"),
  /** The ARM NN execution provider. */
  ARM_NN("ArmNNExecutionProvider"),
  /** The AMD ROCm execution provider. */
  ROCM("ROCMExecutionProvider"),
  /** The Apple CoreML execution provider. */
  CORE_ML("CoreMLExecutionProvider"),
  /** The XNNPACK execution provider. */
  XNNPACK("XnnpackExecutionProvider"),
  /** The Azure remote endpoint execution provider. */
  AZURE("AzureExecutionProvider");

  private static final Map<String, OrtProvider> valueMap = new HashMap<>(values().length);

  static {
    for (OrtProvider p : OrtProvider.values()) {
      valueMap.put(p.name, p);
    }
  }

  private final String name;

  OrtProvider(String name) {
    this.name = name;
  }

  /**
   * Accessor for the internal name of this provider.
   *
   * @return The internal provider name.
   */
  public String getName() {
    return name;
  }

  /**
   * Maps from the name string used by ONNX Runtime into the enum.
   *
   * @param name The provider name string.
   * @return The enum constant.
   */
  public static OrtProvider mapFromName(String name) {
    OrtProvider provider = valueMap.get(name);
    if (provider == null) {
      throw new IllegalArgumentException("Unknown execution provider - " + name);
    } else {
      return provider;
    }
  }
}
