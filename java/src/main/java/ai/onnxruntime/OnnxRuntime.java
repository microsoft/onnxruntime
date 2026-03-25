/*
 * Copyright (c) 2019, 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Static loader for the JNI binding. No public API, but called from various classes in this package
 * to ensure shared libraries are properly loaded.
 */
final class OnnxRuntime {
  private static final Logger logger = Logger.getLogger(OnnxRuntime.class.getName());

  // The initial release of the ORT API.
  private static final int ORT_API_VERSION_1 = 1;
  // Post 1.0 builds of the ORT API.
  private static final int ORT_API_VERSION_2 = 2;
  // Post 1.3 builds of the ORT API
  private static final int ORT_API_VERSION_3 = 3;
  // Post 1.6 builds of the ORT API
  private static final int ORT_API_VERSION_7 = 7;
  // Post 1.7 builds of the ORT API
  private static final int ORT_API_VERSION_8 = 8;
  // Post 1.10 builds of the ORT API
  private static final int ORT_API_VERSION_11 = 11;
  // Post 1.12 builds of the ORT API
  private static final int ORT_API_VERSION_13 = 13;
  // Post 1.13 builds of the ORT API
  private static final int ORT_API_VERSION_14 = 14;
  // Post 1.22 builds of the ORT API
  private static final int ORT_API_VERSION_23 = 23;

  // The initial release of the ORT training API.
  private static final int ORT_TRAINING_API_VERSION_1 = 1;

  private static final String ONNXRUNTIME_NATIVE_PREFIX = "onnxruntime.native";

  private static final String CLASSPATH_BASE = "/ai/onnxruntime/native";

  /** The short name of the ONNX runtime shared library */
  private static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";

  /** The short name of the ONNX runtime JNI shared library */
  private static final String ONNXRUNTIME_JNI_LIBRARY_NAME = "onnxruntime4j_jni";

  /** The short name of the ONNX runtime shared provider library */
  private static final String ONNXRUNTIME_LIBRARY_SHARED_NAME = "onnxruntime_providers_shared";

  /** The short name of the ONNX runtime CUDA provider library */
  private static final String ONNXRUNTIME_LIBRARY_CUDA_NAME = "onnxruntime_providers_cuda";

  /** The short name of the ONNX runtime ROCM provider library */
  private static final String ONNXRUNTIME_LIBRARY_ROCM_NAME = "onnxruntime_providers_rocm";

  /** The short name of the ONNX runtime DNNL provider library */
  private static final String ONNXRUNTIME_LIBRARY_DNNL_NAME = "onnxruntime_providers_dnnl";

  /** The short name of the ONNX runtime OpenVINO provider library */
  private static final String ONNXRUNTIME_LIBRARY_OPENVINO_NAME = "onnxruntime_providers_openvino";

  /** The short name of the ONNX runtime TensorRT provider library */
  private static final String ONNXRUNTIME_LIBRARY_TENSORRT_NAME = "onnxruntime_providers_tensorrt";

  /** The short name of the ONNX runtime QNN provider library */
  private static final String ONNXRUNTIME_LIBRARY_QNN_NAME = "onnxruntime_providers_qnn";

  /** The short name of the WebGPU DAWN library */
  private static final String ONNXRUNTIME_LIBRARY_WEBGPU_DAWN_NAME = "webgpu_dawn";

  /** The short name of the WebGPU DXC library "dxil.dll" */
  private static final String ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXIL_NAME = "dxil";

  /** The short name of the WebGPU DXC library "dxcompiler.dll" */
  private static final String ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXCOMPILER_NAME = "dxcompiler";

  /** Have the core ONNX Runtime native libraries been loaded */
  private static boolean loaded = false;

  /** Tracks if the shared providers have been extracted */
  private static final Set<String> extractedSharedProviders = new HashSet<>();

  private static OnnxStager stager;

  /** The API handle. */
  static long ortApiHandle;

  /** The Training API handle. */
  static long ortTrainingApiHandle;

  /** The Compile API handle. */
  static long ortCompileApiHandle;

  /** Is training enabled in the native library */
  static boolean trainingEnabled;

  /** The available runtime providers */
  static EnumSet<OrtProvider> providers;

  /** The version string. */
  private static String version;

  private OnnxRuntime() {}

  /**
   * Loads the native C library.
   *
   * @throws IOException If it can't write to disk to copy out the library from the jar file.
   * @throws IllegalStateException If the native library failed to load.
   */
  static synchronized void init() throws IOException {
    if (loaded) {
      return;
    }
    if (isAndroid()) {
      System.loadLibrary(ONNXRUNTIME_JNI_LIBRARY_NAME);
    } else {
      stager = new OnnxStager();
      stager.stage(ONNXRUNTIME_NATIVE_PREFIX, ONNXRUNTIME_LIBRARY_NAME, CLASSPATH_BASE);
      stager.stage(ONNXRUNTIME_NATIVE_PREFIX, ONNXRUNTIME_LIBRARY_SHARED_NAME, CLASSPATH_BASE);
      String jniLibraryPath =
          stager
              .stage(ONNXRUNTIME_NATIVE_PREFIX, ONNXRUNTIME_JNI_LIBRARY_NAME, CLASSPATH_BASE)
              .toString();
      if (jniLibraryPath != null) {
        System.load(jniLibraryPath);
      }
    }
    ortApiHandle = initialiseAPIBase(ORT_API_VERSION_23);
    if (ortApiHandle == 0L) {
      throw new IllegalStateException(
          "There is a mismatch between the ORT class files and the ORT native library, and the native library could not be loaded");
    }
    ortTrainingApiHandle = initialiseTrainingAPIBase(ortApiHandle, ORT_API_VERSION_23);
    ortCompileApiHandle = initialiseCompileAPIBase(ortApiHandle);
    trainingEnabled = ortTrainingApiHandle != 0L;
    providers = initialiseProviders(ortApiHandle);
    version = initialiseVersion();
    loaded = true;
  }

  /**
   * Gets the native library version string.
   *
   * @return The version string.
   */
  static String version() {
    return version;
  }

  /**
   * Stages the WebGPU provider libraries according to the default staging procedure specified by
   * {@link ai.onnxruntime}.
   *
   * @return True if the WebGPU provider libraries is ready for loading by the native runtime, false
   *     otherwise.
   */
  static boolean stageWebGPU() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_WEBGPU_DAWN_NAME)
        && stageProviderLibrary(ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXIL_NAME)
        && stageProviderLibrary(ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXCOMPILER_NAME);
  }

  /**
   * Stages the CUDA provider library according to the default staging procedure specified by {@link
   * ai.onnxruntime}.
   *
   * @return True if the CUDA library libraries is ready for loading by the native runtime, false
   *     otherwise.
   */
  static boolean stageCUDA() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_CUDA_NAME);
  }

  /**
   * Stages the ROCM provider library according to the default staging procedure specified by {@link
   * ai.onnxruntime}.
   *
   * @return True if the ROCM library libraries is ready for loading by the native runtime, false
   *     otherwise.
   */
  static boolean stageROCM() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_ROCM_NAME);
  }

  /**
   * Stages the DNNL provider library according to the default staging procedure specified by {@link
   * ai.onnxruntime}.
   *
   * @return True if the DNNL library libraries is ready for loading by the native runtime, false
   *     otherwise.
   */
  static boolean stageDNNL() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_DNNL_NAME);
  }

  /**
   * Stages the OpenVINO provider library according to the default staging procedure specified by
   * {@link ai.onnxruntime}.
   *
   * @return True if the OpenVINO library libraries is ready for loading by the native runtime,
   *     false otherwise.
   */
  static boolean stageOpenVINO() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_OPENVINO_NAME);
  }

  /**
   * Stages the TensorRT provider library according to the default staging procedure specified by
   * {@link ai.onnxruntime}.
   *
   * @return True if the TensorRT library libraries is ready for loading by the native runtime,
   *     false otherwise.
   */
  static boolean stageTensorRT() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_TENSORRT_NAME);
  }

  /**
   * Stages the QNN provider library according to the default staging procedure specified by {@link
   * ai.onnxruntime}.
   *
   * @return True if the QNN library libraries is ready for loading by the native runtime, false
   *     otherwise.
   */
  static boolean stageQNN() {
    return stageProviderLibrary(ONNXRUNTIME_LIBRARY_QNN_NAME);
  }

  /**
   * Stages the ROCM provider library according to the default staging procedure specified by {@link
   * ai.onnxruntime}.
   *
   * @param libraryName The shared provider library to stage.
   * @return True if the library is ready for loading by ORT's native code, false otherwise.
   */
  private static synchronized boolean stageProviderLibrary(String libraryName) {
    // Android does not need to extract provider libraries.
    if (isAndroid()) {
      return false;
    }
    try {
      stager.stage(ONNXRUNTIME_NATIVE_PREFIX, libraryName, CLASSPATH_BASE);
      return true;
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * Check if we're running on Android.
   *
   * @return True if the property java.vendor equals The Android Project, false otherwise.
   */
  static boolean isAndroid() {
    return System.getProperty("java.vendor", "generic").equals("The Android Project");
  }

  /**
   * Extracts the providers array from the C API, converts it into an EnumSet.
   *
   * <p>Throws IllegalArgumentException if a provider isn't recognized (note this exception should
   * only happen during development of ONNX Runtime, if it happens at any other point, file an issue
   * on <a href="https://github.com/microsoft/onnxruntime">GitHub</a>).
   *
   * @param ortApiHandle The API Handle.
   * @return The enum set.
   */
  private static EnumSet<OrtProvider> initialiseProviders(long ortApiHandle) {
    String[] providersArray = getAvailableProviders(ortApiHandle);

    EnumSet<OrtProvider> providers = EnumSet.noneOf(OrtProvider.class);

    for (String provider : providersArray) {
      providers.add(OrtProvider.mapFromName(provider));
    }

    return providers;
  }

  /**
   * Get a reference to the API struct.
   *
   * @param apiVersionNumber The API version to use.
   * @return A pointer to the API struct.
   */
  private static native long initialiseAPIBase(int apiVersionNumber);

  /**
   * Get a reference to the training API struct.
   *
   * @param apiHandle The ORT API struct pointer.
   * @param apiVersionNumber The API version to use.
   * @return A pointer to the training API struct.
   */
  private static native long initialiseTrainingAPIBase(long apiHandle, int apiVersionNumber);

  /**
   * Get a reference to the compile API struct.
   *
   * @param apiHandle The ORT API struct pointer.
   * @return A pointer to the compile API struct.
   */
  private static native long initialiseCompileAPIBase(long apiHandle);

  /**
   * Gets the array of available providers.
   *
   * @param ortApiHandle The API handle
   * @return The array of providers
   */
  private static native String[] getAvailableProviders(long ortApiHandle);

  /**
   * Gets the version string from the native library.
   *
   * @return The version string.
   */
  private static native String initialiseVersion();
}
