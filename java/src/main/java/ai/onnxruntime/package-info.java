/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * A Java interface to the ONNX Runtime.
 *
 * <p>Provides access to the same execution backends as the C library. Non-representable types in
 * Java (such as fp16) are converted into the nearest Java primitive type when accessed through this
 * API.
 *
 * <p>There are several shared libraries required for operation, primarily <code>onnxruntime</code>
 * and <code>onnxruntime4j_jni</code>, as well as <code>onnxruntime_providers_shared</code>. The
 * loader is in {@link ai.onnxruntime.OnnxRuntime} and the logic follows this order:
 *
 * <ol>
 *   <li>The user may signal to skip loading of a shared library using a property in the form <code>
 * onnxruntime.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means the user
 *       has decided to load the library by some other means.
 *   <li>The user may specify an explicit directory containing all native library files using the
 *       property <code>onnxruntime.native.path</code>. This uses {@link java.lang.System#load}.
 *   <li>The user may specify an explicit location of a specific shared library file using a
 *       property in the form <code>onnxruntime.native.LIB_NAME.path</code>. This uses {@link
 *       java.lang.System#load}.
 *   <li>The shared library is autodiscovered:
 *       <ol>
 *         <li>If the shared library is present in the classpath resources, it is extracted and
 *             loaded using {@link java.lang.System#load}.
 *             <ul>
 *               <li>The extraction location can be controlled via <code>
 * onnxruntime.native.extract.path</code>. If unset, a temporary directory is created.
 *               <li>By default, extracted libraries are deleted on JVM termination. This can be
 *                   disabled by setting <code>onnxruntime.native.extract.cleanup</code> to <code>
 * false</code>.
 *             </ul>
 *         <li>If the shared library is not present in the classpath resources, then load using
 *             {@link java.lang.System#loadLibrary}, which looks on the standard library paths
 *             (e.g., <code>java.library.path</code>).
 *       </ol>
 * </ol>
 *
 * For troubleshooting, all shared library loading events are reported to Java logging at the level
 * FINE.
 *
 * <p>Note that CUDA, ROCM, DNNL, OpenVINO, TensorRT, and QNN are all "shared library execution
 * providers." These, along with WebGPU dependencies like Dawn or DXC, must be stored either in the
 * directory containing the ONNX Runtime core native library or as a classpath resource. This is
 * because these providers are loaded by the ONNX Runtime native library itself, and the Java API
 * handles their extraction/preparation to ensure they are available to the native loader.
 */
package ai.onnxruntime;
