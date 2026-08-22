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
 * <p>There are three shared libraries required: <code>onnxruntime</code>, <code>onnxruntime4j_jni
 * </code> and <code>onnxruntime_providers_shared</code>. Additional execution providers such as
 * CUDA, ROCM, DNNL, OpenVINO and TensorRT need additional libraries, e.g. <code>
 * onnxruntime_providers_cuda</code>. On non-Android systems all libraries are linked (or copied if
 * linking fails) into a temporary staging directory. From there {@link java.lang.System#load} is
 * called on <code>onnxruntime4j_jni</code> (except if staging of it was exempted by one of the
 * following rules).
 *
 * <p>This behavior can be influneced; the loader is in {@link ai.onnxruntime.OnnxRuntime} and
 * chooses the libraries to add to the staging direcgory in this order:
 *
 * <ol>
 *   <li>The user may signal to skip staging of all libraries with the name-prefix <code>
 *       onnxruntime.native</code> using the property <code>onnxruntime.native.skip</code> with a
 *       value of <code>true</code>. This means the user has decided to load the library by some
 *       other means.
 *   <li>The user may signal to skip staging of a shared library using a property in the form <code>
 *       onnxruntime.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means the
 *       user has decided to load the library by some other means.
 *   <li>The user may specify an explicit location of all native library files using a property in
 *       the form <code>onnxruntime.native.path</code>.
 *   <li>The user may specify an explicit location of the shared library file using a property in
 *       the form <code>onnxruntime.native.LIB_NAME.path</code>.
 *   <li>Lastly, if no matching property is set, the library is copied from the JARs classpath.
 * </ol>
 *
 * <p>For troubleshooting, all shared library loading events are reported to Java logging at the
 * level FINE.
 */
package ai.onnxruntime;
