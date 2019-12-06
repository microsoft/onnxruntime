/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * A Java interface to the onnxruntime.
 * <p>
 * Provides access to the same execution backends as the C library.
 * Non-representable types in Java (such as fp16) are converted
 * into the nearest Java primitive type when accessed through this API.
 */
package ai.onnxruntime;