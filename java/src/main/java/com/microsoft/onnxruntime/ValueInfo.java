/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

/**
 * Interface for info objects describing an OrtValue.
 *
 * Will be sealed to {@link MapInfo}, {@link TensorInfo} and {@link SequenceInfo}.
 */
public interface ValueInfo { }
