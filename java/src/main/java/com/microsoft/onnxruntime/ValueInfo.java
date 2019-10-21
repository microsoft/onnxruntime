/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
package com.microsoft.onnxruntime;

/**
 * Interface for info objects describing an OrtValue.
 *
 * Will be sealed to {@link MapInfo}, {@link TensorInfo} and {@link SequenceInfo}.
 */
public interface ValueInfo { }
