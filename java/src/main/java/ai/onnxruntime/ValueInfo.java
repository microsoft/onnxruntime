/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/**
 * Interface for info objects describing an {@link OnnxValue}.
 *
 * <p>Will be sealed to {@link MapInfo}, {@link TensorInfo} and {@link SequenceInfo} when Java
 * supports sealed interfaces.
 */
public interface ValueInfo {}
