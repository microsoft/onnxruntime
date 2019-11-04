/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/**
 * Top interface for input and output values from ONNX models.
 * Currently implemented by {@link ONNXTensor}, {@link ONNXSequence} and {@link ONNXMap}. Will be sealed to
 * these types one day.
 *
 * Does not support sparse tensors.
 */
public interface ONNXValue extends AutoCloseable {

    /**
     * The type of the ONNXValue, mirroring the id in the C API.
     */
    public enum ONNXValueType {
        ONNX_TYPE_UNKNOWN(0),
        ONNX_TYPE_TENSOR(1),
        ONNX_TYPE_SEQUENCE(2),
        ONNX_TYPE_MAP(3),
        ONNX_TYPE_OPAQUE(4),
        ONNX_TYPE_SPARSETENSOR(5);

        /**
         * The id number of this type in the C API.
         */
        public final int value;
        ONNXValueType(int value) {
            this.value = value;
        }
    }

    /**
     * Gets the type of this ONNXValue.
     * @return The value type.
     */
    public ONNXValueType getType();

    /**
     * Returns the value as a POJO copying it out of the native heap.
     * This operation can be quite slow for high dimensional tensors.
     *
     * Overridden by the subclasses with a more specific type.
     * @return The value.
     * @throws ONNXException If an error occurred reading the value.
     */
    public Object getValue() throws ONNXException;

    /**
     * Gets the info object associated with this ONNXValue.
     * @return The info.
     */
    public ValueInfo getInfo();

    /**
     * Closes the ONNXValue, freeing it's native memory.
     */
    @Override
    public void close();

    /**
     * Calls close on each element of the iterable.
     * @param itr An iterable of closeable ONNXValues.
     */
    public static void close(Iterable<? extends ONNXValue> itr) {
        for (ONNXValue t : itr) {
            t.close();
        }
    }

}
