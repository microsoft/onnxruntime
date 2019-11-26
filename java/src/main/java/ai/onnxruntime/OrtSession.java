/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Wraps an ONNX model and allows inference calls.
 *
 * Produced by an {@link ONNXEnvironment}.
 */
public class ONNXSession implements AutoCloseable {

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    private final long nativeHandle;

    private final ONNXAllocator allocator;

    private final long inputNamesHandle;

    private final long numInputs;

    private final long outputNamesHandle;

    private final long numOutputs;

    private boolean closed = false;

    ONNXSession(ONNXEnvironment env, String modelPath, ONNXAllocator allocator, SessionOptions options) throws ONNXException {
        nativeHandle = createSession(ONNX.ortApiHandle,env.nativeHandle,modelPath,options.nativeHandle);
        this.allocator = allocator;
        inputNamesHandle = getInputNames(ONNX.ortApiHandle,nativeHandle,allocator.handle);
        numInputs = getNumInputs(ONNX.ortApiHandle,nativeHandle);
        outputNamesHandle = getOutputNames(ONNX.ortApiHandle,nativeHandle,allocator.handle);
        numOutputs = getNumOutputs(ONNX.ortApiHandle,nativeHandle);
    }

    ONNXSession(ONNXEnvironment env, byte[] modelArray, ONNXAllocator allocator, SessionOptions options) throws ONNXException {
        nativeHandle = createSession(ONNX.ortApiHandle,env.nativeHandle,modelArray,options.nativeHandle);
        this.allocator = allocator;
        inputNamesHandle = getInputNames(ONNX.ortApiHandle,nativeHandle,allocator.handle);
        numInputs = getNumInputs(ONNX.ortApiHandle,nativeHandle);
        outputNamesHandle = getOutputNames(ONNX.ortApiHandle,nativeHandle,allocator.handle);
        numOutputs = getNumOutputs(ONNX.ortApiHandle,nativeHandle);
    }

    /**
     * Returns the number of inputs this model expects.
     * @return The number of inputs.
     * @throws ONNXException If there was an error in native code.
     */
    public long getNumInputs() throws ONNXException {
        if (!closed) {
            return getNumInputs(ONNX.ortApiHandle,nativeHandle);
        } else {
            throw new IllegalStateException("Asking for inputs from a closed ONNXSession.");
        }
    }

    /**
     * Returns the number of outputs this model expects.
     * @return The number of outputs.
     * @throws ONNXException If there was an error in native code.
     */
    public long getNumOutputs() throws ONNXException {
        if (!closed) {
            return getNumOutputs(ONNX.ortApiHandle,nativeHandle);
        } else {
            throw new IllegalStateException("Asking for outputs from a closed ONNXSession.");
        }
    }

    /**
     * Returns the info objects for the inputs, including their names and types.
     * @return The input information.
     * @throws ONNXException If there was an error in native code.
     */
    public List<NodeInfo> getInputInfo() throws ONNXException {
        if (!closed) {
            return Arrays.asList(getInputInfo(ONNX.ortApiHandle,nativeHandle, inputNamesHandle));
        } else {
            throw new IllegalStateException("Asking for inputs from a closed ONNXSession.");
        }
    }

    /**
     * Returns the info objects for the outputs, including their names and types.
     * @return The output information.
     * @throws ONNXException If there was an error in native code.
     */
    public List<NodeInfo> getOutputInfo() throws ONNXException {
        if (!closed) {
            return Arrays.asList(getOutputInfo(ONNX.ortApiHandle,nativeHandle, outputNamesHandle));
        } else {
            throw new IllegalStateException("Asking for outputs from a closed ONNXSession.");
        }
    }

    /**
     * Scores an input list, returning the list of inferred outputs.
     * @param inputs The inputs to score.
     * @return The inferred outputs.
     * @throws ONNXException If there was an error in native code, or if there are zero or too many inputs.
     */
    public List<ONNXValue> run(List<ONNXTensor> inputs) throws ONNXException {
        if (!closed) {
            if (inputs.isEmpty() || (inputs.size() > numInputs)) {
                throw new ONNXException("Unexpected number of inputs, expected [1," + numInputs + ") found " + inputs.size());
            }
            long[] inputHandles = new long[inputs.size()];
            int i = 0;
            for (ONNXTensor t : inputs) {
                inputHandles[i] = t.getNativeHandle();
                i++;
            }
            return Arrays.asList(run(ONNX.ortApiHandle,nativeHandle, allocator.handle, inputNamesHandle, numInputs, inputHandles, outputNamesHandle, numOutputs));
        } else {
            throw new IllegalStateException("Trying to score a closed ONNXSession.");
        }
    }

    /**
     * Closes the session, releasing it's resources.
     * @throws ONNXException If it failed to close.
     */
    @Override
    public void close() throws ONNXException {
        if (!closed) {
            releaseNamesHandle(ONNX.ortApiHandle,allocator.handle, inputNamesHandle, getNumInputs(ONNX.ortApiHandle,nativeHandle));
            releaseNamesHandle(ONNX.ortApiHandle,allocator.handle, outputNamesHandle, getNumOutputs(ONNX.ortApiHandle,nativeHandle));
            closeSession(ONNX.ortApiHandle,nativeHandle);
            closed = true;
        } else {
            throw new IllegalStateException("Trying to close an already closed ONNXSession.");
        }
    }

    private native long createSession(long apiHandle, long envHandle, String modelPath, long optsHandle) throws ONNXException;
    private native long createSession(long apiHandle, long envHandle, byte[] modelArray, long optsHandle) throws ONNXException;

    private native long getNumInputs(long apiHandle, long nativeHandle) throws ONNXException;
    private native long getInputNames(long apiHandle, long nativeHandle, long allocatorHandle) throws ONNXException;
    private native NodeInfo[] getInputInfo(long apiHandle, long nativeHandle, long inputNamesHandle) throws ONNXException;

    private native long getNumOutputs(long apiHandle, long nativeHandle) throws ONNXException;
    private native long getOutputNames(long apiHandle, long nativeHandle, long allocatorHandle) throws ONNXException;
    private native NodeInfo[] getOutputInfo(long apiHandle, long nativeHandle, long outputNamesHandle) throws ONNXException;

    private native ONNXValue[] run(long apiHandle, long nativeHandle, long allocatorHandle, long inputNamesHandle, long numInputs, long[] inputs, long outputNamesHandle, long numOutputs) throws ONNXException;

    private native void closeSession(long apiHandle, long nativeHandle) throws ONNXException;
    private native void releaseNamesHandle(long apiHandle, long allocatorHandle, long namesHandle, long numNames) throws ONNXException;

    /**
     * Represents the options used to construct this session.
     * <p>
     * Used to set the number of threads, optimisation level, accelerator backend and other options.
     * <p>
     * Modifying this after the session has been constructed will have no effect.
     */
    public static class SessionOptions implements AutoCloseable {

        /**
         * The optimisation level to use.
         */
        public enum OptLevel { NO_OPT(0), BASIC_OPT(1), EXTENDED_OPT(2), ALL_OPT(99);

            private final int id;

            OptLevel(int id) {
                this.id = id;
            }

            /**
             * Gets the int id used in native code for this optimisation level.
             * @return The int id.
             */
            public int getID() {
                return id;
            }
        }

        private final long nativeHandle;

        /**
         * Create an empty session options.
         */
        public SessionOptions() {
            nativeHandle = createOptions(ONNX.ortApiHandle);
        }

        /**
         * Closes the session options, releasing any memory acquired.
         */
        @Override
        public void close() {
            closeOptions(ONNX.ortApiHandle,nativeHandle);
        }

        /**
         * Turns on sequential execution.
         * @param enable True if the model should execute sequentially.
         * @throws ONNXException If there was an error in native code.
         */
        public void setSequentialExecution(boolean enable) throws ONNXException {
            setSequentialExecution(ONNX.ortApiHandle,nativeHandle,enable);
        }

        /**
         * Sets the optimization level of this options object, overriding the old setting.
         * @param level The optimization level to use.
         * @throws ONNXException If there was an error in native code.
         */
        public void setOptimizationLevel(OptLevel level) throws ONNXException {
            setOptimizationLevel(ONNX.ortApiHandle,nativeHandle, level.getID());
        }

        /**
         * Sets the size of the CPU thread pool used for executing multiple request concurrently, if executing on a CPU.
         * @param numThreads The number of threads to use.
         * @throws ONNXException If there was an error in native code.
         */
        public void setInterOpNumThreads(int numThreads) throws ONNXException {
            setInterOpNumThreads(ONNX.ortApiHandle,nativeHandle,numThreads);
        }

        /**
         * Sets the size of the CPU thread pool used for executing a single graph, if executing on a CPU.
         * @param numThreads The number of threads to use.
         * @throws ONNXException If there was an error in native code.
         */
        public void setIntraOpNumThreads(int numThreads) throws ONNXException {
            setIntraOpNumThreads(ONNX.ortApiHandle,nativeHandle,numThreads);
        }

        /**
         * Add CUDA as an execution backend, using device 0.
         */
        public void addCUDA() throws ONNXException {
            addCUDA(0);
        }

        /**
         * Add CUDA as an execution backend, using the specified CUDA device id.
         * @param deviceNum The CUDA device id.
         */
        public void addCUDA(int deviceNum) throws ONNXException {
            addCUDA(ONNX.ortApiHandle,nativeHandle,deviceNum);
        }

        public void addCPU(boolean useArena) throws ONNXException {
            addCPU(ONNX.ortApiHandle,nativeHandle,useArena?1:0);
        }

        public void addMkldnn(boolean useArena) throws ONNXException {
            addMkldnn(ONNX.ortApiHandle,nativeHandle,useArena?1:0);
        }

        public void addNGraph(String ngBackendType) throws ONNXException {
            addNGraph(ONNX.ortApiHandle,nativeHandle,ngBackendType);
        }

        public void addOpenVINO(String deviceId) throws ONNXException {
            addOpenVINO(ONNX.ortApiHandle,nativeHandle,deviceId);
        }

        public void addTensorrt(int deviceNum) throws ONNXException {
            addTensorrt(ONNX.ortApiHandle,nativeHandle,deviceNum);
        }

        public void addNnapi() throws ONNXException {
            addNnapi(ONNX.ortApiHandle,nativeHandle);
        }

        public void addNuphar(boolean allowUnalignedBuffers, String settings) throws ONNXException {
            addNuphar(ONNX.ortApiHandle,nativeHandle,allowUnalignedBuffers?1:0, settings);
        }

        //ORT_API(void, OrtEnableSequentialExecution, _In_ OrtSessionOptions* options);
        //ORT_API(void, OrtDisableSequentialExecution, _In_ OrtSessionOptions* options);
        private native void setSequentialExecution(long apiHandle, long nativeHandle, boolean enable) throws ONNXException;

        private native void setOptimizationLevel(long apiHandle, long nativeHandle, int level) throws ONNXException;

        private native void setInterOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws ONNXException;
        private native void setIntraOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws ONNXException;

        private native long createOptions(long apiHandle);

        private native void closeOptions(long apiHandle, long nativeHandle);

        /*
         * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
         * functions to enable them in the session:
         *   OrtSessionOptionsAppendExecutionProvider_CPU
         *   OrtSessionOptionsAppendExecutionProvider_CUDA
         *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
         * The order they care called indicates the preference order as well. In other words call this method
         * on your most preferred execution provider first followed by the less preferred ones.
         * If none are called Ort will use its internal CPU execution provider.
         *
         * If a backend is unavailable then it throws an ONNXException
         */
        private native void addCPU(long apiHandle, long nativeHandle, int useArena) throws ONNXException;
        private native void addCUDA(long apiHandle, long nativeHandle, int deviceNum) throws ONNXException;
        private native void addMkldnn(long apiHandle, long nativeHandle, int useArena) throws ONNXException;
        private native void addNGraph(long apiHandle, long nativeHandle, String ngBackendType) throws ONNXException;
        private native void addOpenVINO(long apiHandle, long nativeHandle, String deviceId) throws ONNXException;
        private native void addTensorrt(long apiHandle, long nativeHandle, int deviceNum) throws ONNXException;
        private native void addNnapi(long apiHandle, long nativeHandle) throws ONNXException;
        private native void addNuphar(long apiHandle, long nativeHandle, int allowUnalignedBuffers, String settings) throws ONNXException;
    }
}
