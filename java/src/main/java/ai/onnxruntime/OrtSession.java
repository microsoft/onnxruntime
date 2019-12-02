/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Wraps an ONNX model and allows inference calls.
 *
 * Produced by an {@link OrtEnvironment}.
 */
public class OrtSession implements AutoCloseable {

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    private final long nativeHandle;

    private final OrtAllocator allocator;

    private final long numInputs;

    private final Set<String> inputNames;

    private final long numOutputs;

    private final Set<String> outputNames;

    private boolean closed = false;

    OrtSession(OrtEnvironment env, String modelPath, OrtAllocator allocator, SessionOptions options) throws OrtException {
        nativeHandle = createSession(OnnxRuntime.ortApiHandle,env.nativeHandle,modelPath,options.nativeHandle);
        this.allocator = allocator;
        numInputs = getNumInputs(OnnxRuntime.ortApiHandle,nativeHandle);
        inputNames = new LinkedHashSet<>(Arrays.asList(getInputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
        numOutputs = getNumOutputs(OnnxRuntime.ortApiHandle,nativeHandle);
        outputNames = new LinkedHashSet<>(Arrays.asList(getOutputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
    }

    OrtSession(OrtEnvironment env, byte[] modelArray, OrtAllocator allocator, SessionOptions options) throws OrtException {
        nativeHandle = createSession(OnnxRuntime.ortApiHandle,env.nativeHandle,modelArray,options.nativeHandle);
        this.allocator = allocator;
        numInputs = getNumInputs(OnnxRuntime.ortApiHandle,nativeHandle);
        inputNames = new LinkedHashSet<>(Arrays.asList(getInputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
        numOutputs = getNumOutputs(OnnxRuntime.ortApiHandle,nativeHandle);
        outputNames = new LinkedHashSet<>(Arrays.asList(getOutputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
    }

    /**
     * Returns the number of inputs this model expects.
     * @return The number of inputs.
     */
    public long getNumInputs() {
        if (!closed) {
            return numInputs;
        } else {
            throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
        }
    }

    /**
     * Returns the number of outputs this model expects.
     * @return The number of outputs.
     */
    public long getNumOutputs() {
        if (!closed) {
            return numOutputs;
        } else {
            throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
        }
    }

    /**
     * Returns the input names. The underlying collection is sorted based on the input id number.
     * @return The input names.
     */
    public Set<String> getInputNames() {
        if (!closed) {
            return inputNames;
        } else {
            throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
        }
    }

    /**
     * Returns the output names. The underlying collection is sorted based on the output id number.
     * @return The output names.
     */
    public Set<String> getOutputNames() {
        if (!closed) {
            return outputNames;
        } else {
            throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
        }
    }

    /**
     * Returns the info objects for the inputs, including their names and types.
     * @return The input information.
     * @throws OrtException If there was an error in native code.
     */
    public Map<String,NodeInfo> getInputInfo() throws OrtException {
        if (!closed) {
            return wrapInMap(getInputInfo(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle));
        } else {
            throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
        }
    }

    /**
     * Returns the info objects for the outputs, including their names and types.
     * @return The output information.
     * @throws OrtException If there was an error in native code.
     */
    public Map<String,NodeInfo> getOutputInfo() throws OrtException {
        if (!closed) {
            return wrapInMap(getOutputInfo(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle));
        } else {
            throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
        }
    }

    /**
     * Scores an input feed dict, returning the map of all inferred outputs.
     * @param inputs The inputs to score.
     * @return The inferred outputs.
     * @throws OrtException If there was an error in native code, or if there are zero or too many inputs or outputs.
     */
    public Map<String,OnnxValue> run(Map<String,OnnxTensor> inputs) throws OrtException {
        return run(inputs,outputNames);
    }

    /**
     * Scores an input feed dict, returning the map of requested inferred outputs.
     * @param inputs The inputs to score.
     * @param requestedOutputs The requested outputs.
     * @return The inferred outputs.
     * @throws OrtException If there was an error in native code, or if there are zero or too many inputs or outputs.
     */
    public Map<String,OnnxValue> run(Map<String,OnnxTensor> inputs, Set<String> requestedOutputs) throws OrtException {
        if (!closed) {
            if (inputs.isEmpty() || (inputs.size() > numInputs)) {
                throw new OrtException("Unexpected number of inputs, expected [1," + numInputs + ") found " + inputs.size());
            }
            if (requestedOutputs.isEmpty() || (requestedOutputs.size() > numOutputs)) {
                throw new OrtException("Unexpected number of requestedOutputs, expected [1," + numOutputs + ") found " + requestedOutputs.size());
            }
            String[] inputNamesArray = new String[inputs.size()];
            long[] inputHandles = new long[inputs.size()];
            int i = 0;
            for (Map.Entry<String,OnnxTensor> t : inputs.entrySet()) {
                if (inputNames.contains(t.getKey())) {
                    inputNamesArray[i] = t.getKey();
                    inputHandles[i] = t.getValue().getNativeHandle();
                    i++;
                } else {
                    throw new IllegalArgumentException("Unknown input name " + t.getKey() + ", expected one of " + inputNames.toString());
                }
            }
            String[] outputNamesArray = new String[requestedOutputs.size()];
            i = 0;
            for (String s : requestedOutputs) {
                if (outputNames.contains(s)) {
                    outputNamesArray[i] = s;
                    i++;
                } else {
                    throw new IllegalArgumentException("Unknown output name " + s + ", expected one of " + outputNames.toString());
                }
            }
            OnnxValue[] outputValues = run(OnnxRuntime.ortApiHandle,nativeHandle, allocator.handle, inputNamesArray, inputHandles, numInputs, outputNamesArray, numOutputs);
            return zip(outputNamesArray,outputValues);
        } else {
            throw new IllegalStateException("Trying to score a closed OrtSession.");
        }
    }

    /**
     * Closes the session, releasing it's resources.
     * @throws OrtException If it failed to close.
     */
    @Override
    public void close() throws OrtException {
        if (!closed) {
            closeSession(OnnxRuntime.ortApiHandle,nativeHandle);
            closed = true;
        } else {
            throw new IllegalStateException("Trying to close an already closed OrtSession.");
        }
    }

    private static Map<String,NodeInfo> wrapInMap(NodeInfo[] infos) {
        Map<String,NodeInfo> output = new LinkedHashMap<>();

        for (int i = 0; i < infos.length; i++) {
            output.put(infos[i].getName(),infos[i]);
        }

        return output;
    }

    private static Map<String,OnnxValue> zip(String[] names, OnnxValue[] values) {
        Map<String,OnnxValue> output = new LinkedHashMap<>();

        if (names.length != values.length) {
            throw new IllegalArgumentException("Expected same number of names and values, found names.length = " + names.length + ", values.length = " + values.length);
        }

        for (int i = 0; i < names.length; i++) {
            output.put(names[i],values[i]);
        }

        return output;
    }

    private native long createSession(long apiHandle, long envHandle, String modelPath, long optsHandle) throws OrtException;
    private native long createSession(long apiHandle, long envHandle, byte[] modelArray, long optsHandle) throws OrtException;

    private native long getNumInputs(long apiHandle, long nativeHandle) throws OrtException;
    private native String[] getInputNames(long apiHandle, long nativeHandle, long allocatorHandle) throws OrtException;
    private native NodeInfo[] getInputInfo(long apiHandle, long nativeHandle, long allocatorHandle) throws OrtException;

    private native long getNumOutputs(long apiHandle, long nativeHandle) throws OrtException;
    private native String[] getOutputNames(long apiHandle, long nativeHandle, long allocatorHandle) throws OrtException;
    private native NodeInfo[] getOutputInfo(long apiHandle, long nativeHandle, long allocatorHandle) throws OrtException;

    private native OnnxValue[] run(long apiHandle, long nativeHandle, long allocatorHandle, String[] inputNamesArray, long[] inputs, long numInputs, String[] outputNamesArray, long numOutputs) throws OrtException;

    private native void closeSession(long apiHandle, long nativeHandle) throws OrtException;

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
            nativeHandle = createOptions(OnnxRuntime.ortApiHandle);
        }

        /**
         * Closes the session options, releasing any memory acquired.
         */
        @Override
        public void close() {
            closeOptions(OnnxRuntime.ortApiHandle,nativeHandle);
        }

        /**
         * Turns on sequential execution.
         * @param enable True if the model should execute sequentially.
         * @throws OrtException If there was an error in native code.
         */
        public void setSequentialExecution(boolean enable) throws OrtException {
            setSequentialExecution(OnnxRuntime.ortApiHandle,nativeHandle,enable);
        }

        /**
         * Sets the optimization level of this options object, overriding the old setting.
         * @param level The optimization level to use.
         * @throws OrtException If there was an error in native code.
         */
        public void setOptimizationLevel(OptLevel level) throws OrtException {
            setOptimizationLevel(OnnxRuntime.ortApiHandle,nativeHandle, level.getID());
        }

        /**
         * Sets the size of the CPU thread pool used for executing multiple request concurrently, if executing on a CPU.
         * @param numThreads The number of threads to use.
         * @throws OrtException If there was an error in native code.
         */
        public void setInterOpNumThreads(int numThreads) throws OrtException {
            setInterOpNumThreads(OnnxRuntime.ortApiHandle,nativeHandle,numThreads);
        }

        /**
         * Sets the size of the CPU thread pool used for executing a single graph, if executing on a CPU.
         * @param numThreads The number of threads to use.
         * @throws OrtException If there was an error in native code.
         */
        public void setIntraOpNumThreads(int numThreads) throws OrtException {
            setIntraOpNumThreads(OnnxRuntime.ortApiHandle,nativeHandle,numThreads);
        }

        /**
         * Add CUDA as an execution backend, using device 0.
         */
        public void addCUDA() throws OrtException {
            addCUDA(0);
        }

        /**
         * Add CUDA as an execution backend, using the specified CUDA device id.
         * @param deviceNum The CUDA device id.
         */
        public void addCUDA(int deviceNum) throws OrtException {
            addCUDA(OnnxRuntime.ortApiHandle,nativeHandle,deviceNum);
        }

        public void addCPU(boolean useArena) throws OrtException {
            addCPU(OnnxRuntime.ortApiHandle,nativeHandle,useArena?1:0);
        }

        public void addMkldnn(boolean useArena) throws OrtException {
            addMkldnn(OnnxRuntime.ortApiHandle,nativeHandle,useArena?1:0);
        }

        public void addNGraph(String ngBackendType) throws OrtException {
            addNGraph(OnnxRuntime.ortApiHandle,nativeHandle,ngBackendType);
        }

        public void addOpenVINO(String deviceId) throws OrtException {
            addOpenVINO(OnnxRuntime.ortApiHandle,nativeHandle,deviceId);
        }

        public void addTensorrt(int deviceNum) throws OrtException {
            addTensorrt(OnnxRuntime.ortApiHandle,nativeHandle,deviceNum);
        }

        public void addNnapi() throws OrtException {
            addNnapi(OnnxRuntime.ortApiHandle,nativeHandle);
        }

        public void addNuphar(boolean allowUnalignedBuffers, String settings) throws OrtException {
            addNuphar(OnnxRuntime.ortApiHandle,nativeHandle,allowUnalignedBuffers?1:0, settings);
        }

        //ORT_API(void, OrtEnableSequentialExecution, _In_ OrtSessionOptions* options);
        //ORT_API(void, OrtDisableSequentialExecution, _In_ OrtSessionOptions* options);
        private native void setSequentialExecution(long apiHandle, long nativeHandle, boolean enable) throws OrtException;

        private native void setOptimizationLevel(long apiHandle, long nativeHandle, int level) throws OrtException;

        private native void setInterOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws OrtException;
        private native void setIntraOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws OrtException;

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
         * If a backend is unavailable then it throws an OrtException
         */
        private native void addCPU(long apiHandle, long nativeHandle, int useArena) throws OrtException;
        private native void addCUDA(long apiHandle, long nativeHandle, int deviceNum) throws OrtException;
        private native void addMkldnn(long apiHandle, long nativeHandle, int useArena) throws OrtException;
        private native void addNGraph(long apiHandle, long nativeHandle, String ngBackendType) throws OrtException;
        private native void addOpenVINO(long apiHandle, long nativeHandle, String deviceId) throws OrtException;
        private native void addTensorrt(long apiHandle, long nativeHandle, int deviceNum) throws OrtException;
        private native void addNnapi(long apiHandle, long nativeHandle) throws OrtException;
        private native void addNuphar(long apiHandle, long nativeHandle, int allowUnalignedBuffers, String settings) throws OrtException;
    }
}
