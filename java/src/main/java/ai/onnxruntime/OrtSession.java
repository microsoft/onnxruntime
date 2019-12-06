/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Wraps an ONNX model and allows inference calls.
 * <p>
 * Allows the inspection of the model's input and output nodes.
 * Produced by an {@link OrtEnvironment}.
 * <p>
 * Most instance methods throw {@link IllegalStateException} if the
 * session is closed and the methods are called.
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

    /**
     * Create a session loading the model from disk.
     * @param env The environment.
     * @param modelPath The path to the model.
     * @param allocator The allocator to use.
     * @param options Session configuration options.
     * @throws OrtException If the file could not be read, or the model was corrupted etc.
     */
    OrtSession(OrtEnvironment env, String modelPath, OrtAllocator allocator, SessionOptions options) throws OrtException {
        nativeHandle = createSession(OnnxRuntime.ortApiHandle,env.nativeHandle,modelPath,options.nativeHandle);
        this.allocator = allocator;
        numInputs = getNumInputs(OnnxRuntime.ortApiHandle,nativeHandle);
        inputNames = new LinkedHashSet<>(Arrays.asList(getInputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
        numOutputs = getNumOutputs(OnnxRuntime.ortApiHandle,nativeHandle);
        outputNames = new LinkedHashSet<>(Arrays.asList(getOutputNames(OnnxRuntime.ortApiHandle,nativeHandle,allocator.handle)));
    }

    /**
     * Creates a session reading the model from the supplied byte array.
     * @param env The environment.
     * @param modelArray The model protobuf as a byte array.
     * @param allocator The allocator to use.
     * @param options Session configuration options.
     * @throws OrtException If the mode was corrupted or some other error occurred in native code.
     */
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
     * The underlying collection is sorted based on the input id number.
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
     * The underlying collection is sorted based on the output id number.
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
     * <p>
     * The outputs are sorted based on their id number.
     * @param inputs The inputs to score.
     * @return The inferred outputs.
     * @throws OrtException If there was an error in native code, the input names are invalid, or if there are zero or too many inputs.
     */
    public Result run(Map<String,OnnxTensor> inputs) throws OrtException {
        return run(inputs,outputNames);
    }

    /**
     * Scores an input feed dict, returning the map of requested inferred outputs.
     * <p>
     * The outputs are sorted based on the supplied set traveral order.
     * @param inputs The inputs to score.
     * @param requestedOutputs The requested outputs.
     * @return The inferred outputs.
     * @throws OrtException If there was an error in native code, the input or output names are invalid, or if there are zero or too many inputs or outputs.
     */
    public Result run(Map<String,OnnxTensor> inputs, Set<String> requestedOutputs) throws OrtException {
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
                    throw new OrtException("Unknown input name " + t.getKey() + ", expected one of " + inputNames.toString());
                }
            }
            String[] outputNamesArray = new String[requestedOutputs.size()];
            i = 0;
            for (String s : requestedOutputs) {
                if (outputNames.contains(s)) {
                    outputNamesArray[i] = s;
                    i++;
                } else {
                    throw new OrtException("Unknown output name " + s + ", expected one of " + outputNames.toString());
                }
            }
            OnnxValue[] outputValues = run(OnnxRuntime.ortApiHandle,nativeHandle, allocator.handle, inputNamesArray, inputHandles, numInputs, outputNamesArray, numOutputs);
            return new Result(outputNamesArray,outputValues);
        } else {
            throw new IllegalStateException("Trying to score a closed OrtSession.");
        }
    }

    @Override
    public String toString() {
        return "OrtSession(numInputs="+numInputs+",numOutputs="+numOutputs+")";
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

    /**
     * Converts a NodeInfo array into a map from node name to node info.
     * @param infos The NodeInfo array to convert.
     * @return A Map from String to NodeInfo.
     */
    private static Map<String,NodeInfo> wrapInMap(NodeInfo[] infos) {
        Map<String,NodeInfo> output = new LinkedHashMap<>();

        for (int i = 0; i < infos.length; i++) {
            output.put(infos[i].getName(),infos[i]);
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
     * Used to set the number of threads, optimisation level, computation backend and other options.
     * <p>
     * Modifying this after the session has been constructed will have no effect.
     */
    public static class SessionOptions implements AutoCloseable {

        /**
         * The optimisation level to use. Needs to be kept in sync with the GraphOptimizationLevel enum in the C API.
         */
        public enum OptLevel {
            NO_OPT(0), BASIC_OPT(1), EXTENDED_OPT(2), ALL_OPT(99);

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

        /**
         * The execution mode to use. Needs to be kept in sync with the ExecutionMode enum in the C API.
         */
        public enum ExecutionMode {
            SEQUENTIAL(0), PARALLEL(1);
            private final int id;

            ExecutionMode(int id) {
                this.id = id;
            }

            /**
             * Gets the int id used in native code for the execution mode.
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
         * Sets the execution mode of this options object, overriding the old setting.
         * @param mode The execution mode to use.
         * @throws OrtException If there was an error in native code.
         */
        public void setExecutionMode(ExecutionMode mode) throws OrtException {
            setExecutionMode(OnnxRuntime.ortApiHandle,nativeHandle,mode.getID());
        }

        /**
         * Sets the optimization level of this options object, overriding the old setting.
         * @param level The optimization level to use.
         * @throws OrtException If there was an error in native code.
         */
        public void setOptimizationLevel(OptLevel level) throws OrtException {
            setOptimizationLevel(OnnxRuntime.ortApiHandle,nativeHandle,level.getID());
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
         * Sets the output path for the optimized model.
         * @param outputPath The output path to write the model to.
         * @throws OrtException If there was an error in native code.
         */
        public void setOptimizedModelFilePath(String outputPath) throws OrtException {
            setOptimizationModelFilePath(OnnxRuntime.ortApiHandle,nativeHandle,outputPath);
        }

        /**
         * Add CUDA as an execution backend, using device 0.
         * @throws OrtException If there was an error in native code.
         */
        public void addCUDA() throws OrtException {
            addCUDA(0);
        }

        /**
         * Add CUDA as an execution backend, using the specified CUDA device id.
         * @param deviceNum The CUDA device id.
         * @throws OrtException If there was an error in native code.
         */
        public void addCUDA(int deviceNum) throws OrtException {
            addCUDA(OnnxRuntime.ortApiHandle,nativeHandle,deviceNum);
        }

        /**
         * Adds the CPU as an execution backend, using the arena allocator if desired.
         * <p>
         * By default this backend is used, but if other backends are requested,
         * it should be requested last.
         * @param useArena If true use the arena memory allocator.
         * @throws OrtException If there was an error in native code.
         */
        public void addCPU(boolean useArena) throws OrtException {
            addCPU(OnnxRuntime.ortApiHandle,nativeHandle,useArena?1:0);
        }

        /**
         * Adds Intel's Deep Neural Network Library as an execution backend.
         * @param useArena If true use the arena memory allocator.
         * @throws OrtException If there was an error in native code.
         */
        public void addDnnl(boolean useArena) throws OrtException {
            addDnnl(OnnxRuntime.ortApiHandle,nativeHandle,useArena?1:0);
        }

        /**
         * Adds NGraph as an execution backend.
         * <p>
         * See the documentation for the supported backend types.
         * @param ngBackendType The NGraph backend type.
         * @throws OrtException If there was an error in native code.
         */
        public void addNGraph(String ngBackendType) throws OrtException {
            addNGraph(OnnxRuntime.ortApiHandle,nativeHandle,ngBackendType);
        }

        /**
         * Adds OpenVINO as an execution backend.
         * @param deviceId The id of the OpenVINO execution device.
         * @throws OrtException If there was an error in native code.
         */
        public void addOpenVINO(String deviceId) throws OrtException {
            addOpenVINO(OnnxRuntime.ortApiHandle,nativeHandle,deviceId);
        }

        /**
         * Adds Nvidia's TensorRT as an execution backend.
         * @param deviceNum The id of the CUDA device.
         * @throws OrtException If there was an error in native code.
         */
        public void addTensorrt(int deviceNum) throws OrtException {
            addTensorrt(OnnxRuntime.ortApiHandle,nativeHandle,deviceNum);
        }

        /**
         * Adds Android's NNAPI as an execution backend.
         * @throws OrtException If there was an error in native code.
         */
        public void addNnapi() throws OrtException {
            addNnapi(OnnxRuntime.ortApiHandle,nativeHandle);
        }

        /**
         * Adds Nuphar as an execution backend.
         * @param allowUnalignedBuffers Allow unaligned memory buffers.
         * @param settings See the documentation for valid settings strings.
         * @throws OrtException If there was an error in native code.
         */
        public void addNuphar(boolean allowUnalignedBuffers, String settings) throws OrtException {
            addNuphar(OnnxRuntime.ortApiHandle,nativeHandle,allowUnalignedBuffers?1:0, settings);
        }

        private native void setExecutionMode(long apiHandle, long nativeHandle, int mode) throws OrtException;

        private native void setOptimizationLevel(long apiHandle, long nativeHandle, int level) throws OrtException;

        private native void setInterOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws OrtException;
        private native void setIntraOpNumThreads(long apiHandle, long nativeHandle, int numThreads) throws OrtException;

        private native void setOptimizationModelFilePath(long apiHandle, long nativeHandle, String modelPath) throws OrtException;

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
        private native void addDnnl(long apiHandle, long nativeHandle, int useArena) throws OrtException;
        private native void addNGraph(long apiHandle, long nativeHandle, String ngBackendType) throws OrtException;
        private native void addOpenVINO(long apiHandle, long nativeHandle, String deviceId) throws OrtException;
        private native void addTensorrt(long apiHandle, long nativeHandle, int deviceNum) throws OrtException;
        private native void addNnapi(long apiHandle, long nativeHandle) throws OrtException;
        private native void addNuphar(long apiHandle, long nativeHandle, int allowUnalignedBuffers, String settings) throws OrtException;
    }

    /**
     * An {@link AutoCloseable} wrapper around a {@link Map} containing {@link OnnxValue}s.
     * <p>
     * When this is closed it closes all the {@link OnnxValue}s inside it. If you maintain a reference to a
     * value after this object has been closed it will throw an {@link IllegalStateException} upon access.
     */
    public static class Result implements AutoCloseable, Iterable<Map.Entry<String,OnnxValue>> {

        private static final Logger logger = Logger.getLogger(Result.class.getName());

        private final Map<String,OnnxValue> map;

        private final List<OnnxValue> list;

        private boolean closed;

        /**
         * Creates a Result from the names and values produced by {@link OrtSession#run(Map)}.
         * @param names The output names.
         * @param values The output values.
         */
        Result(String[] names, OnnxValue[] values) {
            map = new LinkedHashMap<>();
            list = new ArrayList<>();

            if (names.length != values.length) {
                throw new IllegalArgumentException("Expected same number of names and values, found names.length = " + names.length + ", values.length = " + values.length);
            }

            for (int i = 0; i < names.length; i++) {
                map.put(names[i],values[i]);
                list.add(values[i]);
            }
            this.closed = false;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                for (OnnxValue t : map.values()) {
                    t.close();
                }
            } else {
                logger.warning("Closing an already closed Result");
            }
        }

        @Override
        public Iterator<Map.Entry<String, OnnxValue>> iterator() {
            if (!closed) {
                return map.entrySet().iterator();
            } else {
                throw new IllegalStateException("Result is closed");
            }
        }

        /**
         * Gets the value from the container at the specified index.
         *
         * Throws {@link IllegalStateException} if the container has been closed, and {@link IndexOutOfBoundsException} if the index is invalid.
         *
         * @param index The index to lookup.
         * @return The value at the index.
         */
        public OnnxValue get(int index) {
            if (!closed) {
                return list.get(index);
            } else {
                throw new IllegalStateException("Result is closed");
            }
        }

        /**
         * Returns the number of outputs in this Result.
         * @return The number of outputs.
         */
        public int size() {
            return map.size();
        }

        /**
         * Gets the value from the container assuming it's not been closed.
         *
         * Throws {@link IllegalStateException} if the container has been closed.
         * @param key The key to lookup.
         * @return Optional.of the value if it exists.
         */
        public Optional<OnnxValue> get(String key) {
            if (!closed) {
                OnnxValue value = map.get(key);
                if (value != null) {
                    return Optional.of(value);
                } else {
                    return Optional.empty();
                }
            } else {
                throw new IllegalStateException("Result is closed");
            }
        }
    }
}
