/*
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Wraps an ONNX model and allows inference calls.
 *
 * <p>Allows the inspection of the model's input and output nodes. Produced by an {@link
 * OrtEnvironment}.
 *
 * <p>Most instance methods throw {@link IllegalStateException} if the session is closed and the
 * methods are called.
 *
 * <p>This class is thread-safe. It is possible to manage sessions in one thread and conduct model
 * evaluations in other threads.
 */
public class OrtSession extends NativeObject {

  private final OrtEnvironment environment;

  private final OrtAllocator allocator;

  private final Map<String, NodeInfo> inputInfo;

  /**
   * a preconstructed array so conversion from outputInfo.keySet() to array does not need to be
   * repeated.
   */
  private final String[] outputNamesArray;

  private final Map<String, NodeInfo> outputInfo;

  private final OnnxModelMetadata metadata;

  private final Set<RunOptions> activeRunOptions;

  /**
   * Create a session loading the model from disk.
   *
   * @param env The environment.
   * @param modelPath The path to the model.
   * @param allocator The allocator to use.
   * @param options Session configuration options.
   * @throws OrtException If the file could not be read, or the model was corrupted etc.
   * @return a new session
   */
  static final OrtSession fromPath(
      OrtEnvironment env, String modelPath, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage optionsReference = options.use()) {
      long sessionHandle =
          createSession(
              OnnxRuntime.ortApiHandle,
              environmentReference.handle(),
              modelPath,
              optionsReference.handle());
      return new OrtSession(sessionHandle, env, allocator);
    }
  }

  /**
   * Creates a session reading the model from the supplied byte array.
   *
   * @param env The environment.
   * @param modelArray The model protobuf as a byte array.
   * @param allocator The allocator to use.
   * @param options Session configuration options.
   * @throws OrtException If the model was corrupted or some other error occurred in native code.
   * @return a new session
   */
  static final OrtSession fromBytes(
      OrtEnvironment env, byte[] modelArray, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage optionsReference = options.use()) {
      long sessionHandle =
          createSession(
              OnnxRuntime.ortApiHandle,
              environmentReference.handle(),
              modelArray,
              optionsReference.handle());
      return new OrtSession(sessionHandle, env, allocator);
    }
  }

  /**
   * Private constructor to build the Java object wrapped around a native session.
   *
   * @param sessionHandle The pointer to the native session.
   * @param environment The environment to check to be open.
   * @param allocator The allocator to use.
   * @throws OrtException If the model's inputs, outputs or metadata could not be read.
   */
  private OrtSession(long sessionHandle, OrtEnvironment environment, OrtAllocator allocator)
      throws OrtException {
    super(sessionHandle);
    this.environment = environment;
    this.allocator = allocator;
    this.activeRunOptions = ConcurrentHashMap.newKeySet();
    try (NativeUsage allocatorReference = allocator.use()) {
      long allocatorHandle = allocatorReference.handle();
      metadata = constructMetadata(OnnxRuntime.ortApiHandle, sessionHandle, allocatorHandle);
      inputInfo = wrapInMap(getInputInfo(OnnxRuntime.ortApiHandle, sessionHandle, allocatorHandle));
      NodeInfo[] outputInfoArray =
          getOutputInfo(OnnxRuntime.ortApiHandle, sessionHandle, allocatorHandle);
      outputInfo = wrapInMap(outputInfoArray);
      int numOutputs = outputInfoArray.length;
      outputNamesArray = new String[numOutputs];
      for (int i = 0; i < numOutputs; i++) {
        outputNamesArray[i] = outputInfoArray[i].getName();
      }
    }
  }

  /**
   * Returns the number of inputs this model expects.
   *
   * @return The number of inputs.
   */
  public long getNumInputs() {
    return inputInfo.size();
  }

  /**
   * Returns the number of outputs this model expects.
   *
   * @return The number of outputs.
   */
  public long getNumOutputs() {
    return outputInfo.size();
  }

  /**
   * Returns the input names. The underlying collection is sorted based on the input id number.
   *
   * @return The input names.
   */
  public Set<String> getInputNames() {
    return inputInfo.keySet();
  }

  /**
   * Returns the output names. The underlying collection is sorted based on the output id number.
   *
   * @return The output names.
   */
  public Set<String> getOutputNames() {
    return outputInfo.keySet();
  }

  /**
   * Returns the info objects for the inputs, including their names and types. The underlying
   * collection is sorted based on the input id number.
   *
   * @return The input information.
   * @throws OrtException If there was an error in native code.
   */
  public Map<String, NodeInfo> getInputInfo() throws OrtException {
    return inputInfo;
  }

  /**
   * Returns the info objects for the outputs, including their names and types. The underlying
   * collection is sorted based on the output id number.
   *
   * @return The output information.
   * @throws OrtException If there was an error in native code.
   */
  public Map<String, NodeInfo> getOutputInfo() throws OrtException {
    return outputInfo;
  }

  /**
   * Scores an input feed dict, returning the map of all inferred outputs.
   *
   * <p>The outputs are sorted based on their id number.
   *
   * @param inputs The inputs to score.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input names are invalid, or if
   *     there are zero or too many inputs.
   */
  public Result run(Map<String, OnnxTensor> inputs) throws OrtException {
    try (RunOptions runOptions = new RunOptions()) {
      return run(inputs, outputNamesArray, runOptions);
    }
  }

  /**
   * Scores an input feed dict, returning the map of all inferred outputs.
   *
   * <p>The outputs are sorted based on their id number.
   *
   * @param inputs The inputs to score.
   * @param runOptions The RunOptions to control this run.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input names are invalid, or if
   *     there are zero or too many inputs.
   */
  public Result run(Map<String, OnnxTensor> inputs, RunOptions runOptions) throws OrtException {
    return run(inputs, outputNamesArray, runOptions);
  }

  /**
   * Scores an input feed dict, returning the map of requested inferred outputs.
   *
   * <p>The outputs are sorted based on the supplied set traversal order.
   *
   * @param inputs The inputs to score.
   * @param requestedOutputs The requested outputs.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input or output names are
   *     invalid, or if there are zero or too many inputs or outputs.
   */
  public Result run(Map<String, OnnxTensor> inputs, Set<String> requestedOutputs)
      throws OrtException {
    try (RunOptions runOptions = new RunOptions()) {
      return run(inputs, convertRequestedOutputs(requestedOutputs), runOptions);
    }
  }

  /**
   * Scores an input feed dict, returning the map of requested inferred outputs.
   *
   * <p>The outputs are sorted based on the supplied set traveral order.
   *
   * @param inputs The inputs to score.
   * @param requestedOutputs The requested outputs.
   * @param runOptions The RunOptions to control this run.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input or output names are
   *     invalid, or if there are zero or too many inputs or outputs.
   */
  public Result run(
      Map<String, OnnxTensor> inputs, Set<String> requestedOutputs, RunOptions runOptions)
      throws OrtException {
    return run(inputs, convertRequestedOutputs(requestedOutputs), runOptions);
  }

  /**
   * Utility method to convert.
   *
   * @param requestedOutputs Set representation
   * @return String[] representation
   * @throws OrtException If there were too many or unrecognized outputs requested.
   */
  private String[] convertRequestedOutputs(Set<String> requestedOutputs) throws OrtException {
    int requestedOutputsLength = requestedOutputs.size();
    int numOutputs = outputInfo.size();
    if (requestedOutputsLength == 0 || (requestedOutputsLength > numOutputs)) {
      throw new OrtException(
          "Unexpected number of requestedOutputs, expected [1,"
              + numOutputs
              + ") found "
              + requestedOutputsLength);
    }
    String[] requestedOutputsArray = new String[requestedOutputsLength];
    int i = 0;
    for (String s : requestedOutputs) {
      if (outputInfo.containsKey(s)) {
        requestedOutputsArray[i++] = s;
      } else {
        throw new OrtException(
            "Unknown output name " + s + ", expected one of " + outputInfo.keySet());
      }
    }
    return requestedOutputsArray;
  }

  private Result run(
      Map<String, OnnxTensor> inputs, String[] requestedOutputsArray, RunOptions runOptions)
      throws OrtException {
    if (runOptions == null) {
      throw new OrtException("RunOptions is required");
    }
    int numRequestedInputs = inputs.size();
    int numInputs = inputInfo.size();
    if (inputs.isEmpty() || (numRequestedInputs > numInputs)) {
      throw new OrtException(
          "Unexpected number of inputs, expected [1,"
              + numInputs
              + ") found "
              + numRequestedInputs);
    }
    try (NativeUsage environmentReference = environment.use();
        NativeUsage allocatorReference = allocator.use();
        NativeUsage sessionReference = use();
        NativeUsage runOptionsReference = runOptions.use();
        NativeInputReferences inputsReferences = new NativeInputReferences(numRequestedInputs)) {
      String[] requestedInputsArray = new String[numRequestedInputs];
      long[] inputHandles = new long[numRequestedInputs];
      int i = 0;
      for (Map.Entry<String, OnnxTensor> t : inputs.entrySet()) {
        String key = t.getKey();
        if (!inputInfo.containsKey(key)) {
          throw new OrtException(
              "Unknown input name " + key + ", expected one of " + inputInfo.keySet());
        }
        requestedInputsArray[i] = key;
        inputHandles[i] = inputsReferences.handle(t.getValue());
        /*
         * This handle will be valid from here until NativeInputReferences.close() which is automatically called
         * in the try-with-resources finally.
         */
        i++;
      }
      OnnxValue[] outputValues;
      /*
       * Collect the RunOptions for terminate method to use, since there is no other way to interrupt JNI
       * execution beyond calling RunOptions.setTerminate(true).
       */
      activeRunOptions.add(runOptions);
      try {
        outputValues =
            run(
                OnnxRuntime.ortApiHandle,
                sessionReference.handle(),
                allocatorReference.handle(),
                allocator,
                requestedInputsArray,
                inputHandles,
                inputHandles.length,
                requestedOutputsArray,
                requestedOutputsArray.length,
                runOptionsReference.handle());
      } finally {
        // the RunOptions does not need to be terminated since the run is complete
        activeRunOptions.remove(runOptions);
      }
      return new Result(requestedOutputsArray, outputValues);
    }
  }

  /**
   * This keeps track of the references for the inputs as the inputs are validated. Nuance: we could
   * have 3 inputs. The last input could be invalid, but we need to ensure the first 2 references
   * are closed.
   */
  private static final class NativeInputReferences implements AutoCloseable {

    private final List<NativeUsage> references;

    public NativeInputReferences(int size) throws OrtException {
      this.references = new ArrayList<>(size);
    }

    public long handle(OnnxTensor tensor) {
      // this could throw if the tensor is closed
      NativeUsage reference = tensor.use();
      references.add(reference);
      return reference.handle();
    }

    @Override
    public void close() {
      for (NativeUsage reference : references) {
        reference.close();
      }
    }
  }

  /**
   * Gets the metadata for the currently loaded model.
   *
   * @return The metadata.
   * @throws OrtException If the native call failed.
   */
  public OnnxModelMetadata getMetadata() throws OrtException {
    return metadata;
  }

  /**
   * Ends the profiling session and returns the output of the profiler.
   *
   * <p>Profiling should be enabled in the {@link SessionOptions} used to construct this {@code
   * Session}.
   *
   * @return The profiling output.
   * @throws OrtException If the native call failed.
   */
  public String endProfiling() throws OrtException {
    try (NativeUsage sessionReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      return endProfiling(
          OnnxRuntime.ortApiHandle, sessionReference.handle(), allocatorReference.handle());
    }
  }

  @Override
  public String toString() {
    return super.toString()
        + "(numInputs="
        + getNumInputs()
        + ",numOutputs="
        + getNumOutputs()
        + ")";
  }

  /**
   * Terminate running model evaluations.
   *
   * <p>This is typically used in multi-threaded environments, where session management and model
   * evaluation are occurring in different threads. This method can be used prior to calling {@link
   * #close()} to prevent the normal behavior of blocking until pending model evaluations are
   * completed.
   *
   * @throws OrtException If the terminate setting failed.
   */
  public synchronized void terminate() throws OrtException {
    for (RunOptions runOptions : activeRunOptions) {
      if (!runOptions.isClosed()) {
        runOptions.setTerminate(true);
      }
    }
  }

  @Override
  protected void doClose(long handle) {
    closeSession(OnnxRuntime.ortApiHandle, handle);
  }

  /**
   * Converts a {@link NodeInfo} array into a map from node name to node info.
   *
   * @param infos The {@link NodeInfo} array to convert.
   * @return An ordered, unmodifiable Map from String to {@link NodeInfo}.
   */
  private static Map<String, NodeInfo> wrapInMap(NodeInfo[] infos) {
    Map<String, NodeInfo> output = new LinkedHashMap<>(infos.length);

    for (NodeInfo info : infos) {
      output.put(info.getName(), info);
    }

    return Collections.unmodifiableMap(output);
  }

  private static native long createSession(
      long apiHandle, long envHandle, String modelPath, long optsHandle) throws OrtException;

  private static native long createSession(
      long apiHandle, long envHandle, byte[] modelArray, long optsHandle) throws OrtException;

  private native NodeInfo[] getInputInfo(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native NodeInfo[] getOutputInfo(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  /**
   * The native run call. runOptionsHandle can be zero (i.e. the null pointer), but all other
   * handles must be valid pointers.
   *
   * @param apiHandle The pointer to the api.
   * @param nativeHandle The pointer to the session.
   * @param allocatorHandle The pointer to the allocator.
   * @param allocatorObject The allocator to use within the OnnxValue implementations.
   * @param inputNamesArray The input names.
   * @param inputs The input tensors.
   * @param numInputs The number of inputs.
   * @param outputNamesArray The requested output names.
   * @param numOutputs The number of requested outputs.
   * @param runOptionsHandle The (possibly null) pointer to the run options.
   * @return The OnnxValues produced by this run.
   * @throws OrtException If the native call failed in some way.
   */
  private native OnnxValue[] run(
      long apiHandle,
      long nativeHandle,
      long allocatorHandle,
      OrtAllocator allocatorObject,
      String[] inputNamesArray,
      long[] inputs,
      long numInputs,
      String[] outputNamesArray,
      long numOutputs,
      long runOptionsHandle)
      throws OrtException;

  private native String endProfiling(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void closeSession(long apiHandle, long nativeHandle);

  /**
   * Builds the {@link OnnxModelMetadata} for this session.
   *
   * @param ortApiHandle The api pointer.
   * @param nativeHandle The native session pointer.
   * @param allocatorHandle The OrtAllocator pointer.
   * @return The metadata.
   * @throws OrtException If the native runtime failed to access or allocate the metadata.
   */
  private native OnnxModelMetadata constructMetadata(
      long ortApiHandle, long nativeHandle, long allocatorHandle) throws OrtException;

  /**
   * Represents the options used to construct this session.
   *
   * <p>Used to set the number of threads, optimisation level, computation backend and other
   * options.
   *
   * <p>Modifying this after the session has been constructed will have no effect.
   *
   * <p>The SessionOptions object must not be closed until all sessions which use it are closed, as
   * otherwise it could release resources that are in use.
   */
  public static class SessionOptions extends NativeObject {

    /**
     * The optimisation level to use. Needs to be kept in sync with the GraphOptimizationLevel enum
     * in the C API.
     */
    public enum OptLevel {
      NO_OPT(0),
      BASIC_OPT(1),
      EXTENDED_OPT(2),
      ALL_OPT(99);

      private final int id;

      OptLevel(int id) {
        this.id = id;
      }

      /**
       * Gets the int id used in native code for this optimisation level.
       *
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
      SEQUENTIAL(0),
      PARALLEL(1);

      private final int id;

      ExecutionMode(int id) {
        this.id = id;
      }

      /**
       * Gets the int id used in native code for the execution mode.
       *
       * @return The int id.
       */
      public int getID() {
        return id;
      }
    }

    private final List<Long> customLibraryHandles;

    /** Create an empty session options. */
    public SessionOptions() {
      super(createOptions(OnnxRuntime.ortApiHandle));
      customLibraryHandles = new ArrayList<>();
    }

    /** Closes the session options, releasing any memory acquired. */
    @Override
    protected void doClose(long handle) {
      if (customLibraryHandles.size() > 0) {
        long[] longArray = new long[customLibraryHandles.size()];
        for (int i = 0; i < customLibraryHandles.size(); i++) {
          longArray[i] = customLibraryHandles.get(i);
        }
        closeCustomLibraries(longArray);
      }
      closeOptions(OnnxRuntime.ortApiHandle, handle);
    }

    /**
     * Sets the execution mode of this options object, overriding the old setting.
     *
     * @param mode The execution mode to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setExecutionMode(ExecutionMode mode) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setExecutionMode(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), mode.getID());
      }
    }

    /**
     * Sets the optimization level of this options object, overriding the old setting.
     *
     * @param level The optimization level to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setOptimizationLevel(OptLevel level) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setOptimizationLevel(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), level.getID());
      }
    }

    /**
     * Sets the size of the CPU thread pool used for executing multiple request concurrently, if
     * executing on a CPU.
     *
     * @param numThreads The number of threads to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setInterOpNumThreads(int numThreads) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setInterOpNumThreads(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), numThreads);
      }
    }

    /**
     * Sets the size of the CPU thread pool used for executing a single graph, if executing on a
     * CPU.
     *
     * @param numThreads The number of threads to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setIntraOpNumThreads(int numThreads) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setIntraOpNumThreads(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), numThreads);
      }
    }

    /**
     * Sets the output path for the optimized model.
     *
     * @param outputPath The output path to write the model to.
     * @throws OrtException If there was an error in native code.
     */
    public void setOptimizedModelFilePath(String outputPath) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setOptimizationModelFilePath(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), outputPath);
      }
    }

    /**
     * Sets the logger id to use.
     *
     * @param loggerId The logger id string.
     * @throws OrtException If there was an error in native code.
     */
    public void setLoggerId(String loggerId) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setLoggerId(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), loggerId);
      }
    }

    /**
     * Enables profiling in sessions using this SessionOptions.
     *
     * @param filePath The file to write profile information to.
     * @throws OrtException If there was an error in native code.
     */
    public void enableProfiling(String filePath) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        enableProfiling(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), filePath);
      }
    }

    /**
     * Disables profiling in sessions using this SessionOptions.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void disableProfiling() throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        disableProfiling(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle());
      }
    }

    /**
     * Turns on memory pattern optimizations, where memory is preallocated if all shapes are known.
     *
     * @param memoryPatternOptimization If true enable memory pattern optimizations.
     * @throws OrtException If there was an error in native code.
     */
    public void setMemoryPatternOptimization(boolean memoryPatternOptimization)
        throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setMemoryPatternOptimization(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), memoryPatternOptimization);
      }
    }

    /**
     * Sets the CPU to use an arena memory allocator.
     *
     * @param useArena If true use an arena memory allocator for the CPU execution provider.
     * @throws OrtException If there was an error in native code.
     */
    public void setCPUArenaAllocator(boolean useArena) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setCPUArenaAllocator(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), useArena);
      }
    }

    /**
     * Sets the Session's logging level.
     *
     * @param logLevel The log level to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setSessionLogLevel(OrtLoggingLevel logLevel) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setSessionLogLevel(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), logLevel.getValue());
      }
    }

    /**
     * Sets the Session's logging verbosity level.
     *
     * @param logLevel The logging verbosity to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setSessionLogVerbosityLevel(int logLevel) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        setSessionLogVerbosityLevel(
            OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), logLevel);
      }
    }

    /**
     * Registers a library of custom ops for use with {@link OrtSession}s using this SessionOptions.
     *
     * @param path The path to the library on disk.
     * @throws OrtException If there was an error loading the library.
     */
    public void registerCustomOpLibrary(String path) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        long customHandle =
            registerCustomOpLibrary(
                OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), path);
        customLibraryHandles.add(customHandle);
      }
    }

    /**
     * Add CUDA as an execution backend, using device 0.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void addCUDA() throws OrtException {
      addCUDA(0);
    }

    /**
     * Add CUDA as an execution backend, using the specified CUDA device id.
     *
     * @param deviceNum The CUDA device id.
     * @throws OrtException If there was an error in native code.
     */
    public void addCUDA(int deviceNum) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addCUDA(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), deviceNum);
      }
    }

    /**
     * Adds the CPU as an execution backend, using the arena allocator if desired.
     *
     * <p>By default this backend is used, but if other backends are requested, it should be
     * requested last.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addCPU(boolean useArena) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addCPU(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), useArena ? 1 : 0);
      }
    }

    /**
     * Adds Intel's Deep Neural Network Library as an execution backend.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addDnnl(boolean useArena) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addDnnl(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), useArena ? 1 : 0);
      }
    }

    /**
     * Adds NGraph as an execution backend.
     *
     * <p>See the documentation for the supported backend types.
     *
     * @param ngBackendType The NGraph backend type.
     * @throws OrtException If there was an error in native code.
     */
    public void addNGraph(String ngBackendType) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addNGraph(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), ngBackendType);
      }
    }

    /**
     * Adds OpenVINO as an execution backend.
     *
     * @param deviceId The id of the OpenVINO execution device.
     * @throws OrtException If there was an error in native code.
     */
    public void addOpenVINO(String deviceId) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addOpenVINO(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), deviceId);
      }
    }

    /**
     * Adds Nvidia's TensorRT as an execution backend.
     *
     * @param deviceNum The id of the CUDA device.
     * @throws OrtException If there was an error in native code.
     */
    public void addTensorrt(int deviceNum) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addTensorrt(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), deviceNum);
      }
    }

    /**
     * Adds Android's NNAPI as an execution backend.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void addNnapi() throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addNnapi(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle());
      }
    }

    /**
     * Adds Nuphar as an execution backend.
     *
     * @param allowUnalignedBuffers Allow unaligned memory buffers.
     * @param settings See the documentation for valid settings strings.
     * @throws OrtException If there was an error in native code.
     */
    public void addNuphar(boolean allowUnalignedBuffers, String settings) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addNuphar(
            OnnxRuntime.ortApiHandle,
            sessionOptionsReference.handle(),
            allowUnalignedBuffers ? 1 : 0,
            settings);
      }
    }

    /**
     * Adds DirectML as an execution backend.
     *
     * @param deviceId The id of the DirectML device.
     * @throws OrtException If there was an error in native code.
     */
    public void addDirectML(int deviceId) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addDirectML(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), deviceId);
      }
    }

    /**
     * Adds the ARM Compute Library as an execution backend.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addACL(boolean useArena) throws OrtException {
      try (NativeUsage sessionOptionsReference = use()) {
        addACL(OnnxRuntime.ortApiHandle, sessionOptionsReference.handle(), useArena ? 1 : 0);
      }
    }

    private native void setExecutionMode(long apiHandle, long nativeHandle, int mode)
        throws OrtException;

    private native void setOptimizationLevel(long apiHandle, long nativeHandle, int level)
        throws OrtException;

    private native void setInterOpNumThreads(long apiHandle, long nativeHandle, int numThreads)
        throws OrtException;

    private native void setIntraOpNumThreads(long apiHandle, long nativeHandle, int numThreads)
        throws OrtException;

    private native void setOptimizationModelFilePath(
        long apiHandle, long nativeHandle, String modelPath) throws OrtException;

    private static native long createOptions(long apiHandle);

    private native void setLoggerId(long apiHandle, long nativeHandle, String loggerId)
        throws OrtException;

    private native void enableProfiling(long apiHandle, long nativeHandle, String filePrefix)
        throws OrtException;

    private native void disableProfiling(long apiHandle, long nativeHandle) throws OrtException;

    private native void setMemoryPatternOptimization(
        long apiHandle, long nativeHandle, boolean memoryPatternOptimization) throws OrtException;

    private native void setCPUArenaAllocator(long apiHandle, long nativeHandle, boolean useArena)
        throws OrtException;

    private native void setSessionLogLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native void setSessionLogVerbosityLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native long registerCustomOpLibrary(long apiHandle, long nativeHandle, String path)
        throws OrtException;

    private native void closeCustomLibraries(long[] nativeHandle);

    private native void closeOptions(long apiHandle, long nativeHandle);

    /**
     * To use additional providers, you must build ORT with the extra providers enabled. Then call
     * one of these functions to enable them in the session:
     *
     * <ul>
     *   <li>OrtSessionOptionsAppendExecutionProvider_CPU
     *   <li>OrtSessionOptionsAppendExecutionProvider_CUDA
     *   <li>OrtSessionOptionsAppendExecutionProvider_(remaining providers...)
     * </ul>
     *
     * The order they care called indicates the preference order as well. In other words call this
     * method on your most preferred execution provider first followed by the less preferred ones.
     * If none are called Ort will use its internal CPU execution provider.
     *
     * <p>If a backend is unavailable then it throws an OrtException
     */
    private native void addCPU(long apiHandle, long nativeHandle, int useArena) throws OrtException;

    private native void addCUDA(long apiHandle, long nativeHandle, int deviceNum)
        throws OrtException;

    private native void addDnnl(long apiHandle, long nativeHandle, int useArena)
        throws OrtException;

    private native void addNGraph(long apiHandle, long nativeHandle, String ngBackendType)
        throws OrtException;

    private native void addOpenVINO(long apiHandle, long nativeHandle, String deviceId)
        throws OrtException;

    private native void addTensorrt(long apiHandle, long nativeHandle, int deviceNum)
        throws OrtException;

    private native void addNnapi(long apiHandle, long nativeHandle) throws OrtException;

    private native void addNuphar(
        long apiHandle, long nativeHandle, int allowUnalignedBuffers, String settings)
        throws OrtException;

    private native void addDirectML(long apiHandle, long nativeHandle, int deviceId)
        throws OrtException;

    private native void addACL(long apiHandle, long nativeHandle, int useArena) throws OrtException;
  }

  /** Used to control logging and termination of a call to {@link OrtSession#run}. */
  public static class RunOptions extends NativeObject {

    /**
     * Creates a RunOptions.
     *
     * @throws OrtException If the construction of the native RunOptions failed.
     */
    public RunOptions() throws OrtException {
      super(createRunOptions(OnnxRuntime.ortApiHandle));
    }

    /**
     * Sets the current logging level on this RunOptions.
     *
     * @param level The new logging level.
     * @throws OrtException If the native call failed.
     */
    public void setLogLevel(OrtLoggingLevel level) throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        setLogLevel(OnnxRuntime.ortApiHandle, runOptionsReference.handle(), level.getValue());
      }
    }

    /**
     * Gets the current logging level set on this RunOptions.
     *
     * @return The logging level.
     * @throws OrtException If the native call failed.
     */
    public OrtLoggingLevel getLogLevel() throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        return OrtLoggingLevel.mapFromInt(
            getLogLevel(OnnxRuntime.ortApiHandle, runOptionsReference.handle()));
      }
    }

    /**
     * Sets the current logging verbosity level on this RunOptions.
     *
     * @param level The new logging verbosity level.
     * @throws OrtException If the native call failed.
     */
    public void setLogVerbosityLevel(int level) throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        setLogVerbosityLevel(OnnxRuntime.ortApiHandle, runOptionsReference.handle(), level);
      }
    }

    /**
     * Gets the current logging verbosity level set on this RunOptions.
     *
     * @return The logging verbosity level.
     * @throws OrtException If the native call failed.
     */
    public int getLogVerbosityLevel() throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        return getLogVerbosityLevel(OnnxRuntime.ortApiHandle, runOptionsReference.handle());
      }
    }

    /**
     * Sets the run tag used in logging.
     *
     * @param runTag The run tag in logging output.
     * @throws OrtException If the native library call failed.
     */
    public void setRunTag(String runTag) throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        setRunTag(OnnxRuntime.ortApiHandle, runOptionsReference.handle(), runTag);
      }
    }

    /**
     * Gets the String used to log information about this run.
     *
     * @return The run tag.
     * @throws OrtException If the native library call failed.
     */
    public String getRunTag() throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        return getRunTag(OnnxRuntime.ortApiHandle, runOptionsReference.handle());
      }
    }

    /**
     * Sets a flag so that all incomplete {@link OrtSession#run} calls using this instance of {@code
     * RunOptions} will terminate as soon as possible. If the flag is false, it resets this {@code
     * RunOptions} so it can be used with other calls to {@link OrtSession#run}.
     *
     * @param terminate If true terminate all runs associated with this RunOptions.
     * @throws OrtException If the native library call failed.
     */
    public void setTerminate(boolean terminate) throws OrtException {
      try (NativeUsage runOptionsReference = use()) {
        setTerminate(OnnxRuntime.ortApiHandle, runOptionsReference.handle(), terminate);
      }
    }

    @Override
    protected void doClose(long handle) {
      close(OnnxRuntime.ortApiHandle, handle);
    }

    private static native long createRunOptions(long apiHandle) throws OrtException;

    private native void setLogLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native int getLogLevel(long apiHandle, long nativeHandle) throws OrtException;

    private native void setLogVerbosityLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native int getLogVerbosityLevel(long apiHandle, long nativeHandle) throws OrtException;

    private native void setRunTag(long apiHandle, long nativeHandle, String runTag)
        throws OrtException;

    private native String getRunTag(long apiHandle, long nativeHandle) throws OrtException;

    private native void setTerminate(long apiHandle, long nativeHandle, boolean terminate)
        throws OrtException;

    private static native void close(long apiHandle, long nativeHandle);
  }

  /**
   * An {@link AutoCloseable} wrapper around a {@link Map} containing {@link OnnxValue}s.
   *
   * <p>When this is closed it closes all the {@link OnnxValue}s inside it. If you maintain a
   * reference to a value after this object has been closed it will throw an {@link
   * IllegalStateException} upon access.
   */
  public static class Result implements AutoCloseable, Iterable<Map.Entry<String, OnnxValue>> {

    private static final Logger logger = Logger.getLogger(Result.class.getName());

    private final Map<String, OnnxValue> map;

    private final List<OnnxValue> list;

    private boolean closed;

    /**
     * Creates a Result from the names and values produced by {@link OrtSession#run(Map)}.
     *
     * @param names The output names.
     * @param values The output values.
     */
    Result(String[] names, OnnxValue[] values) {
      map = new LinkedHashMap<>(names.length);
      list = new ArrayList<>(names.length);

      if (names.length != values.length) {
        throw new IllegalArgumentException(
            "Expected same number of names and values, found names.length = "
                + names.length
                + ", values.length = "
                + values.length);
      }

      for (int i = 0; i < names.length; i++) {
        map.put(names[i], values[i]);
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

    private final void ensureOpen() {
      if (closed) {
        throw new IllegalStateException("Result is closed");
      }
    }

    @Override
    public Iterator<Map.Entry<String, OnnxValue>> iterator() {
      ensureOpen();
      return map.entrySet().iterator();
    }

    /**
     * Gets the value from the container at the specified index.
     *
     * <p>Throws {@link IllegalStateException} if the container has been closed, and {@link
     * IndexOutOfBoundsException} if the index is invalid.
     *
     * @param index The index to lookup.
     * @return The value at the index.
     */
    public OnnxValue get(int index) {
      ensureOpen();
      return list.get(index);
    }

    /**
     * Returns the number of outputs in this Result.
     *
     * @return The number of outputs.
     */
    public int size() {
      ensureOpen();
      return map.size();
    }

    /**
     * Gets the value from the container assuming it's not been closed.
     *
     * <p>Throws {@link IllegalStateException} if the container has been closed.
     *
     * @param key The key to lookup.
     * @return Optional.of the value if it exists.
     */
    public Optional<OnnxValue> get(String key) {
      ensureOpen();
      return Optional.ofNullable(map.get(key));
    }
  }
}
