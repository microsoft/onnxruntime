using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    /// <summary>
    /// This is compensate for the absence of string.Contains() in .NET Standard 2.0
    /// Contains(String, StringComparison)
    /// </summary>
    public static class StringExtensions
    {
        public static bool Contains(this String str, String substring,
                                    StringComparison comp)
        {
            if (substring == null)
                throw new ArgumentNullException("substring",
                                             "substring cannot be null.");
            else if (!Enum.IsDefined(typeof(StringComparison), comp))
                throw new ArgumentException("comp is not a member of StringComparison",
                                         "comp");

            return str.IndexOf(substring, comp) >= 0;
        }
    }

    public partial class InferenceTest
    {
        private const string module = "onnxruntime.dll";
        private const string propertiesFile = "Properties.txt";

        [Fact(DisplayName = "CanCreateAndDisposeSessionWithModelPath")]
        public void CanCreateAndDisposeSessionWithModelPath()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.NotNull(session);
                Assert.NotNull(session.InputMetadata);
                Assert.Equal(1, session.InputMetadata.Count); // 1 input nodeMeta
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input nodeMeta name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output nodeMeta
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output nodeMeta name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].ElementType);
                Assert.True(session.OutputMetadata["softmaxout_1"].IsTensor);
                var expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                Assert.Equal(expectedOutputDimensions.Length, session.OutputMetadata["softmaxout_1"].Dimensions.Length);
                for (int i = 0; i < expectedOutputDimensions.Length; i++)
                {
                    Assert.Equal(expectedOutputDimensions[i], session.OutputMetadata["softmaxout_1"].Dimensions[i]);
                }
            }
        }

#if USE_CUDA
        [Fact(DisplayName = "TestCUDAProviderOptions")]
        private void TestCUDAProviderOptions()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            string defaultDeviceId = "0";
            string deviceIdFromEnv = System.Environment.GetEnvironmentVariable("OnnxruntimeTestGpuDeviceId");
            if (!string.IsNullOrEmpty(deviceIdFromEnv) && int.TryParse(deviceIdFromEnv, out int deviceId) && deviceId >= 0)
            {
                defaultDeviceId = deviceIdFromEnv;
                output.WriteLine($"Parsed ID: {deviceIdFromEnv}");
            }

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                cleanUp.Add(cudaProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = defaultDeviceId;
                // 256MB
                providerOptionsDict["gpu_mem_limit"] = "268435456";
                providerOptionsDict["arena_extend_strategy"] = "kSameAsRequested";
                providerOptionsDict["cudnn_conv_algo_search"] = "DEFAULT";
                providerOptionsDict["do_copy_in_default_stream"] = "1";
                providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";
                providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";
                cudaProviderOptions.UpdateOptions(providerOptionsDict);

                var resultProviderOptionsDict = new Dictionary<string, string>();
                ProviderOptionsValueHelper.StringToDict(cudaProviderOptions.GetOptions(), resultProviderOptionsDict);

                // test provider options configuration
                string value;
                value = resultProviderOptionsDict["device_id"];
                Assert.Equal("0", value);
                value = resultProviderOptionsDict["gpu_mem_limit"];
                Assert.Equal("268435456", value);
                value = resultProviderOptionsDict["arena_extend_strategy"];
                Assert.Equal("kSameAsRequested", value);
                value = resultProviderOptionsDict["cudnn_conv_algo_search"];
                Assert.Equal("DEFAULT", value);
                value = resultProviderOptionsDict["do_copy_in_default_stream"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["cudnn_conv_use_max_workspace"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["cudnn_conv1d_pad_to_nc1d"];
                Assert.Equal("1", value);

                // test correctness of provider options
                SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                float[] inputData = TestDataLoader.LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                session.Run(container);
            }
        }
#endif

#if USE_TENSORRT
        [Fact(DisplayName = "CanRunInferenceOnAModelWithTensorRT")]
        private void CanRunInferenceOnAModelWithTensorRT()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            
            int deviceId = 0;
            string deviceIdStr = System.Environment.GetEnvironmentVariable("ONNXRUNTIME_TEST_GPU_DEVICE_ID");
            if (!string.IsNullOrEmpty(deviceIdStr) && int.TryParse(deviceIdStr, out int parsedValue) && parsedValue >= 0)
            {
                deviceId = parsedValue;
                output.WriteLine($"Parsed ID: {parsedValue}");
            }

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(deviceId);
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                float[] inputData = TestDataLoader.LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }


                using (var results = session.Run(container))
                {
                    ValidateRunResults(results);
                }
            }
        }

        [Fact(DisplayName = "TestTensorRTProviderOptions")]
        private void TestTensorRTProviderOptions()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string calTablePath = "squeezenet_calibration.flatbuffers";
            string enginePath = "./";
            string engineDecrptLibPath = "engine_decryp";
            string defaultDeviceId = "0";
            string deviceIdFromEnv = System.Environment.GetEnvironmentVariable("OnnxruntimeTestGpuDeviceId");
            if (!string.IsNullOrEmpty(deviceIdFromEnv) && int.TryParse(deviceIdFromEnv, out int deviceId) && deviceId >= 0)
            {
                defaultDeviceId = deviceIdFromEnv;
                output.WriteLine($"Parsed ID: {deviceIdFromEnv}");
            }

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var trtProviderOptions = new OrtTensorRTProviderOptions();
                cleanUp.Add(trtProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = defaultDeviceId;
                providerOptionsDict["trt_fp16_enable"] = "1";
                providerOptionsDict["trt_int8_enable"] = "1";
                providerOptionsDict["trt_int8_calibration_table_name"] = calTablePath;
                providerOptionsDict["trt_engine_cache_enable"] = "1";
                providerOptionsDict["trt_engine_cache_path"] = enginePath;
                providerOptionsDict["trt_engine_decryption_enable"] = "0";
                providerOptionsDict["trt_engine_decryption_lib_path"] = engineDecrptLibPath;
                trtProviderOptions.UpdateOptions(providerOptionsDict);

                var resultProviderOptionsDict = new Dictionary<string, string>();
                ProviderOptionsValueHelper.StringToDict(trtProviderOptions.GetOptions(), resultProviderOptionsDict);

                // test provider options configuration
                string value;
                value = resultProviderOptionsDict["device_id"];
                Assert.Equal(defaultDeviceId, value);
                value = resultProviderOptionsDict["trt_fp16_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_int8_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_int8_calibration_table_name"];
                Assert.Equal(calTablePath, value);
                value = resultProviderOptionsDict["trt_engine_cache_enable"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["trt_engine_cache_path"];
                Assert.Equal(enginePath, value);
                value = resultProviderOptionsDict["trt_engine_decryption_enable"];
                Assert.Equal("0", value);
                value = resultProviderOptionsDict["trt_engine_decryption_lib_path"];
                Assert.Equal(engineDecrptLibPath, value);

                // test correctness of provider options
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
                cleanUp.Add(options);

                var session = new InferenceSession(modelPath, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                float[] inputData = TestDataLoader.LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                session.Run(container);
            }
        }
#endif

        private static Func<DirectoryInfo, IEnumerable<DirectoryInfo>> getOpsetDirectories = delegate (DirectoryInfo modelsDirInfo)
        {
            return modelsDirInfo.EnumerateDirectories("opset*", SearchOption.AllDirectories);
        };

        private static Dictionary<string, string> GetSkippedModels(DirectoryInfo modelsDirInfo)
        {
            var skipModels = new Dictionary<string, string>() {
                { "mxnet_arcface", "Model is an invalid ONNX model"},
                { "tf_inception_v2", "TODO: Debug failing model, skipping for now" },
                { "fp16_tiny_yolov2", "Tolerance level for float16 is not known. We now support fp16." },
                { "fp16_test_tiny_yolov2", "ImageScaler is not a registered function/op"},
                { "fp16_coreml_FNS-Candy", "ImageScaler is not a registered function/op" },
                { "fp16_coreml_LinearRegression_NYCTaxi", "Error in Node:featureVectorizer : No Op registered for FeatureVectorizer with domain_version of 1"},
                { "test_mnist", "Does not run in opset9, runs in other opsets. The model runs but I don't have a data set to debug output locally. Tensors of type ElementType not currently supported in the LoadTensorFromFile" },
                { "BERT_Squad", "Could not find an implementation for the nodeMeta bert / embeddings / one_hot:OneHot(9)" },

                { "mlperf_ssd_mobilenet_300", "Could not find file output_0.pb" },
                { "tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied" },
                { "cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose" },
                { "coreml_Imputer-LogisticRegression_sklearn_load_breast_cancer", "Can't determine model file name" },
                { "mask_rcnn_keras", "Model should be edited to remove the extra outputs" },

                { "test_maxunpool_export_with_output_shape", "results mismatch"},

                { "test_min_int8", "Could not find an implementation for Min(13) node with name"},
                { "test_min_uint8", "Could not find an implementation for Min(13) node with name"},
                { "test_min_int16", "Could not find an implementation for Min(13) node with name"},
                { "test_min_uint16", "Could not find an implementation for Min(13) node with name"},

                { "test_max_int8", "Could not find an implementation for Max(13) node with name"},
                { "test_max_uint8", "Could not find an implementation for Max(13) node with name"},
                { "test_max_int16", "Could not find an implementation for Max(13) node with name"},
                { "test_max_uint16", "Could not find an implementation for Max(13) nodeMeta with name '"},

                { "test_mul_uint8", "Could not find an implementation for Mul(14) node with name" },

                { "test_bitshift_right_uint16", "Could not find an implementation for BitShift(11) nodeMeta with name ''"},
                { "test_bitshift_left_uint16", "Could not find an implementation for BitShift(11)"},

                { "test_pow_types_float32_uint64", "Could not find an implementation for Pow(15) node with name ''"},
                { "test_pow_types_float32_uint32", "Could not find an implementation for Pow(15) node with name ''"},

                { "test_resize_downsample_scales_cubic_align_corners", "Results mismatch"},
                { "test_resize_downsample_scales_linear_align_corners", "Results mismatch"},

                { "test_gru_batchwise", "batchwise operations not supported"},
                { "test_lstm_batchwise", "Batchwise recurrent operations(layout == 1) are not supported.If you need support create a github issue with justification."},
                { "test_simple_rnn_batchwise", "batchwise operations not supported"},
                { "test_batchnorm_example_training_mode", "opset14 version not implemented yet"},

                { "test_bernoulli", "random generator, results mismatch"},
                { "test_bernoulli_seed", "random generator, results mismatch"},
                { "test_bernoulli_double", "random generator, results mismatch"},
                { "test_bernoulli_expanded", "random generator, results mismatch"},
                { "test_bernoulli_seed_expanded", "random generator, results mismatch"},
                { "test_bernoulli_double_expanded", "random generator, results mismatch"},

                // the expansion of Softplus uses Exp(1). ORT has a Softplus kernel, so testing the expansion is
                // unnecessary and fails as ORT support for Exp started at opset 6 (as ORT didn't exist until opset 7).

                { "test_clip_default_int8_max_expanded", "Could not find an implementation for Less(13) nodeMeta with name ''" },
                { "test_softplus_expanded", "Could not find an implementation for Exp(1) node with name ''"},
                { "test_softplus_example_expanded", "Could not find an implementation for Exp(1) node with name ''"},
                { "test_div_uint8", "Could not find an implementation for Div(14) nodeMeta with name ''"},
                { "test_add_uint8", "Opset18 Could not find an implementation for Add(14) nodeMeta with name ''"},
                { "test_col2im_pads", "Results mismatch due to a typo in test data"},

                { "test_optional_has_element_empty_optional_input", "OptionalProto test metadata. Unable to load 'optional_input' optional element type of: Undefined type"},
                { "test_loop13_seq", "3rd input is an empty sequence. Ort API does not tolerate empty seq: Number of values should be at least 1" },

                // Training tests
                { "BERT-Squad-int8", "training domain"},
                { "YOLOv3-12-int8", "training_domain"},

                { "test_training_dropout_default", "results mismatch"},
                { "test_training_dropout_default_mask", "Results mismatch"},
                { "test_training_dropout", "results mismatch"},
                { "test_training_dropout_mask", "results mismatch."},

                { "test_momentum", "ai.onnx.preview.training:Momentum(-1) is not a registered function/op"},
                { "test_momentum_multiple", "ai.onnx.preview.training:Momentum(-1) is not a registered function/op"},
                { "test_nesterov_momentum", "ai.onnx.preview.training:Momentum(-1) is not a registered function/op"},

                { "test_adam", "ai.onnx.preview.training:Adam(-1) is not a registered function/op"},
                { "test_adam_multiple", "ai.onnx.preview.training:Adam(-1) is not a registered function/op"},

                { "test_adagrad", "ai.onnx.preview.training:Adagrad(-1) is not a registered function/op"},
                { "test_adagrad_multiple", "ai.onnx.preview.training:Adagrad(-1) is not a registered function/op"},

                { "test_zfnet512", "skip it as ZFNET-512"},
            };

            // The following models fails on nocontribops win CI
            var disableContribOpsEnvVar = Environment.GetEnvironmentVariable("DisableContribOps");
            var isContribOpsDisabled = (disableContribOpsEnvVar != null) ? disableContribOpsEnvVar.Equals("ON") : false;
            if (isContribOpsDisabled)
            {
                skipModels["test_tiny_yolov2"] = "Fails when ContribOps is disabled";
                skipModels["mask_rcnn_keras"] = "Pad is not a registered function/op";
            }

            // Skip traditional ML models
            var disableMlOpsEnvVar = Environment.GetEnvironmentVariable("DisableMlOps");
            var isMlOpsDisabled = (disableMlOpsEnvVar != null) ? disableMlOpsEnvVar.Equals("ON") : false;
            if (isMlOpsDisabled)
            {
                foreach (var opsetDir in getOpsetDirectories(modelsDirInfo))
                {
                    foreach (var modelDir in opsetDir.EnumerateDirectories())
                    {
                        var modelDirName = modelDir.Name;
                        if (modelDirName.StartsWith("scikit_") ||
                        modelDirName.StartsWith("libsvm_") ||
                        modelDirName.StartsWith("coreml_") ||
                        modelDirName.StartsWith("keras2coreml_") ||
                        modelDirName.StartsWith("XGBoost_"))
                        {
                            skipModels[modelDirName] = "Fails when ML ops are disabled";
                        }
                    } //model
                } //opset
            }

            // This model fails on x86 Win CI
            if (System.Environment.Is64BitProcess == false)
            {
                skipModels["test_vgg19"] = "Get preallocated buffer for initializer conv4_4_b_0 failed";
                skipModels["GPT2_LM_HEAD"] = "System out of memory";
                skipModels["GPT2"] = "System out of memory";
                skipModels["test_GPT2"] = "System out of memory";
                skipModels["tf_pnasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_5/comb_iter_1/left/bn_sep_7x7_1/beta:0_203 failed";
                skipModels["tf_nasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_11/beginning_bn/beta:0_331 failed";
                skipModels["ZFNet-512"] = "System out of memory";
                skipModels["test_bvlc_reference_caffenet"] = "System out of memory";
                skipModels["coreml_VGG16_ImageNet"] = "System out of memory";
                skipModels["test_ssd"] = "System out of memory";
                skipModels["roberta_sequence_classification"] = "System out of memory";
                // models from model zoo
                skipModels["VGG 19"] = "bad allocation";
                skipModels["VGG 19-caffe2"] = "bad allocation";
                skipModels["VGG 19-bn"] = "bad allocation";
                skipModels["VGG 16"] = "bad allocation";
                skipModels["VGG 16-bn"] = "bad allocation";
                skipModels["VGG 16-fp32"] = "bad allocation";
            }

            return skipModels;
        }

        public static IEnumerable<object[]> GetModelsForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in getOpsetDirectories(modelsDirInfo))
            {
                //var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in opsetDir.EnumerateDirectories())
                {
                    if (!(skipModels.ContainsKey(modelDir.Name) ||
                          modelDir.Name.Contains("int8", StringComparison.OrdinalIgnoreCase) ||
                          modelDir.Name.Contains("qdq", StringComparison.OrdinalIgnoreCase)))
                    {
                        yield return new object[] { modelDir.Parent.FullName, modelDir.Name };
                    }
                } //model
            } //opset
        }

        public static IEnumerable<object[]> GetSkippedModelForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in getOpsetDirectories(modelsDirInfo))
            {
                foreach (var modelDir in opsetDir.EnumerateDirectories())
                {
                    if (skipModels.ContainsKey(modelDir.Name) ||
                        modelDir.Name.Contains("int8", StringComparison.OrdinalIgnoreCase) ||
                        modelDir.Name.Contains("qdq", StringComparison.OrdinalIgnoreCase))
                    {
                        //Console.WriteLine("Model {0} is skipped due to the error: {1}", modelDir.FullName, skipModels[modelDir.Name]);
                        yield return new object[] { modelDir.Parent.FullName, modelDir.Name };
                    }

                }
            }
        }

        private string MatchInputOutputWithFile(string fileName, InferenceSession session, bool input, out NodeMetadata result)
        {
            string nodeName = string.Empty;
            result = null;
            var names = (input) ? session.InputNames : session.OutputNames;
            var metadata = (input) ? session.InputMetadata : session.OutputMetadata;
            string regEx = (input) ? @"input_(\d{1,}).pb" : @"output_(\d{1,}).pb";
            var inpOut = (input) ? "input" : "output";

            // Extract the number from the file name, if not try to match the input/output name with the name of the file.
            try
            {
                // captures start at index 1
                var group = Regex.Matches(fileName, regEx).Single().Groups[1];
                var num = int.Parse(group.Value);
                if (num >= 0 && num < names.Count)
                {
                    nodeName = names[num];
                    result = metadata[nodeName];
                }
                else
                {
                    throw new InvalidDataException($"Filename '{fileName}' {inpOut} number '{num}' is out of range for '{names.Count}' {inpOut}(s)");
                }
            }
            catch (Exception)
            {
                // Either does not match or can not parse the number
            }

            if (result is null)
            {
                throw new InvalidDataException($"Unable to match file: {fileName} to input/output metadata");
            }
            return nodeName;
        }

        // The numbering of the input files does not match the order of outputs
        // listed in the metadata of test_BERT_Squad. Model metadata order:
        //                 "unique_ids_raw_output___9:0", "segment_ids:0", "input_mask:0", "input_ids:0"
        // The corr input files are: input_0.pb, input_3.pb, input_2.pb, input_1.pb
        // Everything in reverse, but the 0.

        // Previously, it worked because our test data has matching
        // tensor names that we could match to metadata after we load the tensor.
        // But now, we need to know ahead of time what Onnx type we load, and thus match
        // metadata with the test data file before loading. Protobuf can happily load whatever
        // and give you garbage.

        private string MatchBertSquadInputs(string fileName)
        {
            string nodeName = string.Empty;
            switch (fileName)
            {
                case "input_0.pb":
                    nodeName = "unique_ids_raw_output___9:0";
                    break;
                case "input_1.pb":
                    nodeName = "input_ids:0";
                    break;
                case "input_2.pb":
                    nodeName = "input_mask:0";
                    break;
                case "input_3.pb":
                    nodeName = "segment_ids:0";
                    break;
                default:
                    throw new InvalidDataException($"Unhandled input file name: '{fileName}' for test_BERT_Squad");
            }
            return nodeName;
        }

        // The model actually has only 3 outputs, but the Zoo version has 4 files are supplied.
        // The numbering of the output files does not match the order of outputs
        // listed in the metadata.

        // Previously, it worked because our CI test data version has matching
        // tensor names that we could match to metadata after we load the tensor.
        // But now, we need to know ahead of time what Onnx type we load, and thus match
        // metadata with the test data file before loading. Protobuf can happily load whatever
        // and give you garbage.

        // Order in the metadata: unstack:1, unstack:0, unique_ids:0
        // The files are in reverse order
        private string MatchBertSquadOutputs(string fileName)
        {
            string nodeName = string.Empty;
            switch (fileName)
            {
                case "output_0.pb": // Int64
                    nodeName = "unique_ids:0";
                    break;
                case "output_1.pb":
                    nodeName = "unstack:0";
                    break;
                case "output_2.pb":
                    nodeName = "unstack:1";
                    break;
                default:
                    throw new InvalidDataException($"Unhandled output file name: '{fileName}' for test_BERT_Squad");
            }
            return nodeName;
        }

        private const string keras_prelu_ImageNet_small_nodeName_Input = "p_re_lu_3_input";
        private const string keras_prelu_ImageNet_small_nodeName_Output = "p_re_lu_3/add:0";

        private void LoadInputData<T>(string opset, string modelName,
            DirectoryInfo testDataDir,
            InferenceSession session,
            IList<T> inputContainer,
            Func<string, string, NodeMetadata, T> loader)
        {
            var inMeta = session.InputMetadata;
            foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
            {
                if (modelName == "keras_prelu_ImageNet_small" && opset == "opset9")
                {
                    // The model has 1 input, match all file names (they are different in each data set)
                    // to the same input
                    var nodeName = keras_prelu_ImageNet_small_nodeName_Input;
                    var nodeMeta = inMeta[nodeName];
                    inputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
                else if (modelName == "test_BERT_Squad" && opset == "opset8")
                {
                    string nodeName = MatchBertSquadInputs(f.Name);
                    var nodeMeta = inMeta[nodeName];
                    inputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
                else
                {
                    var nodeName = MatchInputOutputWithFile(f.Name, session, true, out NodeMetadata nodeMeta);
                    inputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
            }
        }

        private void LoadOutputData<T>(string opset, string modelName,
                                                DirectoryInfo testDataDir,
                                                InferenceSession session,
                                                IList<T> outputContainer,
                                                Func<string, string, NodeMetadata, T> loader)
        {
            var outMeta = session.OutputMetadata;
            foreach (var f in testDataDir.EnumerateFiles("output_*.pb"))
            {
                if (modelName == "keras_prelu_ImageNet_small" && opset == "opset9")
                {
                    // The model has 1 output, match all file names (they are different in each data set)
                    // to the same output
                    var nodeName = keras_prelu_ImageNet_small_nodeName_Output;
                    var nodeMeta = outMeta[nodeName];
                    outputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
                else if (modelName == "test_BERT_Squad" && opset == "opset8")
                {
                    string nodeName = MatchBertSquadOutputs(f.Name);
                    var nodeMeta = outMeta[nodeName];
                    outputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
                else
                {
                    // Otherwise, just match trailing filename number to the input name -> metadata
                    var nodeName = MatchInputOutputWithFile(f.Name, session, false, out NodeMetadata nodeMeta);
                    outputContainer.Add(loader(f.FullName, nodeName, nodeMeta));
                }
            }
        }

        private void RunPretrainedModel(InferenceSession session,
                     IReadOnlyList<NamedOnnxValue> inputContainer, IReadOnlyList<NamedOnnxValue> outputContainer)
        {
            var outMeta = session.OutputMetadata;

            var orderedOutputNames = new List<string>(outputContainer.Count);
            foreach (var output in outputContainer)
            {
                orderedOutputNames.Add(output.Name);
            }

            using (var resultCollection = session.Run(inputContainer, orderedOutputNames))
            {
                Assert.Equal(outputContainer.Count, resultCollection.Count);
                for (int i = 0; i < resultCollection.Count; ++i)
                {
                    var result = resultCollection[i];
                    var outputValue = outputContainer[i];

                    Assert.NotNull(outputValue);
                    Assert.Equal(result.Name, outputValue.Name);

                    var outputMeta = outMeta[outputValue.Name];
                    if (outputMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
                    {
                        outputMeta = outputMeta.AsOptionalMetadata().ElementMeta;
                    }

                    Assert.Equal(outputValue.ValueType, outputMeta.OnnxValueType);

                    switch (outputValue.ValueType)
                    {
                        case OnnxValueType.ONNX_TYPE_TENSOR:  // Only Dense tensors now
                            {
                                VerifyTensorResults(outputMeta.ElementDataType, result, outputValue);
                            }
                            break;
                        case OnnxValueType.ONNX_TYPE_SEQUENCE:
                            {
                                VerifySequenceResults(result, outputValue, outputMeta);
                            }
                            break;
                        default:
                            Assert.True(false, $"TestPreTrainedModels cannot handle Onnxtype: {outputValue.ValueType}");
                            break;
                    }
                }
            }
        }

        private void RunPretrainedModel(InferenceSession session, RunOptions runOptions,
                     IReadOnlyList<DisposableTestPair<OrtValue>> inputContainer,
                     IReadOnlyList<DisposableTestPair<OrtValue>> outputContainer)
        {
            var outMeta = session.OutputMetadata;

            var orderedInputNames = new List<string>(inputContainer.Count);
            var orderdedInputs = new List<OrtValue>(inputContainer.Count);
            foreach(var pair in inputContainer)
            {
                orderedInputNames.Add(pair.Key);
                orderdedInputs.Add(pair.Value);
            }

            var orderedOutputNames = new List<string>(outputContainer.Count);
            var orderedOutputs = new List<OrtValue>(outputContainer.Count);
            foreach (var pair in outputContainer)
            {
                orderedOutputNames.Add(pair.Key);
                orderedOutputs.Add(pair.Value);
            }

            using (var results = session.Run(runOptions, orderedInputNames, orderdedInputs, orderedOutputNames))
            {
                Assert.Equal(outMeta.Count, results.Count);
                Assert.Equal(outputContainer.Count, results.Count);

                for (int i = 0; i < outputContainer.Count; ++i)
                {
                    var resultValue = results[i];
                    var expectedValue = outputContainer[i].Value;

                    var outputMeta = outMeta[orderedOutputNames[i]];
                    if (outputMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
                    {
                        outputMeta = outputMeta.AsOptionalMetadata().ElementMeta;
                    }

                    if (outputMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                    {
                        VerifyTensorResults(outputMeta.ElementDataType, resultValue, expectedValue);
                    }
                    else if (outputMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                    {
                        VerifySequenceResults(resultValue, expectedValue, outputMeta);
                    }
                    else
                    {
                        Assert.True(false, $"TestPreTrainedModels cannot handle Onnxtype: {outputMeta.OnnxValueType}");
                    }
                }
            }
        }

        [Theory(DisplayName = "TestPretrainedModelsWithOrtValue")]
        [MemberData(nameof(GetModelsForTest))]
        [MemberData(nameof(GetSkippedModelForTest), Skip = "Skipped due to Error, please fix the error and enable the test")]
        public void TestPretrainedModelsWithOrtValue(string opsetDir, string modelName)
        {
            TestPreTrainedModels(opsetDir, modelName, true);
        }

        [Theory(DisplayName = "TestPreTrainedModels")]
        [MemberData(nameof(GetModelsForTest))]
        [MemberData(nameof(GetSkippedModelForTest), Skip = "Skipped due to Error, please fix the error and enable the test")]
        private void TestPreTrainedModels(string opsetDir, string modelName, bool useOrtValueAPIs = false)
        {
            var opsetDirInfo = new DirectoryInfo(opsetDir);
            var opset = opsetDirInfo.Name;
            string onnxModelFileName = null;

            var modelDir = new DirectoryInfo(Path.Combine(opsetDir, modelName));

            try
            {
                var onnxModelNames = modelDir.GetFiles("*.onnx");
                bool validModelFound = false;
                if (onnxModelNames.Length > 0)
                {
                    // TODO remove file "._resnet34v2.onnx" from test set
                    for (int i = 0; i < onnxModelNames.Length; i++)
                    {
                        if (onnxModelNames[i].Name != "._resnet34v2.onnx")
                        {
                            onnxModelNames[0] = onnxModelNames[i];
                            validModelFound = true;
                        }
                    }
                }

                if (validModelFound)
                {
                    onnxModelFileName = Path.Combine(modelDir.FullName, onnxModelNames[0].Name);
                }
                else
                {
                    var modelNamesList = string.Join(",", onnxModelNames.Select(x => x.ToString()));
                    throw new Exception($"Opset {opset} Model {modelName}. Can't determine model file name. Found these :{modelNamesList}");
                }

                using(var runOptions = new RunOptions())
                using (var session = new InferenceSession(onnxModelFileName))
                {
                    string testDataDirNamePattern = "test_data*";
                    if (opset == "opset9" && modelName == "LSTM_Seq_lens_unpacked")
                    {
                        testDataDirNamePattern = "seq_lens*"; // discrepancy in data directory
                    }
                    foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
                    {
                        if (useOrtValueAPIs)
                        {
                            using (var inputOrtValues = new DisposableListTest<DisposableTestPair<OrtValue>>(session.InputMetadata.Count))
                            using (var outputOrtValues = new DisposableListTest<DisposableTestPair<OrtValue>>(session.OutputMetadata.Count))
                            {
                                LoadInputData(opset, modelName, testDataDir, session, inputOrtValues, TestDataLoader.LoadOrtValueFromFilePb);
                                LoadOutputData(opset, modelName, testDataDir, session, outputOrtValues, TestDataLoader.LoadOrtValueFromFilePb);
                                RunPretrainedModel(session, runOptions, inputOrtValues, outputOrtValues);
                            }
                        }
                        else
                        {
                            var inputContainer = new List<NamedOnnxValue>(session.InputMetadata.Count);
                            LoadInputData(opset, modelName, testDataDir, session, inputContainer, TestDataLoader.LoadOnnxValueFromFilePb);
                            var outputContainer = new List<NamedOnnxValue>(session.OutputMetadata.Count);
                            LoadOutputData(opset, modelName, testDataDir, session, outputContainer, TestDataLoader.LoadOnnxValueFromFilePb);
                            RunPretrainedModel(session, inputContainer, outputContainer);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                var msg = $"Opset {opset}, Model {modelName}: ModelFile = {onnxModelFileName} error = {ex.Message}";
                if (ex.Message.Contains("ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions"))
                {
                    // If the exception is thrown because the opset version of the test model is
                    // not supported by ONNXRuntime yet, then ignore the test and proceed.
                    // ORT allows commits from ONNX master and in such cases we do come across new opsets which are
                    // not supported in ORT yet. In order to force these tests to run set env var ALLOW_RELEASED_ONNX_OPSET_ONLY=0
                    output.WriteLine("Skipping the model test as the latest ONNX opset is not supported yet. Error Message: " + msg);
                }
                else
                {
                    throw new Exception(msg + "\n" + ex.StackTrace);
                }
            }
        }

        private static void VerifySequenceResults(NamedOnnxValue result, NamedOnnxValue expectedValue, NodeMetadata metaData)
        {
            var meta = metaData.AsSequenceMetadata();
            var resultSequence = result.AsEnumerable<NamedOnnxValue>();
            var expectedSequence = expectedValue.AsEnumerable<NamedOnnxValue>();
            Assert.Equal(resultSequence.Count(), expectedSequence.Count());

            foreach (var (resultItem, expectedItem) in resultSequence.Zip(expectedSequence, (r, e) => (r, e)))
            {
                Assert.Equal(resultItem.ValueType, expectedItem.ValueType);
                Assert.Equal(resultItem.ValueType, meta.ElementMeta.OnnxValueType);
                switch (resultItem.ValueType)
                {
                    case OnnxValueType.ONNX_TYPE_TENSOR:
                        VerifyTensorResults(meta.ElementMeta.ElementDataType, resultItem, expectedItem);
                        break;
                    case OnnxValueType.ONNX_TYPE_SEQUENCE:
                        {
                            VerifySequenceResults(resultItem, expectedItem, meta.ElementMeta);
                        }
                        break;
                    default:
                        Assert.True(false, "VerifySequenceResults cannot handle Onnxtype: " + resultItem.ValueType.ToString());
                        break;
                }
                Assert.Equal(resultItem.AsTensor<float>(), expectedItem.AsTensor<float>(), new FloatComparer());
            }
        }

        private static void VerifyTensorResults(TensorElementType elementType, NamedOnnxValue result, NamedOnnxValue expectedValue)
        {
            switch (elementType)
            {
                case TensorElementType.Float:
                    Assert.Equal(expectedValue.AsTensor<float>(), result.AsTensor<float>(), new FloatComparer());
                    break;
                case TensorElementType.Double:
                    Assert.Equal(expectedValue.AsTensor<double>(), result.AsTensor<double>(), new DoubleComparer());
                    break;
                case TensorElementType.Int32:
                    Assert.Equal(expectedValue.AsTensor<int>(), result.AsTensor<int>(), new ExactComparer<int>());
                    break;
                case TensorElementType.UInt32:
                    Assert.Equal(expectedValue.AsTensor<uint>(), result.AsTensor<uint>(), new ExactComparer<uint>());
                    break;
                case TensorElementType.Int16:
                    Assert.Equal(expectedValue.AsTensor<short>(), result.AsTensor<short>(), new ExactComparer<short>());
                    break;
                case TensorElementType.UInt16:
                    Assert.Equal(expectedValue.AsTensor<ushort>(), result.AsTensor<ushort>(), new ExactComparer<ushort>());
                    break;
                case TensorElementType.Int64:
                    Assert.Equal(expectedValue.AsTensor<long>(), result.AsTensor<long>(), new ExactComparer<long>());
                    break;
                case TensorElementType.UInt64:
                    Assert.Equal(expectedValue.AsTensor<ulong>(), result.AsTensor<ulong>(), new ExactComparer<ulong>());
                    break;
                case TensorElementType.UInt8:
                    Assert.Equal(expectedValue.AsTensor<byte>(), result.AsTensor<byte>(), new ExactComparer<byte>());
                    break;
                case TensorElementType.Int8:
                    Assert.Equal(result.AsTensor<sbyte>(), result.AsTensor<sbyte>(), new ExactComparer<sbyte>());
                    break;
                case TensorElementType.Bool:
                    Assert.Equal(expectedValue.AsTensor<bool>(), result.AsTensor<bool>(), new ExactComparer<bool>());
                    break;
                case TensorElementType.Float16:
                    Assert.Equal(expectedValue.AsTensor<Float16>(), result.AsTensor<Float16>(), new Float16Comparer { tolerance = 2 });
                    break;
                case TensorElementType.BFloat16:
                    Assert.Equal(expectedValue.AsTensor<BFloat16>(), result.AsTensor<BFloat16>(), new BFloat16Comparer { tolerance = 2 });
                    break;
                case TensorElementType.String:
                    Assert.Equal(expectedValue.AsTensor<string>(), result.AsTensor<string>(), new ExactComparer<string>());
                    break;
                default:
                    Assert.True(false, "TestPreTrainedModels does not yet support output of type: " + elementType.ToString());
                    break;
            }
        }

        private static void VerifySequenceResults(OrtValue resultSequence, OrtValue expectedSequence, NodeMetadata metaData)
        {
            var allocator = OrtAllocator.DefaultInstance;
            Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, resultSequence.OnnxType);
            Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, expectedSequence.OnnxType);

            var elementMeta = metaData.AsSequenceMetadata().ElementMeta;

            var resultCount = resultSequence.GetValueCount();
            Assert.Equal(expectedSequence.GetValueCount(), resultCount);

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                for (int i = 0; i < resultCount; ++i)
                {
                    var resultItem = resultSequence.GetValue(i, allocator);
                    cleanUp.Add(resultItem);

                    var expectedItem = expectedSequence.GetValue(i, allocator);
                    cleanUp.Add(expectedItem);

                    Assert.Equal(elementMeta.OnnxValueType, expectedItem.OnnxType);
                    Assert.Equal(elementMeta.OnnxValueType, resultItem.OnnxType);

                    switch (elementMeta.OnnxValueType)
                    {
                        case OnnxValueType.ONNX_TYPE_TENSOR:
                            VerifyTensorResults(elementMeta.ElementDataType, resultItem, expectedItem);
                            break;
                        case OnnxValueType.ONNX_TYPE_SEQUENCE:
                            {
                                VerifySequenceResults(resultItem, expectedItem, elementMeta);
                            }
                            break;
                        default:
                            Assert.True(false, $"VerifySequenceResults cannot handle Onnxtype: {elementMeta.OnnxValueType}");
                            break;
                    }
                }
            }
        }

        private static void VerifyTensorResults(TensorElementType expectedElementType, OrtValue result, OrtValue expectedValue)
        {
            Assert.True(result.IsTensor);
            Assert.True(expectedValue.IsTensor);

            var resultTypeShape = result.GetTensorTypeAndShape();
            var expectedTypeShape = expectedValue.GetTensorTypeAndShape();
            Assert.Equal(expectedElementType, resultTypeShape.ElementDataType);
            Assert.Equal(expectedElementType, expectedTypeShape.ElementDataType);
            Assert.Equal(expectedTypeShape.Shape, resultTypeShape.Shape);

            if (expectedElementType == TensorElementType.String)
            {
                var resStrings = result.GetStringTensorAsArray();
                var expStrings = expectedValue.GetStringTensorAsArray();
                Assert.Equal(expStrings, resStrings);
                return;
            }

            switch (expectedElementType)
            {
                case TensorElementType.Float:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<float>().ToArray(), result.GetTensorDataAsSpan<float>().ToArray(),
                        new FloatComparer());
                    break;
                case TensorElementType.Double:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<double>().ToArray(), result.GetTensorDataAsSpan<double>().ToArray(),
                        new DoubleComparer());
                    break;
                case TensorElementType.Int32:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<int>().ToArray(), result.GetTensorDataAsSpan<int>().ToArray(), new ExactComparer<int>());
                    break;
                case TensorElementType.UInt32:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<uint>().ToArray(), result.GetTensorDataAsSpan<uint>().ToArray(), new ExactComparer<uint>());
                    break;
                case TensorElementType.Int16:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<short>().ToArray(), result.GetTensorDataAsSpan<short>().ToArray(), new ExactComparer<short>());
                    break;
                case TensorElementType.UInt16:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<ushort>().ToArray(), result.GetTensorDataAsSpan<ushort>().ToArray(), new ExactComparer<ushort>());
                    break;
                case TensorElementType.Int64:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<long>().ToArray(), result.GetTensorDataAsSpan<long>().ToArray(), new ExactComparer<long>());
                    break;
                case TensorElementType.UInt64:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<ulong>().ToArray(), result.GetTensorDataAsSpan<ulong>().ToArray(), new ExactComparer<ulong>());
                    break;
                case TensorElementType.UInt8:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<byte>().ToArray(), result.GetTensorDataAsSpan<byte>().ToArray(), new ExactComparer<byte>());
                    break;
                case TensorElementType.Int8:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<sbyte>().ToArray(), result.GetTensorDataAsSpan<sbyte>().ToArray(), new ExactComparer<sbyte>());
                    break;
                case TensorElementType.Bool:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<bool>().ToArray(), result.GetTensorDataAsSpan<bool>().ToArray(), new ExactComparer<bool>());
                    break;
                case TensorElementType.Float16:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<Float16>().ToArray(), result.GetTensorDataAsSpan<Float16>().ToArray(),
                        new Float16Comparer { tolerance = 2 });
                    break;
                case TensorElementType.BFloat16:
                    Assert.Equal(expectedValue.GetTensorDataAsSpan<BFloat16>().ToArray(), result.GetTensorDataAsSpan<BFloat16>().ToArray(),
                                               new BFloat16Comparer { tolerance = 2 });
                    break;
                default:
                    Assert.True(false, "VerifyTensorResults cannot handle ElementType: " + expectedElementType.ToString());
                    break;
            }
        }

        private static void VerifyContainerContent(IReadOnlyList<OrtValue> results,
            IReadOnlyList<NamedOnnxValue> expectedValues)
        {
            Assert.Equal(results.Count, expectedValues.Count);

            for (int i = 0; i < expectedValues.Count; ++i)
            {
                var result = results[i];

                var resultTypeShape = result.GetTensorTypeAndShape();

                var expectedValue = expectedValues[i];
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, expectedValue.ValueType);

                switch (resultTypeShape.ElementDataType)
                {
                    case TensorElementType.Float:
                        Assert.Equal(result.GetTensorDataAsSpan<float>().ToArray(), expectedValue.AsTensor<float>().ToArray(),
                            new ExactComparer<float>());
                        break;
                    case TensorElementType.Double:
                        Assert.Equal(result.GetTensorDataAsSpan<double>().ToArray(), expectedValue.AsTensor<double>().ToArray(),
                            new DoubleComparer());
                        break;
                    case TensorElementType.Int32:
                        Assert.Equal(result.GetTensorDataAsSpan<int>().ToArray(), expectedValue.AsTensor<int>().ToArray(), new ExactComparer<int>());
                        break;
                    case TensorElementType.UInt32:
                        Assert.Equal(result.GetTensorDataAsSpan<uint>().ToArray(), expectedValue.AsTensor<uint>().ToArray(), new ExactComparer<uint>());
                        break;
                    case TensorElementType.Int16:
                        Assert.Equal(result.GetTensorDataAsSpan<short>().ToArray(), expectedValue.AsTensor<short>().ToArray(), new ExactComparer<short>());
                        break;
                    case TensorElementType.UInt16:
                        Assert.Equal(result.GetTensorDataAsSpan<ushort>().ToArray(), expectedValue.AsTensor<ushort>().ToArray(), new ExactComparer<ushort>());
                        break;
                    case TensorElementType.Int64:
                        Assert.Equal(result.GetTensorDataAsSpan<long>().ToArray(), expectedValue.AsTensor<long>().ToArray(), new ExactComparer<long>());
                        break;
                    case TensorElementType.UInt64:
                        Assert.Equal(result.GetTensorDataAsSpan<ulong>().ToArray(), expectedValue.AsTensor<ulong>().ToArray(), new ExactComparer<ulong>());
                        break;
                    case TensorElementType.UInt8:
                        Assert.Equal(result.GetTensorDataAsSpan<byte>().ToArray(), expectedValue.AsTensor<byte>().ToArray(), new ExactComparer<byte>());
                        break;
                    case TensorElementType.Int8:
                        Assert.Equal(result.GetTensorDataAsSpan<sbyte>().ToArray(), expectedValue.AsTensor<sbyte>().ToArray(), new ExactComparer<sbyte>());
                        break;
                    case TensorElementType.Bool:
                        Assert.Equal(result.GetTensorDataAsSpan<bool>().ToArray(), expectedValue.AsTensor<bool>().ToArray(), new ExactComparer<bool>());
                        break;
                    case TensorElementType.Float16:
                        Assert.Equal(result.GetTensorDataAsSpan<Float16>().ToArray(), expectedValue.AsTensor<Float16>().ToArray(),
                            new Float16Comparer { tolerance = 2 });
                        break;
                    case TensorElementType.BFloat16:
                        Assert.Equal(result.GetTensorDataAsSpan<BFloat16>().ToArray(), expectedValue.AsTensor<BFloat16>().ToArray(),
                                                   new BFloat16Comparer { tolerance = 2 });
                        break;
                    case TensorElementType.String:
                        Assert.Equal(result.GetStringTensorAsArray(), expectedValue.AsTensor<string>().ToArray(), new ExactComparer<string>());
                        break;
                    default:
                        Assert.True(false, $"VerifyTensorResults cannot handle ElementType: { resultTypeShape.ElementDataType}");
                        break;
                }
            }
        }

        // Hint: .NET Core 3.1 has a 'NativeLibrary' class that can be used to free the library handle
        private void UnloadLibrary(IntPtr libraryHandle)
        {
            if (libraryHandle != IntPtr.Zero)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    if (!FreeLibrary(libraryHandle))
                    {
                        throw new Exception("Could not unload the provided shared library using its handle");
                    }
                }

                else
                {
                    // TODO: Deal with non-Windows platforms for the .NET Core use-case
                }
            }
        }

        private string GetCustomOpLibFullPath()
        {
            string libName = "custom_op_library.dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                libName = "custom_op_library.dll";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                libName = "libcustom_op_library.so";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                libName = "libcustom_op_library.dylib";
            }

            string libFullPath = Path.Combine(Directory.GetCurrentDirectory(), libName);
            Assert.True(File.Exists(libFullPath), $"Expected lib {libFullPath} does not exist.");

            return libFullPath;
        }

        private void ValidateModelWithCustomOps(SessionOptions options)
        {
            string modelPath = "custom_op_test.onnx";

            using (var session = new InferenceSession(modelPath, options))
            {
                var inputContainer = new List<NamedOnnxValue>();
                inputContainer.Add(NamedOnnxValue.CreateFromTensor<float>("input_1",
                    new DenseTensor<float>(
                        new float[]
                        {
                                1.1f,   2.2f,   3.3f,   4.4f,   5.5f,
                                6.6f,   7.7f,   8.8f,   9.9f,   10.0f,
                                11.1f,  12.2f,  13.3f,  14.4f,  15.5f
                        },
                        new int[] { 3, 5 }
                        )));

                inputContainer.Add(NamedOnnxValue.CreateFromTensor<float>("input_2",
                    new DenseTensor<float>(
                        new float[]
                        {
                                15.5f,   14.4f,   13.3f,   12.2f,   11.1f,
                                10.0f,   9.9f,    8.8f,    7.7f,    6.6f,
                                5.5f,    4.4f,    3.3f,    2.2f,    1.1f
                        },
                        new int[] { 3, 5 }
                        )));

                using (var result = session.Run(inputContainer))
                {
                    Assert.Equal("output", result.First().Name);
                    var tensorOut = result.First().AsTensor<int>();

                    var expectedOut = new DenseTensor<int>(
                        new int[]
                        {
                                17, 17, 17, 17, 17,
                                17, 18, 18, 18, 17,
                                17, 17, 17, 17, 17
                        },
                        new int[] { 3, 5 }
                        );
                    Assert.True(tensorOut.SequenceEqual(expectedOut));
                }
            }
        }

        [SkipNonPackageTests(DisplayName = "TestRegisterCustomOpLibrary")]
        private void TestRegisterCustomOpLibrary()
        {
            using (var option = new SessionOptions())
            {
                string libFullPath = GetCustomOpLibFullPath();

                try
                {
                    option.RegisterCustomOpLibrary(libFullPath);
                }
                catch (Exception ex)
                {
                    var msg = $"Failed to load custom op library {libFullPath}, error = {ex.Message}";
                    throw new Exception(msg + "\n" + ex.StackTrace);
                }

                var ortEnvInstance = OrtEnv.Instance();
                string[] providers = ortEnvInstance.GetAvailableProviders();
                if (Array.Exists(providers, provider => provider == "CUDAExecutionProvider"))
                {
                    option.AppendExecutionProvider_CUDA(0);
                }

                ValidateModelWithCustomOps(option);
            }
        }

        [SkipNonPackageTests(DisplayName = "TestRegisterCustomOpLibraryV2")]
        private void TestRegisterCustomOpLibraryV2()
        {
            using (var option = new SessionOptions())
            {
                string libFullPath = GetCustomOpLibFullPath();

                var ortEnvInstance = OrtEnv.Instance();
                string[] providers = ortEnvInstance.GetAvailableProviders();
                if (Array.Exists(providers, provider => provider == "CUDAExecutionProvider"))
                {
                    option.AppendExecutionProvider_CUDA(0);
                }

                IntPtr libraryHandle = IntPtr.Zero;
                try
                {
                    option.RegisterCustomOpLibraryV2(libFullPath, out libraryHandle);
                }
                catch (Exception ex)
                {
                    var msg = $"Failed to load custom op library {libFullPath}, error = {ex.Message}";
                    throw new Exception(msg + "\n" + ex.StackTrace);
                }

                ValidateModelWithCustomOps(option);

                // Safe to unload the custom op shared library now
                UnloadLibrary(libraryHandle);
            }
        }

        [Fact(DisplayName = "TestModelSerialization")]
        private void TestModelSerialization()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string modelOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "optimized-squeezenet.onnx");
            // Set the optimized model file path to assert that no exception are thrown.
            using (SessionOptions options = new SessionOptions())
            {
                options.OptimizedModelFilePath = modelOutputPath;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
                using (var session = new InferenceSession(modelPath, options))
                {
                    Assert.NotNull(session);
                    Assert.True(File.Exists(modelOutputPath));
                }
            }
        }

        // TestGpu() will test
        //  - the CUDA EP on CUDA enabled builds
        //  - the DML EP on DML enabled builds
        //  - the ROCm EP on ROCm enabled builds
        [GpuFact(DisplayName = "TestGpu")]
        private void TestGpu()
        {
            var tuple = OpenSessionSqueezeNet(0); // run on deviceID 0
            float[] expectedOutput = TestDataLoader.LoadTensorFromFile(@"bench.expected_out");

            using (var session = tuple.Item1)
            {
                var inputData = tuple.Item2;
                var tensor = tuple.Item3;
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
                var res = session.Run(container);
                var resultArray = res.First().AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput, resultArray, new FloatComparer());
            }
        }

        [DllImport("kernel32", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Ansi)]
        static extern UIntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [Fact(DisplayName = "VerifyNativeMethodsExist")]
        private void VerifyNativeMethodsExist()
        {
            // Check for  external API changes
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            var entryPointNames = new[]{
            "OrtGetApiBase",
            "OrtSessionOptionsAppendExecutionProvider_CPU"
#if USE_DNNL
            ,"OrtSessionOptionsAppendExecutionProvider_Dnnl"
#endif
#if USE_CUDA
            ,"OrtSessionOptionsAppendExecutionProvider_CUDA"
#endif
#if USE_ROCM
            ,"OrtSessionOptionsAppendExecutionProvider_ROCM"
#endif
#if USE_DML
            ,"OrtSessionOptionsAppendExecutionProvider_DML"
#endif
#if USE_OPENVINO
            ,"OrtSessionOptionsAppendExecutionProvider_OpenVINO"
#endif
#if USE_TENSORRT
            ,"OrtSessionOptionsAppendExecutionProvider_Tensorrt"
#endif
#if USE_MIGRAPHX
            ,"OrtSessionOptionsAppendExecutionProvider_MIGraphX"
#endif
#if USE_NNAPI
            ,"OrtSessionOptionsAppendExecutionProvider_Nnapi"
#endif
    };
            IntPtr libraryHandle = IntPtr.Zero;
            try
            {
                libraryHandle = LoadLibrary(module);
                foreach (var ep in entryPointNames)
                {
                    var x = GetProcAddress(libraryHandle, ep);
                    Assert.False(x == UIntPtr.Zero, $"Entrypoint {ep} not found in module {module}");
                }
            }

            finally
            {
                UnloadLibrary(libraryHandle);
            }
        }

        static string GetTestModelsDir()
        {
            // get build directory, append downloaded models location
            var cwd = Directory.GetCurrentDirectory();
            var props = File.ReadAllLines(Path.Combine(cwd, propertiesFile));
            var modelsRelDir = Path.Combine(props[0].Split('=')[1].Trim());
            var modelsDir = Path.Combine(cwd, @"../../..", modelsRelDir, "models");
            return modelsDir;
        }
    }
}
