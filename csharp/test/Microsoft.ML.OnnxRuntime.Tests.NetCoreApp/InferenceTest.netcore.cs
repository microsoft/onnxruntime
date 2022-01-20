using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
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
                Assert.Equal(1, session.InputMetadata.Count); // 1 input node
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input node name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
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

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                cleanUp.Add(cudaProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = "0";
                providerOptionsDict["gpu_mem_limit"] = "20971520";
                providerOptionsDict["arena_extend_strategy"] = "kSameAsRequested";
                providerOptionsDict["cudnn_conv_algo_search"] = "DEFAULT";
                providerOptionsDict["do_copy_in_default_stream"] = "1";
                providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";
                cudaProviderOptions.UpdateOptions(providerOptionsDict);

                var resultProviderOptionsDict = new Dictionary<string, string>();
                ProviderOptionsValueHelper.StringToDict(cudaProviderOptions.GetOptions(), resultProviderOptionsDict);

                // test provider options configuration
                string value;
                value = resultProviderOptionsDict["device_id"];
                Assert.Equal("0", value);
                value = resultProviderOptionsDict["gpu_mem_limit"];
                Assert.Equal("20971520", value);
                value = resultProviderOptionsDict["arena_extend_strategy"];
                Assert.Equal("kSameAsRequested", value);
                value = resultProviderOptionsDict["cudnn_conv_algo_search"];
                Assert.Equal("DEFAULT", value);
                value = resultProviderOptionsDict["do_copy_in_default_stream"];
                Assert.Equal("1", value);
                value = resultProviderOptionsDict["cudnn_conv_use_max_workspace"];
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

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(0);
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

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var trtProviderOptions = new OrtTensorRTProviderOptions();
                cleanUp.Add(trtProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["device_id"] = "0";
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
                Assert.Equal("0", value);
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

        private static Dictionary<string, string> GetSkippedModels(DirectoryInfo modelsDirInfo)
        {
            var skipModels = new Dictionary<string, string>() {
                { "mxnet_arcface", "Model is an invalid ONNX model"},
                { "tf_inception_v2", "TODO: Debug failing model, skipping for now" },
                { "fp16_tiny_yolov2", "Tolerance level for float16 is not known. We now support fp16." },
                { "fp16_test_tiny_yolov2", "ImageScaler is not a registered function/op"},
                { "fp16_coreml_FNS-Candy", "ImageScaler is not a registered function/op" },
                { "fp16_coreml_LinearRegression_NYCTaxi", "Error in Node:featureVectorizer : No Op registered for FeatureVectorizer with domain_version of 1"},
                { "test_bidaf", "Does not run in opset9, runs in other opsets. The model runs but I don't have a data set to debug output locally. Tensors of type ElementType not currently supported in the LoadTensorFromFile." },
                { "test_mnist", "Does not run in opset9, runs in other opsets. The model runs but I don't have a data set to debug output locally. Tensors of type ElementType not currently supported in the LoadTensorFromFile" },
                { "BERT_Squad", "Could not find an implementation for the node bert / embeddings / one_hot:OneHot(9)" },
                { "mlperf_ssd_mobilenet_300", "Could not find file output_0.pb" },
                { "tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied" },
                { "coreml_Imputer-LogisticRegression_sklearn_load_breast_cancer", "Can't determine model file name" },
                { "mask_rcnn_keras", "Model should be edited to remove the extra outputs" },
                { "test_strnormalizer_export_monday_casesensintive_lower", "ElementType not currently supported"},
                { "test_max_float64", "node test error"},
                { "test_min_uint8", "node test error"},
                { "test_mod_mixed_sign_float64", "node test error"},
                { "test_einsum_transpose", "node test error"},
                { "test_momentum", "node test error"},
                { "test_max_uint16", "node test error"},
                { "test_resize_downsample_scales_linear_align_corners", "node test error"},
                { "test_strnormalizer_nostopwords_nochangecase", "node test error"},
                { "test_cumsum_2d_negative_axis", "node test error"},
                { "test_adagrad_multiple", "node test error"},
                { "test_einsum_inner_prod", "node test error"},
                { "test_clip_default_int8_min", "node test error"},
                { "test_max_int8", "node test error"},
                { "test_sequence_insert_at_back", "node test error"},
                { "test_mod_mixed_sign_int8", "node test error"},
                { "test_maxunpool_export_with_output_shape", "node test error"},
                { "test_strnormalizer_export_monday_empty_output", "node test error"},
                { "test_strnormalizer_export_monday_insensintive_upper_twodim", "ElementType not currently supported"},
                { "test_clip_default_int8_max", "node test error"},
                { "test_einsum_sum", "node test error"},
                { "test_min_int16", "node test error"},
                { "test_adagrad", "node test error"},
                { "test_min_float64", "node test error"},
                { "test_max_int16", "node test error"},
                { "test_einsum_batch_diagonal", "node test error"},
                { "test_sequence_insert_at_front", "node test error"},
                { "test_cumsum_1d_exclusive", "node test error"},
                { "test_training_dropout_default", "node test error"},
                { "test_training_dropout", "node test error"},
                { "test_adam", "node test error"},
                { "test_training_dropout_mask", "node test error"},
                { "test_clip_default_int8_inbounds", "node test error"},
                { "test_eyelike_with_dtype", "node test error"},
                { "test_cumsum_1d", "node test error"},
                { "test_conv_with_autopad_same", "node test error"},
                { "test_cumsum_1d_reverse_exclusive", "node test error"},
                { "test_cast_STRING_to_FLOAT", "node test error"},
                { "test_cast_FLOAT16_to_DOUBLE", "node test error"},
                { "test_cast_FLOAT_to_DOUBLE", "node test error"},
                { "test_cast_BFLOAT16_to_FLOAT", "node test error"},
                { "test_cast_FLOAT_to_BFLOAT16", "node test error"},
                { "test_cast_FLOAT_to_STRING", "node test error"},
                { "test_castlike_STRING_to_FLOAT", "node test error"},
                { "test_castlike_STRING_to_FLOAT_expanded", "node test error"},
                { "test_castlike_FLOAT16_to_DOUBLE", "node test error"},
                { "test_castlike_FLOAT16_to_DOUBLE_expanded", "node test error"},
                { "test_castlike_FLOAT_to_DOUBLE", "node test error"},
                { "test_castlike_FLOAT_to_DOUBLE_expanded", "node test error"},
                { "test_castlike_BFLOAT16_to_FLOAT", "node test error"},
                { "test_castlike_BFLOAT16_to_FLOAT_expanded", "node test error"},
                { "test_castlike_FLOAT_to_BFLOAT16", "node test error"},
                { "test_castlike_FLOAT_to_BFLOAT16_expanded", "node test error"},
                { "test_castlike_FLOAT_to_STRING", "node test error"},
                { "test_castlike_FLOAT_to_STRING_expanded", "node test error"},
                { "test_bitshift_right_uint16", "node test error"},
                { "test_bitshift_left_uint16", "node test error"},
                { "test_pow_types_float32_uint64", "node test error"},
                { "test_cumsum_2d_axis_0", "node test error"},
                { "test_max_uint8", "node test error"},
                { "test_strnormalizer_export_monday_casesensintive_nochangecase", "ElementType not currently supported"},
                { "test_momentum_multiple", "node test error"},
                { "test_cumsum_1d_reverse", "node test error"},
                { "test_pow_types_float32_uint32", "node test error"},
                { "test_if_seq", "node test error"},
                { "test_resize_downsample_scales_cubic_align_corners", "node test error"},
                { "test_einsum_batch_matmul", "node test error"},
                { "test_nesterov_momentum", "node test error"},
                { "test_cumsum_2d_axis_1", "node test error"},
                { "test_strnormalizer_export_monday_casesensintive_upper", "node test error"},
                { "test_min_uint16", "node test error"},
                { "test_adam_multiple", "node test error"},
                { "test_loop13_seq", "node test error"},
                { "test_convtranspose_autopad_same", "node test error"},
                { "test_training_dropout_default_mask", "node test error"},
                { "test_min_int8", "node test error"},
                { "test_identity_sequence", "data type not supported"},
                { "test_gru_batchwise", "batchwise operations not supported"},
                { "test_lstm_batchwise", "batchwise operations not supported"},
                { "test_simple_rnn_batchwise", "batchwise operations not supported"},
                { "test_sub_uint8", "data type not supported"},
                { "test_mul_uint8", "data type not supported"},
                { "test_add_uint8", "data type not supported"},
                { "test_div_uint8", "data type not supported"},
                { "test_batchnorm_epsilon", "opset14 version not implemented yet"},
                { "test_batchnorm_epsilon_training_mode", "opset14 version not implemented yet"},
                { "test_batchnorm_example", "opset14 version not implemented yet"},
                { "test_batchnorm_example_training_mode", "opset14 version not implemented yet"},
                { "test_bernoulli", "random generator"},
                { "test_bernoulli_seed", "random generator"},
                { "test_bernoulli_double", "random generator"},
                { "test_bernoulli_expanded", "random generator"},
                { "test_bernoulli_seed_expanded", "random generator"},
                { "test_bernoulli_double_expanded", "random generator"},
                { "test_shape", "opset15 version not implemented yet"},
                { "test_shape_clip_end", "opset15 version not implemented yet"},
                { "test_shape_clip_start", "opset15 version not implemented yet"},
                { "test_shape_end_1", "opset15 version not implemented yet"},
                { "test_shape_end_negative", "opset15 version not implemented yet"},
                { "test_shape_example", "opset15 version not implemented yet"},
                { "test_shape_start_1", "opset15 version not implemented yet"},
                { "test_shape_start_negative_1", "opset15 version not implemented yet"},
                { "test_shape_start_1_end_2", "opset15 version not implemented yet"},
                { "test_shape_start_1_end_negative_1", "opset15 version not implemented yet"},
                { "test_shape_end_negative_1", "opset15 version not implemented yet"},
                { "test_optional_get_element", "not implemented yet"},
                { "test_optional_get_element_sequence", "not implemented yet"},
                { "test_optional_has_element", "not implemented yet"},
                { "test_optional_has_element_empty", "not implemented yet"},
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
                foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
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
                skipModels["test_zfnet512"] = "System out of memory";
                skipModels["test_bvlc_reference_caffenet"] = "System out of memory";
                skipModels["coreml_VGG16_ImageNet"] = "System out of memory";
                skipModels["test_ssd"] = "System out of memory";
                skipModels["roberta_sequence_classification"] = "System out of memory";
            }

            return skipModels;
        }

        public static IEnumerable<object[]> GetModelsForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                //var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in opsetDir.EnumerateDirectories())
                {
                    if (!skipModels.ContainsKey(modelDir.Name))
                    {
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }
                } //model
            } //opset
        }

        public static IEnumerable<object[]> GetSkippedModelForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels(modelsDirInfo);

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in modelRoot.EnumerateDirectories())
                {
                    if (skipModels.ContainsKey(modelDir.Name))
                    {
                        //Console.WriteLine("Model {0} is skipped due to the error: {1}", modelDir.FullName, skipModels[modelDir.Name]);
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }

                }
            }
        }

        [Theory(DisplayName = "TestPreTrainedModels")]
        [MemberData(nameof(GetModelsForTest))]
        [MemberData(nameof(GetSkippedModelForTest), Skip = "Skipped due to Error, please fix the error and enable the test")]
        private void TestPreTrainedModels(string opset, string modelName)
        {
            var modelsDir = GetTestModelsDir();
            string onnxModelFileName = null;

            var modelDir = new DirectoryInfo(Path.Combine(modelsDir, opset, modelName));

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

                using (var session = new InferenceSession(onnxModelFileName))
                {
                    var inMeta = session.InputMetadata;
                    string testDataDirNamePattern = "test_data*";
                    if (opset == "opset9" && modelName == "LSTM_Seq_lens_unpacked")
                    {
                        testDataDirNamePattern = "seq_lens*"; // discrepancy in data directory
                    }
                    foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
                    {
                        var inputContainer = new List<NamedOnnxValue>();
                        var outputContainer = new List<NamedOnnxValue>();
                        foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                        {
                            inputContainer.Add(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                        }
                        foreach (var f in testDataDir.EnumerateFiles("output_*.pb"))
                        {
                            outputContainer.Add(TestDataLoader.LoadTensorFromFilePb(f.FullName, session.OutputMetadata));
                        }

                        using (var resultCollection = session.Run(inputContainer))
                        {
                            foreach (var result in resultCollection)
                            {
                                Assert.True(session.OutputMetadata.ContainsKey(result.Name));
                                var outputMeta = session.OutputMetadata[result.Name];
                                NamedOnnxValue outputValue = null;
                                foreach (var o in outputContainer)
                                {
                                    if (o.Name == result.Name)
                                    {
                                        outputValue = o;
                                        break;
                                    }
                                }
                                if (outputValue == null)
                                {
                                    outputValue = outputContainer.First(); // in case the output data file does not contain the name
                                }
                                if (outputMeta.IsTensor)
                                {
                                    if (outputMeta.ElementType == typeof(float))
                                    {
                                        Assert.Equal(result.AsTensor<float>(), outputValue.AsTensor<float>(), new FloatComparer());
                                    }
                                    else if (outputMeta.ElementType == typeof(int))
                                    {
                                        Assert.Equal(result.AsTensor<int>(), outputValue.AsTensor<int>(), new ExactComparer<int>());
                                    }
                                    else if (outputMeta.ElementType == typeof(uint))
                                    {
                                        Assert.Equal(result.AsTensor<uint>(), outputValue.AsTensor<uint>(), new ExactComparer<uint>());
                                    }
                                    else if (outputMeta.ElementType == typeof(short))
                                    {
                                        Assert.Equal(result.AsTensor<short>(), outputValue.AsTensor<short>(), new ExactComparer<short>());
                                    }
                                    else if (outputMeta.ElementType == typeof(ushort))
                                    {
                                        Assert.Equal(result.AsTensor<ushort>(), outputValue.AsTensor<ushort>(), new ExactComparer<ushort>());
                                    }
                                    else if (outputMeta.ElementType == typeof(long))
                                    {
                                        Assert.Equal(result.AsTensor<long>(), outputValue.AsTensor<long>(), new ExactComparer<long>());
                                    }
                                    else if (outputMeta.ElementType == typeof(ulong))
                                    {
                                        Assert.Equal(result.AsTensor<ulong>(), outputValue.AsTensor<ulong>(), new ExactComparer<ulong>());
                                    }
                                    else if (outputMeta.ElementType == typeof(byte))
                                    {
                                        Assert.Equal(result.AsTensor<byte>(), outputValue.AsTensor<byte>(), new ExactComparer<byte>());
                                    }
                                    else if (outputMeta.ElementType == typeof(bool))
                                    {
                                        Assert.Equal(result.AsTensor<bool>(), outputValue.AsTensor<bool>(), new ExactComparer<bool>());
                                    }
                                    else if (outputMeta.ElementType == typeof(Float16))
                                    {
                                        Assert.Equal(result.AsTensor<Float16>(), outputValue.AsTensor<Float16>(), new Float16Comparer { tolerance = 2 });
                                    }
                                    else if (outputMeta.ElementType == typeof(BFloat16))
                                    {
                                        Assert.Equal(result.AsTensor<BFloat16>(), outputValue.AsTensor<BFloat16>(), new BFloat16Comparer { tolerance = 2 });
                                    }
                                    else
                                    {
                                        Assert.True(false, "The TestPretrainedModels does not yet support output of type " + nameof(outputMeta.ElementType));
                                    }
                                }
                                else
                                {
                                    Assert.True(false, "TestPretrainedModel cannot handle non-tensor outputs yet");
                                }
                            }
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

        [SkipNonPackageTests(DisplayName = "TestRegisterCustomOpLibrary")]
        private void TestRegisterCustomOpLibrary()
        {
            using (var option = new SessionOptions())
            {
                string libName = "custom_op_library.dll";
                string modelPath = "custom_op_test.onnx";
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

                var ortEnvInstance = OrtEnv.Instance();
                string[] providers = ortEnvInstance.GetAvailableProviders();
                if (Array.Exists(providers, provider => provider == "CUDAExecutionProvider")) {
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


                using (var session = new InferenceSession(modelPath, option))
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

        // TestGpu() will test the CUDA EP on CUDA enabled builds and
        // the DML EP on DML enabled builds
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