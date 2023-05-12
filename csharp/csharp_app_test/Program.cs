using System;
using System.Threading;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests {

   class TRTEPTest {
        // Copy of the class that is internal in the main package
        internal class DisposableListTest<T> : List<T>, IDisposableReadOnlyCollection<T>
            where T : IDisposable
        {
            public DisposableListTest() { }
            public DisposableListTest(int count) : base(count) { }

            #region IDisposable Support
            private bool disposedValue = false; // To detect redundant calls

            protected virtual void Dispose(bool disposing)
            {
                if (!disposedValue)
                {
                    if (disposing)
                    {
                        // Dispose in the reverse order.
                        // Objects should typically be destroyed/disposed
                        // in the reverse order of its creation
                        // especially if the objects created later refer to the
                        // objects created earlier. For homogeneous collections of objects
                        // it would not matter.
                        for (int i = this.Count - 1; i >= 0; --i)
                        {
                            this[i]?.Dispose();
                        }
                        this.Clear();
                    }

                    disposedValue = true;
                }
            }

            // This code added to correctly implement the disposable pattern.
            public void Dispose()
            {
                // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
                Dispose(true);
                GC.SuppressFinalize(this);
            }
            #endregion
        }

        /* Recently added and not support for old ORT
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
        */

        protected void DecodeWithIOBinding() {
            var allocator = OrtAllocator.DefaultInstance;
            string Dir  = "/home/azureuser/disk/models/zcode_model/decoder"; 
            var modelDir = new DirectoryInfo(Dir);
            using var ioBinding = this.encoder.CreateIoBinding();
            var inMeta = this.encoder.InputMetadata;
            var outMeta = this.encoder.OutputMetadata;
            string testDataDirNamePattern = "test_data*";
            foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
            {
                var inputContainer = new List<NamedOnnxValue>(inMeta.Count);
                var outputContainer = new List<NamedOnnxValue>(outMeta.Count);
                int index = 0;
                foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                {
                    // Bind inputs
                    Console.WriteLine(f.Name);
                    Console.WriteLine(f.FullName);
                    if (f.Name == "input_0.pb") {
                        //using var encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                        using FixedBufferOnnxValue encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_long(f.FullName, inMeta));
                        ioBinding.BindInput("input_ids", encoderInputIds);
                    } else if (f.Name == "input_1.pb") {
                        //using var encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                        using FixedBufferOnnxValue encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_bool(f.FullName, inMeta));
                        ioBinding.BindInput("input_mask", encoderMask);
                    } else if (f.Name == "input_2.pb") {
                        var dimensions1 = new[] { 16, 256, 1024 };
                        var emptyDenseTensor1 = new DenseTensor<float>(new Span<int>(dimensions1));
                        using FixedBufferOnnxValue encoderStates = FixedBufferOnnxValue.CreateFromTensor(emptyDenseTensor1);
                        ioBinding.BindInput("encoder_states", encoderStates);
                    } else if (f.Name == "input_3.pb") {
                        var dimensions1 = new[] { 16, 256 };
                        var emptyDenseTensor1 = new DenseTensor<bool>(new Span<int>(dimensions1));
                        using FixedBufferOnnxValue encoderInputMask = FixedBufferOnnxValue.CreateFromTensor(emptyDenseTensor1);
                        ioBinding.BindInput("encoder_input_mask", encoderInputMask);
                    } else {
                        var dimensions1 = new[] { 16, 256, 1024 };
                        var emptyDenseTensor1 = new DenseTensor<float>(new Span<int>(dimensions1));
                        using FixedBufferOnnxValue historyStates = FixedBufferOnnxValue.CreateFromTensor(emptyDenseTensor1);
                        string input = "history_states_" + index.ToString();
                        ioBinding.BindInput(input, historyStates);
                        index += 1;
                    }
                }

                ioBinding.BindOutputToDevice("lm_logits", allocator.Info);
                ioBinding.BindOutputToDevice("log_lm_logits", allocator.Info);

                for (int i = 0; i < 25; i++) {
                    var input = "hidden_states_" + i.ToString();
                    ioBinding.BindOutputToDevice(input, allocator.Info);
                }
                ioBinding.SynchronizeBoundInputs();

                var runOptions = new RunOptions();
                this.encoder.RunWithBinding(runOptions, ioBinding);
                ioBinding.SynchronizeBoundOutputs();
            }
        }

        protected void EncodeWithIOBinding() {
            var allocator = OrtAllocator.DefaultInstance;
            string Dir  = "/home/azureuser/disk/models/zcode_model/encoder"; 
            var modelDir = new DirectoryInfo(Dir);
            using var ioBinding = this.encoder.CreateIoBinding();
            var inMeta = this.encoder.InputMetadata;
            var outMeta = this.encoder.OutputMetadata;
            string testDataDirNamePattern = "test_data*";
            foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
            {
                var inputContainer = new List<NamedOnnxValue>(inMeta.Count);
                var outputContainer = new List<NamedOnnxValue>(outMeta.Count);
                foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                {
                    // Bind inputs
                    Console.WriteLine(f.Name);
                    Console.WriteLine(f.FullName);
                    if (f.Name == "input_0.pb") {
                        //using var encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                        using var encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_long(f.FullName, inMeta));
                        ioBinding.BindInput("input_ids", encoderInputIds);
                    }

                    if (f.Name == "input_1.pb") {
                        //using var encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                        using FixedBufferOnnxValue encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_bool(f.FullName, inMeta));
                        ioBinding.BindInput("input_mask", encoderMask);
                    }
                }

                ioBinding.BindOutputToDevice("hidden_states", allocator.Info);
                ioBinding.SynchronizeBoundInputs();
                var runOptions = new RunOptions();
                this.encoder.RunWithBinding(runOptions, ioBinding);
                ioBinding.SynchronizeBoundOutputs();
            }
        }

        protected void Encode() {
            string Dir  = "/home/azureuser/disk/models/zcode_model/encoder"; 
            //string Dir  = "/home/azureuser/disk/models/wbs"; 
            var modelDir = new DirectoryInfo(Dir);
            var inMeta = this.encoder.InputMetadata;
            var outMeta = this.encoder.OutputMetadata;
            string testDataDirNamePattern = "test_data*";
            foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
            {
                var inputContainer = new List<NamedOnnxValue>(inMeta.Count);
                var outputContainer = new List<NamedOnnxValue>(outMeta.Count);
                foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                {
                    Console.WriteLine(f.Name);
                    Console.WriteLine(f.FullName);

                    ////var nodeName = MatchInputOutputWithFile(f.Name, session, true, out NodeMetadata nodeMeta);
                    ////inputContainer.Add(TestDataLoader.LoadOnnxValueFromFilePb(f.FullName, nodeName, nodeMeta));
                    
                    //inputContainer.Add(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));

                    if (f.Name == "input_0.pb") {
                        var dimensions1 = new[] { 1, 256 };
                        var emptyDenseTensor1 = new DenseTensor<long>(new Span<int>(dimensions1));
                        //inputContainer.Add(emptyDenseTensor1);
                        inputContainer.Add(NamedOnnxValue.CreateFromTensor<long>("input_ids", emptyDenseTensor1));
                    }

                    if (f.Name == "input_1.pb") {
                        var dimensions2 = new[] { 1, 256 };
                        var emptyDenseTensor2 = new DenseTensor<bool>(new Span<int>(dimensions2));
                        inputContainer.Add(NamedOnnxValue.CreateFromTensor<bool>("input_mask", emptyDenseTensor2));
                    }
                    
                }
                this.encoder.Run(inputContainer);
            }
        }

        private InferenceSession encoder;
        private InferenceSession decoder;

        static void Main(string[] args) {
            var ort = new TRTEPTest();
            ort.Run();
        }

        protected void Run() {
            Console.WriteLine("Start ...");

            //string modelName  = "encoder.onnx";
            //string Dir  = "/home/azureuser/disk/models/zcode_model/encoder"; 
            string modelName  = "decoder.onnx";
            string Dir  = "/home/azureuser/disk/models/zcode_model/decoder"; 
            var modelDir = new DirectoryInfo(Dir);
            string onnxModelFileName = null;
            onnxModelFileName = Path.Combine(modelDir.FullName, modelName);

            var ortEnvInstance = OrtEnv.Instance();
            //ortEnvInstance.EnvLogLevel = LogLevel.Verbose;
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var trtProviderOptions = new OrtTensorRTProviderOptions();
                cleanUp.Add(trtProviderOptions);

                var providerOptionsDict = new Dictionary<string, string>();
                providerOptionsDict["trt_fp16_enable"] = "1";
                providerOptionsDict["trt_engine_cache_enable"] = "1";
                trtProviderOptions.UpdateOptions(providerOptionsDict);
                var allocator = OrtAllocator.DefaultInstance;
                //var ortAllocationOutput = allocator.Allocate(1*256*1024 * sizeof(float));

                // test correctness of provider options
                SessionOptions options = SessionOptions.MakeSessionOptionWithTensorrtProvider(trtProviderOptions);
                //SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(0);
                cleanUp.Add(options);
                //cleanUp.Add(ortAllocationOutput);
                //
                
                const int num_threads = 10;
                Thread[] workerThreads = new Thread[num_threads];


                using (this.encoder = new InferenceSession(onnxModelFileName, options)) {
                    //DecodeWithIOBinding();

                    for (int i = 0; i < num_threads; i++) {
                        //Thread ThreadObject = new Thread(EncodeWithIOBinding); //Creating the Thread    
                        workerThreads[i] = new Thread(DecodeWithIOBinding); //Creating the Thread    
                        workerThreads[i].Start();
                    }

                   for (int i = 0; i < num_threads; i++) {
                       workerThreads[i].Join();
                   }
                }


                /*
                using (var session = new InferenceSession(onnxModelFileName, options))
                {
                    using var ioBinding = session.CreateIoBinding();
                    var inMeta = session.InputMetadata;
                    var outMeta = session.OutputMetadata;
                    string testDataDirNamePattern = "test_data*";
                    foreach (var testDataDir in modelDir.EnumerateDirectories(testDataDirNamePattern))
                    {
                        var inputContainer = new List<NamedOnnxValue>(inMeta.Count);
                        var outputContainer = new List<NamedOnnxValue>(outMeta.Count);
                        foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                        {
                            //var nodeName = MatchInputOutputWithFile(f.Name, session, true, out NodeMetadata nodeMeta);
                            //inputContainer.Add(TestDataLoader.LoadOnnxValueFromFilePb(f.FullName, nodeName, nodeMeta));
                            
                            //inputContainer.Add(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                            
                            // Bind inputs
                            Console.WriteLine(f.Name);
                            Console.WriteLine(f.FullName);
                            if (f.Name == "input_0.pb") {
                                //using var encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                                using var encoderInputIds = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_long(f.FullName, inMeta));
                                ioBinding.BindInput("input_ids", encoderInputIds);
                            }

                            if (f.Name == "input_1.pb") {
                                //using var encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePb(f.FullName, inMeta));
                                using FixedBufferOnnxValue encoderMask = FixedBufferOnnxValue.CreateFromTensor(TestDataLoader.LoadTensorFromFilePbV2_bool(f.FullName, inMeta));
                                ioBinding.BindInput("input_mask", encoderMask);
                            }
                        }

                        ioBinding.BindOutputToDevice("hidden_states", allocator.Info);
                        ioBinding.SynchronizeBoundInputs();
                        var runOptions = new RunOptions();
                        session.RunWithBinding(runOptions, ioBinding);
                        ioBinding.SynchronizeBoundOutputs();

                        //using (var resultCollection = session.Run(inputContainer))
                        //{
                        //}
                    }
                }
                */
            }
     }
   }
}
