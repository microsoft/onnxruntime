using System;

namespace Microsoft.AI.MachineLearning.Tests.Lib.DotNet5_0
{
    public class SessionAPITests
    {
        public static void LoadBindEval()
        {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            Console.WriteLine("Load kitten_224.png as StorageFile.");
            var name = AppDomain.CurrentDomain.BaseDirectory + "kitten_224.png";
            var getFileFromPathTask = Windows.Storage.StorageFile.GetFileFromPathAsync(name);
            getFileFromPathTask.AsTask()
                .ContinueWith<Windows.Storage.Streams.IRandomAccessStreamWithContentType>(
                    (task) =>
                    {
                        var image = task.Result;
                        Console.WriteLine("Load StorageFile into Stream.");
                        var stream_task = image.OpenReadAsync();
                        return stream_task.AsTask().Result;
                    })
                .ContinueWith<Windows.Graphics.Imaging.BitmapDecoder>(
                    (task) =>
                    {
                        using (var stream = task.Result)
                        {
                            Console.WriteLine("Create SoftwareBitmap from decoded Stream.");
                            var decoder_task = Windows.Graphics.Imaging.BitmapDecoder.CreateAsync(stream);
                            return decoder_task.AsTask().Result;
                        }
                    })
                .ContinueWith<Windows.Graphics.Imaging.SoftwareBitmap>(
                    (task) =>
                    {
                        var decoder = task.Result;
                        var software_bitmap_task = decoder.GetSoftwareBitmapAsync();
                        return software_bitmap_task.AsTask().Result;
                    })
                .ContinueWith(
                    (task) =>
                    {
                        using (var software_bitmap = task.Result)
                        {
                            Console.WriteLine("Create VideoFrame.");
                            var frame = Windows.Media.VideoFrame.CreateWithSoftwareBitmap(software_bitmap);
                            Console.WriteLine("Load squeezenet.onnx.");
                            using (var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx"))
                            {
                                Console.WriteLine("Create LearningModelSession.");
                                using (var session = new Microsoft.AI.MachineLearning.LearningModelSession(model))
                                {
                                    Console.WriteLine("Create LearningModelBinding.");
                                    var binding = new Microsoft.AI.MachineLearning.LearningModelBinding(session);
                                    Console.WriteLine("Bind data_0.");
                                    binding.Bind("data_0", frame);
                                    Console.WriteLine("Evaluate.");
                                    var results = session.Evaluate(binding, "");
                                }
                                Console.WriteLine("Success!\n");
                            }
                        }
                    })
                .Wait();  
        }

        public static void CreateSessionDeviceDefault()
        {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx");
            Console.WriteLine("Creating LearningModelSession.");
            var session = new Microsoft.AI.MachineLearning.LearningModelSession(model, new LearningModelDevice(LearningModelDeviceKind.Default));
            Console.WriteLine("Created LearningModelSession.");
        }

        public static void CreateSessionDeviceCpu() {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx");
            Console.WriteLine("Creating LearningModelSession.");
            var session = new Microsoft.AI.MachineLearning.LearningModelSession(model, new LearningModelDevice(LearningModelDeviceKind.Cpu));
            Console.WriteLine("Created LearningModelSession.");
        }

        public static void CreateSessionDeviceDirectX()
        {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx");
            Console.WriteLine("Creating LearningModelSession.");
            var session = new Microsoft.AI.MachineLearning.LearningModelSession(model, new LearningModelDevice(LearningModelDeviceKind.DirectX));
            Console.WriteLine("Created LearningModelSession.");
        }

        public static void CreateSessionDeviceDirectXHighPerformance()
        {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx");
            Console.WriteLine("Creating LearningModelSession.");
            var session = new Microsoft.AI.MachineLearning.LearningModelSession(model, new LearningModelDevice(LearningModelDeviceKind.DirectXHighPerformance));
            Console.WriteLine("Created LearningModelSession.");
        }

        public static void CreateSessionDeviceDirectXMinimumPower()
        {
            Console.WriteLine("Running test: " + System.Reflection.MethodBase.GetCurrentMethod().Name);
            var model = Microsoft.AI.MachineLearning.LearningModel.LoadFromFilePath("squeezenet.onnx");
            Console.WriteLine("Creating LearningModelSession.");
            var session = new Microsoft.AI.MachineLearning.LearningModelSession(model, new LearningModelDevice(LearningModelDeviceKind.DirectXMinPower));
            Console.WriteLine("Created LearningModelSession.");
        }

    }
}
