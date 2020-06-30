using System;
using System.IO;

using Microsoft.AI.MachineLearning;
using WinRT;

namespace Microsoft.AI.MachineLearning.Tests
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Load squeezenet.onnx.");
            using (var model = LearningModel.LoadFromFilePath("squeezenet.onnx"))
            {
               Console.WriteLine("Load kitten_224.png as StorageFile.");
               var name = AppDomain.CurrentDomain.BaseDirectory + "kitten_224.png";
               var image_task = Windows.Storage.StorageFile.GetFileFromPathAsync(name);
               image_task.AsTask().Wait();
               var image = image_task.GetResults();
               Console.WriteLine("Load StorageFile into Stream.");
               var stream_task = image.OpenReadAsync();
               System.Threading.Thread.Sleep(1000);
               // stream_task.AsTask().Wait();
               //
               // Unable to call AsTask on IAsyncOperation<IRandomAccessStreamWithContentType>... 
               // System.TypeInitializationException: 'The type initializer for 'ABI.Windows.Foundation.AsyncOperationCompletedHandler`1' threw an exception.'
               // This exception was originally thrown at this call stack:
               //   System.RuntimeType.ThrowIfTypeNeverValidGenericArgument(System.RuntimeType)
               //   System.RuntimeType.SanityCheckGenericArguments(System.RuntimeType[], System.RuntimeType[])
               //   System.RuntimeType.MakeGenericType(System.Type[])
               //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeNewDelegate(System.Type[])
               //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeDelegateType(System.Type[])
               //   ABI.Windows.Foundation.AsyncOperationCompletedHandler<TResult>.AsyncOperationCompletedHandler()
               // 
               // So sleep instead...
               using (var stream = stream_task.GetResults())
               {
                   Console.WriteLine("Create SoftwareBitmap from decoded Stream.");
                   var decoder_task = Windows.Graphics.Imaging.BitmapDecoder.CreateAsync(stream);
                   System.Threading.Thread.Sleep(1000);
                   // decoder_task.AsTask().Wait();
                   //
                   // Unable to call AsTask on IAsyncOperation<SoftwareBitmap>... 
                   // System.TypeInitializationException: 'The type initializer for 'ABI.Windows.Foundation.AsyncOperationCompletedHandler`1' threw an exception.'
                   // This exception was originally thrown at this call stack:
                   //   System.RuntimeType.ThrowIfTypeNeverValidGenericArgument(System.RuntimeType)
                   //   System.RuntimeType.SanityCheckGenericArguments(System.RuntimeType[], System.RuntimeType[])
                   //   System.RuntimeType.MakeGenericType(System.Type[])
                   //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeNewDelegate(System.Type[])
                   //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeDelegateType(System.Type[])
                   //   ABI.Windows.Foundation.AsyncOperationCompletedHandler<TResult>.AsyncOperationCompletedHandler()
                   // 
                   // So sleep instead...
                   var decoder = decoder_task.GetResults();
                   var software_bitmap_task = decoder.GetSoftwareBitmapAsync();
                   System.Threading.Thread.Sleep(1000);
                   // software_bitmap_task.AsTask().Wait();
                   //
                   // Unable to call AsTask on IAsyncOperation<SoftwareBitmap>... 
                   // System.TypeInitializationException: 'The type initializer for 'ABI.Windows.Foundation.AsyncOperationCompletedHandler`1' threw an exception.'
                   // This exception was originally thrown at this call stack:
                   //   System.RuntimeType.ThrowIfTypeNeverValidGenericArgument(System.RuntimeType)
                   //   System.RuntimeType.SanityCheckGenericArguments(System.RuntimeType[], System.RuntimeType[])
                   //   System.RuntimeType.MakeGenericType(System.Type[])
                   //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeNewDelegate(System.Type[])
                   //   System.Linq.Expressions.Compiler.DelegateHelpers.MakeDelegateType(System.Type[])
                   //   ABI.Windows.Foundation.AsyncOperationCompletedHandler<TResult>.AsyncOperationCompletedHandler()
                   // 
                   // So sleep instead...
                   using (var software_bitmap = software_bitmap_task.GetResults())
                   {
                       Console.WriteLine("Create VideoFrame.");
                       var frame = Windows.Media.VideoFrame.CreateWithSoftwareBitmap(software_bitmap);

                       Console.WriteLine("Create LearningModelSession.");
                       using (var session = new LearningModelSession(model))
                       {
                           Console.WriteLine("Create LearningModelBinding.");
                           var binding = new LearningModelBinding(session);
                           Console.WriteLine("Bind data_0.");
                           binding.Bind("data_0", frame);
                           Console.WriteLine("Evaluate.");
                           var results = session.Evaluate(binding, "");
                       }
                       Console.WriteLine("Success!\n");
                   }
               }
            }
        }
    }
}
