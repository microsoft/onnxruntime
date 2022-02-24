using System;
using Microsoft.AI.MachineLearning.Tests.Lib.DotNet5_0;

namespace DotNet5_0Tests
{
    public static class ExceptionHandler
    {
        public static void StdTryCatch(Action act)
        {
            try
            {
                Console.WriteLine("\nSTART TEST");
                act();
                Console.WriteLine("SUCCESS");
            }
            catch
            {
                Console.WriteLine("TEST FAILED");
            }
        }

    }
    class Program
    {
        static void Main(string[] args)
        {
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.LoadBindEval();});
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.CreateSessionDeviceDefault(); });
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.CreateSessionDeviceCpu();});
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.CreateSessionDeviceDirectX(); });
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.CreateSessionDeviceDirectXHighPerformance(); });
            ExceptionHandler.StdTryCatch(() => { SessionAPITests.CreateSessionDeviceDirectXMinimumPower(); });
        }
    }
}
