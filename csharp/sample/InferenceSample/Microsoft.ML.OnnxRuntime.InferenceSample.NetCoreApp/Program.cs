// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Microsoft.ML.OnnxRuntime.InferenceSample;

namespace CSharpUsage
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Using API");
            using (var inferenceSampleApi = new InferenceSampleApi())
            {
                inferenceSampleApi.Execute();
            }
            Console.WriteLine("Done");
        }
    }
}