using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class FixedBufferOnnxValueTests
    {
        [Fact]
        public void TestCreateFromStringTensor()
        {
            var tensor = new DenseTensor<string>(new string[] { "a", "b" }, new int[] { 1, 2 });

            Assert.Throws<ArgumentException>("value", () =>
            {
                FixedBufferOnnxValue.CreateFromTensor(tensor);
            });
        }

    }
}
