using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    // Copy of the class that is internal in the main package
    public class DisposableListTest<T> : List<T>, IDisposableReadOnlyCollection<T>
        where T : IDisposable
    {
        public DisposableListTest()
        { }

        public DisposableListTest(IEnumerable<T> enumerable) : base(enumerable)
        { }

        public DisposableListTest(int count)
            : base(count)
        { }

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

    internal struct DisposableTestPair<TValue> : IDisposable
        where TValue : IDisposable
    {
        public string Key;
        public TValue Value;
        public DisposableTestPair(string key, TValue value)
        {
            Key = key;
            Value = value;
        }
        public void Dispose()
        {
            Value?.Dispose();
        }
    }

    internal static class TestDataLoader
    {
        internal static byte[] LoadModelFromEmbeddedResource(string path)
        {
            var assembly = typeof(TestDataLoader).Assembly;
            byte[] model = null;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                using (MemoryStream memoryStream = new MemoryStream())
                {
                    stream.CopyTo(memoryStream);
                    model = memoryStream.ToArray();
                }
            }

            return model;
        }

        internal static float[] LoadTensorFromEmbeddedResource(string path)
        {
            var tensorData = new List<float>();
            var assembly = typeof(TestDataLoader).Assembly;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (StreamReader inputFile = new StreamReader(assembly.GetManifestResourceStream(resourceName)))
            {
                inputFile.ReadLine(); // skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static NamedOnnxValue LoadTensorPb(Onnx.TensorProto tensor, string nodeName, NodeMetadata nodeMeta)
        {
            if (nodeMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new InvalidDataException($"Metadata for: '{nodeName}' has a type: '{nodeMeta.OnnxValueType}'" +
                                               $" but loading as tensor: '{tensor.Name}'");
            }

            var protoDt = (Tensors.TensorElementType)tensor.DataType;
            var metaElementType = nodeMeta.ElementDataType;
            if (!((protoDt == metaElementType) ||
                  (protoDt == TensorElementType.UInt16 &&
                   (metaElementType == TensorElementType.BFloat16 || metaElementType == TensorElementType.Float16))))
                throw new InvalidDataException($"For node: '{nodeName}' metadata expects: '{metaElementType}' but loaded loaded tensor type: '{protoDt}'");

            // Tensors within Sequences may have no dimensions as the standard allows
            // different dimensions for each tensor element of the sequence
            if (nodeMeta.Dimensions.Length > 0 && nodeMeta.Dimensions.Length != tensor.Dims.Count)
            {
                throw new InvalidDataException($"node: '{nodeName}' nodeMeta.Dim.Length: {nodeMeta.Dimensions.Length} " +
                                               $"is expected to be equal to tensor.Dims.Count {tensor.Dims.Count}");
            }

            var intDims = new int[tensor.Dims.Count];
            for (int i = 0; i < tensor.Dims.Count; i++)
            {
                intDims[i] = (int)tensor.Dims[i];
            }

            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                if ((nodeMeta.Dimensions[i] != -1) && (nodeMeta.Dimensions[i] != tensor.Dims[i]))
                    throw new InvalidDataException($"Node: '{nodeName}' dimension at idx {i} is {nodeMeta.Dimensions}[{i}] " +
                                                   $"is expected to either be -1 or {tensor.Dims[i]}");
            }

            // element type for Float16 and BFloat16 in the loaded tensor would always be uint16, so
            // we want to use element type from metadata
            if (protoDt == TensorElementType.String)
                return CreateNamedOnnxValueFromStringTensor(tensor.StringData, nodeName, intDims);

            return CreateNamedOnnxValueFromTensorRawData(nodeName, tensor.RawData.Span, metaElementType, intDims);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromTensorRawData(string nodeName, ReadOnlySpan<byte> rawData,
                                                                             TensorElementType elementType, int[] intDims)
        {
            switch (elementType)
            {
                case TensorElementType.Float:
                    return CreateNamedOnnxValueFromRawData<float>(nodeName, rawData, intDims);
                case TensorElementType.Double:
                    return CreateNamedOnnxValueFromRawData<double>(nodeName, rawData, intDims);
                case TensorElementType.Int32:
                    return CreateNamedOnnxValueFromRawData<int>(nodeName, rawData, intDims);
                case TensorElementType.UInt32:
                    return CreateNamedOnnxValueFromRawData<uint>(nodeName, rawData, intDims);
                case TensorElementType.Int16:
                    return CreateNamedOnnxValueFromRawData<short>(nodeName, rawData, intDims);
                case TensorElementType.UInt16:
                    return CreateNamedOnnxValueFromRawData<ushort>(nodeName, rawData, intDims);
                case TensorElementType.Int64:
                    return CreateNamedOnnxValueFromRawData<long>(nodeName, rawData, intDims);
                case TensorElementType.UInt64:
                    return CreateNamedOnnxValueFromRawData<ulong>(nodeName, rawData, intDims);
                case TensorElementType.UInt8:
                    return CreateNamedOnnxValueFromRawData<byte>(nodeName, rawData, intDims);
                case TensorElementType.Int8:
                    return CreateNamedOnnxValueFromRawData<sbyte>(nodeName, rawData, intDims);
                case TensorElementType.Bool:
                    return CreateNamedOnnxValueFromRawData<bool>(nodeName, rawData, intDims);
                case TensorElementType.Float16:
                    return CreateNamedOnnxValueFromRawData<Float16>(nodeName, rawData, intDims);
                case TensorElementType.BFloat16:
                    return CreateNamedOnnxValueFromRawData<BFloat16>(nodeName, rawData, intDims);
                case TensorElementType.String:
                    throw new ArgumentException("For string tensors of type use: CreateNamedOnnxValueFromStringTensor.");
                default:
                    throw new NotImplementedException($"Tensors of type: {elementType} not currently supported by this function");
            }
        }

        internal static NamedOnnxValue LoadTensorFromEmbeddedResourcePb(string path, string nodeName, NodeMetadata nodeMeta)
        {
            Onnx.TensorProto tensor = null;

            var assembly = typeof(TestDataLoader).Assembly;

            using (Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.TestData.{path}"))
            {
                tensor = Onnx.TensorProto.Parser.ParseFrom(stream);
            }

            return LoadTensorPb(tensor, nodeName, nodeMeta);
        }

        internal static NamedOnnxValue LoadOnnxValueFromFilePb(string fullFilename, string nodeName, NodeMetadata nodeMeta)
        {
            // No sparse tensor support yet
            // Set buffer size to 4MB
            const int readBufferSize = 4194304;
            using (var file = new FileStream(fullFilename, FileMode.Open, FileAccess.Read, FileShare.Read, readBufferSize))
            {
                switch (nodeMeta.OnnxValueType)
                {
                    case OnnxValueType.ONNX_TYPE_TENSOR:
                        {
                            var tensor = Onnx.TensorProto.Parser.ParseFrom(file);
                            return LoadTensorPb(tensor, nodeName, nodeMeta);
                        }
                    case OnnxValueType.ONNX_TYPE_SEQUENCE:
                        {
                            var sequence = Onnx.SequenceProto.Parser.ParseFrom(file);
                            return CreateNamedOnnxValueFromSequence(sequence, nodeName, nodeMeta);
                        }
                    case OnnxValueType.ONNX_TYPE_MAP:
                        {
                            throw new NotImplementedException(
                                "Map test data format requires clarification: https://github.com/onnx/onnx/issues/5072");
                        }

                    case OnnxValueType.ONNX_TYPE_OPTIONAL:
                        {
                            var opt = Onnx.OptionalProto.Parser.ParseFrom(file);
                            return CreateNamedOnnxValueFromOptional(opt, nodeName, nodeMeta);
                        }
                    default:
                        throw new NotImplementedException($"Unable to load value type: {nodeMeta.OnnxValueType} not implemented");
                }
            }
        }

        internal static DisposableTestPair<OrtValue> LoadOrtValueFromFilePb(string fullFilename, string nodeName, NodeMetadata nodeMeta)
        {
            // No sparse tensor support yet
            // Set buffer size to 4MB
            const int readBufferSize = 4194304;
            using (var file = new FileStream(fullFilename, FileMode.Open, FileAccess.Read, FileShare.Read, readBufferSize))
            {
                switch (nodeMeta.OnnxValueType)
                {
                    case OnnxValueType.ONNX_TYPE_TENSOR:
                        {
                            var tensor = Onnx.TensorProto.Parser.ParseFrom(file);
                            return new DisposableTestPair<OrtValue>(nodeName, LoadOrValueTensorPb(tensor, nodeName, nodeMeta));
                        }
                    case OnnxValueType.ONNX_TYPE_SEQUENCE:
                        {
                            var sequence = Onnx.SequenceProto.Parser.ParseFrom(file);
                            return new DisposableTestPair<OrtValue>(nodeName, CreateOrtValueFromSequence(sequence, nodeName, nodeMeta));
                        }
                    case OnnxValueType.ONNX_TYPE_MAP:
                        {
                            throw new NotImplementedException(
                                "Map test data format requires clarification: https://github.com/onnx/onnx/issues/5072");
                        }

                    case OnnxValueType.ONNX_TYPE_OPTIONAL:
                        {
                            var opt = Onnx.OptionalProto.Parser.ParseFrom(file);
                            return new DisposableTestPair<OrtValue>(nodeName, CreateOrtValueFromOptional(opt, nodeName, nodeMeta));
                        }
                    default:
                        throw new NotImplementedException($"Unable to load value type: {nodeMeta.OnnxValueType} not implemented");
                }
            }
        }

        private static void SequenceCheckMatchOnnxType(string nodeName, SequenceMetadata meta,
                                                       OnnxValueType onnxType)
        {
            if (meta.ElementMeta.OnnxValueType == onnxType)
                return;

            throw new InvalidDataException($"Sequence node: '{nodeName}' " +
                                           $"has element type: '{onnxType}'" +
                                           $" expected: '{meta.ElementMeta.OnnxValueType}'");
        }

        private static string MakeSequenceElementName(string nodeName, string seqName, int seqNum)
        {
            if (seqName.Length > 0)
                return $"seq.{nodeName}.data.{seqName}.{seqNum}";
            else
                return $"seq.{nodeName}.data._.{seqNum}";
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromSequence(Onnx.SequenceProto sequence, string nodeName, NodeMetadata nodeMeta)
        {
            var sequenceMeta = nodeMeta.AsSequenceMetadata();
            var elemMeta = sequenceMeta.ElementMeta;

            int seqNum = 0;
            var seqElemType = (Onnx.SequenceProto.Types.DataType)sequence.ElemType;
            switch (seqElemType)
            {
                case Onnx.SequenceProto.Types.DataType.Tensor:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_TENSOR);
                        var sequenceOfTensors = new List<NamedOnnxValue>(sequence.TensorValues.Count);
                        foreach (var tensor in sequence.TensorValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            var namedOnnxValue = LoadTensorPb(tensor, elemName, elemMeta);
                            sequenceOfTensors.Add(namedOnnxValue);
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, sequenceOfTensors);
                    }
                case Onnx.SequenceProto.Types.DataType.Sequence:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_SEQUENCE);
                        var seqOfSequences = new List<NamedOnnxValue>(sequence.SequenceValues.Count);
                        foreach (var s in sequence.SequenceValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            seqOfSequences.Add(CreateNamedOnnxValueFromSequence(s, elemName, elemMeta));
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, seqOfSequences);
                    }
                case Onnx.SequenceProto.Types.DataType.Map:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_MAP);
                        var seqOfMaps = new List<NamedOnnxValue>(sequence.MapValues.Count);
                        foreach (var m in sequence.MapValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            seqOfMaps.Add(CreateNamedOnnxValueFromMap(m, elemName, elemMeta));
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, seqOfMaps);
                    }
                case Onnx.SequenceProto.Types.DataType.Optional:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_OPTIONAL);
                        var seqOfOpts = new List<NamedOnnxValue>(sequence.OptionalValues.Count);
                        foreach (var opt in sequence.OptionalValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            seqOfOpts.Add(CreateNamedOnnxValueFromOptional(opt, elemName, elemMeta));
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, seqOfOpts);
                    }
                default:
                    throw new NotImplementedException($"Sequence test data loading does not support element type: " +
                                                      $"'{seqElemType}'");
            }
        }
        internal static NamedOnnxValue CreateNamedOnnxValueFromMap(Onnx.MapProto map, string nodeName, NodeMetadata nodeMetadata)
        {
            // See GH issue https://github.com/onnx/onnx/issues/5072
            throw new NotImplementedException($"Loading map node: '{nodeName}' not implemented yet");
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromOptional(Onnx.OptionalProto optional, string nodeName, NodeMetadata nodeMetadata)
        {
            var meta = nodeMetadata.AsOptionalMetadata().ElementMeta;
            switch ((Onnx.OptionalProto.Types.DataType)optional.ElemType)
            {
                case Onnx.OptionalProto.Types.DataType.Tensor:
                    {
                        var tensor = optional.TensorValue;
                        return LoadTensorPb(tensor, nodeName, meta);
                    }
                case Onnx.OptionalProto.Types.DataType.Sequence:
                    {
                        var sequence = optional.SequenceValue;
                        return CreateNamedOnnxValueFromSequence(sequence, nodeName, meta);
                    }
                case Onnx.OptionalProto.Types.DataType.Map:
                    {
                        var map = optional.MapValue;
                        return CreateNamedOnnxValueFromMap(map, nodeName, meta);
                    }
                case Onnx.OptionalProto.Types.DataType.Optional:
                    throw new NotImplementedException($"Unable to load '{nodeName}' optional contained within optional");
                default:
                    // Test data contains OptionalProto with the contained element type undefined.
                    // the premise is, if the element is not fed as an input, we should not care
                    // what Onnx type it is. However, we do not need to support AFAIK such inputs
                    // since the value for them could never be supplied.
                    throw new NotImplementedException($"Unable to load '{nodeName}' optional element type of: {(Onnx.OptionalProto.Types.DataType)optional.ElemType} type");
            }
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, ReadOnlySpan<byte> rawData,
                                                                          int[] dimensions)
            where T : struct
        {
            var typedSrcSpan = MemoryMarshal.Cast<byte, T>(rawData);
            var dt = new DenseTensor<T>(typedSrcSpan.ToArray(), dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        static OrtValue LoadOrValueTensorPb(Onnx.TensorProto tensor, string nodeName, NodeMetadata nodeMeta)
        {
            if (nodeMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new InvalidDataException($"Metadata for: '{nodeName}' has a type: '{nodeMeta.OnnxValueType}'" +
                                               $" but loading as tensor: {tensor.Name}");
            }

            var protoDt = (Tensors.TensorElementType)tensor.DataType;
            var metaElementType = nodeMeta.ElementDataType;
            if (!((protoDt == metaElementType) ||
                  (protoDt == TensorElementType.UInt16 &&
                   (metaElementType == TensorElementType.BFloat16 || metaElementType == TensorElementType.Float16))))
                throw new InvalidDataException($"For node: '{nodeName}' metadata expects: '{metaElementType}' but loaded loaded tensor type: '{protoDt}'");

            // Tensors within Sequences may have no dimensions as the standard allows
            // different dimensions for each tensor element of the sequence
            if (nodeMeta.Dimensions.Length > 0 && nodeMeta.Dimensions.Length != tensor.Dims.Count)
            {
                throw new InvalidDataException($"node: '{nodeName}' nodeMeta.Dim.Length: {nodeMeta.Dimensions.Length} " +
                                               $"is expected to be equal to tensor.Dims.Count {tensor.Dims.Count}");
            }

            var shape = tensor.Dims.ToArray();

            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                if ((nodeMeta.Dimensions[i] != -1) && (nodeMeta.Dimensions[i] != shape[i]))
                    throw new InvalidDataException($"Node: '{nodeName}' dimension at idx {i} is {nodeMeta.Dimensions}[{i}] " +
                                                   $"is expected to either be -1 or {shape[i]}");
            }

            // element type for Float16 and BFloat16 in the loaded tensor would always be uint16, so
            // we want to use element type from metadata
            if (protoDt == TensorElementType.String)
                return CreateOrtValueFromStringTensor(tensor.StringData, shape);

            return CreateOrtValueFromRawData(OrtAllocator.DefaultInstance, tensor.RawData.Span, metaElementType, shape);
        }

        internal static OrtValue CreateOrtValueFromSequence(Onnx.SequenceProto sequence, string nodeName, NodeMetadata nodeMeta)
        {
            var sequenceMeta = nodeMeta.AsSequenceMetadata();
            var elemMeta = sequenceMeta.ElementMeta;

            int seqNum = 0;
            var seqElemType = (Onnx.SequenceProto.Types.DataType)sequence.ElemType;
            switch (seqElemType)
            {
                case Onnx.SequenceProto.Types.DataType.Tensor:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_TENSOR);
                        using DisposableListTest<OrtValue> sequenceOfTensors = new(sequence.TensorValues.Count);
                        foreach (var tensor in sequence.TensorValues)
                        {
                            var element = LoadOrValueTensorPb(tensor, sequence.Name, elemMeta);
                            sequenceOfTensors.Add(element);
                        }
                        // Will take possession of ortValues in the sequence and will clear this container
                        return OrtValue.CreateSequence(sequenceOfTensors);
                    }
                case Onnx.SequenceProto.Types.DataType.Sequence: // Sequence of sequences
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_SEQUENCE);
                        using DisposableListTest<OrtValue> seqOfSequences = new(sequence.TensorValues.Count);
                        foreach (var s in sequence.SequenceValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            var ortValue = CreateOrtValueFromSequence(s, elemName, elemMeta);
                            seqOfSequences.Add(ortValue);
                        }
                        return OrtValue.CreateSequence(seqOfSequences);
                    }
                case Onnx.SequenceProto.Types.DataType.Map:
                    {
                        throw new NotImplementedException(
                            "Test data format for maps is under investigation");
                    }
                case Onnx.SequenceProto.Types.DataType.Optional:
                    {
                        SequenceCheckMatchOnnxType(nodeName, sequenceMeta, OnnxValueType.ONNX_TYPE_OPTIONAL);
                        using DisposableListTest<OrtValue> seqOfSequences = new(sequence.TensorValues.Count);
                        foreach (var opt in sequence.OptionalValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            var ortValue = CreateOrtValueFromOptional(opt, elemName, elemMeta);
                            seqOfSequences.Add(ortValue);
                        }
                        return OrtValue.CreateSequence(seqOfSequences);
                    }
                default:
                    throw new NotImplementedException($"Sequence test data loading does not support element type: " +
                                                      $"'{seqElemType}'");
            }
        }

        internal static OrtValue CreateOrtValueFromOptional(Onnx.OptionalProto optional, string nodeName, NodeMetadata nodeMetadata)
        {
            var meta = nodeMetadata.AsOptionalMetadata().ElementMeta;
            switch ((Onnx.OptionalProto.Types.DataType)optional.ElemType)
            {
                case Onnx.OptionalProto.Types.DataType.Tensor:
                    {
                        var tensor = optional.TensorValue;
                        return LoadOrValueTensorPb(tensor, nodeName, meta);
                    }
                case Onnx.OptionalProto.Types.DataType.Sequence:
                    {
                        var sequence = optional.SequenceValue;
                        return CreateOrtValueFromSequence(sequence, nodeName, meta);
                    }
                case Onnx.OptionalProto.Types.DataType.Map:
                    {
                        throw new NotImplementedException(
                            "Test data format for maps is under investigation");
                    }
                case Onnx.OptionalProto.Types.DataType.Optional:
                    throw new NotImplementedException($"Unable to load '{nodeName}' optional contained within optional");
                default:
                    // Test data contains OptionalProto with the contained element type undefined.
                    // the premise is, if the element is not fed as an input, we should not care
                    // what Onnx type it is. However, we do not need to support AFAIK such inputs
                    // since the value for them could never be supplied.
                    throw new NotImplementedException($"Unable to load '{nodeName}' optional element type of: {(Onnx.OptionalProto.Types.DataType)optional.ElemType} type");
            }
        }

        internal static OrtValue CreateOrtValueFromRawData(OrtAllocator allocator, ReadOnlySpan<byte> rawData, TensorElementType elementType, long[] shape)
        {
            Debug.Assert(elementType != TensorElementType.String, "Does not support strings");
            var typeInfo = TensorBase.GetElementTypeInfo(elementType);
            Assert.NotNull(typeInfo);

            // ArrayUtilities not accessible in all builds
            var shapeSize = ShapeUtils.GetSizeForShape(shape);
            var inferredSize = rawData.Length / typeInfo.TypeSize;
            Assert.Equal(shapeSize, inferredSize);
            Assert.Equal(0, rawData.Length % typeInfo.TypeSize);

            var ortValue = OrtValue.CreateAllocatedTensorValue(allocator, elementType, shape);
            try
            {
                // The endianess data in protobuf is little endian.
                // We simply copy raw memory into the tensor raw data.
                var span = ortValue.GetTensorMutableRawData();
                Assert.Equal(rawData.Length, span.Length);
                rawData.CopyTo(span);
                return ortValue;
            }
            catch (Exception)
            {
                ortValue.Dispose();
                throw;
            }
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromStringTensor(IList<Google.Protobuf.ByteString> strings,
                                                                            string nodeName, int[] dimensions)
        {
            string[] strArray = new string[strings.Count];
            for (int i = 0; i < strings.Count; ++i)
            {
#if NET6_0_OR_GREATER
                strArray[i] = Encoding.UTF8.GetString(strings[i].Span);
#else
                strArray[i] = Encoding.UTF8.GetString(strings[i].ToByteArray());
#endif
            }

            var dt = new DenseTensor<string>(strArray, dimensions);
            return NamedOnnxValue.CreateFromTensor<string>(nodeName, dt);
        }
        internal static OrtValue CreateOrtValueFromStringTensor(IList<Google.Protobuf.ByteString> strings,
                                                                long[] shape)
        {
            var ortValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
            try
            {
                for (int i = 0; i < strings.Count; ++i)
                {
                    ortValue.StringTensorSetElementAt(strings[i].Span, i);
                }
                return ortValue;
            }
            catch (Exception)
            {
                ortValue.Dispose();
                throw;
            }
        }

        internal static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); // skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }
    }
}
