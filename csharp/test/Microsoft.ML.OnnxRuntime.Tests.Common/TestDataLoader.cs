using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Microsoft.ML.OnnxRuntime.Tests
{
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
                inputFile.ReadLine(); //skip the input name
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

            return CreateNamedOnnxValueFromTensorRawData(nodeName, tensor.RawData.ToArray(), metaElementType, intDims);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromTensorRawData(string nodeName, byte[] rawData, TensorElementType elementType, int[] intDims)
        {
            switch (elementType)
            {
                case TensorElementType.Float:
                    return CreateNamedOnnxValueFromRawData<float>(nodeName, rawData, sizeof(float), intDims);
                case TensorElementType.Double:
                    return CreateNamedOnnxValueFromRawData<double>(nodeName, rawData, sizeof(double), intDims);
                case TensorElementType.Int32:
                    return CreateNamedOnnxValueFromRawData<int>(nodeName, rawData, sizeof(int), intDims);
                case TensorElementType.UInt32:
                    return CreateNamedOnnxValueFromRawData<uint>(nodeName, rawData, sizeof(uint), intDims);
                case TensorElementType.Int16:
                    return CreateNamedOnnxValueFromRawData<short>(nodeName, rawData, sizeof(short), intDims);
                case TensorElementType.UInt16:
                    return CreateNamedOnnxValueFromRawData<ushort>(nodeName, rawData, sizeof(ushort), intDims);
                case TensorElementType.Int64:
                    return CreateNamedOnnxValueFromRawData<long>(nodeName, rawData, sizeof(long), intDims);
                case TensorElementType.UInt64:
                    return CreateNamedOnnxValueFromRawData<ulong>(nodeName, rawData, sizeof(ulong), intDims);
                case TensorElementType.UInt8:
                    return CreateNamedOnnxValueFromRawData<byte>(nodeName, rawData, sizeof(byte), intDims);
                case TensorElementType.Int8:
                    return CreateNamedOnnxValueFromRawData<sbyte>(nodeName, rawData, sizeof(sbyte), intDims);
                case TensorElementType.Bool:
                    return CreateNamedOnnxValueFromRawData<bool>(nodeName, rawData, sizeof(bool), intDims);
                case TensorElementType.Float16:
                    return CreateNamedOnnxValueFromRawData<Float16>(nodeName, rawData, sizeof(ushort), intDims);
                case TensorElementType.BFloat16:
                    return CreateNamedOnnxValueFromRawData<BFloat16>(nodeName, rawData, sizeof(ushort), intDims);
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
            //Set buffer size to 4MB
            int readBufferSize = 4194304;
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
                            var map = Onnx.MapProto.Parser.ParseFrom(file);
                            return CreateNamedOnnxValueFromMap(map, nodeName, nodeMeta);
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

        internal static NamedOnnxValue CastAndCreateFromMapKeys(string name, TensorElementType elementType, IList<long> keys)
        {
            switch (elementType)
            {
                case TensorElementType.Float:
                    return CastAndCreateTensor<long, float>(name, keys);
                case TensorElementType.Double:
                    return CastAndCreateTensor<long, double>(name, keys);
                case TensorElementType.Int32:
                    return CastAndCreateTensor<long, int>(name, keys);
                case TensorElementType.UInt32:
                    return CastAndCreateTensor<long, uint>(name, keys);
                case TensorElementType.Int16:
                    return CastAndCreateTensor<long, short>(name, keys);
                case TensorElementType.UInt16:
                    return CastAndCreateTensor<long, ushort>(name, keys);
                case TensorElementType.Int64:
                    return CastAndCreateTensor<long, long>(name, keys);
                case TensorElementType.UInt64:
                    return CastAndCreateTensor<long, ulong>(name, keys);
                case TensorElementType.UInt8:
                    return CastAndCreateTensor<long, byte>(name, keys);
                case TensorElementType.Int8:
                    return CastAndCreateTensor<long, sbyte>(name, keys);
                case TensorElementType.Bool:
                    return CastAndCreateTensor<long, bool>(name, keys);
                case TensorElementType.Float16:
                    return CastAndCreateTensor<long, Float16>(name, keys);
                case TensorElementType.BFloat16:
                    return CastAndCreateTensor<long, BFloat16>(name, keys);
                default:
                    throw new NotImplementedException($"Tensors of type: " + elementType.ToString() +
                        " not currently supported here, use: CreateNamedOnnxValueFromStringTensor.");
            }
        }

        /// <summary>
        /// All the keys in maps are stored as an array of longs, so
        /// to create a real tensor we need to cast to create a continuous buffer
        /// essentially packing it as a raw data.
        /// </summary>
        /// <typeparam name="E"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="name"></param>
        /// <param name="elements"></param>
        /// <returns></returns>
        /// <exception cref="InvalidDataException"></exception>
        internal static NamedOnnxValue CastAndCreateTensor<E, T>(string name, IList<E> elements)
        {
            // Create raw data
            T[] castKeys = new T[elements.Count];
            if (typeof(T) == typeof(Float16) || typeof(T) == typeof(BFloat16))
            {
                for (int i = 0; i < elements.Count; i++)
                {
                    var obj = Convert.ChangeType(elements[i], typeof(ushort));
                    if (obj == null)
                    {
                        throw new InvalidDataException($"Conversion from long to {typeof(T)} failed");
                    }
                    castKeys[i] = (T)obj;
                }
            }
            else
            {
                for (int i = 0; i < elements.Count; i++)
                {
                    var obj = (T)Convert.ChangeType(elements[i], typeof(T));
                    if (obj == null)
                    {
                        throw new InvalidDataException($"Conversion from long to {typeof(T)} failed");
                    }
                    castKeys[i] = (T)obj;
                }
            }
            var tensor = new DenseTensor<T>(castKeys, new int[] { elements.Count });
            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromMap(Onnx.MapProto map, string nodeName, NodeMetadata nodeMetadata)
        {
            // See GH issue https://github.com/onnx/onnx/issues/5072
            throw new NotImplementedException($"Loading map node: '{nodeName}' not implemented yet");

            //var mapMeta = nodeMetadata.AsMapMetadata();

            //if ((TensorElementType)map.KeyType != mapMeta.KeyDataType)
            //{
            //    throw new InvalidDataException($"Node: '{nodeName}' map key type expected: " +
            //                           $"'{mapMeta.KeyDataType}', loaded from test data: '{(TensorElementType)map.KeyType}'");
            //}

            //// temp non-generic(!) container
            //NamedOnnxValue keysTensor;
            //if (mapMeta.KeyDataType == TensorElementType.String)
            //{
            //    keysTensor = CreateNamedOnnxValueFromStringTensor(map.StringKeys, nodeName, new int[] { map.StringKeys.Count });
            //}
            //else
            //{
            //    keysTensor = CastAndCreateFromMapKeys(nodeName, mapMeta.KeyDataType, map.Keys);
            //}

            //switch ((Onnx.SequenceProto.Types.DataType)map.Values.ElemType)
            //{
            //    case Onnx.SequenceProto.Types.DataType.Tensor:
            //        var tensorCount = map.Values.TensorValues.Count;
            //        break;
            //    default:
            //        throw new NotImplementedException("Does not support map value type other than a tensor");
            //}

            //return new NamedOnnxValue(string.Empty, new Object(), OnnxValueType.ONNX_TYPE_UNKNOWN);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromOptional(Onnx.OptionalProto optional, string nodeName, NodeMetadata nodeMetadata)
        {
            var meta = nodeMetadata.AsOptionalMetadata().ElementMeta;
            switch((Onnx.OptionalProto.Types.DataType)optional.ElemType)
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

        internal static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, int[] dimensions)
        {
            T[] typedArr = new T[rawData.Length / elemWidth];
            var typeOf = typeof(T);
            if (typeOf == typeof(Float16) || typeOf == typeof(BFloat16))
            {
                using (var memSrcHandle = new Memory<byte>(rawData).Pin())
                using (var memDstHandle = new Memory<T>(typedArr).Pin())
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(memSrcHandle.Pointer, memDstHandle.Pointer, typedArr.Length * elemWidth, rawData.Length);
                    }
                }
            }
            else
            {
                Buffer.BlockCopy(rawData, 0, typedArr, 0, rawData.Length);
            }
            var dt = new DenseTensor<T>(typedArr, dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromStringTensor(IList<Google.Protobuf.ByteString> strings,
            string nodeName, int[] dimensions)
        {
            string[] strArray = new string[strings.Count];
            for (int i = 0; i < strings.Count; ++i)
            {
                strArray[i] = System.Text.Encoding.UTF8.GetString(strings[i].ToByteArray());
            }

            var dt = new DenseTensor<string>(strArray, dimensions);
            return NamedOnnxValue.CreateFromTensor<string>(nodeName, dt);
        }

        internal static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
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