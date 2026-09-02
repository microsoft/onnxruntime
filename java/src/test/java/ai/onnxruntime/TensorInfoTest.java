package ai.onnxruntime;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TensorInfoTest {
  @Test
  public void testConstructFromJavaArray_UnexpectedType() {
    Object obj = new Object();
    Throwable t =
        Assertions.assertThrows(OrtException.class, () -> TensorInfo.constructFromJavaArray(obj));
    Assertions.assertEquals(
        "Cannot convert class java.lang.Object to a OnnxTensor.", t.getMessage());
  }

  @Test
  public void testConstructFromJavaArray_ScalarType() throws OrtException {
    float obj = 1.0f;
    TensorInfo tensorInfo = TensorInfo.constructFromJavaArray(obj);
    Assertions.assertArrayEquals(new long[0], tensorInfo.shape);
    Assertions.assertEquals(OnnxJavaType.FLOAT, tensorInfo.type);
    Assertions.assertEquals(
        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensorInfo.onnxType);
  }

  @Test
  public void testConstructFromJavaArray_1DArrayOfNonPrimitiveNorString() {
    Object[] obj = new Object[] {new Object(), new Object()};
    Throwable t =
        Assertions.assertThrows(OrtException.class, () -> TensorInfo.constructFromJavaArray(obj));
    Assertions.assertEquals(
        "Cannot create an OnnxTensor from a base type of class java.lang.Object", t.getMessage());
  }

  @Test
  public void testConstructFromJavaArray_NineDimensions() {
    float[][][][][][][][][] obj = new float[1][1][1][1][1][1][1][1][1];
    Throwable t =
        Assertions.assertThrows(OrtException.class, () -> TensorInfo.constructFromJavaArray(obj));
    Assertions.assertEquals(
        "Cannot create an OnnxTensor with more than 8 dimensions. Found 9 dimensions.",
        t.getMessage());
  }

  @Test
  public void testConstructFromJavaArray_RaggedArray() {
    float[][] obj = new float[][] {new float[1], new float[2]};
    Throwable t =
        Assertions.assertThrows(OrtException.class, () -> TensorInfo.constructFromJavaArray(obj));
    Assertions.assertEquals("Supplied array is ragged, expected 1, found 2", t.getMessage());
  }

  @Test
  public void testConstructFromJavaArray_ExtractRecursive() throws OrtException {
    float[][][] obj = new float[3][2][3];
    TensorInfo tensorInfo = TensorInfo.constructFromJavaArray(obj);

    Assertions.assertArrayEquals(new long[] {3, 2, 3}, tensorInfo.shape);
    Assertions.assertEquals(OnnxJavaType.FLOAT, tensorInfo.type);
    Assertions.assertEquals(
        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensorInfo.onnxType);
  }
}
