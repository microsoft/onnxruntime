/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** */
public class UtilTest {

  @Test
  public void reshapeTest() {
    float[] input =
        new float[] {
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29
        };

    long[] candidateShape = new long[] {1, 3, 2, 5};

    float[][][][] reshapedArray = (float[][][][]) OrtUtil.reshape(input, candidateShape);

    float[] firstTestArray = new float[] {0, 1, 2, 3, 4};
    float[] secondTestArray = new float[] {5, 6, 7, 8, 9};
    float[] thirdTestArray = new float[] {10, 11, 12, 13, 14};
    float[] fourthTestArray = new float[] {15, 16, 17, 18, 19};
    float[] fifthTestArray = new float[] {20, 21, 22, 23, 24};
    float[] sixthTestArray = new float[] {25, 26, 27, 28, 29};

    Assertions.assertArrayEquals(firstTestArray, reshapedArray[0][0][0]);
    Assertions.assertArrayEquals(secondTestArray, reshapedArray[0][0][1]);
    Assertions.assertArrayEquals(thirdTestArray, reshapedArray[0][1][0]);
    Assertions.assertArrayEquals(fourthTestArray, reshapedArray[0][1][1]);
    Assertions.assertArrayEquals(fifthTestArray, reshapedArray[0][2][0]);
    Assertions.assertArrayEquals(sixthTestArray, reshapedArray[0][2][1]);
  }

  @Test
  public void reshape4DTest() {
    float[] input =
        new float[] {
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
        };

    long[] candidateShape = new long[] {3, 3, 2, 2};

    float[][][][] reshapedArray = (float[][][][]) OrtUtil.reshape(input, candidateShape);

    float[] oneTestArray = new float[] {0, 1};
    float[] twoTestArray = new float[] {2, 3};
    float[] threeTestArray = new float[] {4, 5};
    float[] fourTestArray = new float[] {6, 7};
    float[] fiveTestArray = new float[] {8, 9};
    float[] sixTestArray = new float[] {10, 11};
    float[] sevenTestArray = new float[] {12, 13};
    float[] eightTestArray = new float[] {14, 15};
    float[] nineTestArray = new float[] {16, 17};
    float[] tenTestArray = new float[] {18, 19};
    float[] elevenTestArray = new float[] {20, 21};
    float[] twelveTestArray = new float[] {22, 23};
    float[] thirteenTestArray = new float[] {24, 25};
    float[] fourteenTestArray = new float[] {26, 27};
    float[] fifteenTestArray = new float[] {28, 29};
    float[] sixteenTestArray = new float[] {30, 31};
    float[] seventeenTestArray = new float[] {32, 33};
    float[] eighteenTestArray = new float[] {34, 35};

    Assertions.assertArrayEquals(oneTestArray, reshapedArray[0][0][0]);
    Assertions.assertArrayEquals(twoTestArray, reshapedArray[0][0][1]);
    Assertions.assertArrayEquals(threeTestArray, reshapedArray[0][1][0]);
    Assertions.assertArrayEquals(fourTestArray, reshapedArray[0][1][1]);
    Assertions.assertArrayEquals(fiveTestArray, reshapedArray[0][2][0]);
    Assertions.assertArrayEquals(sixTestArray, reshapedArray[0][2][1]);
    Assertions.assertArrayEquals(sevenTestArray, reshapedArray[1][0][0]);
    Assertions.assertArrayEquals(eightTestArray, reshapedArray[1][0][1]);
    Assertions.assertArrayEquals(nineTestArray, reshapedArray[1][1][0]);
    Assertions.assertArrayEquals(tenTestArray, reshapedArray[1][1][1]);
    Assertions.assertArrayEquals(elevenTestArray, reshapedArray[1][2][0]);
    Assertions.assertArrayEquals(twelveTestArray, reshapedArray[1][2][1]);
    Assertions.assertArrayEquals(thirteenTestArray, reshapedArray[2][0][0]);
    Assertions.assertArrayEquals(fourteenTestArray, reshapedArray[2][0][1]);
    Assertions.assertArrayEquals(fifteenTestArray, reshapedArray[2][1][0]);
    Assertions.assertArrayEquals(sixteenTestArray, reshapedArray[2][1][1]);
    Assertions.assertArrayEquals(seventeenTestArray, reshapedArray[2][2][0]);
    Assertions.assertArrayEquals(eighteenTestArray, reshapedArray[2][2][1]);
  }

  @Test
  void capacityFromSize() {
    // Capacity remainder below 0.5
    Assertions.assertEquals(10, OrtUtil.capacityFromSize(7));
    // Capacity remainder above 0.5
    Assertions.assertEquals(11, OrtUtil.capacityFromSize(8));
    // Capacity remainder equals 0
    Assertions.assertEquals(13, OrtUtil.capacityFromSize(9));
  }
}
