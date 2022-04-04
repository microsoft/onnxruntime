package ai.onnxruntime;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class MapUtilTest {

  @Test
  void capacityFromSize() {
    // Capacity remainder below 0.5
    assertEquals(10, MapUtil.capacityFromSize(7));
    // Capacity remainder above 0.5
    assertEquals(11, MapUtil.capacityFromSize(8));
    // Capacity remainder equals 0
    assertEquals(13, MapUtil.capacityFromSize(9));
  }
}
