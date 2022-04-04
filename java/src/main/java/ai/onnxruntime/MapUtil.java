package ai.onnxruntime;

final class MapUtil {

  private MapUtil() {
    throw new UnsupportedOperationException(
        "MapUtil is a utility class and should not be instantiated");
  }

  static int capacityFromSize(int size) {
    // 0.75 is the default JDK load factor
    return (int) (size / 0.75 + 1);
  }
}
