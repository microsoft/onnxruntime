package ml.microsoft.onnxruntime.sample;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ml.microsoft.onnxruntime.Allocator;
import ml.microsoft.onnxruntime.AllocatorInfo;
import ml.microsoft.onnxruntime.AllocatorType;
import ml.microsoft.onnxruntime.Env;
import ml.microsoft.onnxruntime.LoggingLevel;
import ml.microsoft.onnxruntime.MemType;
import ml.microsoft.onnxruntime.OrtException;
import ml.microsoft.onnxruntime.RunOptions;
import ml.microsoft.onnxruntime.Session;
import ml.microsoft.onnxruntime.SessionOptions;
import ml.microsoft.onnxruntime.TensorElementDataType;
import ml.microsoft.onnxruntime.TensorTypeAndShapeInfo;
import ml.microsoft.onnxruntime.Value;

public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    try {
      Env env = new Env(LoggingLevel.INFO, "test");
      RunOptions runOptions = new RunOptions();
      String modelPath = "/data/local/tmp/squeezenet.onnx";
      SessionOptions sessionOptionsWithNnapi = new SessionOptions();
      sessionOptionsWithNnapi.appendNnapiExecutionProvider();
      Session sessionWithNnapi = new Session(env, modelPath, sessionOptionsWithNnapi);

      AllocatorInfo allocatorInfo = AllocatorInfo.createCpu(AllocatorType.ARENA_ALLOCATOR, MemType.DEFAULT);
      final TensorTypeAndShapeInfo inputInfo =
          sessionWithNnapi.getInputTypeInfo(0).getTensorTypeAndShapeInfo();
      final long input_size = inputInfo.getElementCount();
      final long[] shape = inputInfo.getShape();
      ByteBuffer byteBuffer = ByteBuffer.allocateDirect((int) input_size * 4);
      byteBuffer.order(ByteOrder.nativeOrder());
      for (int i = 0; i < input_size; i++) {
        byteBuffer.putFloat(((float) i) / (input_size + 1));
      }
      Value value = Value.createTensor(allocatorInfo, byteBuffer, shape, TensorElementDataType.FLOAT);

      Allocator allocator = Allocator.createDefault();
      String input = sessionWithNnapi.getInputName(0, allocator);
      String output = sessionWithNnapi.getOutputName(0, allocator);

      Log.d(TAG, "onCreate: nnapi start");
      for (int i = 0; i < 100; i++) {
        Value[] output_values = sessionWithNnapi.run(runOptions, new String[] {input}, new Value[] {value}, new String[] {output});
      }
      Log.d(TAG, "onCreate: nnapi end");
      SessionOptions sessionOptionsWithCpu = new SessionOptions();
      Session sessionWithCpu = new Session(env, modelPath, sessionOptionsWithCpu);
      Log.d(TAG, "onCreate: cpu start");
      for (int i = 0; i < 100; i++) {
        Value[] outputValues = sessionWithCpu.run(runOptions, new String[] {input}, new Value[] {value}, new String[] {output});
      }
      Log.d(TAG, "onCreate: cpu end");
      Value[] outputValuesNnapi = sessionWithNnapi.run(runOptions, new String[] {input}, new Value[] {value}, new String[] {output});
      Value[] outputValuesCpu = sessionWithCpu.run(runOptions, new String[] {input}, new Value[] {value}, new String[] {output});
      ByteBuffer byteBufferNnapi = outputValuesNnapi[0].getTensorMutableData().order(ByteOrder.nativeOrder());
      ByteBuffer byteBufferCpu = outputValuesCpu[0].getTensorMutableData().order(ByteOrder.nativeOrder());
      if (byteBufferCpu.capacity() != byteBufferNnapi.capacity()) {
        Log.e(TAG, "onCreate: capacity not the same");
      }
      for (int i = 0; byteBufferCpu.hasRemaining(); i++) {
        final float cpuValue = byteBufferCpu.getFloat();
        final float nnapiValue = byteBufferNnapi.getFloat();
        if (Math.abs(nnapiValue - cpuValue) > 1e-4) {
          Log.e(TAG, "onCreate: value not the same, cpu: " + cpuValue + ", nnapi: " + nnapiValue);
        }
        if (i < 5) {
          Log.d(TAG, "onCreate: " + cpuValue);
        }
      }
    } catch (OrtException e) {
      Log.e(TAG, "onCreate: ", e);
    }
  }
}
