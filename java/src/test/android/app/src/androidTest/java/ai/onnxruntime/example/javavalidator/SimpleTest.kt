package ai.onnxruntime.example.javavalidator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession.SessionOptions
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException
import java.util.*

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @Test
    @Throws(OrtException::class, IOException::class)
    fun runSigmoidModelTest() {
        val env = OrtEnvironment.getEnvironment()
        env.use {
            val opts = SessionOptions()
            opts.use {
                val session = env.createSession(readModel("sigmoid.ort"), opts)
                session.use {
                    val inputName = session.inputNames.iterator().next()
                    val testdata = Array(3) { Array(4) { FloatArray(5) } }
                    val expected = Array(3) { Array(4) { FloatArray(5) } }
                    for (i in 0..2) {
                        for (j in 0..3) {
                            for (k in 0..4) {
                                testdata[i][j][k] = (i + j + k).toFloat()
                                //expected sigmoid output is y = 1.0 / (1.0 + exp(-x))
                                expected[i][j][k] =
                                    (1.0 / (1.0 + kotlin.math.exp(-testdata[i][j][k]))).toFloat()
                            }
                        }
                    }
                    val inputTensor = OnnxTensor.createTensor(env, testdata)
                    inputTensor.use {
                        val output = session.run(Collections.singletonMap(inputName, inputTensor))
                        output.use {
                            val rawOutput = output[0].value as Array<Array<FloatArray>>
                            for (i in 0..2) {
                                for (j in 0..3) {
                                    for (k in 0..4) {
                                        Assert.assertEquals(
                                            rawOutput[i][j][k],
                                            expected[i][j][k],
                                            1e-6.toFloat()
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    @Throws(IOException::class)
    private fun readModel(fileName: String): ByteArray {
        return InstrumentationRegistry.getInstrumentation().context.assets.open(fileName)
            .readBytes()
    }
}
