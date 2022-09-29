package ai.onnxruntime.example.javavalidator

import ai.onnxruntime.*
import ai.onnxruntime.OrtSession.SessionOptions
import android.os.Build;
import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.microsoft.appcenter.espresso.Factory
import com.microsoft.appcenter.espresso.ReportHelper
import org.junit.*
import org.junit.runner.RunWith
import java.io.IOException
import java.util.*

private const val TAG = "ORTAndroidTest"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    val activityTestRule = ActivityScenarioRule(MainActivity::class.java)

    @get:Rule
    var reportHelper: ReportHelper = Factory.getReportHelper()

    @Before
    fun Start() {
        reportHelper.label("Starting App")
        Log.println(Log.INFO, TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @After
    fun TearDown() {
        reportHelper.label("Stopping App")
    }

    @Test
    fun runSigmoidModelTest() {
        for (intraOpNumThreads in 1..4) {
            runSigmoidModelTestImpl(intraOpNumThreads)
        }
    }

    @Test
    fun runSigmoidModelTestNNAPI() {
        runSigmoidModelTestImpl(1, true)
    }

    @Throws(IOException::class)
    private fun readModel(fileName: String): ByteArray {
        return InstrumentationRegistry.getInstrumentation().context.assets.open(fileName)
            .readBytes()
    }

    @Throws(OrtException::class, IOException::class)
    fun runSigmoidModelTestImpl(intraOpNumThreads: Int, useNNAPI: Boolean = false) {
        reportHelper.label("Start Running Test with intraOpNumThreads=$intraOpNumThreads, useNNAPI=$useNNAPI")
        Log.println(Log.INFO, TAG, "Testing with intraOpNumThreads=$intraOpNumThreads")
        Log.println(Log.INFO, TAG, "Testing with useNNAPI=$useNNAPI")
        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        env.use {
            val opts = SessionOptions()
            opts.setIntraOpNumThreads(intraOpNumThreads)
            if (useNNAPI) {
                if (OrtEnvironment.getAvailableProviders().contains(OrtProvider.NNAPI)) {
                    opts.addNnapi()
                } else {
                    Log.println(Log.INFO, TAG, "NO NNAPI EP available, skip the test")
                    return
                }
            }
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
                            @Suppress("UNCHECKED_CAST")
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
}
