package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.ActivityTestRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert
import java.io.IOException
import java.io.InputStream
import java.util.*


@RunWith(AndroidJUnit4::class)
class ClassifierTest{

    @get:Rule
    val activityRule = ActivityTestRule(MainActivity::class.java)

    private val inputModelName = "mobilenet_v2_float.ort"
    private val inputImageName = "fox.jpg"

    @Test
    fun classificationSampleTest() {
        val mainActivity = activityRule.activity
        Log.i("Test", "Start running check classification result test")

        val imgBitmap = loadImage(inputImageName)
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val imgData = preProcess(rawBitmap)

        val labelData: List<String> by lazy { mainActivity.readLabels() }

        val ortEnv = OrtEnvironment.getEnvironment()
        ortEnv.use {
            val opts = SessionOptions()
            opts.setOptimizationLevel(SessionOptions.OptLevel.BASIC_OPT)
            Log.i("Test", "Loading Model:" + inputModelName)
            val session = ortEnv.createSession(mainActivity.readModel(), opts)
            val inputName = session?.inputNames?.iterator()?.next()
            val shape = longArrayOf(1, 3, 224, 224)
            val tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
            tensor.use {
                val output = session?.run(Collections.singletonMap(inputName, tensor))
                val outputProbs = output!![0].value as Array<FloatArray>
                val probs = outputProbs[0]
                val predIdx = predict(probs)
                val actual = labelData[predIdx]
                //Log.i("Test", "DetectedItem:$actual")
                val detectedScore = probs[predIdx]
                //Log.i("Test", "DetectedTopScore:$detectedScore")
                val expected = "red fox"
                Assert.assertEquals("Expected Item is equal to actual detected item.", expected, actual)
            }
        }
    }

    private fun loadImage(fileName: String): Bitmap {
        val assetManager = InstrumentationRegistry.getInstrumentation().context.assets
        var inputStream: InputStream? = null
        try {
            inputStream = assetManager.open(fileName)
        } catch (e: IOException) {
            Log.e("Test", "Load Test Image Failed.")
        }
        return BitmapFactory.decodeStream(inputStream)
    }

    private fun predict(probabilities: FloatArray): Int {
        var maxVal = Float.MIN_VALUE
        var idx = 0
        for (i in probabilities.indices) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i]
                idx = i
            }
        }
        return idx
    }
}