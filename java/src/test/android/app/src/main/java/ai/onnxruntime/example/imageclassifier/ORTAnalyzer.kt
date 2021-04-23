package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*
import kotlin.math.exp


internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    // Get index of top 3 values
    // This is for demo purpose only, there are more efficient algorithms for topK problems
    private fun getTop3(labelVals: FloatArray): List<Int> {
        var indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    // Calculate the SoftMax for the input array
    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max!!)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            var result = Result()

            val imgData = preProcess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            val shape = longArrayOf(1, 3, 224, 224)
            val env = OrtEnvironment.getEnvironment()
            env.use {
                val tensor = OnnxTensor.createTensor(env, imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                tensor.use {
                    val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                    output.use {
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        val rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                        val probabilities = softMax(rawOutput)
                        result.detectedIndices = getTop3(probabilities)
                        for (idx in result.detectedIndices) {
                            result.detectedScore.add(probabilities[idx])
                        }
                    }
                }
            }
            callBack(result)
        }

        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}