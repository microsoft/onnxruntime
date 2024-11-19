package ai.onnxruntime.example.javavalidator

import android.os.Bundle
import android.system.Os
import androidx.appcompat.app.AppCompatActivity

/*Empty activity app mainly used for testing*/
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        if (BuildConfig.IS_QNN_BUILD) {
            val adspLibraryPath = applicationContext.applicationInfo.nativeLibraryDir
            // set the path variable to the native library directory
            // so that any native libraries downloaded as dependencies
            // (like qnn libs) are found
            Os.setenv("ADSP_LIBRARY_PATH", adspLibraryPath, true)
        }
        super.onCreate(savedInstanceState)
    }
}
