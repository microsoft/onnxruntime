package ai.onnxruntime.example.javavalidator

import android.os.Bundle
import android.system.Os
import androidx.appcompat.app.AppCompatActivity

/*Empty activity app mainly used for testing*/
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        val adspLibraryPath = applicationContext.applicationInfo.nativeLibraryDir
        // set the ADSP_LIBRARY_PATH environment variable to the native library directory
        // so that the any native libraries downloaded as dependenciesthat the app may
        // use (like qnn libs) are found
        //Os.setenv("ADSP_LIBRARY_PATH", adspLibraryPath, true)
        super.onCreate(savedInstanceState)
    }
}
