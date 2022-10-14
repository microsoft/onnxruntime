package com.example.reactnativeonnxruntimemodule;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.test.runner.screenshot.BasicScreenCaptureProcessor;
import androidx.test.runner.screenshot.ScreenCapture;
import androidx.test.runner.screenshot.Screenshot;

import org.junit.rules.TestWatcher;
import org.junit.runner.Description;

import java.io.IOException;

import java.util.Collections;

public class ScreenshottingTestWatcher extends TestWatcher {
    private static final String TAG = ScreenshottingTestWatcher.class.getSimpleName();

    private final class ScreenshotProcessor extends BasicScreenCaptureProcessor {
        @Override
        public String process(ScreenCapture capture) throws IOException {
            String captureFileName = super.process(capture);
            Log.d(TAG, "Screenshot file name: " + captureFileName);
            return captureFileName;
        }
    }

    @Override
    protected void failed(Throwable e, Description description) {
        captureScreenshot(description, "failed");
    }

    @Override
    protected void succeeded(Description description) {
        captureScreenshot(description, "succeeded");
    }

    @Override
    protected void starting(Description description) {
        captureScreenshot(description, "starting");
    }

    private void captureScreenshot(Description description, String reason) {
        String captureName = description.getClassName() + "." + description.getMethodName() + "-" + reason;
        Log.d(TAG, "Capturing screenshot: " + captureName);

        ScreenCapture capture = Screenshot.capture();
        capture.setFormat(Bitmap.CompressFormat.JPEG);
        capture.setName(captureName);

        try {
            capture.process(Collections.singleton(new ScreenshotProcessor()));
        } catch (IOException e) {
            Log.e(TAG, "Failed to save screen capture.", e);
        }
    }
}
