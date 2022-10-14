package com.example.reactnativeonnxruntimemodule;

import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.test.espresso.NoMatchingViewException;
import androidx.test.espresso.UiController;
import androidx.test.espresso.ViewAction;
import androidx.test.espresso.ViewInteraction;
import androidx.test.espresso.util.TreeIterables;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.filters.LargeTest;

import org.hamcrest.Matcher;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.matcher.ViewMatchers.isAssignableFrom;
import static androidx.test.espresso.matcher.ViewMatchers.isDisplayed;
import static androidx.test.espresso.matcher.ViewMatchers.isRoot;
import static androidx.test.espresso.matcher.ViewMatchers.withContentDescription;
import static org.hamcrest.Matchers.allOf;

@RunWith(AndroidJUnit4.class)
@LargeTest
public class OnnxruntimeModuleExampleUITests {
    public static final String TAG = OnnxruntimeModuleExampleUITests.class.getSimpleName();

    @Rule
    public ActivityScenarioRule<MainActivity> activityScenarioRule = new ActivityScenarioRule<>(MainActivity.class);

    @Test
    public void testExample() {
        try {
            // Wait for a view displayed
            int waitTime = 0;
            final int sleepTime = 1000;
            do {
                try {
                    ViewInteraction view = onView(allOf(withContentDescription("output"), isDisplayed()));
                    if (getText(view) != null) {
                        break;
                    }
                } catch (NoMatchingViewException ne) {
                    try {
                        Thread.sleep(sleepTime);
                    } catch (InterruptedException ie) {
                    }
                    waitTime += sleepTime;
                }
            } while (waitTime < 180000);

            ViewInteraction view = onView(allOf(withContentDescription("output"), isDisplayed()));
            Assert.assertEquals("Result: 3", getText(view));
        } finally {
            dumpRootViewHierarchy();
        }
    }

    private String getText(ViewInteraction matcher) {
        final String[] text = {null};

        matcher.perform(new ViewAction() {
            @Override
            public Matcher<View> getConstraints() {
                return isAssignableFrom(TextView.class);
            }

            @Override
            public String getDescription() {
                return "get a text from a TextView";
            }

            @Override
            public void perform(UiController uiController, View view) {
                TextView textView = (TextView)view;
                text[0] = textView.getText().toString();
            }
        });

        return text[0];
    }

    private static void dumpRootViewHierarchy() {
        ViewInteraction rootViewInteraction = onView(isRoot());
        rootViewInteraction.perform(new ViewAction() {
            @Override
            public Matcher<View> getConstraints() {
                return isRoot();
            }

            @Override
            public String getDescription() {
                return "dump view hierarchy";
            }

            @Override
            public void perform(UiController uiController, View view) {
                if (view == null) {
                    Log.w(TAG, "view is null, unable to dump view hierarchy");
                    return;
                }

                Log.d(TAG, "view hierarchy:");
                for (TreeIterables.ViewAndDistance viewAndDistance : TreeIterables.depthFirstViewTraversalWithDistance(view)) {
                    StringBuilder builder = new StringBuilder();
                    for (int i = 0; i < viewAndDistance.getDistanceFromRoot(); i++) builder.append(" ");
                    builder.append(viewAndDistance.getView().toString());
                    Log.d(TAG, builder.toString());
                }
            }
        });
    }
}
