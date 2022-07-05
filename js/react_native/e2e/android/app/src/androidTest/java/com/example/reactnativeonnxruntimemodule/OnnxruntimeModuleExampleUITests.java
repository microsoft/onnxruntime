package com.example.reactnativeonnxruntimemodule;

import android.view.View;
import android.widget.TextView;

import androidx.test.espresso.NoMatchingViewException;
import androidx.test.espresso.UiController;
import androidx.test.espresso.ViewAction;
import androidx.test.espresso.ViewInteraction;
import androidx.test.filters.LargeTest;
import androidx.test.rule.ActivityTestRule;
import androidx.test.runner.AndroidJUnit4;

import org.hamcrest.Matcher;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.matcher.ViewMatchers.isAssignableFrom;
import static androidx.test.espresso.matcher.ViewMatchers.isDisplayed;
import static androidx.test.espresso.matcher.ViewMatchers.withContentDescription;
import static org.hamcrest.Matchers.allOf;

@RunWith(AndroidJUnit4.class)
@LargeTest
public class OnnxruntimeModuleExampleUITests {
    @Rule
    public ActivityTestRule<MainActivity> activityTestRule = new ActivityTestRule<>(MainActivity.class);

    @Test
    public void testExample() {
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
        Assert.assertEquals(getText(view), "Result: 3");
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
}
