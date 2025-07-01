using OpenQA.Selenium.Appium;
using OpenQA.Selenium;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.OnnxRuntime.Tests.BrowserStack.Android
{
    /// <summary>
    /// This class contains a single test: RunAll, which interacts with the UI from
    /// https://github.com/mattleibow/DeviceRunners/tree/main by clicking the "Run All" button and checking the number
    /// of passed and failed tests.
    ///
    /// It searches for elements on the page using Appium's WebDriver. These searches use the XPath attributes.
    ///
    /// Launching the MAUI test app in Appium Inspector will allow you to see the exact XPath attributes for each
    /// element.
    /// </summary>
    [TestFixture]
    public class RunAllTest : BrowserStackTest
    {
        public AppiumElement FindAppiumElement(String xpathQuery, String text)
        {
            IReadOnlyCollection<AppiumElement> appiumElements = driver.FindElements(By.XPath(xpathQuery));

            foreach (var element in appiumElements)
            {
                if (element.Text.Contains(text))
                {
                    return element;
                }
            }

            Assert.Fail("Could not find element with xpath: {0} and text: {1}", xpathQuery, text);
      // was unable to find given element
      throw new Exception(String.Format("Could not find {0}: {1} on the page.", xpathQuery, text));
        }

        public AppiumElement FindAppiumElementThenClick(String xpathQuery, String text)
        {
            AppiumElement appiumElement = FindAppiumElement(xpathQuery, text);
      appiumElement.Click();
            return appiumElement;
        }

        public (int, int) GetPassFailCount()
        {
            int numPassed = -1;
            int numFailed = -1;

            IReadOnlyCollection<AppiumElement> labelElements =
                driver.FindElements(By.XPath("//android.widget.TextView"));

            for (int i = 0; i < labelElements.Count; i++)
            {
                AppiumElement element = labelElements.ElementAt(i);

                if (element.Text.Equals("✔"))
                {
                    i++;
                    numPassed = int.Parse(labelElements.ElementAt(i).Text);
                }

                if (element.Text.Equals("⛔"))
                {
                    i++;
                    numFailed = int.Parse(labelElements.ElementAt(i).Text);
                    break;
                }
            }

            Assert.That(numPassed, Is.GreaterThanOrEqualTo(0), "Could not find number passed label.");
            Assert.That(numFailed, Is.GreaterThanOrEqualTo(0), "Could not find number failed label.");

            return (numPassed, numFailed);
        }

        [Test]
        public async Task ClickRunAllTest()
        {
            await Task.Delay(500);

      // XAML for the main page:
      // https://github.com/mattleibow/DeviceRunners/blob/cba7644e07b305ba64dc930b01c3eee55ef2b93d/src/DeviceRunners.VisualRunners.Maui/App/Pages/HomePage.xaml
      AppiumElement runAllButton = FindAppiumElementThenClick("//android.widget.Button", "Run All");

            while (!runAllButton.Enabled)
            {
                // waiting for unit tests to execute
                await Task.Delay(500);
            }

            (int numPassed, int numFailed) = GetPassFailCount();

            if (numFailed == 0)
            {
                return;
            }

            // click into test results if tests have failed
            FindAppiumElementThenClick("//android.widget.TextView", "⛔");
            await Task.Delay(500);

            // Brings you to the test assembly page
            // XAML for test assembly page:
            // https://github.com/mattleibow/DeviceRunners/blob/cba7644e07b305ba64dc930b01c3eee55ef2b93d/src/DeviceRunners.VisualRunners.Maui/App/Pages/TestAssemblyPage.xaml
            FindAppiumElementThenClick("//android.widget.EditText", "All");
            await Task.Delay(100);
            FindAppiumElementThenClick("//android.widget.TextView", "Failed");
            await Task.Delay(500);

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("PASSED TESTS: " + numPassed + " | FAILED TESTS: " + numFailed);

            IReadOnlyCollection<AppiumElement> textResults = driver.FindElements(By.XPath("//android.widget.TextView"));
            foreach (var element in textResults)
            {
                sb.AppendLine(element.Text);
            }

            Assert.That(numFailed, Is.EqualTo(0), sb.ToString());
        }
    }
}
