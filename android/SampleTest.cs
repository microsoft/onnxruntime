using System;
using System.Threading;
using System.Collections.ObjectModel;
using NUnit.Framework;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Android;
using OpenQA.Selenium.Support.UI;

namespace BrowserStack
{
  [TestFixture]
  [Category("sample-test")]
  public class SampleTest : BrowserStackNUnitTest
  {
    public SampleTest() : base(){}

    [Test]
    public void searchWikipedia()
    {
      AndroidElement searchElement = (AndroidElement)new WebDriverWait(driver, TimeSpan.FromSeconds(30)).Until(ExpectedConditions.ElementToBeClickable(MobileBy.AccessibilityId("Search Wikipedia")));
      searchElement.Click();
      AndroidElement insertTextElement = (AndroidElement)new WebDriverWait(driver, TimeSpan.FromSeconds(30)).Until(ExpectedConditions.ElementToBeClickable(By.Id("org.wikipedia.alpha:id/search_src_text")));
      insertTextElement.SendKeys("BrowserStack");
      Thread.Sleep(5000);

      ReadOnlyCollection<AndroidElement> allProductsName = driver.FindElements(By.ClassName("android.widget.TextView"));
      Assert.True(allProductsName.Count > 0);
    }
  }
}
