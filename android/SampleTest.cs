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
      Assert.True(true);
    }
  }
}
