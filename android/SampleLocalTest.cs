using System;
using System.Threading;
using System.Collections.ObjectModel;
using NUnit.Framework;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium.Android;
using OpenQA.Selenium.Support.UI;

namespace BrowserStack
{
    [TestFixture]
    [Category("sample-local-test")]
    public class LocalTest : BrowserStackNUnitTest
    {
        public LocalTest() : base() {}

        [Test]
        public void testLocal()
        {
            Assert.True(true);
        }
    }
}
