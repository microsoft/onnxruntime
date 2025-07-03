using NUnit.Framework;
using OpenQA.Selenium.Remote;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Configuration;
using OpenQA.Selenium.Appium.Android;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Enums;
using OpenQA.Selenium.Appium.iOS;

namespace BrowserStack
{
	public class BrowserStackNUnitTest
	{
		protected AndroidDriver<AndroidElement> driver;
		public BrowserStackNUnitTest() {}

		[SetUp]
		public void Init()
		{
			AppiumOptions appiumOptions = new AppiumOptions();
			appiumOptions.AddAdditionalCapability(MobileCapabilityType.DeviceName, "Samsung Galaxy S20");
			appiumOptions.AddAdditionalCapability(MobileCapabilityType.PlatformName, "Android");
			appiumOptions.AddAdditionalCapability(MobileCapabilityType.PlatformVersion, "10");
			driver = new AndroidDriver<AndroidElement>(new Uri("http://127.0.0.1:4723/wd/hub"), appiumOptions);
		}

		[TearDown]
		public void Cleanup()
		{
			driver.Quit();
		}

	}
}
