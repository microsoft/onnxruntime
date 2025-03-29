using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
  internal static class AssertUtils
  {

    /// <summary>
    /// Check if the action throws the expected exception. If it doesn't, the method passes. If it does, check for
    /// the exception type and the expected exception message. More detailed Assert method to be used for unit tests
    /// written with XUnit.
    /// </summary>
    /// <typeparam name="T">Type of exception expected to be thrown.</typeparam>
    /// <param name="action">Action to be executed or tested.</param>
    /// <param name="feedbackMessage">Feedback message if an unexpected exception happens.</param>
    /// <param name="expectedExceptionMessage">Expected exception message. If null, the exception message is not
    // checked.</param>
    public static void IfThrowsCheckException<T>(Action action, string feedbackMessage, string expectedExceptionMessage = null) where T : Exception
    {
      try
      {
        action();
      }
      catch (T ex)
      {
        if (expectedExceptionMessage == null)
        {
          return;
        }
        else
        {
          Assert.True(ex.Message.Contains(expectedExceptionMessage),
            $"{feedbackMessage}\nExpected exception message to contain '{expectedExceptionMessage}', but got '{ex.Message}'");
        }
      }
      catch (Exception ex)
      {
        Assert.Fail($"{feedbackMessage}\nExpected {typeof(T).Name} but got {ex.GetType().Name}. ");
      }
    }


    /// <summary>
    /// Check if the action throws the expected exception. If it doesn't, the method fails with the feedbackMessage.
    /// If it does, check for the exception type and the expected exception message. More detailed Assert method to be
    /// used for unit tests written with XUnit.
    /// </summary>
    /// <typeparam name="T">Type of exception expected to be thrown.</typeparam>
    /// <param name="action">Action to be executed or tested. It is expected that the action will throw.</param>
    /// <param name="feedbackMessage">Feedback message if an unexpected exception happens.</param>
    /// <param name="expectedExceptionMessage">Expected exception message. If null, the exception message is not
    // checked.</param>
    public static void AssertThrowsCheckException<T>(Action action, string feedbackMessage, string expectedExceptionMessage = null) where T : Exception
    {
      try
      {
        action();
        Assert.Fail($"{feedbackMessage}\nExpected {typeof(T).Name} but no exception was thrown.");
      }
      catch (T ex)
      {
        if (expectedExceptionMessage == null)
        {
          return;
        }
        else
        {
          Assert.True(ex.Message.Contains(expectedExceptionMessage),
            $"{feedbackMessage}\nExpected exception message to contain '{expectedExceptionMessage}', but got '{ex.Message}'");
        }
      }
      catch (Exception ex)
      {
        Assert.Fail($"{feedbackMessage}\nExpected {typeof(T).Name} but got {ex.GetType().Name}. ");
      }
    }
  }
}
