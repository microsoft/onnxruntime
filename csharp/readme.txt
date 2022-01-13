The main solution file is OnnxRuntime.CSharp.sln. This includes desktop and Xamarin mobile projects.
OnnxRuntime.DesktopOnly.CSharp.sln is a copy of that with all the mobile projects removed. This is 
due to there being no way to selectively exclude a csproj from the sln if Xamarin isn't available. 

If changes are required, either update the main solution first and copy the relevant changes across,
 or copy the entire file and remove the mobile projects (anything with iOS, Android or Droid in the name). 