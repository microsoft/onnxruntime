// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

// Diagnostic-only proxy: preserve the compiler's exact arguments and byte streams.
internal static class CaptureWindowsCompiler
{
    private static int Main()
    {
        string command = Environment.CommandLine;
        int end = command[0] == '"' ? command.IndexOf('"', 1) + 1 : command.IndexOf(' ');
        string arguments = end < 0 ? "" : command.Substring(end);
        var start = new ProcessStartInfo(
            Environment.GetEnvironmentVariable("SCCACHE_DIAGNOSTIC_COMPILER"), arguments);
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        using (var process = Process.Start(start))
        using (var stdout = new MemoryStream())
        using (var stderr = new MemoryStream())
        {
            Task.WaitAll(
                process.StandardOutput.BaseStream.CopyToAsync(stdout),
                process.StandardError.BaseStream.CopyToAsync(stderr));
            process.WaitForExit();
            byte[] output = stdout.ToArray();
            byte[] errors = stderr.ToArray();
            string[] argv = Environment.GetCommandLineArgs();
            if (Array.IndexOf(argv, "-EP") >= 0 || Array.IndexOf(argv, "-E") >= 0)
            {
                string path = Path.Combine(
                    Environment.GetEnvironmentVariable("SCCACHE_DIAGNOSTIC_OUTPUT"),
                    process.Id.ToString());
                File.WriteAllBytes(path + ".stdout", output);
                File.WriteAllBytes(path + ".stderr", errors);
                var inputs = new StringBuilder(arguments + Environment.NewLine);
                foreach (string key in new[] {
                    "CL", "_CL_", "INCLUDE", "LIB", "TEMP", "TMP", "VS_UNICODE_OUTPUT",
                    "MSBUILDPRESERVETOOLTEMPFILES"
                })
                {
                    inputs.AppendLine(key + "=" + Environment.GetEnvironmentVariable(key));
                }
                File.WriteAllText(path + ".inputs.txt", inputs.ToString());
            }
            Console.OpenStandardOutput().Write(output, 0, output.Length);
            Console.OpenStandardError().Write(errors, 0, errors.Length);
            return process.ExitCode;
        }
    }
}
