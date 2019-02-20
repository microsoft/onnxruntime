# Steps for Debugging Firmware Using Emulator

## Prepare Firware Source Files

It is assumed that we have firmware bond files, and source code (*.c, *.h). Create a folder given the name of firware, put *.bond/c/h into it.

## Build .dll for Emulator

In this step, we will compile the onnx_rnns bond files and source code into x86 Dll which can be run in the emulator. 

Change working directory to onnxruntime\firmware\emulator\, run command:

`build_debug_firmware.bat <path-to-firware-source-code> <nuget-root-path>`

Here is an example:
`build_debug_firmware.bat C:\...\onnx_rnns  C:\...\onnxruntime\nuget_root`

There are two parameters:
1. C:\...\onnx_rnns: the firmware folder containing bond files and source files.
2. C:\...\onnxruntime\nuget_root: nuget root folder

## Start Giano Process

Before starting Giano process, please build onnxruntime firstly. 

To start Giano process, in working directory onnxruntime\firmware\emulator\, run command:

`start_giano.bat <firmware-name> <path-to-onnxruntime-root>`

For example:

`start_giano.bat onnx_rnns C:\...\onnxruntime`

Be noted: each time you rebuild onnxruntime, stop existing Giano process and re-run this bat file.

## Attach to Giano process with Visual Studio

Open the folder containing firware source files, in Visual Studio, click "Debug" >> "Attach To Process" >> Choose "giano.exe".

Add breakpoints in *.c/h.

## Trigger Test Case

Use whatever approaches (UT or other script) to trigger the firmware code run, the breakpoints should be hit.