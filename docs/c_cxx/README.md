How to build the Doxygen HTML pages on Windows:

* Install Doxygen (https://www.doxygen.nl/download.html)
* Running from the command line:
  * cd to (Repository Root)\docs\c_cxx
  * Run "\Program Files\doxygen\bin\doxygen.exe" Doxyfile
* Using the Doxygen GUI app:
  * Launch Doxygen GUI (Doxywizard)
  * File->Open (Repository Root)\docs\c_cxx\Doxyfile
  * Switch to Run tab, click 'Run doxygen'
* Generated docs are written to (Repository Root)\build\doxygen\html

The generated documentation is online in the gh-pages branch of the project: https://github.com/microsoft/onnxruntime/tree/gh-pages/docs/api/c
