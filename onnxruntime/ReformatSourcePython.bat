:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

:: Before running this, please make sure python.exe is in path, and black is installed like the following
::    pip install --upgrade black

:: For more info about black, see https://github.com/psf/black

python -m black ./python
python -m black ./test

if errorlevel 1 echo please install python, then pip install black
