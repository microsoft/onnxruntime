:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

:: Before running this, please make sure python.exe is in path, and yapf is installed like the following
::    pip install --upgrade yapf

:: If you use Visual Studio Code, you can configure like the following:
::    "python.formatting.provider": "yapf"
::    "python.formatting.yapfArgs": ["--style", "{based_on_style: pep8, column_limit: 120}"]
:: See https://code.visualstudio.com/docs/python/editing for more info

:: The style configuration file is .style.yapf in current directory.

:: You can setup git pre-commit hook following https://github.com/google/yapf/tree/master/plugins.

:: For more info about YAPF, see https://github.com/google/yapf

yapf -ir ./python
yapf -ir ./test

if errorlevel 1 echo please install python, then pip install yapf
