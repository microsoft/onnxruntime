# Contributing

We're always looking for your help to fix bugs and improve the product. Create a pull request and we'll be happy to take a look.
Start by reading the [Engineering Design](docs/HighLevelDesign.md).

# Checkin procedure
```
git clone --recursive https://github.com/Microsoft/onnxruntime
git checkout -b feature_branch
# make your changes
# write unit tests
# make sure it builds and all tests pass
git commit -m "my changes"
git push origin HEAD
```
To request merge into master send a pull request from the web ui
https://github.com/Microsoft/onnxruntime and add 'Microsoft/onnxruntime' as a reviewer.

New code *must* be accompanied by unit tests.

# Build
[Build](BUILD.md)

# Coding guidelines
Please see [Coding Conventions and Standards](./docs/Coding_Conventions_and_Standards.md)

# Licensing guidelines
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

# Code of conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Reporting Security Issues
Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).
