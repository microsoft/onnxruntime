# Contributing

We're always looking for your help to fix bugs and improve the product. Create a pull request and we'll be happy to take a look.
Start by reading the [Engineering Design](docs/HighLevelDesign.md). You can find the doxygen generated documentation [here](https://microsoft.github.io/onnxruntime/).

# Proposing new public APIs
The ONNX runtime has a collection of [public API's](docs/HighLevelDesign.md).  Some of these API's make their way back into the Windows OS.  We make compatability committments for these API's and follow a structured process when adding to them.  Please use the [Feature Request issue template](issues/new?template=feature_request.md) before starting any PR's that affect any of the public API's.

# Process details

Please search the [issue tracker](https://github.com/microsoft/onnxruntime/issues) for a similar idea first: there may already be an issue you can contribute to.

1. **Create Issue**  
All code changes must be tied to an Issue. 
To propose a new feature or API please start by filing a new issue in the [issue tracker](https://github.com/microsoft/onnxruntime/issues).  
Include as much detail as you have. It's fine if it's not a complete design: just a summary and rationale is a good starting point.

2. **Wait for Team Owner**  
We will assign an ONNX Runtime team owner to your issue. The ONNX Runtime team regularly triages all incoming issues.  

3. **Discussion**  
We'll keep the issue open for community discussion until the team owner decides it's ready or should be closed.  
Note that if an issue isn't a high priority or has many open questions then it might stay open for a long time.

4. **Owner Review**  
The ONNX Runtime team will review the proposal and either approve or close the issue based on whether it broadly aligns with the [Onnx Runtime Roadmap](../docs/Roadmap.md) and contribution guidelines.

5. **API Review**  
If the feature adds new APIs then we'll start an API review. 
All new public APIs must be reviewed before merging.  

6. **Implementation**  
A feature can be implemented by you, the ONNX Runtime team, or other community members.  
Code contributions are greatly appreciated: feel free to work on any reviewed feature you proposed, or choose one in the backlog and send us a PR. Please let us know in the issue comments if you are actively working on implementing a feature so we can ensure it's assigned to you.   

7. **Checkin Procedure**  
Once a feature is complete and tested according to the contribution guidelines follow these steps:

 * Fork the repo
 * git clone your fork
 * Create feature branch
 * Make and checkin your changes along with unit tests
 * git commit your changes
 * git push origin HEAD
 * To request merge into master, send a pull request from the [web ui](https://github.com/Microsoft/onnxruntime).
 * Add 'Microsoft/onnxruntime' as a reviewer.

8. **Documentation and sample updates**  
We will update the documentation and if applicable add a sample to the samples repository.
Feel free to also contribute to docs and samples!  
Once the docs and samples are updated we'll close the issue.

9. **Binaries**  
We periodically produce signed prerelease binaries from the master branch to validate new features and APIs.  
After the feature has been sufficiently validated as part of a prerelease package we will include it in the next stable binary release on NuGet.


New code *must* be accompanied by unit tests.

*Note*: After creating a pull request, you might not see a build getting triggered right away. One of the
onnxruntime team members will trigger the build for you.
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
