# Contributing

We're always looking for your help to improve the product (bug fixes, new features, documentation, etc).

## Contribute a code change
* Start by reading the [Engineering Design](./docs/InferenceHighLevelDesign.md). More documentation can be found in the [docs folder](./docs/) and [here](https://microsoft.github.io/onnxruntime/).
* If your change is non-trivial or introduces new public facing APIs (discussed in more detail below) please use the [feature request issue template](https://github.com/microsoft/onnxruntime/issues/new?template=feature_request.md) to discuss it with the team and get consensus on the basic design and direction first. For all other changes, you can directly create a pull request (PR) and we'll be happy to take a look.
* Make sure your PR adheres to the [PR Guidelines](./docs/PR_Guidelines.md) and [Coding Conventions and Standards](./docs/Coding_Conventions_and_Standards.md) established by the team.
* If you're unsure about any of the above and want to contribute, you're welcome to [start a discussion](https://github.com/microsoft/onnxruntime/discussions) with the team.

## Propose a new public API

 ONNX Runtime has a collection of [public APIs](./README.md#api-documentation).  Some of these APIs make their way back into the Windows OS.  We make compatibility commitments for these APIs and follow a structured process when adding to them.  Please use the [Feature Request issue template](https://github.com/microsoft/onnxruntime/issues/new?template=feature_request.md) before starting any PRs that affect any of the public APIs.
 
 If your proposal contributes the Windows Machine Learning WinRT API and will affect the Windows.AI.MachineLearning namespace, please review [Windows ML main principles](./docs/WinML_principles.md) before you submit your request. 

## Process details

Please search the [issue tracker](https://github.com/microsoft/onnxruntime/issues) for a similar idea first: there may already be an issue you can contribute to.

1. **Create Issue**  
To propose a new feature or API please start by filing a new issue in the [issue tracker](https://github.com/microsoft/onnxruntime/issues).  
Include as much detail as you have. It's fine if it's not a complete design: just a summary and rationale is a good starting point.

2. **Discussion**  
We'll keep the issue open for community discussion until it has been resolved or is deemed no longer relevant.
Note that if an issue isn't a high priority or has many open questions then it might stay open for a long time.

3. **Owner Review**  
The ONNX Runtime team will review the proposal and either approve or close the issue based on whether it broadly aligns with the [Onnx Runtime Roadmap - High Level Goals section](./docs/Roadmap.md) and contribution guidelines.

4. **API Review**  
If the feature adds new APIs then we'll start an API review. All new public APIs must be reviewed before merging. 
For making changes to the C API refer to guidance [here](./docs/C_API_Guidelines.md).
For making changes to the Windows Machine Learning WinRT API, please label your request as *api:WinML*, and someone from the Windows ML team will work with you. 

5. **Implementation**
* A feature can be implemented by you, the ONNX Runtime team, or other community members.  Code contributions are greatly appreciated: feel free to work on any reviewed feature you proposed, or choose one in the backlog and send us a PR. If you are new to the project and want to work on an existing issue, we recommend starting with issues that are tagged with “good first issue”. Please let us know in the issue comments if you are actively working on implementing a feature so we can ensure it's assigned to you.  
* Unit tests: New code *must* be accompanied by unit tests.
* Documentation and sample updates: If the PR affects any of the documentation or samples then include those updates in the same PR.
* Build instructions are [here](https://www.onnxruntime.ai/docs/how-to/build.html).
* Checkin Procedure:  Once a feature is complete and tested according to the contribution guidelines follow these steps:
   * Fork the repo
   * git clone your fork
   * Create feature branch
   * Make and checkin your changes along with unit tests
   * git commit your changes
   * git push origin HEAD
   * To request merge into master, send a pull request from the [web ui](https://github.com/Microsoft/onnxruntime).
  * Add 'Microsoft/onnxruntime' as a reviewer.
* Binaries: We periodically produce signed prerelease binaries from the master branch to validate new features and APIs.  After the feature has been sufficiently validated as part of a prerelease package we will include it in the next stable binary release.
* Note: After creating a pull request, you might not see a build getting triggered right away. One of the
onnxruntime team members will trigger the build for you.

## Contribute to ONNX Runtime docs

Tutorials, how-to guides and API reference docs for ONNX Runtime are published at https://onnxruntime.ai/docs.

The html docs are generated from markdown using Jekyll and published using GitHub Pages.

To update the docs, create a Pull Request against the [gh-pages](https://github.com/microsoft/onnxruntime/tree/gh-pages) branch of the [ONNX Runtime repo](https://github.com/microsoft/onnxruntime).

Once your PR is approved and merged, your changes will be automatically published to https://onnxruntime.ai/docs.

Note: technical reference docs for developers of ONNX Runtime source code can be found [here](https://github.com/microsoft/onnxruntime/docs)

## Licensing guidelines

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Report a security issue

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).
