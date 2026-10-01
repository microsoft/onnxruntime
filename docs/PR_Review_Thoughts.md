## PR Review Thoughts

### Pre-PR
- AI is going to generate most if not all of the code but you (the developer) MUST understand the changes
    - if you do not understand why something is a certain way or where it is ask AI to clarify
    - e.g. 'why did this change from `std::mutex` to a `std::recursive_mutex`'
- Previously the primary concern was AI slop
  - poor quality implementation
  - unnecessary duplication
    - lack of refactoring to separate out common functionality
    - re-implementing simple helpers instead of looking for existing implementations
  - etc., etc.
  - less of an issue with newer models and well planned changes
- New concern is unnecessary over-engineering
  - added code and complexity to handle unlikely scenarios and edge cases
    - are we adding/changing something that normal usage would never hit?
    - are we making things unnecessarily complicated?
    - consider cost/benefit and risk
  - once checked in it creates a signal going forward that we need to care about these scenarios
  - if unsure, ask AI to justify the inclusion and be specific about why and when it is necessary

### Review efficiency
- suggestion: use a multi-agent review approach
  - GH Copilot Cloud Review Agent is one agent
  - use a local agent to provide an additional perspective and explanations when reviewing
    - checkout PR locally with the GH CLI is easiest. `gh pr checkout <PR_NUMBER>`
      - from VS Code, use Copilot Chat with a suitable model
    - ask the local agent to provide its review of changes on the branch first
      - don't have it read the existing review initially so you get two independent perspectives
      - contrast with any comments from the Copilot Cloud Review agent
    - while reviewing, if you have questions or are unsure about something ask the local model to explain
      - faster than trying to find all the relevant pieces from the existing code and PR manually
      - don't assume an AI generated explanation is 100% correct.
        - still need to validate and ask PR author if unsure.
      - if AI can explain it avoids a loop back to the PR author which takes time and requires you to re-establish context to understand their reply
    - ask the local agent to write the PR comment/provide suggested fix directly where applicable
      - e.g. the agent session has meaningful context for the comment
    - if as part of the local review process you do some experimentation (e.g. you think there's a better way and you use the local agent to prove that out) and the changes seem good:
      - create a branch off the PR branch
      - add the changes
      - create a PR for them targeting the PR branch so the PR author can easily pickup some or all of the changes

- keep iterating on the AI setup around reviews to refine and augment
  - add new skills/instructions/review specific agents as needed
    - e.g. .github\skills\code-review\SKILL.md
  - automate iterative enhancements to the code review agent
    - [Add agent guidance update skills by edgchen1 · Pull Request #32366 · microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/pull/32366)

- human review should largely focus on higher level things and be risk based
  - 'what' is being added/changed and 'why'
    - any backwards compatibility issues?
    - tests should demonstrate/show expected usage
      - make sure there are no gaps
    - identify the integration/functionality tests vs low level unit tests
      - primarily review the integration/functionality tests as they cover the paths production usage will hit
        - make sure all expected usage is covered by these tests
      - low level unit tests generally need less attention during human review if integration/functionality tests are comprehensive.

- request for large PRs to be broken up into smaller, more manageable pieces
  - https://docs.github.com/en/pull-requests/get-started/about-stacked-prs
    - caveat: I haven't used it but it's now auto-suggested at times

- watch out for meaningless AI generated PRs that have no production use-case/value
  - ensure that all PRs have clear and valid production use-case before approving or merging


### Key Review Areas

- public C API is ABI and any changes must clearly justify their inclusion
  - once it's in the API it's essentially there forever
  - should provide general purpose capabilities for production scenarios
  - for any API change...
    - is it well thought through and the expected usage clear?
    - is it stable or likely to require further modifications?
        - use the new experimental API functionality until sure
    - is it flexible enough?
        - use OrtKeyValuePairs as string:string property bag as a flexible future-proof way to provide options
    - is it well documented?
- Any changes must be backwards compatible
- Is the change overly complicated/convoluted
  - Could the change be simplified without losing functionality?
    - requests to simplify are one of my most common
- Binary size is an important consideration
  - we do NOT implement support for every single aspect of the ONNX spec
    - e.g. data types supported by any particular operator are added based on production usage
    - also consider if that production usage can be handled via a model change
      - "I have a production model that uses `double` for XYZ" is not a justification unless it can be demonstrated that converting the model to `float` would have insufficient accuracy
  - overly templatized code is a frequent cause of unnecessary growth
    - templatizing a value that can easily be passed in as an argument
      - e.g. templatizing a `bool` value is generally avoidable
    - where possible split out code that is not dependent on the template parameters to file local helper functions
      - in theory the compiler should be able to do this, but from experience it generally doesn't do it well
    - [SizeBench](https://github.com/microsoft/SizeBench) (Windows) and [Bloaty](https://github.com/google/bloaty) (*nix) can be used to investigate growth
- ORT core code should not have EP specific code
  - there's some legacy stuff, but that is the exception and not the justification for adding more
  - design smell if someone is claiming this is required
    - fix the design instead
