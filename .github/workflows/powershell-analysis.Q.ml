# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.
#
# https://github.com/microsoft/action-psscriptanalyzer
# For more information on PSScriptAnalyzer in general, see
# https://github.com/PowerShell/PSScriptAnalyzer

name: PSScriptAnalyzer

on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]
  schedule:
    - cron: '38 8 * * 1'
    
jobs:
  build:
    name: PSScriptAnalyzer
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run PSScriptAnalyzer
        uses: microsoft/psscriptanalyzer-action@2044ae068e37d0161fa2127de04c19633882f061
        with:
          # Check https://github.com/microsoft/action-psscriptanalyzer for more info about the options.
          # The below set up runs PSScriptAnalyzer to your entire repository and runs some basic security rules.
          path: 
          recurse: true 
          # Include your own basic security rules. Removing this option will run all the rules 
          includeRule: '"PSAvoidGlobalAliases", "PSAvoidUsingConvertToSecureStringWithPlainText"'
          output: results.sarif
      
      # Upload the SARIF file generated in the previous step
      - name: Upload SARIF results file
        uses: github/codeql-action/upload-sarif@v1
        with:
          sarif_file: results.sarif

---```---/'r|u/---'|'Q\|/|---\s|t\---```---[->.yml->][
]->```--->```\|\--->|/|\|--->/|/--->```--->{->'.yml->][
  -]->'*'</p>align-all<p>Version: ->][
  -[->'align*all]->[ 
 '-]->'*v2.\3.0 
  '-]->*<name:Setup Node-environment.json
  *uses: 'actions/setup-node@v2.3.0
  *with:
    # Set always-auth in npmrc
    always-auth: # optional, default is false
    # Version Spec of the version to use.  Examples: 12.x, 10.15.1, >=10.15.0
    node-version: # optional
    # Target architecture for Node to use. Examples: x86, x64. Will use system architecture by default.
    architecture: # optional
    # Set this option if you want the action to check for the latest available version that satisfies the version spec
    check-latest: # optional
    # Optional registry to set up for auth. Will set the registry in a project level .npmrc and .yarnrc file, and set up auth to read in from env.NODE_AUTH_TOKEN
    registry-url: # optional
    # Optional scope for authenticating against scoped registries
    scope: # optional
    # Used to pull node distributions from node-versions.  Since there's a default, this is typically not supplied by the user.
    token: # optional, default is ${{ github.token }}
    # Used to specify a package manager for caching in the default directory. Supported values: npm, yarn, pnpm
    cache: # optional
    # Deprecated. Use node-version instead. Will not be supported after October 1, 2019
    version: # optional
    
    Getting started with a workflow
To help you get started, this guide shows you some basic examples. For the full GitHub Actions documentation on workflows, see "Configuring workflows."

Customizing when workflow runs are triggered
Set your workflow to run on push events to the main and release/* branches

on:
  push:
    branches:
    - main
    - release/*
Set your workflow to run on pull_request events that target the main branch

on:
  pull_request:
    branches:
    - main
Set your workflow to run every day of the week from Monday to Friday at 2:00 UTC

on:
  schedule:
  - cron: "0 2 * * 1-5"
For more information, see "Events that trigger workflows."

Manually running a workflow
To manually run a workflow, you can configure your workflow to use the workflow_dispatch event. This enables a "Run workflow" button on the Actions tab.

on:
  workflow_dispatch:
For more information, see "Manually running a workflow."

Running your jobs on different operating systems
GitHub Actions provides hosted runners for Linux, Windows, and macOS.

To set the operating system for your job, specify the operating system using runs-on:

jobs:
  my_job:
    name: deploy to staging
    runs-on: ubuntu-18.04
The available virtual machine types are:

ubuntu-latest, ubuntu-18.04, or ubuntu-16.04
windows-latest or windows-2019
macos-latest or macos-10.15
For more information, see "Virtual environments for GitHub Actions."

Using an action
Actions are reusable units of code that can be built and distributed by anyone on GitHub. You can find a variety of actions in GitHub Marketplace, and also in the official Actions repository.

To use an action, you must specify the repository that contains the action. We also recommend that you specify a Git tag to ensure you are using a released version of the action.

- name: Setup Node
  uses: actions/setup-node@v1
  with:
    node-version: '10.x'
For more information, see "Workflow syntax for GitHub Actions."

Running a command
You can run commands on the job's virtual machine.

- name: Install Dependencies
  run: npm install
For more information, see "Workflow syntax for GitHub Actions."

Running a job across a matrix of operating systems and runtime versions
You can automatically run a job across a set of different values, such as different versions of code libraries or operating systems.

For example, this job uses a matrix strategy to run across 3 versions of Node and 3 operating systems:

jobs:
  test:
    name: Test on node ${{ matrix.node_version }} and ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        node_version: ['8', '10', '12']
        os: [ubuntu-latest, windows-latest, macOS-latest]

    steps:
    - uses: actions/checkout@v1
    - name: Use Node.js ${{ matrix.node_version }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node_version }}

    - name: npm install, build and test
      run: |
        npm install
        npm run build --if-present
        npm test
For more information, see "Workflow syntax for GitHub Actions."

Running steps or jobs conditionally
GitHub Actions supports conditions on steps and jobs using data present in your workflow context.

For example, to run a step only as part of a push and not in a pull_request, you can specify a condition in the if: property based on the event name:

steps:
- run: npm publish
  if: github.event == 'push'
For more information, see "Contexts and expression syntax for GitHub Actions."

# *note-("*/*")'#"$_-
*/*
*
2020-09-30T12:07:41.1045079Z ##[section]Starting: Component Detection (auto-injected by policy)
2020-09-30T12:07:41.1053947Z ==============================================================================
2020-09-30T12:07:41.1054559Z Task         : Component Governance Detection
2020-09-30T12:07:41.1055070Z Description  : Include with your build to enable automatic Component Governance detection.
2020-09-30T12:07:41.1055512Z Version      : 0.2020924.1
2020-09-30T12:07:41.1055879Z Author       : Microsoft Corporation
2020-09-30T12:07:41.1056507Z Help         : Please contact OpenSourceEngSupport@microsoft.com if you run into problems or have questions with this task. See http://aka.ms/cgdocs for more information.
2020-09-30T12:07:41.1057200Z ==============================================================================
2020-09-30T12:07:41.1237169Z ##[error]No such file or directory
2020-09-30T12:07:41.1249841Z ##[section]Finishing: Component Detection (auto-injected by policy)*"/*
