# VM Setup

## Access Groups

Before proceeding, please join the following:

* `ort-qnn-ep-ci-admin-contacts`
* `pam.ort-qnn-ep.rw`

## Windows

### Provisioning

Virtual machines may be provisioned from Phoenix Virtualization Services. Just
[open a ticket](https://qualcomm.service-now.com/sp?id=sc_cat_item&sys_id=3261c161877a05903742ff78cebb3558) with
the following values:

* Request Type: `New Server VM Deployment`
* Operating System: `Windows`
* OS Version Windows: `Windows Server 2022`
* Number of Processors: `12` or more
* Memory Size (GBs): `64`
* Data Drive Size (GBs): `10` (builds are small so we just use `C:`)
* Number of Servers: up to you
* Server Name 1: `ort-ep-win-XX`
* Lease Duration (months): `12 months` (or more if possible)
* Backup: `No Backup`
* Admin Contact Group: `ort-qnn-ep-ci-admin-contacts`
* Criticality: `1`
* Environment Level: `dev`
* Project: `ONNX Runtime QNN Execution Provider`
* Zone: `San Diego Zone 01`
* Network: `Qualnet`
* PAM Access Group: `pam.ort-qnn-ep.rw`
* Windows Domain: `NA.QUALCOMM.COM`

### Logging In

Once the VM has been provisioned, it'll be added to the safe shared by the PAM group.

1. [go/pam](http://go/pam)
2. Find your new host in the list; you might need to search for it by server name.
3. Click on the VM's record to open its overview panel.
4. Click on the `Connect` button on the right side.
5. Put any non-empty short string (e.g., `asdf`, `tha server`) into the `Log On To` box.
6. Hit Connect. At this point, either Remote Desktop will open or you'll download an `.rdp` file you can use to connect.
7. If prompted for credentials, leave the password blank.

Congratulations, you are now logged in as administrator.

### Intial Setup

#### Add Service Account

Add `NA\OrtQnnEpCi` as a local `Standard`:

Add user:

1. Right-click on Start button and select `Settings`.
2. Navigate to `Accounts` --> `Other Users` and click on `Add a work or school user`.
3. Enter `NA\OrtQnnEpCi` and ensure that `Account type` is set to `Standard`.

Enable Symlinks for `NA\OrtQnnEpCi`:

1. Open the `Local Security Policy` app
2. Navigate to `Security Settings` --> `Local Policies` --> `User Rights Assignment` and add `NA\OrtQnnEpCi` to
   `Create symbolic links`. You'll probably be prompted for _your_ credentials so that `NA\OrtQnnEpCi` can be
   validated as a member of the domain.

#### Visual Studio

Ensure you have a valid license for Visual Studio Professional. You might need to request one from IT. The form
will want to know the name of the machine it will be installed on, but it might be impossible to enter the name
of the VM, even after several days. Just use the name of a machine you _can_ enter and add comments in the notes
section describing the situation and mentioning the server name of the VM.

Software Center probably isn't availble on your VM so find a link to the Visual Studio 2022 Professional installer
in the [Microsoft docs](https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2022).
Run the installer, ensuring to add the following:

* Workloads
  * `Desktop development with C++`
* Individual Components
  * `C++ Universal Windows Platform support for v143 build tools (ARM64/ARM64EC)`

#### Other Software

* [Python 3.12.10 for Windows](https://www.python.org/downloads/windows/)
  * Note: Uses the same version of Python as Microsoft's external CI
  * Check `Add python.exe to PATH`
  * Customize Installation
    * `for all users (requires admin privileges)`
    * `Install Python 3.12 for all users` (yes, a second time on the second page)
  * (After files are copied): `Disable path length limit`
* [Git for Windows](https://git-scm.com/download/win)
  * Select "Git from the command line and also from 3rd-party software" during install.
  * Use bundled OpenSSH.
  * Use the native Windows Secure Channel Library.
  * Checkout files as-is, commit Unix-style ("core.autocrlf" is set to "input").
  * Enable symbolic links.
  * Optional, but very handy: Add Git Prompt to Terminal.
  * (Use defaults otherwise.)

#### GitHub Actions Runner Service

Install the Service

The GitHub runner runs as a Windows service. Log in as the local administrator and follow the
[standard instructions](https://github.qualcomm.com/MLG/onnxruntime-qnn-ep/settings/actions/runners/new?arch=x64&os=win).

#### Build Configuration

* Set the environment variable `ORT_BUILD_TOOLS_PATH` to `C:\Users\OrtQnnEpCi\.ort-build-tools`.
* Set `REQUESTS_CA_BUNDLE` to the location of the Netskope combined certificate as
  [described here](https://qualcomm.sharepoint.com/teams/cloudproxy/SitePages/Certificate-Management.aspx).

## Linux

### Provisioning

We haven't had much luck with self-service provisioning of Linux VMs. As of June 2025, they are short on capacity
in San Diego and Las Vegas. After weeks of them ignoring our failed tickets, Krishna escalated and we suddenly got
five VMs, four of which we actually wanted. In case it comes up again, here's the config we sent them via the
[request form](https://qualcomm.service-now.com/sp?id=sc_cat_item&sys_id=3261c161877a05903742ff78cebb3558).

* Request Type: `New Server VM Deployment`
* Operating System: `Linux`
* OS Version Windows: `Ubuntu 22.04`
* Number of Processors: `12`
* Memory Size (GBs): `32`
* Data Drive Size (GBs): `100`
* Number of Servers: up to you
* Server Name 1: `ort-ep-win-XX`
* Lease Duration (months): `12 months` (or more if possible)
* Backup: `No Backup`
* Admin Contact Group: `ort-qnn-ep-ci-admin-contacts`
* Criticality: `1`
* Environment Level: `dev`
* Project: `ONNX Runtime QNN Execution Provider`
* Zone: `San Diego Zone 01`
* Network: `Qualnet`
* User ID(s): `jkilpat`, `kromero`, `muchhsu`
* Site: `sandiego`
* GV Image Type: `GV`
* GV Cluster: `corp_it`
* Login duty: `ort-qnn-ep-runner.guests-login`
* Sudo duty: `ort-qnn-ep.guests.sudo`
* Additional duties: (none)

### Initial Setup

1. Install but **do not run** the GitHub Actions Runner into `/local/mnt/workspace/actions-runner` following the
   [usual instructions](https://github.qualcomm.com/MLG/onnxruntime-qnn-ep/settings/actions/runners/new?arch=x64&os=linux).
   It's important that `ortqnnepci` owns everything so consider running GitHub's instructions under `sudo -u ortqnnepci bash`.
2. Copy `./setup_linux_vm.sh` to the host and run it as your user.
3. Install and start the Runner service using the commands printed at the end of the above script.
