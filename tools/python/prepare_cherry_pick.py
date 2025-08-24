# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
prepare_cherry_picks.py

Generates commit SHAs and PR descriptions for cherry-picking GitHub PRs tagged with a specific release label.

🛠 Requirements:
- Python 3
- GitHub CLI (`gh`) installed and authenticated
- A file named 'already_picked.txt' containing PR numbers already cherry-picked (one per line)

📦 What it does:
1. Fetches all PRs from `main` tagged with the specified label (e.g. `release:1.22.0`)
2. Filters out already cherry-picked PRs using `already_picked.txt`
3. Outputs:
   - `cherry_pick_list.txt`: Commit SHAs and PR titles (for reference)
   - `cherry_pick_shas.txt`: Raw list of SHAs for scripting or manual use
   - `cherry_pick_pr_description.md`: PR-ready markdown summary of cherry-picked PRs

📋 Usage:
$ python3 prepare_cherry_picks.py
"""

import json
import subprocess
from pathlib import Path

# Config
RELEASE_BRANCH = "rel-1.22.0"  # The branch to cherry-pick into
LABEL = "release:1.22.0"  # The label to search for in PRs
ALREADY_PICKED_FILE = "already_picked.txt"  # File containing PR numbers already cherry-picked

# Output files
LIST_FILE = "cherry_pick_list.txt"
SHAS_FILE = "cherry_pick_shas.txt"
DESC_FILE = "cherry_pick_pr_description.md"

# Step 1: Fetch PRs with label
print(f"Fetching PRs with label: {LABEL}")
result = subprocess.run(
    [
        "gh",
        "pr",
        "list",
        "--search",
        f"label:{LABEL} is:merged base:main",
        "--json",
        "number,title,mergeCommit,mergedAt,url",
        "--limit",
        "1000",
    ],
    stdout=subprocess.PIPE,
    check=True,
)
all_prs = json.loads(result.stdout)

# Step 2: Load already cherry-picked PRs
already_picked = set()
if Path(ALREADY_PICKED_FILE).exists():
    with open(ALREADY_PICKED_FILE) as f:
        already_picked = {int(line.strip()) for line in f if line.strip().isdigit()}
else:
    print(f"⚠️  {ALREADY_PICKED_FILE} not found. Assuming no PRs have been picked.")

# Step 3: Filter and sort
to_pick = [pr for pr in all_prs if pr["number"] not in already_picked and pr["mergeCommit"]]
to_pick.sort(key=lambda pr: pr["mergedAt"])

# Step 4: Write cherry_pick_list.txt (SHA + info)
with open(LIST_FILE, "w") as f:
    for pr in to_pick:
        sha = pr["mergeCommit"]["oid"]
        f.write(f"{sha} # PR #{pr['number']}: {pr['title']} ({pr['url']})\n")

# Step 5: Write cherry_pick_shas.txt (only SHAs)
with open(SHAS_FILE, "w") as f:
    for pr in to_pick:
        f.write(f"{pr['mergeCommit']['oid']}\n")

# Step 6: Write PR description markdown
with open(DESC_FILE, "w") as f:
    f.write("### Description\n\n")
    f.write("Cherry pick the following into\n")
    f.write(f"[{RELEASE_BRANCH}](https://github.com/microsoft/onnxruntime/tree/{RELEASE_BRANCH})\n\n")
    for pr in to_pick:
        f.write(f"- (#{pr['number']})\n")


# Final summary
total_tagged = len(all_prs)
missing_sha = len([pr for pr in all_prs if not pr["mergeCommit"]])
already_picked_count = len(already_picked)
to_pick_count = len(to_pick)

print("\n📊 Summary")
print("---------")
print(f"🏷️  Total PRs with label '{LABEL}': {total_tagged}")
print(f"✅ Already cherry-picked PRs (from file): {already_picked_count}")
print(f"🔍 Missing merge commit SHA: {missing_sha}")
print(f"📦 PRs to cherry-pick: {to_pick_count}")

print("\n✅ Output files generated:")
print(f"{LIST_FILE} - full commit info")
print(f"{SHAS_FILE} - clean SHAs for xargs or manual cherry-pick")
print(f"{DESC_FILE} - ready-to-paste PR description markdown")
