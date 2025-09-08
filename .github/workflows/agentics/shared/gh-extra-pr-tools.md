---
---

## Creating and Updating Pull Requests

To create a branch, add changes to your branch, use Bash `git branch...` `git add ...`, `git commit ...` etc.

When using `git commit`, ensure you set the author name and email appropriately. Do this by using a `--author` flag with `git commit`, for example `git commit --author "${{ github.workflow }} <github-actions[bot]@users.noreply.github.com>" ...`.

