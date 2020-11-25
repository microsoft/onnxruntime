# Guidelines for creating a good pull request

1. A PR should describe the change clearly and most importantly it should mention the motivation behind the change. Filling out the PR template should satisfy this guideline.
2. If the PR is fixing a performance issue, mention the improvement and how the measurement was done (for educational purposes).
3. Do not leave comments unresolved. If PR comments have been addressed without making the requested code changes, explicitly mark them resolved with an appropriate comment explaining why you're resolving it. If you intend to resolve it in a follow up PR, create a task and mention why this comment cannot be fixed in this PR. Leaving comments unresolved sets a wrong precedent for other contributors that it's ok to ignore comments. 
4. In the interest of time, discuss the PR/comments in person/phone if it's difficult to explain in writing. Document the resolution in the PR for the educational benefit of others. Don't just mark the comment resolved saying 'based on offline discussion'.
5. Add comments, if not obvious, in the PR to help the reviewer navigate your PR faster. If this is a big change, include a short design doc (docs/ folder).
6. Unit tests are mandatory for all PRs (except when the proposed changes are already covered by existing unit tests).
7. Do not use PRs as scratch pads for development as they consume valuable build/CI cycles for every commit. Build and test your changes for at least one environment (windows/linux/mac) before creating a PR.
8. Keep it small. If the feature is big, it's best to split into multiple PRs. Modulo cosmetic changes, a PR with more than 10 files is notoriously hard to review. Be kind to the reviewers.
9. Separate cosmetic changes from functional changes by making them separate PRs.
10. The PR author is responsible for merging the changes once they're approved.
11. If you co-author a PR, seek review from someone else. Do not self-approve PRs.