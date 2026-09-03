# Agent Coding Guidance

ONNX Runtime uses layered agent customizations so that guidance is available to GitHub Copilot and local agents without
duplicating subsystem knowledge. This document explains how maintainers should extend that system.

## Guidance Layers

| Location | Purpose |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | Repository-wide guidance. |
| [`.github/instructions/`](../.github/instructions/) | Guidance for working with specific paths. |
| [`.github/skills/code-review/SKILL.md`](../.github/skills/code-review/SKILL.md) | Generic code review workflow. |
| [`.github/skills/`](../.github/skills/) | Deeper workflows and knowledge for particular tasks or subsystems. |
| [`.github/skills/collect-agent-guidance-from-reviews/SKILL.md`](../.github/skills/collect-agent-guidance-from-reviews/SKILL.md) | Propose guidance changes from accepted PR review feedback. |
| [`.github/skills/audit-agent-guidance/SKILL.md`](../.github/skills/audit-agent-guidance/SKILL.md) | Audit existing guidance for revisions or retirement. |

Keep each piece of guidance in one canonical location and use the narrowest layer and scope that reliably load for the
affected work. Other layers should point to the canonical location rather than restating it.

Keep `.github/copilot-instructions.md` minimal. It should route GitHub Copilot to canonical guidance in AGENTS.md
rather than contain guidance that other agents cannot discover.

## Guidance Units

Organize guidance into independently maintainable units under descriptive headings. Treat the repository-relative file
path and heading text together as the unit's identity for audits and updates. Keep headings stable when practical so
references remain useful, but rename them when clarity improves; Git history preserves the transition.

## Adding Path-Scoped Instructions

Create a descriptively named `*.instructions.md` file under `.github/instructions/`. Include YAML frontmatter with a
meaningful `description` and an `applyTo` string. Separate multiple patterns with commas.

```markdown
---
description: "Guidance for Example subsystem changes."
applyTo: "onnxruntime/core/example/**/*.cc,onnxruntime/core/example/**/*.h"
---

# Example Subsystem

State the invariant, why it matters when that is not obvious, and what a correct change must update.
```

Use the narrowest paths that reliably identify relevant changes. Avoid `applyTo: "**"`; repository-wide guidance belongs
in `AGENTS.md`. Unless the text explicitly limits their scope, matching instructions apply to both implementation and
review.

Prefer actionable invariants over broad reminders. Good guidance identifies a concrete failure mode and the required
correction. Link to existing design documentation or domain skills for detailed background instead of copying it.

## Guidance Lifecycle

Agent guidance is maintained through reviewed pull requests. Review comments and agent analysis are evidence for a
change, not authoritative instructions by themselves.

Use this lifecycle for repository guidance:

1. **Collect evidence.** Identify an accepted review comment, recurring failure, architectural change, or enforcement
   mechanism that may affect guidance.
2. **Classify the response.** Prefer a test, lint rule, safer API, or other mechanical enforcement when practical.
   Otherwise update the narrowest applicable guidance layer.
3. **Review the proposal.** Submit complete proposals as pull requests; they become guidance only after maintainer
   approval and merge.
4. **Refine or relocate.** Narrow, broaden, split, merge, or move guidance when its scope or wording is ineffective.
5. **Retire responsibly.** Remove guidance when it is obsolete, contradicted by current behavior, or fully replaced by
   mechanical enforcement. Preserve the rationale and replacement in pull request history.
