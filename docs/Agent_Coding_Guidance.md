# Extending Agent Coding Guidance

ONNX Runtime uses layered agent customizations so that guidance is available to GitHub Copilot and local agents without
duplicating subsystem knowledge. This document explains how maintainers should extend that system.

## Guidance Layers

| Location | Purpose |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | Repository-wide guidance. |
| [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) | Thin GitHub Copilot adapter that points to canonical guidance. |
| [`.github/instructions/`](../.github/instructions/) | Guidance for agents working with specific paths. |
| [`.agents/skills/code-review/SKILL.md`](../.agents/skills/code-review/SKILL.md) | Generic diff-driven review workflow and finding format. |
| [`.agents/skills/`](../.agents/skills/) | Deeper workflows and knowledge for particular tasks or subsystems. |

Keep each piece of guidance in one canonical location. Other layers should point to it rather than restating it.

## Choosing a Location

Add guidance to:

- `AGENTS.md` when it applies across the repository, languages, or subsystems;
- `.github/instructions/<area>.instructions.md` when it applies to changes under specific paths;
- `.agents/skills/code-review/SKILL.md` when it changes the generic process for reviewing any diff;
- a domain skill when it requires a specialized workflow or substantial subsystem knowledge.

Keep `.github/copilot-instructions.md` concise. It should route GitHub Copilot to canonical guidance rather than contain
guidance that other agents cannot discover.

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
