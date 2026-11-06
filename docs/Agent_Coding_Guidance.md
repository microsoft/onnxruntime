# Extending Agent Coding Guidance

ONNX Runtime uses layered agent customizations so that guidance is available to GitHub Copilot and local agents without
duplicating subsystem knowledge. This document explains how maintainers should extend that system.

## Guidance Layers

| Location | Purpose |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | Repository-wide guidance. |
| [`.github/instructions/`](../.github/instructions/) | Guidance for working with specific paths. |
| [`.github/skills/code-review/SKILL.md`](../.github/skills/code-review/SKILL.md) | Generic code review workflow. |
| [`.github/skills/`](../.github/skills/) | Deeper workflows and knowledge for particular tasks or subsystems. |

Keep each piece of guidance in one canonical location. Other layers should point to it rather than restating it.

Keep `.github/copilot-instructions.md` minimal. It should route GitHub Copilot to canonical guidance in AGENTS.md
rather than contain guidance that other agents cannot discover.

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

## Keeping Subsystem Guidance Selective and Complete

Keep a path-scoped instruction file short when it can route to one owning subsystem skill. Put the skill's shared
procedure and completion criteria in `SKILL.md`, and conditional checks in focused references. For example,
[`code-review-webgpu`](../.github/skills/code-review-webgpu/SKILL.md) routes kernel, arithmetic, layout, cache, runtime,
device, plugin, build, graph, profiling, registration, and validation work to focused references. Its
[coverage audit](WebGPU_Review_Coverage.md) records the source boundaries and historical evidence behind the categories.
The routes apply to implementation and review; the generic `code-review` skill
continues to own the review workflow and finding format.

Give each reference a clear ownership boundary and a predicate describing which changes can invalidate its meaning.
The entry point owns a reading table whose "When to read" predicates can be evaluated from the task or diff without
opening the targets. Select every matching row, including indirect effects through shared callers. Do not treat
the rows as mutually exclusive or load every reference by default.

Gate domain-skill selection on a direct or traceable effect on that domain. A shared filename or passing keyword
does not establish applicability. For example, unrelated CPU/CUDA changes and shared optimizer/test edits without
a WebGPU consumer must not select `code-review-webgpu` or its references. In mixed PRs, scope the domain checks to
the affected portion. Keep this boundary explicit in the skill description as well as its workflow.

Keep each rule in one maintained location. When reorganizing guidance, maintain a temporary requirement-to-destination
map covering the original checks, workflows, validation commands, and provenance. Treat an unmapped
requirement as lost until checked. Keep migration evidence and PR status in the task handoff rather than startup guidance.

Before handoff, check link and route completeness and account for each applicable check with evidence or an explicit
validation gap. When scope changes, repeat reference selection. This preserves checks without requiring every task
to load all subsystem history.

Save a new verified review lesson in the reference that owns the affected invariant. Retain its applicability,
failure trigger, corrective rule, and regression check. Keep immutable source/review provenance in maintenance
evidence; include examples in active references only when they clarify an otherwise ambiguous check. Keep raw transcripts,
logs, experiments, and current task status in the task handoff. Update routing when the lesson introduces a new
reading condition. A lesson in a separate local learning store becomes available to a hosted reviewer only after
its reusable check is incorporated into repository guidance and included in the reviewed branch.

## Copilot Pull Request Reviews on GitHub

GitHub Copilot code review supports repository skills under `.github/skills`, including linked supporting files.
Keep the skill directory name and description review-focused, as in `code-review-webgpu`. The repository-wide
Copilot entry point conditionally routes WebGPU changes to `code-review` and `code-review-webgpu`; path instructions select WebGPU work,
and the domain description also covers indirect effects such as shared operator tests.

Include the instructions, `SKILL.md`, and all linked references in the pull request's head branch. GitHub's documented
behavior is to read review guidance from that branch, so guidance can be evaluated before it merges. Keep custom
instructions enabled in the repository's Copilot code-review settings. Local personal skills, worktree progress,
and ignored evidence are not substitutes for repository files available to the hosted reviewer.

After requesting a review, inspect any skill attributions on review comments and the linked review session for
available evidence of which guidance was used. A local structure/link check establishes discoverability, not that
a hosted review selected every reference or detected every defect. Record inaccessible references and unavailable
GPU/runtime checks as validation gaps. Guidance does not provide a GPU runner by itself.

See GitHub's [skill setup](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills),
[repository instruction settings](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions),
and [Copilot review guidance](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-code-review)
when checking current platform support. Do not assume IDE review surfaces have identical capabilities.
