---
name: audit-agent-guidance
description: "Audit ONNX Runtime agent guidance for conflicts, stale claims, ineffective scope, duplication, misplaced detail, and opportunities for mechanical enforcement; open a PR that refines, consolidates, relocates, or retires guidance with one logical change per commit."
---

# Audit ONNX Runtime Agent Guidance

Use this skill for periodic maintenance of the repository's agent-guidance corpus. Unlike
`collect-agent-guidance-from-reviews`, which starts from PR feedback, this skill starts from committed guidance and asks
whether each unit is still correct, well-scoped, discoverable, non-duplicative, and worth its context cost.

Follow the lifecycle and guidance-layer rules in
[`docs/Agent_Coding_Guidance.md`](../../../docs/Agent_Coding_Guidance.md).

## Safety and Authority

- Validate guidance claims against the current source code.
- Do not remove guidance merely because it has not been cited recently.
- Do not preserve guidance merely because it is old or was written by an expert.
- Never weaken a security, ABI, memory-safety, correctness, or compatibility invariant without concrete evidence and
  appropriate owner review.
- Limit the audit targets to committed repository content, while consulting external references that the guidance
  depends on when necessary.

## Scope

Inventory all agent guidance:

- `AGENTS.md`;
- `.github/copilot-instructions.md`;
- `.github/instructions/**/*.instructions.md`;
- `.github/skills/**/SKILL.md`;
- documents explicitly linked from those files as canonical guidance.

Record the full commit ID of the default-branch state being audited before making changes. Audit only the committed
guidance and repository state at that snapshot; do not account for changes proposed in open PRs.

## Audit Dimensions

Evaluate each guidance unit against all applicable dimensions.

### Technical validity

- Do referenced paths, symbols, commands, flags, APIs, and tests still exist?
- Does current implementation behavior still support the stated rationale and correction?
- Has an architectural change invalidated the guidance?
- Are version-sensitive statements clearly bounded and current?

### Scope and loading

- Is repository-wide guidance truly universal?
- Does each `applyTo` pattern cover all intended files without loading for unrelated work?
- Is a skill description specific enough to trigger for the relevant task?
- Is detailed subsystem knowledge misplaced in `AGENTS.md` or a broad instruction file?
- Would moving guidance improve discoverability without duplicating it?

### Actionability and discoverability

- Does the guidance state the triggering condition, failure mode, and required correction?
- Is the `applyTo` scope or skill description sufficient for an agent to load it for the relevant work?
- Does the guidance distinguish important exceptions and adjacent cases?
- Is essential context buried in narrative that an agent is unlikely to apply?

### Duplication and conflict

- Find semantically overlapping guidance, not only repeated wording.
- Check whether two instructions prescribe incompatible behavior under an overlapping scope.
- Prefer the narrower, more technically precise invariant.
- Preserve useful detail by merging into one canonical location and replacing other copies with links when needed.

### Enforceability

- Identify guidance that could be replaced by tests, linters, type-system constraints, safer APIs, schema validation,
  or CI checks.
- Once mechanical enforcement is established and discoverable, remove redundant prose guidance.

### Context value

- Consolidate historical narrative that does not change agent behavior.
- Keep rationale, constraints, and non-obvious failure modes that are necessary for correct application.

## Audit Outcomes

Classify each guidance unit as one of:

| Outcome | Use when |
|---|---|
| Retain | The guidance remains correct, useful, and appropriately scoped. |
| Revise | The guidance needs correction, clarification, rescoping, relocation, consolidation, or separation. |
| Retire | Guidance is obsolete or fully superseded by maintained mechanical enforcement. |

Retirement is a normal lifecycle operation, but every retirement requires evidence and a documented successor when one
exists.

## Subagent Delegation

If detailed validation will not fit reliably in one context, partition the guidance corpus into coherent,
non-overlapping groups and delegate those groups to subagents using the same audit dimensions. Have subagents return
findings without editing files. The coordinating agent remains responsible for cross-corpus conflict and duplication
analysis, final outcomes, and all edits.

## Workflow

### 1. Build the inventory

For each guidance unit, record:

- file and durable heading;
- guidance layer and effective scope;
- invariant and rationale;
- referenced paths, symbols, tools, and documents;
- related guidance units;
- relevant origin or refinement PRs available from Git history.

Use the path and heading as the stable identity defined in
[`docs/Agent_Coding_Guidance.md`](../../../docs/Agent_Coding_Guidance.md).

### 2. Validate current claims

Read the relevant source and documentation. Use code navigation and history to verify behavioral claims and determine
whether the guidance remains current. Do not infer technical validity from wording alone.

### 3. Inspect provenance and enforcement

Use the guidance text, its visible source links, current tests and tooling, and Git history to determine:

- why the guidance was introduced and whether that rationale still applies;
- whether later repository changes refined or contradicted it;
- whether mechanical enforcement now makes some or all of the prose redundant;
- whether a source link or replacement reference is missing or broken.

Do not broadly rescan PR review comments in this workflow. Review-derived additions and recurrence evidence belong in
`collect-agent-guidance-from-reviews`.

### 4. Design the smallest coherent changes

For each proposed revision:

- identify the evidence;
- state why the current guidance is deficient;
- select the narrowest canonical destination;
- check for downstream links or references that must be updated;
- preserve technically useful rationale and examples;
- avoid unrelated rewriting or stylistic cleanup.

### 5. Prepare independent commits

Create a branch such as `copilot/audit-agent-guidance-YYYY-MM-DD`. Use one commit per logical guidance change.
Several files may belong in one commit when moving or deduplicating one invariant.

Each commit message must identify:

- the guidance unit being changed;
- the audit outcome and change;
- the evidence or current-code basis;
- source PRs or review comments when relevant.

### 6. Validate the proposal

Before opening the PR:

- verify every technical claim against the current default branch;
- ensure moves and merges leave one canonical source without broken links;
- ensure narrowed scopes still load for every intended path;
- ensure broadened scopes do not impose subsystem-specific rules on unrelated work;
- ensure retired guidance has concrete evidence and names its replacement when applicable;
- ensure each commit corresponds to one row in the proposed-changes table;
- request appropriate subsystem owners for changes to specialized guidance.

Open no PR when the audit finds no warranted changes. Report the audited scope and validation performed instead.

### 7. Open the PR

Use this structure:

```markdown
## Audit scope

<guidance layers, paths, or subsystems examined>

## Snapshot

Audited commit: `<full commit ID>`.

## Proposed changes

| Commit | Guidance unit | Outcome | Evidence | Replacement or destination |
|---|---|---|---|---|

## Follow-up enforcement work

| Guidance unit | Recommended test/tooling change |
|---|---|

```

Ready-for-review status means the audit proposal is complete enough for maintainers to evaluate. It does not mean the
proposed guidance changes have been accepted; normal PR approval and merge determine acceptance.
