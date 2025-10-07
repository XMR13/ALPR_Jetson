# Progress Tracking

This folder maintains a session-by-session record aligned with `plan.md` Section 5 (Detailed Timeline & Tasks). The goal is to keep operational progress in lockstep with the plan, making it easy to audit what changed, when, and why.

## Naming & Location
- Files live under this `progress/` folder.
- File name format: `YYYY-MM-DD_session-N.md` (increment `N` per session in the same day).
- Always include a new session file when a session starts or wraps up significant work.

## Session File Template
Each session file should contain the following sections:

1) Summary
- One-paragraph summary of what was done and key outcomes.

2) Changes
- Added: list of new files/dirs created in this session (workspace-relative paths).
- Modified: list of files updated in this session.
- Deleted: list of files removed in this session.

3) Plan Snapshot (from `plan.md` Section 5)
- Mirror the plan’s structure (Week/Day items) and assign a status to each relevant bullet.
- Recommended statuses: `checked` (verified done), `in_progress`, `open`, `blocked`.
- Keep wording identical to the plan; add short parenthetical notes if needed.

4) Decisions & Blockers
- Note any design decisions, constraints, or open issues affecting scope/timeline.

5) Next Actions
- Concrete, verifiable steps to take before or during the next session.

6) Sync Check
- Brief note that this session’s snapshot was cross-checked against `plan.md` Section 5 and the repository state.

## Sync Rules
- Source of truth for tasks is `plan.md` Section 5. If the plan changes, reflect updates in subsequent session files.
- Do not rewrite past sessions unless correcting factual errors; instead, add a note in the latest session.
- Default behavior: explicitly mark verified items as `checked`. Use `open` for not-started, `in_progress` for ongoing, and `blocked` for external dependencies.

## Quick Workflow
1) Start session → create new `YYYY-MM-DD_session-N.md`.
2) Work → record Adds/Mods/Deletes as you go.
3) Before ending → update Plan Snapshot statuses and Next Actions.
4) Ensure Sync Check passes (plan alignment + repo state verified).

Note: Commit and testing policies are documented in `AGENTS.md`.

## References
- Plan: `plan.md`
- Key code: `src/deepstream_app/`, `src/ocr_service/`, `src/api_server/`
- Key configs: `configs/deepstream/`, `configs/ocr/`
