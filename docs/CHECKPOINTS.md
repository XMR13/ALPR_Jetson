# Checkpoints — How To Create, List, Restore

This repo uses lightweight Git tags and companion branches to mark safe restore points you can jump back to if needed.

## TL;DR
- Create checkpoint at current HEAD:
  - Tag: `git tag checkpoint/<name>`
  - Branch: `git branch checkpoint/<name>`
- List checkpoints: `git tag -l 'checkpoint/*'`
- Restore to a checkpoint in a new working branch:
  - `git switch -c restore/<name> checkpoint/<name>`

## Creating a Checkpoint
Use a descriptive name with date, e.g., `yolov9-pivot-2025-10-11`.

```
git tag checkpoint/yolov9-pivot-2025-10-11
git branch checkpoint/yolov9-pivot-2025-10-11
```

Notes:
- Tags are immutable pointers to a commit; branches can move. We create both so you can restore with either method.
- Lightweight tags do not require Git user identity setup and are enough for local workflows.

## Listing and Inspecting
```
git tag -l 'checkpoint/*'
git show checkpoint/yolov9-pivot-2025-10-11 --no-patch --pretty=fuller
```

## Restoring (Non-Destructive)
Create a new branch starting from the checkpoint without altering main/history:

```
git switch -c restore/yolov9-pivot-2025-10-11 checkpoint/yolov9-pivot-2025-10-11
```

Alternatively, use the tag:

```
git switch -c restore/yolov9-pivot-2025-10-11 tags/checkpoint/yolov9-pivot-2025-10-11
```

## Cleaning Up (Optional)
Remove a checkpoint tag/branch when it is no longer needed:

```
git tag -d checkpoint/yolov9-pivot-2025-10-11
git branch -D checkpoint/yolov9-pivot-2025-10-11
```

## Safety Tips
- Stash local changes before switching: `git stash -u`
- Verify working state with `git status` after restoring.
- Avoid force operations (`reset --hard`) unless you are confident and have backups.
