#!/usr/bin/env bash
# PreToolUse hook: snapshot the working tree before Claude edits a file.
#
# Commits any uncommitted changes as a "checkpoint:" commit so every agent
# edit has a known-good commit immediately before it. Does nothing when the
# tree is already clean, so a run of edits produces at most one checkpoint.
#
# Revert with:  git revert <sha>        (safe, keeps history)
#           or  git reset --hard <sha>  (discards everything after it)

set -uo pipefail

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

# Nothing staged or unstaged (untracked included) -> nothing to snapshot.
if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
  exit 0
fi

git add -A >/dev/null 2>&1
git -c core.hooksPath=/dev/null commit -q --no-verify \
  -m "checkpoint: before agent edit ($(date '+%Y-%m-%d %H:%M:%S'))" >/dev/null 2>&1

sha=$(git rev-parse --short HEAD 2>/dev/null)
printf '{"systemMessage":"Checkpoint committed: %s","suppressOutput":true}\n' "$sha"
