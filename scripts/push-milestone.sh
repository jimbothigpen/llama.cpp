#!/usr/bin/env bash
# push-milestone.sh — tag the current HEAD and push branch + tag to origin.
#
# Usage: scripts/push-milestone.sh <label>
#
# Creates an annotated tag milestone/<label> at HEAD, then pushes the current
# branch and the tag to origin. Run from the repo root.
#
# Examples:
#   scripts/push-milestone.sh phase-0.7-sidecar-engine
#   scripts/push-milestone.sh phase-1.0-ppl-harness

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/push-milestone.sh <label>" >&2
    exit 1
fi

LABEL="$1"
TAG="milestone/${LABEL}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"

if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    echo "error: detached HEAD — check out a branch first" >&2
    exit 1
fi

if git tag --list | grep -qx "$TAG"; then
    echo "error: tag '$TAG' already exists — delete it first if you want to re-tag" >&2
    exit 1
fi

echo "Branch : $BRANCH"
echo "Commit : $COMMIT"
echo "Tag    : $TAG"
echo ""

git tag -a "$TAG" -m "Milestone: ${LABEL}"
git push origin "$BRANCH"
git push origin "$TAG"

echo ""
echo "Pushed branch '$BRANCH' and tag '$TAG' to origin."
