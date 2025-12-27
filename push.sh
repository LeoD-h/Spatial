#!/usr/bin/env bash
set -e

# Usage: ./push.sh [branch]
# Default branch: main

branch="${1:-main}"
remote="https://github.com/LeoD-h/Spatial.git"

git remote set-url origin "$remote"
git status

# -A pour inclure suppressions
git add -A

# Commit seulement s'il y a quelque chose à committer
if git diff --cached --quiet; then
  echo "Rien à committer."
else
  git commit -m "Organize project, add training/inference scripts and GUI"
fi

git push origin "$branch"
