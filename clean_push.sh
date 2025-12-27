#!/usr/bin/env bash
set -e

# Usage: ./clean_push.sh [branch]
# Default branch: main

branch="${1:-main}"
remote="https://github.com/LeoD-h/Spatial.git"

echo ">> Mise à jour du remote origin vers $remote"
git remote set-url origin "$remote"

echo ">> Soft reset du dernier commit (conserve vos fichiers)"
git reset --soft HEAD~1 || true

echo ">> Génération du .gitignore"
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
*.DS_Store
*.zip
*.pt
data/raw/
data/processed/
models/
outputs/
EOF

echo ">> Nettoyage de l'index (pas des fichiers sur disque)"
git rm -r --cached data/raw data/processed models outputs || true
git rm --cached *.zip *.pt 2>/dev/null || true
git rm -r --cached __pycache__ app/__pycache__ spatial/__pycache__ || true

echo ">> Ajout des fichiers restants"
git add -A

if git diff --cached --quiet; then
  echo ">> Rien à committer."
else
  echo ">> Commit..."
  git commit -m "Organize project, add training/inference scripts and GUI"
fi

echo ">> Push vers $branch"
git push origin "$branch"

echo ">> Terminé."
