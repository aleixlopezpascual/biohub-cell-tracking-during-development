#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f AGENTS.md ]]; then
  echo "AGENTS.md not found in $repo_root" >&2
  exit 1
fi

mkdir -p .github
ln -sf ../AGENTS.md .github/copilot-instructions.md
ln -sf AGENTS.md CLAUDE.md
ln -sf AGENTS.md GEMINI.md

printf 'Linked assistant guidance files to AGENTS.md:\n'
ls -l .github/copilot-instructions.md CLAUDE.md GEMINI.md
