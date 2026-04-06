#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"
cd "$ROOT_DIR"

release_files=(
  "inference.py"
  "README.md"
  "Dockerfile"
  "scripts/validate-submission.sh"
  "scripts/pre_submission_validate.sh"
)

for file in "${release_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "FAILED -- missing required release file: $file"
    exit 1
  fi
  if rg -n -i "groq|GROQ_" "$file" >/dev/null; then
    echo "FAILED -- Groq-only marker found in release file: $file"
    rg -n -i "groq|GROQ_" "$file"
    exit 1
  fi
done

for blocked in test_groq_inference.py setup_groq_inference.sh GROQ_SETUP.md GROQ_INTEGRATION_SUMMARY.md; do
  if [ -e "$blocked" ]; then
    echo "FAILED -- Groq-only artifact present on release branch: $blocked"
    exit 1
  fi
done

echo "PASSED -- release hygiene check"
