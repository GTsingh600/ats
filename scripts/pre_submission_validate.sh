#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-.}"
PING_URL="${PING_URL:-}"

if [ -n "$PING_URL" ]; then
  echo "Step 1/4: Pinging deployed Space"
  python "$REPO_DIR/scripts/ping_env.py" "$PING_URL"
fi

echo "Step 2/4: Building Docker image"
docker build "$REPO_DIR"

echo "Step 3/4: Running OpenEnv validation"
python -m openenv.cli validate "$REPO_DIR"

echo "Step 4/4: Running task graders"
python "$REPO_DIR/scripts/run_graders.py"

echo "Validation completed successfully."
