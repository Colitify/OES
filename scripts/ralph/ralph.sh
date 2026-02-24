#!/bin/bash
# Ralph Wiggum - Long-running AI agent loop
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]

# NOTE: set -e removed intentionally.
# With set -e, ANY unprotected command that exits non-zero (including bare
# `cd` calls in if-then/else branches) silently terminates the entire script,
# not just the current iteration.  All critical paths use explicit guards.

# Parse arguments
TOOL="amp"  # Default to amp for backwards compatibility
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    *)
      # Assume it's max_iterations if it's a number
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Python environment - use Anaconda pytorch_env
PYTHON="/d/Develop/Anaconda/envs/pytorch_env/python.exe"
export PYTHON

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")
  
  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    # Archive the previous run
    DATE=$(date +%Y-%m-%d)
    # Strip "ralph/" prefix from branch name for folder
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"
    
    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"
    
    # Reset progress file for new run
    echo "# Ralph Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"

# Get repo root directory
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="

  # Save current git state for potential rollback
  BEFORE_SHA=$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo "")

  # Run the selected tool with the ralph prompt
  TEMP_OUTPUT="$SCRIPT_DIR/.claude_output.tmp"

  if [[ "$TOOL" == "amp" ]]; then
    cat "$SCRIPT_DIR/prompt.md" | amp --dangerously-allow-all 2>&1 | tee "$TEMP_OUTPUT" || true
    OUTPUT=$(cat "$TEMP_OUTPUT" 2>/dev/null || echo "")
  else
    # Claude Code: use --dangerously-skip-permissions for autonomous operation
    cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT — aborting iteration $i"; continue; }
    # Unset CLAUDECODE so that a nested Claude Code invocation is allowed.
    # If ralph.sh is run inside a Claude Code session, the env var is set and
    # claude refuses to start ("nested sessions share runtime resources").
    unset CLAUDECODE
    claude --dangerously-skip-permissions -p "请阅读 scripts/ralph/CLAUDE.md 获取完整指令，然后执行其中描述的任务。完成后输出 <promise>COMPLETE</promise> 或继续下一个 story。" 2>&1 | tee "$TEMP_OUTPUT" || true
    OUTPUT=$(cat "$TEMP_OUTPUT" 2>/dev/null || echo "")
  fi

  # Clean up temp file
  rm -f "$TEMP_OUTPUT"

  # Check for completion signal
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "Ralph completed all tasks!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    exit 0
  fi

  # Run guardrail check if metrics.json exists
  METRICS_FILE="$REPO_ROOT/results/metrics.json"
  if [ -f "$METRICS_FILE" ]; then
    echo "Running guardrail check..."
    cd "$REPO_ROOT" || { echo "WARNING: cannot cd to $REPO_ROOT for guardrail — skipping"; continue; }
    # --tol 0.05: allow up to 5% RMSE regression between iterations.
    # tol=0.0 caused infinite rollback loops because Optuna's TPE sampler is
    # not fully deterministic, so re-running the same pipeline can return
    # RMSE values that differ by ~0.001, always failing the guardrail.
    if $PYTHON -m src.guardrail "$METRICS_FILE" --tol 0.05; then
      echo "Guardrail PASSED - changes accepted"
    else
      echo "Guardrail FAILED - rolling back changes"
      # Rollback uncommitted changes
      git checkout -- . 2>/dev/null || true
      # If a new commit was made, reset to before
      AFTER_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
      if [ -n "$BEFORE_SHA" ] && [ -n "$AFTER_SHA" ] && [ "$BEFORE_SHA" != "$AFTER_SHA" ]; then
        echo "Resetting to previous commit: $BEFORE_SHA"
        git reset --hard "$BEFORE_SHA" 2>/dev/null || true
      fi
    fi
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
exit 1
