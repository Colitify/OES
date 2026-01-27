# Ralph Agent Instructions (Claude Code)

## Your Task (ONE story per iteration)

1. Read `scripts/ralph/prd.json`
2. Read `scripts/ralph/progress.txt` (check Codebase Patterns first)
3. Ensure you're on the correct branch from `prd.json.branchName` (create it if missing)
4. Pick the highest priority story where `passes: false`
5. Implement that ONE story (keep scope tight)
6. Run the project evaluation loop (see Quality Checks)
7. Update relevant `AGENTS.md` with reusable learnings (only if worth preserving)
8. Commit: `feat: [ID] - [Title]` (ONLY if guardrail passes)
9. Update `scripts/ralph/prd.json`: set this story `passes: true`
10. Append learnings to `scripts/ralph/progress.txt`

## Quality Checks (this project)

From repo root:

1) Run pipeline and emit machine metrics:
```bash
python main.py --train "C:\Users\PC\libs-spectral-analysis\data\train_dataset_RAW.csv" --cv 5 --model all --optimize \
  --output_dir outputs --seed 42 --metrics_out results/metrics.json
