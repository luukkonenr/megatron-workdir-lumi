## Minimal Automated Evaluation Setup

**1 file to edit, 1 command to run**

### Quick Start

1. Edit config in `scripts/evals/eval.sbatch`:
```bash
CHECKPOINT="/path/to/checkpoint"
TASKS="arc_easy"
BATCH_SIZE=64
```

2. Submit:
```bash
sbatch scripts/evals/eval.sbatch
```

3. Monitor:
```bash
tail -f logs/eval-<job_id>.out
```

### What It Does

- Launches server in background
- Waits 2 minutes for ready
- Runs eval tasks
- Auto cleanup via trap
- Saves to `results/`

### Fixed Issue

The `AttributeError: 'InferenceRequest' object has no attribute 'merge'` has been fixed in `run_mcore_engine.py` to handle both legacy and dynamic engine return types.
