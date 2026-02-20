# Evaluation Speed Optimizations Applied

## Summary

Successfully implemented **config-only speed optimizations** to `scripts/evals/eval.sbatch` without modifying any Megatron core code.

## Changes Made

### 1. Increased Batch Size (4x increase)
```bash
BATCH_SIZE=256  # Changed from 64
```
- **Expected gain**: 1.5-2x speedup
- Processes more samples per HTTP request
- Better GPU utilization

### 2. Increased Inference Engine Limits (8x increase)
```bash
INFERENCE_MAX_BATCH=256  # Changed from 32
--inference-max-batch-size $INFERENCE_MAX_BATCH
--inference-max-requests $INFERENCE_MAX_BATCH
```
- **Expected gain**: 10-20% speedup
- Allows engine to handle larger batches internally
- Prevents batch size from being a bottleneck

### 3. Enabled Tokenized Requests
```bash
tokenized_requests=True  # Changed from False
```
- **Expected gain**: 10-20% speedup
- Server doesn't need to tokenize on every request
- Eval harness pre-tokenizes and sends token IDs

### 4. Increased Concurrent Requests
```bash
NUM_CONCURRENT=4  # Changed from 1
```
- **Expected gain**: 0-30% speedup (uncertain due to Flask LOCK)
- Allows request pipelining
- Client can prepare next batch while server processes current batch

## Expected Performance

### Before Optimizations
- **Original**: ~0.75 samples/s (1.32s/iteration)
- **After first batch increase (to 64)**: ~5.25 samples/s (7x speedup)

### After These Optimizations
- **Target**: 10-15 samples/s
- **Total speedup from start**: 13-20x faster
- **Additional speedup from 5.25 s/s**: 2-3x more

### Performance Table

| Configuration | Samples/s | Speedup | Time for 1000 samples |
|--------------|-----------|---------|----------------------|
| Original (batch=8) | 0.75 | 1x | 22 minutes |
| First improvement (batch=64) | 5.25 | 7x | 3.2 minutes |
| **After all optimizations** | **10-15** | **13-20x** | **~1.5 minutes** |
| HuggingFace baseline | 127 | 169x | 8 seconds |

## Testing the Changes

### Test 1: Run with Current Settings
```bash
sbatch scripts/evals/eval.sbatch
# Monitor with: tail -f logs/eval-*.out
```

### Test 2: If BATCH_SIZE=256 Causes OOM
Reduce in steps:
```bash
# Edit eval.sbatch:
BATCH_SIZE=128  # Try this first
BATCH_SIZE=192  # Or this
```

### Test 3: Quick Testing with Smaller Limit
For faster iteration during tuning:
```bash
# Edit eval.sbatch line 102:
--limit 50 \  # Instead of 100
```

## Tuning Variables

You can easily adjust these at the top of `eval.sbatch`:

```bash
# For maximum speed (if VRAM allows)
BATCH_SIZE=512
INFERENCE_MAX_BATCH=512
NUM_CONCURRENT=8

# For stability (if OOM occurs)
BATCH_SIZE=128
INFERENCE_MAX_BATCH=128
NUM_CONCURRENT=2

# For testing
BATCH_SIZE=64
INFERENCE_MAX_BATCH=64
NUM_CONCURRENT=1
```

## Monitoring Performance

Watch the logs to verify batching:
```bash
# Check server logs for batch sizes
grep "BATCHING DEBUG" logs/server-*.log

# Monitor eval progress
tail -f logs/eval-*.out
```

Look for:
- `[BATCHING DEBUG] run_mcore_engine called with 256 prompts` ← Confirms large batches
- `400/400 [XX:XX<00:00, X.XXit/s]` ← Shows throughput

## What This Doesn't Include

These optimizations require code changes and were NOT implemented:
- ❌ Removing debug logging (requires editing run_mcore_engine.py)
- ❌ Removing Flask LOCK (requires editing completions.py)
- ❌ Adding /loglikelihood endpoint (requires new endpoint)
- ❌ Production WSGI server (requires server changes)

## Next Steps

1. **Test the current config** - See if BATCH_SIZE=256 works with your VRAM
2. **Monitor and adjust** - If OOM, reduce batch size; if more VRAM available, increase
3. **Measure results** - Compare iteration speed to the 5.25 samples/s baseline
4. **Fine-tune** - Adjust NUM_CONCURRENT and other parameters based on results

## Reverting Changes

If you need to go back to the previous configuration:
```bash
BATCH_SIZE=64
INFERENCE_MAX_BATCH=32
NUM_CONCURRENT=1
# Change tokenized_requests=False in line 103
```

## Expected Outcome

With these changes, you should see:
- ✅ 10-15 samples/second (2-3x faster than current 5.25 s/s)
- ✅ Evaluation of 1000 samples in ~1.5 minutes
- ✅ Still slower than HuggingFace but fast enough for practical use
- ✅ No code modifications required - easy to revert if needed
