# Kimi-K2.5 Decode Latency Optimization - Reproduction Guide

## Summary

47% decode latency improvement for Kimi-K2.5 (1T MoE) on AMD MI355X GPUs with TP=8.

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Decode median latency (batch=1) | 38.3ms | 20.3ms | 47.0% |
| Single-user throughput | ~26 tok/s | ~54.2 tok/s | ~108% |

**Verified**: 2026-04-06, 3 consecutive runs at 20.3ms. Real server stress test: 54.2 tok/s sustained.

## Hardware

- **Required**: 8x AMD MI355X GPUs (gfx950)
- ROCm 6.3+ with Docker GPU access
- Hugging Face model cache with `moonshotai/Kimi-K2.5`

The GEMM M-threshold configs and MoE kernel tuning target MI355X (gfx950) specifically. Other AMD GPUs (MI300X, MI325X) should benefit but optimal values may differ.

## Optimizations

### 1. AITER GEMM M-threshold configs (-9.0ms, -23.5%)
**File**: `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC.json`

Added M_LEQ_1 through M_LEQ_256 entries with BLOCK_SIZE_M=16-64 for small-batch decode. The original config only had an `any` fallback with BLOCK_SIZE_M=256, wasting 255/256 MFMA rows for M=1 decode.

### 2. MoE kernel config (-5.3ms, -18.1%)
**File**: `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`

AMD-specific small-M decode config: BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, num_warps=2. Default BLOCK_SIZE_K=64 underutilizes AMD MFMA units.

### 3. Triton attention num_kv_splits (-3.7ms, -15.4%)
**File**: `sglang/python/sglang/srt/server_args.py`

Changed `triton_attention_num_kv_splits` from 16 to 64. More KV splits enable better parallelism across MI355X compute units for long-context decode (8192+ tokens).

### Compatibility fixes (0ms, required for correctness)
- Disabled GLUON kernel path in `pa_mqa_logits.py` for Kimi-K2.5 compatibility
- Fixed config mutation bug in `fused_gemm_afp4wfp4_split_cat.py`

## Step-by-Step Reproduction

### Step 1: Start container with upstream base image

```bash
docker run -d \
    --name kimi_k25_optimized \
    --device /dev/kfd --device /dev/dri \
    --shm-size 64g \
    --group-add video \
    -e TOKENIZERS_PARALLELISM=false \
    -e GPU_COREDUMP_ENABLE=0 \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -v /path/to/huggingface:/root/.cache/huggingface \
    lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326 \
    sleep infinity
```

> **Note**: Replace `/path/to/huggingface` with your local HF cache path containing `moonshotai/Kimi-K2.5`.

### Step 2: Run baseline benchmark (batch=1)

```bash
docker exec kimi_k25_optimized /opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 2048 \
    --dtype bfloat16 \
    --decode-attention-backend triton \
    --prefill-attention-backend aiter \
    --mem-fraction-static 0.9
# Expected: Decode median ~38.3ms
```

### Step 3: Apply sglang optimizations

```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/sglang
    git remote add fork https://github.com/Arist12/sglang.git || true
    git fetch fork kimi-k25-optimize-v4
    git cherry-pick fork/kimi-k25-optimize-v4~2..fork/kimi-k25-optimize-v4
"
```

If cherry-pick fails, use file checkout:
```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/sglang
    git remote add fork https://github.com/Arist12/sglang.git || true
    git fetch fork kimi-k25-optimize-v4
    git checkout fork/kimi-k25-optimize-v4 -- python/sglang/srt/server_args.py
    git checkout fork/kimi-k25-optimize-v4 -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py
"
```

### Step 4: Apply aiter optimizations

```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/aiter
    git remote add fork https://github.com/Arist12/aiter.git || true
    git fetch fork kimi-k25-optimize-v4
    git cherry-pick fork/kimi-k25-optimize-v4
"
```

If cherry-pick fails, use file checkout:
```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/aiter
    git remote add fork https://github.com/Arist12/aiter.git || true
    git fetch fork kimi-k25-optimize-v4
    git checkout fork/kimi-k25-optimize-v4 -- aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC.json
    git checkout fork/kimi-k25-optimize-v4 -- aiter/ops/triton/attention/pa_mqa_logits.py
    git checkout fork/kimi-k25-optimize-v4 -- aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py
"
```

### Step 5: Run optimized benchmark (batch=1)

```bash
docker exec kimi_k25_optimized /opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 2048 \
    --dtype bfloat16 \
    --decode-attention-backend triton \
    --prefill-attention-backend aiter \
    --mem-fraction-static 0.9
# Expected: Decode median ~20.3ms (47% improvement)
```

Run 3 times to confirm consistency.

### Step 6: Batch size sweep (batch=1,2,4,8)

```bash
for BS in 1 2 4 8; do
    echo "=== batch_size=$BS ==="
    docker exec kimi_k25_optimized /opt/venv/bin/python3 -m sglang.bench_one_batch \
        --model-path moonshotai/Kimi-K2.5 \
        --tensor-parallel-size 8 \
        --trust-remote-code \
        --batch-size $BS \
        --input-len 8192 \
        --output-len 2048 \
        --dtype bfloat16 \
        --decode-attention-backend triton \
        --prefill-attention-backend aiter \
        --mem-fraction-static 0.9
done
```

### Step 7: Real server test (optional)

```bash
# Start server
docker exec -d kimi_k25_optimized bash -c "
    /opt/venv/bin/python3 -m sglang.launch_server \
        --model-path moonshotai/Kimi-K2.5 \
        --tensor-parallel-size 8 \
        --trust-remote-code \
        --dtype bfloat16 \
        --decode-attention-backend triton \
        --prefill-attention-backend aiter \
        --mem-fraction-static 0.9 \
        --host 0.0.0.0 \
        --port 8000 \
        > /tmp/server.log 2>&1
"

# Wait for server (~8-10 minutes for model loading)
# Check: docker exec kimi_k25_optimized curl -s http://localhost:8000/health

# Send test request
docker exec kimi_k25_optimized curl -s \
    http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "moonshotai/Kimi-K2.5", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 128, "temperature": 0.7}'
```

## Forked Repository Branches

- **sglang**: https://github.com/Arist12/sglang/tree/kimi-k25-optimize-v4
  - Base: latest upstream main (`b311db2`)
  - 3 commits: MoE config (x2) + num_kv_splits tuning
- **aiter**: https://github.com/Arist12/aiter/tree/kimi-k25-optimize-v4
  - Base: latest upstream main (`06de6e9`)
  - 1 commit: GEMM M-threshold configs + compatibility fixes

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Model | moonshotai/Kimi-K2.5 |
| Tensor Parallelism | 8 |
| Input Length | 8192 tokens |
| Output Length | 2048 tokens |
| Decode Backend | triton |
| Prefill Backend | aiter |
| Memory Fraction | 0.9 |
| dtype | bfloat16 |

## Verification Results (batch=1, 2026-04-06)

### Benchmark A/B
| Run | Latency | Notes |
|-----|---------|-------|
| Baseline (unmodified) | 38.3ms | Fresh container |
| Optimized Run 1 | 20.3ms | After patches |
| Optimized Run 2 | 20.3ms | Consistency |
| Optimized Run 3 | 20.3ms | Consistency |

### Real Server Stress Test
| Test | Result |
|------|--------|
| Single request (128 tokens) | 49.6 tok/s |
| Single request (512 tokens) | 54.2 tok/s |
| 5 concurrent requests | 114.8 tok/s aggregate |
| Long context (5K prompt) | 42.5 tok/s |
| 10 sequential (sustained) | 54.2 tok/s, zero variance |

### Independent Cross-Validation (by @Alex)
| Run | Latency |
|-----|---------|
| Baseline | 38.3ms |
| Optimized x3 | 20.3ms, 20.3ms, 20.3ms |

## Cleanup

```bash
docker rm -f kimi_k25_optimized
```
