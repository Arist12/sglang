# Kimi-K2.5 Decode Latency Optimization - Reproduction Guide

## Summary

31.6% decode latency improvement for Kimi-K2.5 (1T MoE) on AMD MI355X GPUs with TP=8.

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Decode median latency (batch=1) | 26.6ms | 18.2ms | 31.6% |
| Single-user throughput | ~37.6 tok/s | ~55.0 tok/s | ~46.3% |

**Verified**: 2026-04-06, 4 consecutive optimized runs at 18.18-18.28ms. Baseline consistent at 26.59-27.19ms.

**Note**: Baseline improved from 38.3ms (sglang `7f99319`, March 17) to 26.6ms (sglang `0906e45ce`, March 26) due to upstream sglang improvements. Our optimized absolute latency also improved: 20.3ms -> 18.2ms.

## Batch Size Scaling

| Batch Size | Baseline Latency | Optimized Latency | Improvement | Optimized Throughput |
|------------|-----------------|-------------------|-------------|---------------------|
| 1 | 26.6ms | 18.2ms | 31.6% | 55.0 tok/s |
| 2 | 29.0ms | 24.3ms | 16.2% | 82.2 tok/s |
| 4 | 36.0ms | 33.0ms | 8.3% | 121.1 tok/s |
| 8 | N/A* | N/A* | - | - |

*bsz=8 hits `num_experts must be a power of 2` error (Kimi-K2.5 has 384 experts). This is a known sglang limitation, not related to our patches.

## Hardware

- **Required**: 8x AMD MI355X GPUs (gfx950)
- ROCm 6.3+ with Docker GPU access
- Hugging Face model cache with `moonshotai/Kimi-K2.5`

The GEMM M-threshold configs and MoE kernel tuning target MI355X (gfx950) specifically. Other AMD GPUs (MI300X, MI325X) should benefit but optimal values may differ.

## Optimizations

### 1. AITER GEMM M-threshold configs
**File**: `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC.json`

Added M_LEQ_1 through M_LEQ_256 entries with BLOCK_SIZE_M=16-64 for small-batch decode. The original config only had an `any` fallback with BLOCK_SIZE_M=256, wasting 255/256 MFMA rows for M=1 decode.

### 2. MoE kernel config
**File**: `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`

AMD-specific small-M decode config: BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, num_warps=2. Default BLOCK_SIZE_K=64 underutilizes AMD MFMA units.

### 3. Triton attention num_kv_splits
**File**: `sglang/python/sglang/srt/server_args.py`

Changed `triton_attention_num_kv_splits` from 16 to 64. More KV splits enable better parallelism across MI355X compute units for long-context decode (8192+ tokens).

### Compatibility fixes (required for correctness)
- Disabled GLUON kernel path in `pa_mqa_logits.py` for Kimi-K2.5 compatibility
- Fixed config mutation bug in `fused_gemm_afp4wfp4_split_cat.py`

### Upstream aiter FP4 configs (e742eeb8)

Upstream aiter commit `e742eeb8` adds FP4 MoE tuned configs for Kimi-K2.5 (385 experts, topk=9, inter_dim=512, fp4 per_1x32). These target A4W4 quantized mode, **not bfloat16**. Verified: zero effect on bfloat16 decode latency. Our bfloat16 (A16W16) optimizations are complementary, not competing.

## Step-by-Step Reproduction

### Step 1: Start container

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

### Step 2: Install aiter from source

The base image has aiter pre-installed as a wheel without full C++ extensions. Re-install from source to enable all quantization functions:

```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/aiter && pip install . --no-build-isolation
"
```

This takes ~1-2 minutes. Verify with:
```bash
docker exec kimi_k25_optimized python3 -c "from aiter import dynamic_per_tensor_quant; print('OK')"
```

### Step 3: Run baseline benchmark (batch=1)

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
# Expected: Decode median ~26.6ms
```

### Step 4: Apply sglang optimizations

```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/sglang
    git remote add fork https://github.com/Arist12/sglang.git || true
    git fetch fork kimi-k25-optimize-v4
    git cherry-pick fork/kimi-k25-optimize-v4~3..fork/kimi-k25-optimize-v4~1 --no-commit
"
```

If cherry-pick has conflicts in `fused_moe_triton_config.py`, resolve by keeping the fork's version (with `_is_hip` AMD-specific branch).

### Step 5: Apply aiter optimizations

```bash
docker exec kimi_k25_optimized bash -c "
    cd /sgl-workspace/aiter
    git remote add fork https://github.com/Arist12/aiter.git || true
    git fetch fork kimi-k25-optimize-v4
    git checkout fork/kimi-k25-optimize-v4 -- \
        aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC.json \
        aiter/ops/triton/attention/pa_mqa_logits.py \
        aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py
    # Copy to installed location
    SITE=/opt/venv/lib/python3.10/site-packages
    cp aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC.json \$SITE/aiter/ops/triton/configs/gemm/
    cp aiter/ops/triton/attention/pa_mqa_logits.py \$SITE/aiter/ops/triton/attention/
    cp aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py \$SITE/aiter/ops/triton/gemm/fused/
"
```

### Step 6: Run optimized benchmark (batch=1)

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
# Expected: Decode median ~18.2ms (31.6% improvement)
```

Run 3 times to confirm consistency.

### Step 7: Batch size sweep (batch=1,2,4)

```bash
for BS in 1 2 4; do
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

> **Note**: bsz=8 is not supported for Kimi-K2.5 (384 experts, non-power-of-2).

### Step 8: Real server test (optional)

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
  - Base: upstream main (`b311db2`)
  - 3 commits: MoE config (x2) + num_kv_splits tuning
- **aiter**: https://github.com/Arist12/aiter/tree/kimi-k25-optimize-v4
  - Base: upstream main (`06de6e9`)
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
| Base Image | lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326 |
| sglang version | `0906e45ce` (2026-03-26) |
| aiter version | `417de6df4` (2026-03-05) |

## Verification Results (2026-04-06)

### Benchmark A/B (batch=1)

| Config | Run 1 | Run 2 | Run 3 | Run 4 |
|--------|-------|-------|-------|-------|
| Baseline | 26.59ms | 27.19ms | - | - |
| Optimized | 18.18ms | 18.26ms | 18.19ms | 18.28ms |
| Baseline + upstream FP4 | 26.59ms | 27.21ms | - | - |
| Combined (ours + upstream FP4) | 18.19ms | 18.29ms | - | - |

### Batch Size Sweep

| Batch Size | Baseline | Optimized | Improvement |
|------------|----------|-----------|-------------|
| 1 | 26.6ms (37.6 tok/s) | 18.2ms (55.0 tok/s) | 31.6% latency, +46.3% throughput |
| 2 | 29.0ms (69.1 tok/s) | 24.3ms (82.2 tok/s) | 16.2% latency, +19.0% throughput |
| 4 | 36.0ms (111.1 tok/s) | 33.0ms (121.1 tok/s) | 8.3% latency, +9.0% throughput |
| 8 | Error* | Error* | - |

*Kimi-K2.5 (384 experts) hits `num_experts must be a power of 2` at bsz=8.

### Upstream aiter FP4 Configs (e742eeb8)

Verified no effect on bfloat16 mode:
- Baseline: 26.6ms -> Baseline + FP4 configs: 26.6ms (no change)
- Optimized: 18.2ms -> Optimized + FP4 configs: 18.2ms (no change)

The upstream FP4 configs target A4W4 quantized inference, not bfloat16. They are complementary to our optimizations.

## Cleanup

```bash
docker rm -f kimi_k25_optimized
```
