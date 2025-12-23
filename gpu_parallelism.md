# GPU Parallelism Investigation

This document records the current state of our custom multi-device training loop, the expected behaviour, and the actual observations when running on an 8× NVIDIA B200 setup.

## What We Implemented

1. **Threaded Worker Pool**  
   - Each GPU has a long-lived worker thread (spawned once at startup).  
   - Training loop splits each physical batch into per-device chunks and dispatches them via channels.  
   - Worker pulls a request, runs `forward_classification` + backward locally, then sends back gradients and a `ChessOutput`.

2. **Gradients Aggregation Strategy**  
   - Worker computes gradients (`GradientsParams::from_grads`) and *keeps them on the worker device*.  
   - Main thread receives the response, calls `to_device(main_device)` to move the gradient tensors to GPU 0, then immediately accumulates them and drops the worker copy.  
   - Goal: GPU 0 should only see **one** gradient buffer at a time (plus the model), not N simultaneous copies.

3. **Output Handling**  
   - `ChessOutput` is moved to the main device for metrics.  
   - Outputs still include tensors sized `local_batch_size × 4096` (logits, masks, etc.), so GPU 0 retains the entire batch worth of predictions every iteration.

4. **Model Copies**  
   - Before sending a request we clone the main model, fork it to the worker’s device, and pass that fork inside the request.  
   - Workers therefore hold a full model replica during execution (expected; needed for forward/backward).  
   - No additional cloning happens on GPU 0 during worker startup.

## Our Expectations

With the above setup we expect:

1. **GPU 0 footprint** ≈ `model parameters + one gradient buffer + one batch of logits`  
   - Roughly the same order of magnitude as the other GPUs, perhaps modestly higher due to the optimizer copy.

2. **Worker GPUs footprint** ≈ `model parameters + local batch tensors`.  
   - Should scale proportionally with batch size but stay well below GPU 0.

3. **Raising `physical_batch_size`** should be limited by per-device memory, not massively constrained by GPU 0 alone.

## What Actually Happens

From the NVIDIA-SMI snapshot:

```
GPU 0 (main): ~169,598 MiB used
GPU 1–7:      ~35,500 MiB used each
```

Observations:

1. **GPU 0 is still 4–5× heavier** than the workers, despite gradient buffers being streamed one at a time.  
   - Suggests there is another source of duplication on the main device.

2. **Workers sit ~35 GB** even when idle (no active training).  
   - That aligns with one model replica plus cached activations/optimizer state, not an entire batch’s worth of duplicated data.

3. **GPU 0 memory remains high even when utilization is 0%** (training paused).  
   - Implies memory isn’t freed after a batch; tensors linger on GPU 0 between iterations.

4. **Batch size increases still trigger OOM on GPU 0 first.**  
   - Confirms GPU 0 holds significantly more state than a single batch + model.

## Code Walkthrough Findings

- **GradNorm probe replays the full batch on GPU 0** – Every iteration we materialised a `ChessBatch` for the entire physical batch on the lead device so we could call `gradnorm_state.maybe_update_weights(..., &batch)`. Even when no GradNorm update was scheduled the tensors were allocated, and when an update did fire the helper ran *three* full forward/backward passes (policy/value/time) on GPU 0. CUDA’s caching allocator held on to those activations, so peak reserved memory quickly grew to the size of roughly three extra training steps of the whole batch.
- **Metrics only need CPU copies** – The rest of the loop only consumed `items_all: Vec<ChessItem>` for accuracy buckets and logging. Keeping an additional GPU batch offered no benefit, meaning the extra allocation was pure overhead for the main device.
- **Centralised optimizer state is smaller but still present** – GPU 0 continues to own the AdamW moment buffers (one pair per parameter), which explains a modest overhead versus the worker devices but not the 8–10× spike we observed.

## Likely Culprit(s)

1. **GradNorm forward/backward on the full physical batch**  
   - Each weight update replays three end-to-end passes on GPU 0 (one per task).  
   - CUDA caching keeps the activation buffers alive, so the lead device permanently reserves memory for those extra graphs.

2. **Always-on full-batch allocation for metrics bookkeeping**  
   - We created the `ChessBatch` on GPU 0 every iteration even when no GradNorm update was due.  
   - That kept ~8× more board/policy tensors resident on GPU 0 than on any worker.

3. **Centralised optimizer state**  
   - AdamW moment buffers still live on GPU 0. They account for a smaller, steady overhead and remain future optimisation work, but they did not cause the extreme imbalance.

## Fix Implemented

- Added `gradnorm_probe_size` to the config and sample at most that many `ChessItem`s when a GradNorm update is required.  
- The sampled items stay on the CPU until we hand them to a worker; GPU 0 never materialises a probe batch.  
- GradNorm updates now run the three task-specific passes on a worker GPU and return only aggregate scalars, keeping GPU 0 within the same memory envelope as the rest of the cluster.

## Next Steps / Investigation Ideas

1. **Validate probe quality**  
   - Track GradNorm weights and training metrics while varying `gradnorm_probe_size` to ensure the smaller sample still provides stable task balancing.

2. **Profile allocator behaviour after the change**  
   - Capture NVIDIA-SMI snapshots before/after a long run to confirm GPU 0 stays flat when probes execute on the workers.

3. **Tackle centralised optimizer state**  
   - Explore ZeRO-style sharding or CPU offloading so the AdamW moment buffers are not concentrated on GPU 0.

## Summary

- GPU 0’s runaway memory use was caused by full-batch GradNorm probes repeatedly executed on the lead device.  
- We now sample a capped subset for GradNorm and execute the probe on a worker GPU, so GPU 0 never allocates those tensors.  
- Memory usage across devices should converge, leaving the optimizer state as the primary remaining imbalance to address in future work.

This doc should be updated as we gather profiling data with the new GradNorm sampling strategy.

## Remaining Issue: GPU 0 Still Grows Over Time

Even with worker-side GradNorm probes, GPU 0 still creeps upward by ~50–100 MiB every training iteration until it eventually OOMs (first reproduced around iteration 32 at `gradnorm_interval = 8`). Current hypotheses:

1. **Model cloning on GPU 0**  
   - `model.clone().fork(&devices[device_index])` still allocates the clone on GPU 0 before the fork hops to the worker. CUDA keeps that temporary buffer cached, so each iteration leaves another full-parameter allocation resident.

2. **Main-device gradient accumulation**  
   - We aggregate incoming gradients on GPU 0 via `GradientsAccumulator::accumulate`. The tensors themselves should be reused, but the optimizer split (`weight_decay_groups.split_grads`) may allocate fresh buffers that linger in the allocator cache each step.

3. **Output concatenation**  
   - `combine_outputs` cats every worker output on GPU 0, keeping around multiple copies of logits/masks (≈100 MiB per physical batch). While small compared to the model, the allocator may not recycle these immediately.

What we’ve ruled out so far:

- GradNorm replays on the lead device (moved entirely off GPU 0).  
- Metrics/debug logging (single-sample inference only; no persistent tensors).

Next experiment ideas:

1. Persist one model replica per worker (spawn at startup); broadcast parameter updates instead of cloning in the hot loop.  
2. Profile CUDA allocations after each major stage (post-clone, post-grad-accumulation, post-optimizer) to pinpoint which section increases GPU 0’s reserved bytes.  
3. Move `combine_outputs` to CPU or perform reductions on the worker devices, returning scalars only.
