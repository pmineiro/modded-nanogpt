# Plan: Implement SoftmaxMuon Optimizer for lm_head

## 1. Understanding the Goal
- **Optimizer Replacement**: Switch `lm_head` from Adam to a new `softmax_muon` optimizer.
- **Efficient Accumulation**: Leverage the existing `FusedSoftcappedCrossEntropy` Triton kernel to accumulate the average probability distribution (`p_bar`) across the vocabulary during the backward pass.
- **External Library Integration**: Import and use the tested, sharding-aware `softmax_muon` function from `softmaxmuon.py`.
- **Momentum & Cautious Update**: Apply a momentum strategy and cautious weight decay (with mantissa tracking) to the `softmax_muon` update, mirroring the robustness of the `normuon` optimizer while skipping its variance reduction step.
- **Maintain Tied State**: Keep `lm_head` and `embed` tied for now, ensuring their gradients and updates are synchronized.

## 2. Investigation Findings
- **Files Involved**: `train_gpt.py`, `triton_kernels.py`, and `softmaxmuon.py`.
- **Sharding Strategy**: `lm_head.weight` is sharded along the model dimension (shape `(model_dim // world_size, vocab_size)`).
- **Optimization Opportunity**: `softmaxmuon.py` supports distributed computation. The Triton backward kernel can efficiently accumulate `p_bar` statistics.

## 3. Step-by-Step Implementation Plan

### Phase 1: Triton Kernel Modification (`triton_kernels.py`)
1. **Global Registry**: Define `_active_optimizer = None`.
2. **Update Backward Kernel**:
   - Modify `fused_softcapped_entropy_bwd_kernel` to accept an optional `p_sum_ptr`.
   - In the column loop, after computing `p = tl.exp(z - lse)`, add:
     `if p_sum_ptr is not None: tl.atomic_add(p_sum_ptr + cols, p, mask=mask)`
3. **Update Autograd Function**:
   - In `FusedSoftcappedCrossEntropy.backward`:
     - If `_active_optimizer` is set, allocate a zeroed `p_sum` buffer of size `vocab_size` (float32).
     - Pass the `p_sum` pointer to the Triton kernel.
     - Call `_active_optimizer.accumulate_p_bar(p_sum, n_rows)` after execution.

### Phase 2: Optimizer State Management (`train_gpt.py`)
1. **Imports**: `from softmaxmuon import softmax_muon`.
2. **State Initialization**:
   - In `_init_state`, initialize the following for `softmaxmuon` parameters:
     - `p_bar_acc` (size `vocab_size`, FP32) and `p_bar_count` (float).
     - `momentum_buffer` (size of the parameter shard, FP32).
     - `mantissa` (size of the parameter shard, uint16).
3. **Accumulate Method**:
   - Implement `accumulate_p_bar(self, p_sum, n_rows)` to add the batch-level statistics to the running totals.
4. **Reset Logic**:
   - Update `NorMuonAndAdam.reset()` to zero out `p_bar_acc`, `p_bar_count`, `momentum_buffer`, and `mantissa`.
5. **Update Logic**:
   - Implement `_softmax_muon_update(self, param, grad_chunk, p_cfg, rank)`.
   - **Momentum Update**:
     - `momentum_buffer.lerp_(grad_chunk.float(), 1 - p_cfg.momentum)`
     - `updated_grads = grad_chunk.float().lerp_(momentum_buffer, p_cfg.momentum)`
   - **Global Sync (p_bar)**: `dist.all_reduce` both `p_bar_acc` and `p_bar_count`.
   - **Normalize (p_bar)**: `p_bar = p_bar_acc / p_bar_count`.
   - **Distributed Call**:
     - Invoke `softmax_muon(p_bar, updated_grads.T, ...)` with sharding-aware callbacks (all-gather B, all-gather K, localize sqrt K).
     - Result `W` is `(vocab, shard)`; use `W.T` for the parameter update.
   - **Update Parameter**: Use `NorMuonAndAdam._cautious_wd_and_update_inplace` with the tracked mantissa and effective learning rate/weight decay.
   - **Reset**: Zero out `p_bar_acc` and `p_bar_count` after the update.

### Phase 3: Configuration and Lifecycle
1. **Initialize Hook**: In `train_gpt.py`, set `triton_kernels._active_optimizer = training_manager.optimizer` before training.
2. **Param Table**: Update `lm_head` entry: `optim: "softmaxmuon"`, `adam_betas: None`, `wd_mul: 1.2` (matching `normuon` default).
3. **Schedule Alignment**: Ensure `softmaxmuon` follows the `do_adam` (odd step) schedule to stay tied with `embed`.

## 4. Verification Strategy
- **Probability Sum**: Assert that the globally synchronized `p_bar` sums to 1.0.
- **Tie Consistency**: Verify `lm_head.weight == embed.weight.T` after each update.
- **Convergence**: Run a short training window to ensure loss decreases without NaNs.
