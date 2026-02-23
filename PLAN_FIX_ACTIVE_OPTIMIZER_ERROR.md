# Plan: Fix HigherOrderOperator Mutation Error in Triton Kernels

## 1. Understanding the Goal
- **Problem**: `torch.compile` fails with `HigherOrderOperator: Mutating a variable not in the current scope` because the `FusedSoftcappedCrossEntropy.backward` function attempts to mutate a global `_active_optimizer` state.
- **Solution**: Refactor the accumulation of `p_bar_acc` and `p_bar_count` to use explicit PyTorch buffers passed through the autograd graph. This makes the mutations local to the traced scope and compatible with TorchDynamo.

## 2. Investigation Findings
- **Files Involved**: `train_gpt.py` (Model/Optimizer) and `triton_kernels.py` (Triton Kernels/Autograd).
- **Root Cause**: Dynamo prohibits side effects on global variables during the backward pass of a compiled `autograd.Function`.
- **Refactoring Strategy**:
  - Move `p_bar_acc` and `p_bar_count` from the optimizer state to `nn.Buffer` objects in the `lm_head` module.
  - Pass these buffers as arguments to `FusedSoftcappedCrossEntropy.apply`.
  - Update Triton kernels to perform `tl.atomic_add` directly on these buffers.

## 3. Step-by-Step Implementation Plan

### Phase 1: Model Buffer Integration (`train_gpt.py`)
1. **Modify `CastedLinearT`**:
   - In `__init__`, register two buffers: `p_bar_acc` (zeros, size `vocab_size`) and `p_bar_count` (zeros, size 1).
2. **Update `GPT.forward`**:
   - Pass `self.lm_head.p_bar_acc` and `self.lm_head.p_bar_count` as additional arguments to `FusedSoftcappedCrossEntropy.apply`.

### Phase 2: Triton and Autograd Refactoring (`triton_kernels.py`)
1. **Update Triton Kernel**:
   - Modify `fused_softcapped_entropy_bwd_kernel` signature to accept `p_sum_ptr` and `p_count_ptr`.
   - Inside the kernel, add `tl.atomic_add(p_count_ptr, 1.0)` to increment the global token count.
2. **Update Autograd Function**:
   - **Forward**: Accept `p_bar_acc` and `p_bar_count` as arguments. Save them to `ctx` using `ctx.save_for_backward`.
   - **Backward**: Retrieve the buffers from `ctx` and pass their data pointers to the Triton kernel.
3. **Cleanup**: Remove the global `_active_optimizer` registry and all references to it.

### Phase 3: Optimizer Cleanup (`train_gpt.py`)
1. **Refactor `NorMuonAndAdam._init_state`**:
   - Update `softmaxmuon` initialization to use the buffers attached to the parameter if they exist.
2. **Remove Accumulation Hook**:
   - Delete the `accumulate_p_bar` method from `NorMuonAndAdam`.
   - Remove the `triton_kernels._active_optimizer` initialization in the main script.
3. **Update Reset/Step Logic**:
   - Use `.zero_()` on the `p_bar_count` tensor instead of scalar assignment.

## 4. Verification Strategy
- **Compilation Check**: Verify the "Warming up kernels" phase completes without Dynamo errors.
- **State Inspection**: Confirm `p_bar_acc` and `p_bar_count` are non-zero after a few training iterations.
- **Numerical Consistency**: Assert that the synchronized `p_bar` sums to 1.0.
- **Convergence**: Run a short 50-step training loop to ensure the loss decreases as expected.
