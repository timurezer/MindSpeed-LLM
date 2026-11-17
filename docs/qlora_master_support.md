# QLoRA support restored on master

## Purpose and scope

This branch restores basic QLoRA support on current master and adapts it for DeepSeek-V4-Flash. It intentionally contains no performance or memory optimizations beyond the original QLoRA design.

The starting point is master `90f862cd`. The implementation is split into:

1. `22a7581a` (`Restore base QLoRA support`) restores the generic QLoRA arguments, patches, NF4 linear and grouped-expert handling, dependency declaration, and original integration tests.
2. `032fd468` (`Add basic DeepSeek-V4 QLoRA support`) adds DeepSeek-specific NF4 conversion, mixed-precision loading, and required device/dequantization handling.

The old `ds_qlora_support` branch was used as a reference, but its patches were not applied wholesale. In particular, debug prints, breakpoints, old export code, bnb patches, expert chunking, CPU quantization, fused DSA kernels, and long-context optimizations were excluded.

## Implementation map

### Generic QLoRA

- `mindspeed_llm/features_manager/finetune/lora.py`
  - Adds `--qlora` and `--qlora-save-dequantize`.
  - Registers the QLoRA model builder and linear/grouped-GEMM patches only when QLoRA is enabled.
  - Ordinary LoRA continues to use master's `get_model_wrapper`.
- `mindspeed_llm/tasks/posttrain/lora/qlora.py`
  - Restored from the pre-sunset implementation.
  - Keeps frozen target linear weights on CPU until the prequantized checkpoint is loaded.
  - Loads `Params4bit` weights and dequantizes them for linear and grouped-GEMM forward/backward paths.
  - Detects NF4 using the current module prefix. This is required because DeepSeek checkpoints deliberately contain both NF4 and bf16 linears.
- `mindspeed_llm/core/tensor_parallel/layers.py`
  - Restores NF4 handling for `LinearNoTP`.
- `mindspeed_llm/tasks/posttrain/lora/moe/experts.py` and `cc_lora_forward.py`
  - Restore dequantization of frozen grouped-expert and fused-LoRA base weights.
- `convert_ckpt_v2.py`
  - Restores `--qlora-nf4`.

### DeepSeek-V4-Flash

- `mindspeed_llm/tasks/checkpoint/convert_ckpt_deepseek4.py`
  - `DeepSeek4Hf2MgConvert._store_weight` is the single conversion helper.
  - Without `--qlora-nf4`, it stores a clone of the original tensor, preserving existing bf16 conversion.
  - With `--qlora-nf4`, it stores packed `uint8` data plus `absmax`, `quant_map`, and `quant_state.bitsandbytes__nf4`.
  - It is used for the output layer, MTP projection, the same four attention projections as the old fork, shared experts, and routed experts in grouped and non-grouped layouts.
  - Transformer Engine conversion is rejected for NF4 because this implementation currently supports the local transformer path only.
- `mindspeed_llm/tasks/models/transformer/deepseek4/csa.py`
  - Master uses CSA rather than the old fork's `g2_attention.py`.
  - `_linear_o_down_proj` explicitly dequantizes its frozen weight before rearranging it. Master's existing LoRA A/B handling is retained.
- `mindspeed_llm/tasks/models/transformer/deepseek4/compressor.py` and `mindspeed_llm/core/transformer/moe/moe_utils.py`
  - Handle parameters/tables that can remain on CPU under QLoRA's selective model placement.
- Master's own DeepSeek LoRA export implementation is unchanged.

## Validation completed

- Python syntax compilation passed for the changed modules.
- `git -c core.whitespace=cr-at-eol diff --check master..HEAD` passed.
- The working tree was clean after the two implementation commits.
- Imports passed in `lab-th_va25_qlora` with bitsandbytes 0.45.3.
- The bf16 `_store_weight` path preserved dtype and values and returned an independent tensor.
- Before NPU access was paused because colleagues were using the devices, two small smoke checks passed:
  - a 64x64 bf16 tensor produced packed NF4 data and all three expected quantization-state entries;
  - a mixed state dict loaded one prefix as NF4 while leaving another prefix bf16.

## Not yet verified

- Full HF-to-Megatron DeepSeek NF4 conversion.
- Checkpoint key inspection on a real full-model shard.
- Distributed forward/backward and optimizer steps.
- Loss agreement with bf16 LoRA.
- Adapter saving and export.

No full conversion or training run has been performed. Do not claim end-to-end support is validated until these checks are completed. NPU workloads require the user's explicit go-ahead and must not run while devices are in use by colleagues.
