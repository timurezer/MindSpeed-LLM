# QLoRA support on current master

## Current state

- Work on branch `dev/qlora_support`, based on master commit `90f862cd`.
- Commit `22a7581a` restores the base QLoRA infrastructure removed by `07e65e55`.
- Commit `032fd468` adds minimal DeepSeek-V4-Flash QLoRA conversion and runtime support.
- This is basic functionality only. Do not add bnb patches, expert chunking, fused kernels, or other optimizations unless requested.
- Full DeepSeek checkpoint conversion and distributed training have not been verified.

See `docs/qlora_master_support.md` for the implementation map and validation status.

## Constraints

- **Do not run anything on the NPUs.** The user explicitly prohibited NPU execution. Use local/static or CPU-only checks unless the user gives new explicit permission.
- Preserve current master fixes, including the sequence auxiliary-loss and LoRA pipeline-stage-0 fixes. Do not replace current files wholesale with versions from the old QLoRA fork.
- Do not apply `patches/shanghai_changes.patch` from the old project directly; it contains debug code and obsolete implementation details.
- `convert_ckpt_v2.py` and some checkpoint files use CRLF line endings. Preserve them. For whitespace checks, use `git -c core.whitespace=cr-at-eol diff --check`.
- Keep ordinary bf16 conversion and LoRA behavior unchanged when QLoRA is disabled.

## Repository locations

- Local: `/home/timur/projects/qlora_mastered/MindSpeed-LLM`
- node1 mirror: `/data/lab-shared/timur/qlora_mastered/MindSpeed-LLM`

The mirror is for inspection and future authorized testing. Keep it synchronized after changes.
