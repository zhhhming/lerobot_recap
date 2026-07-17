# PI0 / PI0.5 Subtask M5 Completion Notes

Date: 2026-07-03

## Scope

Completed milestone M5 from `pi0_pi05_subtask_generation_plan.md` by mirroring the PI0.5 subtask AR training and inference paths into PI0.

- `src/lerobot/policies/pi0/modeling_pi0.py`
  - Added PI0 subtask CE helper and subtask attention dropout helper.
  - `PI0Pytorch.embed_prefix()` now accepts optional `subtask_tokens` and `subtask_masks`.
  - `PI0Pytorch.forward()` now accepts optional subtask tensors and returns `(fm_losses, ce_loss_per_sample)`.
  - Teacher-forced subtask CE is computed from the PaliGemma prefix output over the appended subtask segment.
  - Subtask dropout masks the full PI0 suffix from subtask columns.
  - PI0-specific suffix handling uses `suffix_len = 1 + chunk_size`, so both the state token row and all action rows are dropped when selected.
  - `PI0Pytorch.sample_actions()` now optionally runs inference-time subtask AR generation before denoising.
  - Added PI0 `_generate_subtask()` with KV-cache seed prefill, seed causal mask, greedy/sampling decode, EOS stop, and prefix mask extension.
  - Existing `denoise_step()` remains unchanged and receives the extended `prefix_pad_masks`.
  - `PI0Policy` loads the PaliGemma tokenizer only when `predict_subtask=True`.
  - `PI0Policy.forward()` passes subtask tensors when enabled and aggregates `fm_loss + subtask_ce_loss_weight * ce_loss` for both `mean` and `none`.
  - `PI0Policy.predict_action_chunk()` decodes generated subtask tokens into `policy.last_subtask_text` and logs only when the text changes.

## Tests Added

- `tests/policies/pi0_pi05/test_pi0_subtask_training.py`
  - PI0 prefix/subtask/state/action mask layout.
  - Masked shifted subtask CE.
  - Dropout masking of state + action suffix rows.
  - PI0 policy loss aggregation for `mean` and `none`.
  - FM-only behavior when subtask prediction is disabled.
- `tests/policies/pi0_pi05/test_pi0_subtask_inference.py`
  - Seed prefill attention mask.
  - KV-cache subtask generation and EOS stop.
  - `sample_actions()` AR gating by config flags.
  - Extended prefix mask reaching denoise.
  - State tensor still passed through the PI0 denoise path.
  - `PI0Policy.predict_action_chunk()` decoded text exposure.

## Verification

Passed syntax verification:

```bash
conda run -n lerobot-main python -m py_compile \
  src/lerobot/policies/pi0/modeling_pi0.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py
```

Passed PI0 M5 tests:

```bash
conda run -n lerobot-main python -m pytest \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py -q
```

Result:

```text
10 passed in 1.50s
```

Passed M1-M5 focused subtask tests:

```bash
conda run -n lerobot-main python -m pytest \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  tests/processor/test_subtask_ar_processors.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py -q
```

Result:

```text
36 passed, 2 warnings in 1.87s
```

The warnings are the expected config warnings from the processor test case that intentionally sets `subtask_max_decode_tokens > subtask_max_tokens`.

## Not Run

`ruff` is not installed in the current `lerobot-main` environment:

```text
No module named ruff
```

No full-library test suite, GPU smoke, real PaliGemma checkpoint inference, overfit run, or real robot deployment was run. Those remain part of M6/integration validation.

## Remaining Work

M6 remains: run small labeled-data overfit training and real deployment smoke for PI0/PI0.5 subtask AR behavior.
