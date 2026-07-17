# PI0 / PI0.5 Subtask M4 Completion Notes

Date: 2026-07-03

## Scope

Completed milestone M4 from `pi0_pi05_subtask_generation_plan.md` for PI0.5 inference:

- `src/lerobot/policies/pi05/modeling_pi05.py`
  - Added PI0.5 inference-time AR subtask generation with KV cache reuse.
  - Added seed-token prefill for `"Subtask:"` using a mask that attends to valid prefix tokens and is causal inside the seed segment.
  - Added incremental token decoding with the existing PaliGemma `lm_head`.
  - Added greedy decoding by default, with `subtask_decode_temperature > 0` sampling support.
  - Added EOS stopping while ensuring every valid generated token is also inserted into the KV cache.
  - Extended `prefix_pad_masks` after AR generation so the existing denoise path can attend to generated subtask tokens without changing `denoise_step()`.
  - Kept `sample_actions()` return type unchanged.
  - Kept default `predict_subtask=False` inference behavior on the original path.
- `PI05Policy`
  - Loads the PaliGemma tokenizer only when `config.predict_subtask=True`.
  - Precomputes subtask seed token ids and EOS id for the model.
  - Decodes the generated subtask tokens after `predict_action_chunk()`.
  - Stores the latest decoded text in `policy.last_subtask_text`.
  - Logs `[subtask] ...` only when the decoded text changes.
- Added `tests/policies/pi0_pi05/test_pi05_subtask_inference.py`.

## Verification

Installed local pytest dependencies into the existing `lerobot-main` conda environment using the configured `127.0.0.1:1080` proxy:

```bash
env PIP_REQUIRE_VIRTUALENV=false \
  http_proxy=http://127.0.0.1:1080 \
  https_proxy=http://127.0.0.1:1080 \
  conda run -n lerobot-main python -m pip install 'pytest>=8.1.0,<9.0.0' 'pytest-timeout>=2.4.0,<3.0.0'
```

The sandboxed network path could not connect to the proxy, so the install was run with approved escalated network access.

Passed syntax verification:

```bash
conda run -n lerobot-main python -m py_compile \
  src/lerobot/policies/pi05/modeling_pi05.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py
```

Passed PI0.5 M3+M4 policy tests:

```bash
conda run -n lerobot-main python -m pytest \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py -q
```

Result:

```text
10 passed in 1.45s
```

Passed M1-M4 focused subtask tests:

```bash
conda run -n lerobot-main python -m pytest \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  tests/processor/test_subtask_ar_processors.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py -q
```

Result:

```text
26 passed, 2 warnings in 1.80s
```

The warnings are the expected config warnings from the existing processor test where `subtask_max_decode_tokens > subtask_max_tokens` is intentionally exercised.

## Not Run

`ruff` is not installed in the current `lerobot-main` environment:

```text
No module named ruff
```

No full-library test suite, GPU smoke, or real PaliGemma checkpoint inference was run. M4 verification is limited to the focused PI0.5 AR inference behavior and the already completed M1-M3 local tests.

## Remaining Work

M5 remains: mirror the PI0.5 training and inference subtask AR changes into PI0, accounting for PI0's state token in the suffix sequence.
