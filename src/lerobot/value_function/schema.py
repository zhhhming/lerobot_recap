"""Shared names for value-function raw dataset artifacts."""

from lerobot.datasets.raw_media import RAW_FORMAT_VERSION
PIPELINE_SCHEMA_VERSION = 2

TARGET_STAGE = "targets"
ADVANTAGE_STAGE_PREFIX = "advantage"
ADVANTAGE_LABELING_STAGE_PREFIX = "advantage_labeling"
ADVANTAGE_WEIGHTS_STAGE_PREFIX = "advantage_weights"
VALUE_INFERENCE_STAGE_PREFIX = "value_inference"

PREDICTION_SOURCE_GT = "gt"
PREDICTION_SOURCE_MOCK = "mock_pred"
PREDICTION_SOURCE_MODEL = "model_pred"
MOCK_PREDICTIONS_STAGE = "mock_predictions"

RUN_META_FILENAME = "run_meta.json"
ANNOTATION_CONFIG_FILENAME = "annotation_config.json"
FRAMES_FILENAME = "frames.parquet"
EXTRAS_FILENAME = "extras.parquet"
VALUE_FUNCTION_META_FILENAME = "value_function_meta.json"

EPISODE_DIR_PREFIX = "ep_"
SUBTASK_COLUMN = "subtask"

VALUE_GLOBAL_REMAINING_FRAMES_GT = "value_global_remaining_frames_gt"
VALUE_GLOBAL_REMAINING_NORM_GT = "value_global_remaining_norm_gt"
VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED = "value_global_remaining_norm_gt_is_clipped"
VALUE_GLOBAL_ELAPSED_FRAMES_GT = "value_global_elapsed_frames_gt"
VALUE_GLOBAL_ELAPSED_NORM_GT = "value_global_elapsed_norm_gt"

VALUE_SUBTASK_ID_GT = "value_subtask_id_gt"
VALUE_SUBTASK_NAME_GT = "value_subtask_name_gt"
VALUE_SUBTASK_REMAINING_FRAMES_GT = "value_subtask_remaining_frames_gt"
VALUE_SUBTASK_REMAINING_NORM_GT = "value_subtask_remaining_norm_gt"
VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED = "value_subtask_remaining_norm_gt_is_clipped"
VALUE_SUBTASK_ELAPSED_FRAMES_GT = "value_subtask_elapsed_frames_gt"
VALUE_SUBTASK_ELAPSED_NORM_GT = "value_subtask_elapsed_norm_gt"

VALUE_GLOBAL_REMAINING_NORM_PRED = "value_global_remaining_norm_pred"
VALUE_GLOBAL_REMAINING_FRAMES_PRED = "value_global_remaining_frames_pred"
VALUE_SUBTASK_ID_PRED = "value_subtask_id_pred"
VALUE_SUBTASK_CONFIDENCE = "value_subtask_confidence"
VALUE_SUBTASK_ID_PRED_SMOOTH = "value_subtask_id_pred_smooth"
VALUE_SUBTASK_NAME_PRED_SMOOTH = "value_subtask_name_pred_smooth"
VALUE_SUBTASK_REMAINING_NORM_PRED = "value_subtask_remaining_norm_pred"
VALUE_SUBTASK_REMAINING_FRAMES_PRED = "value_subtask_remaining_frames_pred"
VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD = "value_subtask_remaining_norm_pred_gt_head"
VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD = "value_subtask_remaining_frames_pred_gt_head"
VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD = (
    "value_subtask_remaining_norm_pred_smooth_head"
)
VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD = (
    "value_subtask_remaining_frames_pred_smooth_head"
)

VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED = "value_global_remaining_frames_mock_pred"
VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED = "value_subtask_remaining_frames_mock_pred"

ADVANTAGE_GLOBAL_CHUNK = "advantage_global_chunk"
ADVANTAGE_GLOBAL_VALID_HORIZON = "advantage_global_valid_horizon"
ADVANTAGE_GLOBAL_IS_VALID = "advantage_global_is_valid"
ADVANTAGE_GLOBAL_START_VALUE = "advantage_global_start_value"
ADVANTAGE_GLOBAL_END_VALUE = "advantage_global_end_value"

ADVANTAGE_SUBTASK_CHUNK = "advantage_subtask_chunk"
ADVANTAGE_SUBTASK_VALID_HORIZON = "advantage_subtask_valid_horizon"
ADVANTAGE_SUBTASK_IS_VALID = "advantage_subtask_is_valid"
ADVANTAGE_SUBTASK_START_VALUE = "advantage_subtask_start_value"
ADVANTAGE_SUBTASK_END_VALUE = "advantage_subtask_end_value"
ADVANTAGE_SUBTASK_NUM_CROSSINGS = "advantage_subtask_num_crossings"
ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON = "advantage_subtask_within_subtask_horizon"
ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS = "advantage_subtask_boundary_progress"

ADVANTAGE_LABEL_GLOBAL = "advantage_label_global"
ADVANTAGE_LABEL_SUBTASK = "advantage_label_subtask"

ADVANTAGE_GROUP_ID_GLOBAL = "advantage_group_id_global"
ADVANTAGE_GROUP_ID_SUBTASK = "advantage_group_id_subtask"
ADVANTAGE_LOSS_WEIGHT_GLOBAL = "advantage_loss_weight_global"
ADVANTAGE_LOSS_WEIGHT_SUBTASK = "advantage_loss_weight_subtask"

GLOBAL_GT_COLUMNS = (
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
)

SUBTASK_GT_COLUMNS = (
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_NAME_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
)

TARGET_COLUMNS = (
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
    VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_NAME_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
    VALUE_SUBTASK_ELAPSED_FRAMES_GT,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
)
