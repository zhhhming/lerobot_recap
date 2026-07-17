# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import builtins
import datetime as dt
import math
import numbers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, PeftConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # RA-BC (Reward-Aligned Behavior Cloning) parameters
    use_rabc: bool = False  # Enable reward-weighted training
    rabc_progress_path: str | None = None  # Path to precomputed SARM progress parquet file
    rabc_kappa: float = 0.01  # Hard threshold for high-quality samples
    rabc_epsilon: float = 1e-6  # Small constant for numerical stability
    rabc_head_mode: str | None = "sparse"  # For dual-head models: "sparse" or "dense"

    # Offline group-relative advantage weighting and train-only condition dropout.
    use_advantage_weighting: bool = False
    advantage_loss_weight_key: str = "advantage_loss_weight_global"
    advantage_label_key: str = "advantage_label_global"
    advantage_condition_dropout_prob: float = 0.1
    advantage_ignore_label: str = "ignore"
    advantage_disable_weight_when_condition_dropped: bool = True

    # Dynamic previous-subtask history and train-only memory condition dropout.
    memory_lookback_min_frames: int = 1
    memory_lookback_max_frames: int = 12
    memory_dropout_prob: float = 0.2

    # Current-subtask elapsed-time noise and independent condition dropout.
    subtask_time_noise_ratio: float = 0.4
    subtask_time_noise_max_seconds: float = 5.0
    subtask_time_dropout_prob: float = 0.2

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    def validate(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )

            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )

            policy_dir = Path(config_path).parent
            if self.policy is not None:
                self.policy.pretrained_path = policy_dir
            self.checkpoint_path = policy_dir.parent

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with `--policy.path`."
            )

        if (
            isinstance(self.memory_lookback_min_frames, bool)
            or isinstance(self.memory_lookback_max_frames, bool)
            or not isinstance(self.memory_lookback_min_frames, numbers.Integral)
            or not isinstance(self.memory_lookback_max_frames, numbers.Integral)
            or self.memory_lookback_min_frames < 1
            or self.memory_lookback_min_frames > self.memory_lookback_max_frames
        ):
            raise ValueError(
                "memory lookback must satisfy 1 <= memory_lookback_min_frames <= "
                "memory_lookback_max_frames, got "
                f"{self.memory_lookback_min_frames} and {self.memory_lookback_max_frames}"
            )
        if not 0.0 <= self.memory_dropout_prob <= 1.0:
            raise ValueError(
                f"memory_dropout_prob must be in [0, 1], got {self.memory_dropout_prob}"
            )
        use_memory_conditioning = bool(
            getattr(self.policy, "use_memory_conditioning", False)
        )
        if use_memory_conditioning:
            if self.policy.type not in {"pi0", "pi05"}:
                raise ValueError(
                    "Memory conditioning currently supports only pi0 and pi05 policies, "
                    f"got {self.policy.type!r}"
                )
            if self.dataset.streaming:
                raise ValueError(
                    "Memory conditioning requires a non-streaming dataset; "
                    "set --dataset.streaming=false."
                )

        for field_name in ("subtask_time_noise_ratio", "subtask_time_noise_max_seconds"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and >= 0, got {value!r}")
        if (
            isinstance(self.subtask_time_dropout_prob, bool)
            or not isinstance(self.subtask_time_dropout_prob, numbers.Real)
            or not math.isfinite(float(self.subtask_time_dropout_prob))
            or not 0.0 <= self.subtask_time_dropout_prob <= 1.0
        ):
            raise ValueError(
                "subtask_time_dropout_prob must be finite and in [0, 1], got "
                f"{self.subtask_time_dropout_prob!r}"
            )
        use_subtask_time_conditioning = bool(
            getattr(self.policy, "use_subtask_time_conditioning", False)
        )
        if use_subtask_time_conditioning:
            if self.policy.type not in {"pi0", "pi05"}:
                raise ValueError(
                    "Subtask time conditioning currently supports only pi0 and pi05 policies, "
                    f"got {self.policy.type!r}"
                )
            if not bool(getattr(self.policy, "predict_subtask", False)):
                raise ValueError("Subtask time conditioning requires predict_subtask=True.")
            if self.dataset.streaming:
                raise ValueError(
                    "Subtask time conditioning requires a non-streaming dataset; "
                    "set --dataset.streaming=false."
                )

        if not 0.0 <= self.advantage_condition_dropout_prob <= 1.0:
            raise ValueError(
                "advantage_condition_dropout_prob must be in [0, 1], got "
                f"{self.advantage_condition_dropout_prob}"
            )
        if not self.advantage_loss_weight_key:
            raise ValueError("advantage_loss_weight_key must be non-empty")
        if not self.advantage_label_key:
            raise ValueError("advantage_label_key must be non-empty")
        if self.advantage_ignore_label != "ignore":
            raise ValueError("The first advantage-weighting version only supports ignore label 'ignore'")
        if self.use_rabc and self.use_advantage_weighting:
            raise ValueError("use_rabc and use_advantage_weighting cannot be enabled together")
        if self.use_advantage_weighting and self.policy.type not in {"pi0", "pi05"}:
            raise ValueError(
                "use_advantage_weighting currently supports only pi0 and pi05 policies, got "
                f"{self.policy.type!r}"
            )

        use_advantage_conditioning = getattr(self.policy, "use_advantage_conditioning", False)
        if use_advantage_conditioning:
            policy_label_key = getattr(self.policy, "advantage_label_key", None)
            if policy_label_key != self.advantage_label_key:
                raise ValueError(
                    "Training and policy advantage label keys must match: "
                    f"{self.advantage_label_key!r} != {policy_label_key!r}"
                )
        if self.use_advantage_weighting:
            policy_weight_key = getattr(self.policy, "advantage_loss_weight_key", None)
            if policy_weight_key != self.advantage_loss_weight_key:
                raise ValueError(
                    "Training and policy advantage loss weight keys must match: "
                    f"{self.advantage_loss_weight_key!r} != {policy_weight_key!r}"
                )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

        if self.use_rabc and not self.rabc_progress_path:
            # Auto-detect from dataset path
            repo_id = self.dataset.repo_id
            if self.dataset.root:
                self.rabc_progress_path = str(Path(self.dataset.root) / "sarm_progress.parquet")
            else:
                self.rabc_progress_path = f"hf://datasets/{repo_id}/sarm_progress.parquet"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional
