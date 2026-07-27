#!/usr/bin/env python

"""Manual episode recording into a raw on-disk format (not LeRobotDataset).

Each episode is written to its own folder so individual episodes can be
inspected, deleted, or annotated before being assembled into a LeRobotDataset
by `lerobot-build-dataset`.

Layout:

    <root>/                       (default: $HF_LEROBOT_HOME/raw/<repo_id>)
        run_meta.json             # feature schema + run-level config
        ep_000000/
            info.json             # per-episode metadata
            frames.parquet        # per-frame scalar features
            events.jsonl          # source switches, subtask markers, pauses
            <cam_key>/000000.jpg  # one folder per camera (PNG for legacy runs)
        ep_000001/ ...

This script reuses the control loop, state machine, and action sources from
`lerobot_hil_record` and only swaps the recording backend.
"""

import os

_no_proxy = os.environ.get("no_proxy", "")
for _host in ("127.0.0.1", "localhost"):
    if _host not in _no_proxy.split(","):
        _no_proxy = (_no_proxy + "," + _host).strip(",")
os.environ["no_proxy"] = _no_proxy
os.environ["NO_PROXY"] = _no_proxy

import json
import logging
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from threading import Event
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import AsyncImageWriter
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.raw_media import (
    RAW_FORMAT_VERSION,
    RawImageEncoding,
    RawImageFormat,
    camera_subdir_name,
    make_raw_image_encoding,
    raw_frame_image_path,
    raw_image_encoding_from_meta,
    validate_raw_format_version,
)
from lerobot.inference_engines.robot_wrapper import ThreadSafeRobot
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, RTCConfig
from lerobot.processor import make_default_processors

# Import config modules so draccus can resolve --robot.type / --teleop.type.
from lerobot.robots import (  # noqa: F401,E402
    RobotConfig,
    bi_nero_follower,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    nero_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.scripts.lerobot_hil_record import (
    ActionSmoother,
    ControlMode,
    HILStateMachine,
    InferenceType,
    KeyboardEvents,
    LoopTimer,
    ObsCache,
    PolicySource,
    TeleopSource,
    _action_from_observation,
    _build_engine,
    _init_keyboard_listener,
    _ordered_action_keys,
    _resolve_control_rate,
    _smooth_action_keys,
)
from lerobot.scripts.lerobot_policy_deploy import (
    _check_policy_compatibility,
    _default_home_joints_from_robot_config,
    _load_policy_from_model_dir,
    _start_homing,
)
from lerobot.teleoperators import (  # noqa: F401,E402
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_pico_nero_teleop,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    pico_nero_teleop,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


RUN_META_FILENAME = "run_meta.json"
EP_DIR_PATTERN = "ep_{:06d}"


@dataclass
class RawRecordDatasetConfig:
    repo_id: str
    single_task: str = (
        "Pick up the match in front, strike it to light it, then use it to light the small candle "
        "on the cake in front."
    )
    root: str | Path | None = None
    fps: int = 30
    tags: list[str] | None = None
    force: bool = False
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    image_format: RawImageFormat = "jpeg"
    jpeg_quality: int = 95
    jpeg_subsampling: int = 0
    png_compress_level: int = 3

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide --dataset.single_task.")
        make_raw_image_encoding(
            self.image_format,
            jpeg_quality=self.jpeg_quality,
            jpeg_subsampling=self.jpeg_subsampling,
            png_compress_level=self.png_compress_level,
        )


@dataclass
class RawRecordConfig:
    robot: RobotConfig
    dataset: RawRecordDatasetConfig
    mode: ControlMode = "hil"
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    inference_type: InferenceType = "sync"
    rtc: RTCConfig = field(default_factory=RTCConfig)
    rtc_queue_threshold: int = 40
    interpolation_multiplier: int = 1
    control_multiplier: int | None = 3
    control_hz: float | None = None
    smoother_alpha: float = 0.2
    policy_gripper_max_width_m: float = 0.1
    display_data: bool = True
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    home_joints_rad: list[float] | None = None
    home_speed_rad_s: float = 0.5

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.mode in ("policy", "hil") and self.policy is None:
            raise ValueError(f"mode={self.mode!r} requires --policy.path.")
        if self.mode in ("teleop", "hil") and self.teleop is None:
            raise ValueError(f"mode={self.mode!r} requires --teleop.type.")
        if self.inference_type == "rtc":
            self.rtc.enabled = True
        if self.interpolation_multiplier < 1:
            raise ValueError("--interpolation_multiplier must be >= 1.")
        if self.control_multiplier is not None and self.control_multiplier < 1:
            raise ValueError("--control_multiplier must be >= 1.")
        if self.control_hz is not None and self.control_hz <= 0:
            raise ValueError("--control_hz must be > 0.")
        if not 0 < self.smoother_alpha <= 1:
            raise ValueError("--smoother_alpha must be in (0, 1].")
        if not 0 < self.policy_gripper_max_width_m <= 0.1:
            raise ValueError("--policy_gripper_max_width_m must be in (0, 0.1].")
        if self.home_speed_rad_s <= 0:
            raise ValueError("--home_speed_rad_s must be > 0.")
        if self.resume and self.dataset.force:
            raise ValueError("--dataset.force cannot be used with --resume.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _resolve_raw_root(repo_id: str, root: str | Path | None) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / "raw" / repo_id


def _camera_subdir_name(image_key: str) -> str:
    """observation.images.cam_top -> cam_top."""
    return camera_subdir_name(image_key)


def _serialize_features(features: dict) -> dict:
    """Convert feature dict (with tuples) to a JSON-friendly dict."""
    out = {}
    for k, v in features.items():
        ft = dict(v)
        shape = ft.get("shape")
        if isinstance(shape, tuple):
            ft["shape"] = list(shape)
        out[k] = ft
    return out


def _deserialize_features(features: dict) -> dict:
    out = {}
    for k, v in features.items():
        ft = dict(v)
        shape = ft.get("shape")
        if isinstance(shape, list):
            ft["shape"] = tuple(shape)
        out[k] = ft
    return out


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class RawRun:
    """Owns the on-disk raw run directory and per-episode lifecycle."""

    def __init__(
        self,
        root: Path,
        features: dict,
        fps: int,
        task: str,
        robot_type: str,
        run_config: dict,
        image_encoding: RawImageEncoding,
    ) -> None:
        self.root = Path(root)
        self.features = features
        self.fps = fps
        self.task = task
        self.robot_type = robot_type
        self.run_config = run_config
        self.image_encoding = image_encoding

        self.image_keys = [k for k, v in features.items() if v["dtype"] in ("image", "video")]
        self.scalar_keys = [k for k, v in features.items() if v["dtype"] not in ("image", "video")]
        self._next_episode_index = self._scan_existing_episodes()

    @classmethod
    def create(
        cls,
        root: Path,
        features: dict,
        fps: int,
        task: str,
        robot_type: str,
        run_config: dict,
        image_encoding: RawImageEncoding,
    ) -> "RawRun":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Raw run directory '{root}' is not empty. Use --resume to continue or "
                f"--dataset.force to wipe it."
            )
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": RAW_FORMAT_VERSION,
            "fps": fps,
            "task": task,
            "robot_type": robot_type,
            "features": _serialize_features(features),
            "image_encoding": image_encoding.to_metadata(),
            "run_config": run_config,
            "created_at": _iso_now(),
        }
        with open(root / RUN_META_FILENAME, "w") as f:
            json.dump(meta, f, indent=2)
        return cls(root, features, fps, task, robot_type, run_config, image_encoding)

    @classmethod
    def resume(cls, root: Path) -> "RawRun":
        meta_path = root / RUN_META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"No {RUN_META_FILENAME} in '{root}'; cannot resume.")
        with open(meta_path) as f:
            meta = json.load(f)
        validate_raw_format_version(meta, root)
        return cls(
            root=root,
            features=_deserialize_features(meta["features"]),
            fps=meta["fps"],
            task=meta["task"],
            robot_type=meta["robot_type"],
            run_config=meta.get("run_config", {}),
            image_encoding=raw_image_encoding_from_meta(meta),
        )

    def _scan_existing_episodes(self) -> int:
        if not self.root.exists():
            return 0
        existing = sorted(d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("ep_"))
        if not existing:
            return 0
        last = existing[-1]
        try:
            return int(last.split("_", 1)[1]) + 1
        except (IndexError, ValueError):
            return 0

    @property
    def num_episodes(self) -> int:
        return self._next_episode_index

    @property
    def next_episode_index(self) -> int:
        return self._next_episode_index

    def episode_dir(self, episode_index: int) -> Path:
        return self.root / EP_DIR_PATTERN.format(episode_index)

    def commit_episode(self) -> int:
        committed = self._next_episode_index
        self._next_episode_index += 1
        return committed


class RawEpisodeRecorder:
    """Per-episode recorder writing independently addressable image frames and parquet rows."""

    def __init__(
        self,
        run: RawRun,
        display_data: bool,
        display_compressed_images: bool,
        image_writer: AsyncImageWriter | None,
        timer: LoopTimer | None = None,
    ) -> None:
        self._run = run
        self._display_data = display_data
        self._display_compressed_images = display_compressed_images
        self._image_writer = image_writer
        self._timer = timer

        self._episode_dir: Path | None = None
        self._episode_index: int | None = None
        self._frame_rows: list[dict] = []
        self._events: list[dict] = []
        self._started_at: str | None = None
        self._episode_start_perf: float | None = None
        self.frames = 0

    def _time(self, name: str):
        if self._timer is None:
            return _NullContext()
        return self._timer.block(name)

    def push_event(self, event_type: str, **payload) -> None:
        if self._episode_dir is None:
            return
        self._events.append(
            {
                "frame_index": self.frames,
                "wall_time_s": _wall_time(self._episode_start_perf),
                "event": event_type,
                **payload,
            }
        )

    def _ensure_episode_started(self, source: str) -> None:
        if self._episode_dir is not None:
            return
        idx = self._run.next_episode_index
        ep_dir = self._run.episode_dir(idx)
        ep_dir.mkdir(parents=True, exist_ok=False)
        for cam_key in self._run.image_keys:
            (ep_dir / _camera_subdir_name(cam_key)).mkdir(parents=True, exist_ok=True)

        self._episode_dir = ep_dir
        self._episode_index = idx
        self._frame_rows = []
        self._events = []
        self._started_at = _iso_now()
        self._episode_start_perf = time.perf_counter()
        self.frames = 0
        self.push_event("episode_started", source=source)

    def add(
        self,
        obs_processed: dict[str, Any],
        robot_action: dict[str, float],
        source: str,
    ) -> None:
        self._ensure_episode_started(source)
        assert self._episode_dir is not None and self._episode_index is not None

        frame_index = self.frames
        wall_time = _wall_time(self._episode_start_perf)

        with self._time("rec_build_frame"):
            obs_frame = build_dataset_frame(self._run.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(self._run.features, robot_action, prefix=ACTION)

        with self._time("rec_save_image"):
            for cam_key in self._run.image_keys:
                if cam_key not in obs_frame:
                    continue
                img = obs_frame[cam_key]
                fpath = raw_frame_image_path(
                    self._episode_dir,
                    _camera_subdir_name(cam_key),
                    frame_index,
                    self._run.image_encoding,
                )
                save_kwargs = self._run.image_encoding.pillow_save_kwargs()
                if self._image_writer is not None:
                    self._image_writer.save_image(img, fpath, **save_kwargs)
                else:
                    from lerobot.datasets.image_writer import write_image

                    write_image(img, fpath, **save_kwargs)

        row: dict[str, Any] = {
            "frame_index": frame_index,
            "wall_time_s": wall_time,
            "source": source,
            "task": self._run.task,
        }
        for k in self._run.scalar_keys:
            if k in obs_frame:
                row[k] = obs_frame[k].astype(np.float32, copy=False).tolist()
            elif k in action_frame:
                row[k] = action_frame[k].astype(np.float32, copy=False).tolist()
        self._frame_rows.append(row)
        self.frames += 1

        if self._display_data:
            with self._time("rec_rerun"):
                log_rerun_data(
                    observation=obs_processed,
                    action=robot_action,
                    compress_images=self._display_compressed_images,
                )

    def save(self) -> int | None:
        if self._episode_dir is None or self.frames == 0:
            self.cancel()
            return None
        assert self._episode_index is not None

        if self._image_writer is not None:
            self._image_writer.wait_until_done()

        self.push_event("episode_saved")

        self._write_frames_parquet()
        self._write_events_jsonl()
        self._write_info_json()

        committed = self._run.commit_episode()
        ep_dir = self._episode_dir
        self._reset_state()
        logger.info("Saved raw episode %d to %s", committed, ep_dir)
        return committed

    def cancel(self) -> None:
        if self._episode_dir is not None:
            if self._image_writer is not None:
                self._image_writer.wait_until_done()
            if self._episode_dir.exists():
                shutil.rmtree(self._episode_dir, ignore_errors=True)
        self._reset_state()

    def _reset_state(self) -> None:
        self._episode_dir = None
        self._episode_index = None
        self._frame_rows = []
        self._events = []
        self._started_at = None
        self._episode_start_perf = None
        self.frames = 0

    def _write_frames_parquet(self) -> None:
        assert self._episode_dir is not None
        fields = [
            pa.field("frame_index", pa.int64()),
            pa.field("wall_time_s", pa.float64()),
            pa.field("source", pa.string()),
            pa.field("task", pa.string()),
        ]
        for k in self._run.scalar_keys:
            fields.append(pa.field(k, pa.list_(pa.float32())))
        schema = pa.schema(fields)
        table = pa.Table.from_pylist(self._frame_rows, schema=schema)
        pq.write_table(table, self._episode_dir / "frames.parquet")

    def _write_events_jsonl(self) -> None:
        assert self._episode_dir is not None
        with open(self._episode_dir / "events.jsonl", "w") as f:
            for e in self._events:
                f.write(json.dumps(e) + "\n")

    def _write_info_json(self) -> None:
        assert self._episode_dir is not None and self._episode_index is not None
        ended_at_perf = time.perf_counter()
        duration_s = ended_at_perf - self._episode_start_perf if self._episode_start_perf is not None else 0.0
        source_counts: dict[str, int] = {}
        for row in self._frame_rows:
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        info = {
            "episode_index": self._episode_index,
            "length": self.frames,
            "task": self._run.task,
            "fps": self._run.fps,
            "started_at": self._started_at,
            "ended_at": _iso_now(),
            "duration_s": duration_s,
            "source_counts": source_counts,
        }
        with open(self._episode_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _wall_time(start_perf: float | None) -> float:
    if start_perf is None:
        return 0.0
    return time.perf_counter() - start_perf


def _is_safe_to_remove_root(root: Path) -> bool:
    if root.is_symlink() or not root.is_dir():
        return False
    try:
        next(root.iterdir())
    except StopIteration:
        return True
    return (root / RUN_META_FILENAME).is_file()


def _remove_root(root: Path, *, reason: str) -> None:
    if not root.exists():
        return
    if not _is_safe_to_remove_root(root):
        raise FileExistsError(
            f"Refusing to remove existing directory '{root}'. "
            f"{reason} only removes empty directories or raw run directories "
            f"containing '{RUN_META_FILENAME}'."
        )
    logger.warning("Removing raw run directory '%s' (%s).", root, reason)
    shutil.rmtree(root)


@parser.wrap()
def raw_record(cfg: RawRecordConfig) -> RawRun:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="raw_record", ip=cfg.display_ip, port=cfg.display_port)

    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = teleop_action_processor.transform_features(
        create_initial_features(action=robot.action_features)
    )
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=action_features,
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    run: RawRun | None = None
    listener = None
    engine = None
    inference_shutdown_event = Event()
    image_writer: AsyncImageWriter | None = None
    created_root: Path | None = None

    try:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        root = _resolve_raw_root(cfg.dataset.repo_id, cfg.dataset.root)

        run_config = {
            "mode": cfg.mode,
            "control_multiplier": cfg.control_multiplier,
            "control_hz": cfg.control_hz,
            "interpolation_multiplier": cfg.interpolation_multiplier,
            "smoother_alpha": cfg.smoother_alpha,
            "repo_id": cfg.dataset.repo_id,
        }

        if cfg.resume:
            run = RawRun.resume(root)
            if run.fps != cfg.dataset.fps:
                raise ValueError(f"Resume fps mismatch: existing={run.fps}, requested={cfg.dataset.fps}")
            if run.robot_type != robot.name:
                raise ValueError(f"Resume robot mismatch: existing={run.robot_type}, requested={robot.name}")
            logger.info(
                "Resuming raw run at %s (next episode = %d, image encoding = %s)",
                root,
                run.num_episodes,
                run.image_encoding.to_metadata(),
            )
        else:
            if cfg.dataset.force:
                _remove_root(root, reason="--dataset.force was set")
            root_existed_before = root.exists() and any(root.iterdir()) if root.exists() else False
            try:
                run = RawRun.create(
                    root=root,
                    features=dataset_features,
                    fps=cfg.dataset.fps,
                    task=cfg.dataset.single_task,
                    robot_type=robot.name,
                    run_config=run_config,
                    image_encoding=make_raw_image_encoding(
                        cfg.dataset.image_format,
                        jpeg_quality=cfg.dataset.jpeg_quality,
                        jpeg_subsampling=cfg.dataset.jpeg_subsampling,
                        png_compress_level=cfg.dataset.png_compress_level,
                    ),
                )
                created_root = run.root
            except Exception:
                if not root_existed_before and root.exists():
                    _remove_root(root, reason="raw run creation failed")
                raise

        if num_cameras > 0:
            image_writer = AsyncImageWriter(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )

        policy = None
        preprocessor = postprocessor = None
        policy_cfg = cfg.policy if cfg.mode in ("policy", "hil") else None
        if policy_cfg is not None:
            if cfg.inference_type == "rtc" and hasattr(policy_cfg, "rtc_config"):
                cfg.rtc.enabled = True
                policy_cfg.rtc_config = cfg.rtc
            policy = _load_policy_from_model_dir(policy_cfg)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                preprocessor_overrides={
                    "device_processor": {"device": policy_cfg.device},
                },
            )

        robot.connect()
        robot_wrapper = ThreadSafeRobot(robot)

        if teleop is not None:
            attach = getattr(teleop, "attach", None)
            if callable(attach):
                logger.info("Attaching teleoperator to robot before connect")
                attach(robot)
            teleop.connect()

        if policy_cfg is not None:
            _check_policy_compatibility(policy_cfg, dataset_features)

        if policy is not None:
            engine = _build_engine(
                cfg,
                policy,
                preprocessor,
                postprocessor,
                robot_wrapper,
                dataset_features,
                shutdown_event=inference_shutdown_event,
            )
            engine.start()

        events = KeyboardEvents()
        listener = _init_keyboard_listener(events)

        control_hz, obs_stride = _resolve_control_rate(cfg)
        logger.info(
            "Raw record loop: control_hz=%.3f, dataset_fps=%d, control_multiplier=%d, smoother_alpha=%.3f",
            control_hz,
            cfg.dataset.fps,
            obs_stride,
            cfg.smoother_alpha,
        )

        if policy_cfg is not None:
            action_keys = list(
                getattr(policy_cfg, "action_feature_names", None) or _ordered_action_keys(dataset_features)
            )
        else:
            action_keys = _ordered_action_keys(dataset_features)
        sources = {}
        if teleop is not None:
            sources["teleop"] = TeleopSource(
                "teleop", teleop, teleop_action_processor, robot_action_processor
            )
            sources["correction"] = TeleopSource(
                "correction", teleop, teleop_action_processor, robot_action_processor
            )
        if engine is not None:
            sources["autonomous"] = PolicySource(
                engine=engine,
                interpolator=ActionInterpolator(multiplier=obs_stride),
                action_keys=action_keys,
                dataset_features=dataset_features,
                robot_action_processor=robot_action_processor,
                gripper_max_width_m=cfg.policy_gripper_max_width_m,
            )

        control_interval = 1.0 / control_hz
        loop_timer = LoopTimer(window=int(control_hz), control_interval_s=control_interval)
        recorder = RawEpisodeRecorder(
            run=run,
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
            image_writer=image_writer,
            timer=loop_timer,
        )

        smoother = ActionSmoother(
            alpha=cfg.smoother_alpha,
            smooth_keys=_smooth_action_keys(dataset_features),
        )
        sm = HILStateMachine(cfg.mode)
        # Extend the state machine in place to support a `waiting -> homing`
        # transition that drives the arm back to a known pose before recording.
        sm._transitions[("waiting", "h")] = "home"
        sm._transitions[("homing", "esc")] = "stop"

        obs_cache = ObsCache()
        last_action_to_send: dict[str, float] | None = None
        prepare_ready_announced = False
        home_by_prefix = _default_home_joints_from_robot_config(cfg.robot, cfg.home_joints_rad)
        homing_waypoints: deque[dict[str, float]] = deque()

        def deactivate_current_source() -> None:
            if sm.source is not None:
                sources[sm.source].on_deactivate()
                sm.source = None

        def deactivate_all_sources() -> None:
            targets = {target for target in (sm.source, sm.prepare_target) if target is not None}
            for target in targets:
                sources[target].on_deactivate()
            sm.source = None
            sm.prepare_target = None

        def teleop_handoff_action(target) -> dict[str, float] | None:
            if target not in ("teleop", "correction"):
                return None
            if last_action_to_send is not None:
                return dict(last_action_to_send)
            with loop_timer.block("handoff_obs_read"):
                obs = robot_wrapper.get_observation()
            with loop_timer.block("handoff_obs_proc"):
                obs_cache.raw = obs
                obs_cache.processed = robot_observation_processor(obs)
            return _action_from_observation(obs_cache.raw, action_keys)

        def prepare_source(target):
            nonlocal prepare_ready_announced
            if target not in sources:
                logger.warning("Ignoring unavailable control source: %s", target)
                return
            deactivate_all_sources()
            sm.state = "preparing"
            sm.prepare_target = target
            prepare_ready_announced = False
            sources[target].prepare()
            if target == "autonomous":
                log_say("Preparing policy action", cfg.play_sounds)
            else:
                prepare_ready_announced = True
                log_say("Ready for human correction. Press right arrow to record.", cfg.play_sounds)

        def activate_source(target):
            nonlocal last_action_to_send
            if target not in sources:
                logger.warning("Ignoring unavailable control source: %s", target)
                return
            if sm.source is not None and sm.source != target:
                sources[sm.source].on_deactivate()
            if sm.prepare_target is not None and sm.prepare_target != target:
                sources[sm.prepare_target].on_deactivate()
            handoff_action = teleop_handoff_action(target)
            sources[target].on_activate(handoff_action)
            smoother.reset()
            previous_source = sm.source
            sm.source = target
            sm.prepare_target = None
            sm.state = "recording"
            last_action_to_send = handoff_action
            log_say(f"Recording {target}", cfg.play_sounds)
            if previous_source is not None and previous_source != target:
                recorder.push_event("source_switched", from_source=previous_source, to_source=target)

        def start() -> None:
            if cfg.mode == "teleop":
                activate_source("teleop")
            else:
                prepare_source("autonomous")

        def confirm_prepare() -> None:
            target = sm.prepare_target
            if target is None:
                return
            if sources[target].ready:
                activate_source(target)

        def pause() -> None:
            if cfg.mode != "hil":
                return
            deactivate_current_source()
            sm.state = "paused"
            sm.prepare_target = None
            recorder.push_event("paused")
            log_say("Paused. Press q for policy or e for correction.", cfg.play_sounds)

        def pause_unavailable() -> None:
            log_say("Pause is only available in HIL mode.", cfg.play_sounds)

        def prepare_policy() -> None:
            if cfg.mode == "hil":
                prepare_source("autonomous")

        def prepare_correction() -> None:
            if cfg.mode == "hil":
                prepare_source("correction")

        def cancel_episode() -> None:
            nonlocal last_action_to_send
            deactivate_all_sources()
            recorder.cancel()
            smoother.reset()
            last_action_to_send = None
            sm.state = "waiting"
            log_say("Episode cancelled", cfg.play_sounds)

        def save_episode() -> None:
            nonlocal last_action_to_send
            deactivate_all_sources()
            if recorder.frames > 0:
                committed = recorder.save()
                if committed is not None:
                    log_say(f"Episode {committed} saved", cfg.play_sounds)
                else:
                    log_say("No frames recorded", cfg.play_sounds)
            else:
                recorder.cancel()
                log_say("No frames recorded", cfg.play_sounds)
            smoother.reset()
            last_action_to_send = None
            sm.state = "waiting"

        def stop() -> None:
            sm.stop_requested = True

        def home() -> None:
            nonlocal homing_waypoints
            if not obs_cache.ready or obs_cache.raw is None:
                logger.warning("Cannot home before the first observation is available.")
                return
            homing_waypoints = _start_homing(
                obs_cache.raw,
                action_keys,
                home_by_prefix,
                cfg.policy_gripper_max_width_m,
                cfg.home_speed_rad_s,
                control_interval,
            )
            if not homing_waypoints:
                logger.warning("Homing aborted: no waypoints generated.")
                return
            smoother.reset()
            sm.state = "homing"
            log_say("Homing", cfg.play_sounds)

        sm_actions = {
            "start": start,
            "confirm_prepare": confirm_prepare,
            "pause": pause,
            "pause_unavailable": pause_unavailable,
            "prepare_policy": prepare_policy,
            "prepare_correction": prepare_correction,
            "cancel": cancel_episode,
            "save": save_episode,
            "stop": stop,
            "home": home,
        }

        log_say(
            "Waiting. Press right arrow to start; press h to home the robot.",
            cfg.play_sounds,
        )

        loop_i = 0
        while not sm.stop_requested:
            loop_start = time.perf_counter()

            if inference_shutdown_event.is_set() or (engine is not None and engine.failed):
                logger.error("Inference engine failed; aborting raw recording.")
                log_say("Inference engine failed; stopping raw recording", cfg.play_sounds)
                break

            with loop_timer.block("event_handle"):
                event = events.pop_latest()
                if event is not None:
                    sm.handle(event, sm_actions)
            if sm.stop_requested:
                break

            is_obs_tick = loop_i % obs_stride == 0
            if is_obs_tick:
                with loop_timer.block("obs_read"):
                    obs = robot_wrapper.get_observation()
                with loop_timer.block("obs_proc"):
                    obs_cache.raw = obs
                    obs_cache.processed = robot_observation_processor(obs)

            if sm.state == "homing":
                if homing_waypoints:
                    waypoint = homing_waypoints.popleft()
                    with loop_timer.block("send_action"):
                        robot_wrapper.send_action(waypoint)
                    last_action_to_send = waypoint
                else:
                    sm.state = "waiting"
                    log_say("Homing complete", cfg.play_sounds)

            if is_obs_tick and sm.state == "preparing" and sm.prepare_target is not None:
                source = sources[sm.prepare_target]
                was_ready = source.ready
                with loop_timer.block("prepare_update"):
                    source.update_prepare(obs_cache)
                if (
                    sm.prepare_target == "autonomous"
                    and source.ready
                    and not was_ready
                    and not prepare_ready_announced
                ):
                    prepare_ready_announced = True
                    log_say("Policy action ready. Press right arrow to record.", cfg.play_sounds)

            action_step = None
            action_to_record = None
            if sm.state == "recording" and sm.source is not None:
                if is_obs_tick:
                    with loop_timer.block("source_on_obs"):
                        sources[sm.source].on_observation(obs_cache)
                with loop_timer.block("source_step"):
                    action_step = sources[sm.source].step(obs_cache)
                if action_step is not None:
                    with loop_timer.block("smoother"):
                        action_to_send = smoother.step(action_step.robot_action)
                    action_to_record = action_to_send
                    with loop_timer.block("send_action"):
                        robot_wrapper.send_action(action_to_send)
                    last_action_to_send = action_to_send
                elif last_action_to_send is not None:
                    with loop_timer.block("send_action"):
                        robot_wrapper.send_action(last_action_to_send)
            elif sm.state == "paused" and last_action_to_send is not None:
                with loop_timer.block("send_action"):
                    robot_wrapper.send_action(last_action_to_send)

            if (
                is_obs_tick
                and sm.state == "recording"
                and action_to_record is not None
                and obs_cache.processed is not None
                and sm.source is not None
            ):
                recorder.add(obs_cache.processed, action_to_record, source=sm.source)

            dt = time.perf_counter() - loop_start
            loop_timer.record("loop_compute", dt)
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            elif sm.state == "recording":
                logger.warning(
                    "Loop is running slower (%.1f Hz) than target control rate (%.1f Hz)",
                    1 / max(dt, 1e-6),
                    1 / control_interval,
                )
            loop_timer.record("loop_total", time.perf_counter() - loop_start)
            if sm.state == "recording":
                loop_timer.tick()
            loop_i += 1
    finally:
        log_say("Stopping raw recording", cfg.play_sounds, blocking=True)
        if engine is not None:
            engine.stop()
        if image_writer is not None:
            try:
                image_writer.stop()
            except Exception:
                logger.exception("Failed to stop image writer cleanly.")
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        if run is not None and created_root is not None and run.num_episodes == 0:
            try:
                _remove_root(created_root, reason="no episodes were recorded")
            except Exception:
                logger.exception("Failed to remove empty raw run directory '%s'.", created_root)

    return run


def main():
    register_third_party_plugins()
    raw_record()


if __name__ == "__main__":
    main()
