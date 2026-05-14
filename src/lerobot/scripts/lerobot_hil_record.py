#!/usr/bin/env python

"""Manual episode recording with teleop, policy, or HIL control."""

import logging
import math
import os
import select
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Any, Literal

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import INFO_PATH
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.inference_engines import RTCInferenceEngine, SyncInferenceEngine
from lerobot.inference_engines.robot_wrapper import ThreadSafeRobot
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, RTCConfig
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Import config modules so draccus can resolve --robot.type / --teleop.type.
from lerobot.robots import (  # noqa: F401,E402
    bi_nero_follower,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    nero_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401,E402
    bi_openarm_leader,
    bi_pico_nero_teleop,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    pico_nero_teleop,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)

logger = logging.getLogger(__name__)


ControlMode = Literal["teleop", "policy", "hil"]
InferenceType = Literal["sync", "rtc"]
ControlSource = Literal["teleop", "autonomous", "correction"]
LoopState = Literal["waiting", "preparing", "recording", "paused"]


@dataclass
class HILRecordDatasetConfig:
    repo_id: str
    single_task: str = (
        "Pick up the match in front, strike it to light it, then use it to light the small candle "
        "on the cake in front."
    )
    root: str | Path | None = None
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    force: bool = False

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide --dataset.single_task.")


@dataclass
class HILRecordConfig:
    robot: RobotConfig
    dataset: HILRecordDatasetConfig
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
    display_data: bool = True
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False

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
        if self.interpolation_multiplier < 1:
            raise ValueError("--interpolation_multiplier must be >= 1.")
        if self.control_multiplier is not None and self.control_multiplier < 1:
            raise ValueError("--control_multiplier must be >= 1.")
        if self.control_hz is not None and self.control_hz <= 0:
            raise ValueError("--control_hz must be > 0.")
        if not 0 < self.smoother_alpha <= 1:
            raise ValueError("--smoother_alpha must be in (0, 1].")
        if self.resume and self.dataset.force:
            raise ValueError("--dataset.force cannot be used with --resume.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class KeyboardEvents:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: deque[str] = deque()

    def push(self, event: str) -> None:
        with self._lock:
            self._events.append(event)

    def pop_latest(self) -> str | None:
        with self._lock:
            if not self._events:
                return None
            event = self._events[-1]
            self._events.clear()
        return event


class _TerminalKeyboardListener:
    _ESCAPE_SEQUENCE_TIMEOUT_S = 0.1
    _MAX_ESCAPE_SEQUENCE_CHARS = 12
    _ESCAPE_SEQUENCES = {
        "\x1b[C": "right",
        "\x1bOC": "right",
        "\x1b[D": "left",
        "\x1bOD": "left",
    }
    _SINGLE_CHAR_EVENTS = {
        "\r": "enter",
        "\n": "enter",
        " ": "space",
        "\x1b": "esc",
        "q": "q",
        "e": "e",
    }

    def __init__(self, events: KeyboardEvents, stdin=None) -> None:
        self._events = events
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._old_termios = None
        self._fd: int | None = None

    @classmethod
    def parse_key(cls, text: str) -> str | None:
        if text in cls._ESCAPE_SEQUENCES:
            return cls._ESCAPE_SEQUENCES[text]
        if (text.startswith("\x1b[") or text.startswith("\x1bO")) and len(text) >= 3:
            if text[-1] == "C":
                return "right"
            if text[-1] == "D":
                return "left"
        if text in cls._SINGLE_CHAR_EVENTS:
            return cls._SINGLE_CHAR_EVENTS[text]
        return None

    def start(self) -> bool:
        if not hasattr(self._stdin, "isatty") or not self._stdin.isatty():
            return False

        try:
            import termios
            import tty

            fd = self._stdin.fileno()
            self._fd = fd
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            logger.exception("Failed to enable terminal keyboard control.")
            return False

        self._thread = Thread(target=self._run, name="hil-terminal-keyboard-listener", daemon=True)
        self._thread.start()
        logger.info("Terminal keyboard control enabled.")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._old_termios is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                logger.exception("Failed to restore terminal settings.")
            self._old_termios = None
        self._fd = None

    def _read_char(self) -> str:
        if self._fd is None:
            return ""
        return os.read(self._fd, 1).decode(errors="ignore")

    def _read_available_escape_sequence(self, first_char: str) -> str:
        if self._fd is None:
            return first_char
        chars = [first_char]
        deadline = time.monotonic() + self._ESCAPE_SEQUENCE_TIMEOUT_S
        while len(chars) < self._MAX_ESCAPE_SEQUENCE_CHARS and not self._stop_event.is_set():
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            readable, _, _ = select.select([self._fd], [], [], timeout)
            if not readable:
                break
            chars.append(self._read_char())
            if len(chars) >= 3 and (chars[-1].isalpha() or chars[-1] == "~"):
                break
        return "".join(chars)

    def _run(self) -> None:
        while self._fd is not None and not self._stop_event.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.05)
            if not readable:
                continue
            text = self._read_char()
            if not text:
                continue
            if text == "\x1b":
                text = self._read_available_escape_sequence(text)
            event = self.parse_key(text)
            if text.startswith("\x1b"):
                logger.info("Keyboard escape sequence: %r", text)
            if event is not None:
                logger.info("Keyboard event: %s", event)
                self._events.push(event)


def _init_keyboard_listener(events: KeyboardEvents):
    terminal_listener = _TerminalKeyboardListener(events)
    if terminal_listener.start():
        logger.info(
            "Keyboard: right=start/confirm, enter=save, left=cancel, space=pause, q=policy, e=teleop, esc=exit"
        )
        return terminal_listener

    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.right:
                events.push("right")
            elif key == keyboard.Key.left:
                events.push("left")
            elif key == keyboard.Key.enter:
                events.push("enter")
            elif key == keyboard.Key.space:
                events.push("space")
            elif key == keyboard.Key.esc:
                events.push("esc")
            elif hasattr(key, "char") and key.char in ("q", "e"):
                events.push(key.char)

        pynput_listener = keyboard.Listener(on_press=on_press)
        pynput_listener.start()
        logger.info("pynput keyboard control enabled.")
        logger.info(
            "Keyboard: right=start/confirm, enter=save, left=cancel, space=pause, q=policy, e=teleop, esc=exit"
        )
        return pynput_listener
    except Exception as exc:
        logger.warning("pynput keyboard control unavailable: %s", exc)

    logger.warning("Keyboard control is disabled because neither terminal stdin nor pynput is available.")
    return None


def _ordered_action_keys(dataset_features: dict) -> list[str]:
    return list(dataset_features[ACTION]["names"])


def _tensor_to_action_dict(action: torch.Tensor, keys: list[str]) -> dict[str, float]:
    action = action.squeeze().cpu()
    if len(action) != len(keys):
        raise ValueError(f"Action dim ({len(action)}) does not match action keys ({len(keys)}).")
    return {key: float(action[i]) for i, key in enumerate(keys)}


def _clamp_policy_action(action: dict[str, float]) -> dict[str, float]:
    out = dict(action)
    for key, value in out.items():
        if "gripper" in key.lower():
            out[key] = float(max(0.0, min(0.1, value)))
    return out


def _resolve_dataset_root(repo_id: str, root: str | Path | None) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / repo_id


def _is_safe_to_remove_dataset_root(root: Path) -> bool:
    if root.is_symlink() or not root.is_dir():
        return False
    try:
        next(root.iterdir())
    except StopIteration:
        return True
    return (root / INFO_PATH).is_file()


def _remove_dataset_root(root: Path, *, reason: str) -> None:
    if not root.exists():
        return
    if not _is_safe_to_remove_dataset_root(root):
        raise FileExistsError(
            f"Refusing to remove existing directory '{root}'. "
            f"{reason} only removes empty directories or LeRobot dataset directories "
            f"containing '{INFO_PATH}'."
        )
    logger.warning("Removing dataset directory '%s' (%s).", root, reason)
    shutil.rmtree(root)


def _resolve_control_rate(cfg: HILRecordConfig) -> tuple[float, int]:
    """Return a control rate that is exactly dataset.fps * multiplier."""
    if cfg.control_multiplier is not None:
        if cfg.control_hz is not None:
            logger.warning(
                "Both control_multiplier and control_hz are set; using control_multiplier=%d.",
                cfg.control_multiplier,
            )
        if cfg.interpolation_multiplier != 1:
            logger.info(
                "control_multiplier=%d overrides interpolation_multiplier=%d.",
                cfg.control_multiplier,
                cfg.interpolation_multiplier,
            )
        multiplier = cfg.control_multiplier
    elif cfg.control_hz is not None:
        multiplier = max(1, math.floor(cfg.control_hz / cfg.dataset.fps + 0.5))
        adjusted_hz = cfg.dataset.fps * multiplier
        if not math.isclose(adjusted_hz, cfg.control_hz, rel_tol=0.0, abs_tol=1e-6):
            logger.warning(
                "Adjusted control_hz from %.3f to %.3f so dataset frames remain exactly %d FPS.",
                cfg.control_hz,
                adjusted_hz,
                cfg.dataset.fps,
            )
        if cfg.interpolation_multiplier != 1:
            logger.info(
                "control_hz overrides interpolation_multiplier=%d; using multiplier=%d.",
                cfg.interpolation_multiplier,
                multiplier,
            )
    else:
        multiplier = cfg.interpolation_multiplier

    return float(cfg.dataset.fps * multiplier), multiplier


def _build_engine(
    cfg: HILRecordConfig,
    policy,
    preprocessor,
    postprocessor,
    robot_wrapper: ThreadSafeRobot,
    dataset_features: dict,
    shutdown_event=None,
):
    keys = _ordered_action_keys(dataset_features)
    if cfg.inference_type == "sync":
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=keys,
            task=cfg.dataset.single_task,
            device=cfg.policy.device,
            robot_type=robot_wrapper.robot_type,
        )

    cfg.rtc.enabled = True
    if hasattr(policy.config, "rtc_config"):
        policy.config.rtc_config = cfg.rtc
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        rtc_config=cfg.rtc,
        hw_features=dataset_features,
        task=cfg.dataset.single_task,
        fps=cfg.dataset.fps,
        device=cfg.policy.device,
        rtc_queue_threshold=cfg.rtc_queue_threshold,
        shutdown_event=shutdown_event,
    )


@dataclass
class ObsCache:
    raw: dict[str, Any] | None = None
    processed: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self.raw is not None and self.processed is not None


@dataclass
class ActionStep:
    robot_action: dict[str, float]


def _smooth_action_keys(dataset_features: dict) -> set[str]:
    keys = _ordered_action_keys(dataset_features)
    return {
        key
        for key in keys
        if "gripper" not in key.lower() and (key.endswith(".pos") or key.endswith(".q"))
    }


class ActionSmoother:
    """EMA smoother for final robot joint targets; gripper commands pass through."""

    def __init__(self, alpha: float, smooth_keys: set[str]) -> None:
        self.alpha = float(alpha)
        self._smooth_keys = smooth_keys
        self._prev: dict[str, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def step(self, action: dict[str, float]) -> dict[str, float]:
        if self.alpha >= 1.0:
            return dict(action)
        if self._prev is None:
            self._prev = dict(action)
            return dict(action)

        out: dict[str, float] = {}
        for key, value in action.items():
            v = float(value)
            if key in self._smooth_keys:
                prev = float(self._prev.get(key, v))
                out[key] = self.alpha * v + (1.0 - self.alpha) * prev
            else:
                out[key] = v

        self._prev = dict(out)
        return out


class ActionSource:
    name: ControlSource

    @property
    def ready(self) -> bool:
        return True

    def prepare(self) -> None:
        pass

    def update_prepare(self, obs_cache: ObsCache) -> None:
        pass

    def on_observation(self, obs_cache: ObsCache) -> None:
        pass

    def on_activate(self, handoff_action: dict[str, float] | None = None) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def step(self, obs_cache: ObsCache) -> ActionStep | None:
        raise NotImplementedError


class TeleopSource(ActionSource):
    def __init__(self, name: ControlSource, teleop, teleop_action_processor, robot_action_processor) -> None:
        self.name = name
        self._teleop = teleop
        self._teleop_action_processor = teleop_action_processor
        self._robot_action_processor = robot_action_processor

    def on_activate(self, handoff_action: dict[str, float] | None = None) -> None:
        if handoff_action is None:
            return
        prime_action = getattr(self._teleop, "prime_action", None)
        if not callable(prime_action):
            return
        if not prime_action(handoff_action, lock_reference=True):
            logger.warning("Failed to prime teleop source %s from handoff action.", self.name)

    def on_deactivate(self) -> None:
        release_primed_action = getattr(self._teleop, "release_primed_action", None)
        if callable(release_primed_action):
            release_primed_action()

    def step(self, obs_cache: ObsCache) -> ActionStep | None:
        if not obs_cache.ready:
            return None
        teleop_action = self._teleop.get_action()
        processed_teleop = self._teleop_action_processor((teleop_action, obs_cache.raw))
        robot_action = self._robot_action_processor((processed_teleop, obs_cache.raw))
        return ActionStep(robot_action=robot_action)


class PolicySource(ActionSource):
    def __init__(
        self,
        engine,
        interpolator: ActionInterpolator,
        action_keys: list[str],
        dataset_features: dict,
        robot_action_processor,
    ) -> None:
        self.name: ControlSource = "autonomous"
        self._engine = engine
        self._interpolator = interpolator
        self._action_keys = action_keys
        self._dataset_features = dataset_features
        self._robot_action_processor = robot_action_processor
        self._pending_action: torch.Tensor | None = None
        self._ready = False
        self._active = False

    @property
    def ready(self) -> bool:
        return self._ready

    def prepare(self) -> None:
        self._pending_action = None
        self._ready = False
        self._active = False
        self._interpolator.reset()
        self._engine.reset()
        self._engine.resume()

    def update_prepare(self, obs_cache: ObsCache) -> None:
        if self._ready or not obs_cache.ready:
            return
        self._engine.notify_observation(obs_cache.processed)
        obs_frame = build_dataset_frame(self._dataset_features, obs_cache.processed, prefix=OBS_STR)
        action = self._engine.get_action(obs_frame)
        if action is not None:
            self._pending_action = action
            self._ready = True
            self._engine.pause()

    def on_activate(self, handoff_action: dict[str, float] | None = None) -> None:
        if self._pending_action is not None:
            self._interpolator.add(self._pending_action.cpu())
        self._active = True
        self._ready = False
        self._engine.resume()

    def on_deactivate(self) -> None:
        self._active = False
        self._ready = False
        self._pending_action = None
        self._engine.pause()

    def on_observation(self, obs_cache: ObsCache) -> None:
        if not self._active or not obs_cache.ready:
            return

        self._engine.notify_observation(obs_cache.processed)
        if self._interpolator.needs_new_action():
            obs_frame = build_dataset_frame(self._dataset_features, obs_cache.processed, prefix=OBS_STR)
            action = self._engine.get_action(obs_frame)
            if action is not None:
                self._interpolator.add(action.cpu())

    def step(self, obs_cache: ObsCache) -> ActionStep | None:
        if not self._active or not obs_cache.ready:
            return None

        interp = self._interpolator.get()
        if interp is None:
            return None

        action_dict = _tensor_to_action_dict(interp, self._action_keys)
        action_dict = _clamp_policy_action(action_dict)
        robot_action = self._robot_action_processor((action_dict, obs_cache.raw))
        return ActionStep(robot_action=robot_action)


class EpisodeRecorder:
    def __init__(
        self,
        dataset: LeRobotDataset,
        dataset_features: dict,
        task: str,
        display_data: bool,
        display_compressed_images: bool,
    ) -> None:
        self._dataset = dataset
        self._dataset_features = dataset_features
        self._task = task
        self._display_data = display_data
        self._display_compressed_images = display_compressed_images
        self.frames = 0

    def add(self, obs_processed: dict[str, Any], robot_action: dict[str, float]) -> None:
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        action_frame = build_dataset_frame(self._dataset_features, robot_action, prefix=ACTION)
        self._dataset.add_frame({**obs_frame, **action_frame, "task": self._task})
        self.frames += 1

        if self._display_data:
            log_rerun_data(
                observation=obs_processed,
                action=robot_action,
                compress_images=self._display_compressed_images,
            )

    def save(self) -> None:
        self._dataset.save_episode()
        self.frames = 0

    def cancel(self) -> None:
        self._dataset.clear_episode_buffer()
        self.frames = 0


class HILStateMachine:
    def __init__(self, mode: ControlMode) -> None:
        self.state: LoopState = "waiting"
        self.source: ControlSource | None = None
        self.prepare_target: ControlSource | None = None
        self.stop_requested = False
        self._transitions = {
            ("waiting", "right"): "start",
            ("waiting", "esc"): "stop",
            ("preparing", "right"): "confirm_prepare",
            ("preparing", "left"): "cancel",
            ("preparing", "enter"): "save",
            ("recording", "left"): "cancel",
            ("recording", "enter"): "save",
            ("paused", "q"): "prepare_policy",
            ("paused", "e"): "prepare_correction",
            ("paused", "left"): "cancel",
            ("paused", "enter"): "save",
        }
        self._transitions[("recording", "space")] = "pause" if mode == "hil" else "pause_unavailable"

    def handle(self, event: str, actions: dict[str, Any]) -> None:
        action_name = self._transitions.get((self.state, event))
        if action_name is not None:
            actions[action_name]()


@parser.wrap()
def hil_record(cfg: HILRecordConfig) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="hil_record", ip=cfg.display_ip, port=cfg.display_port)

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
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    engine = None
    inference_shutdown_event = Event()
    created_dataset_root = None

    try:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        policy_cfg = cfg.policy if cfg.mode in ("policy", "hil") else None
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, policy_cfg)
            dataset_root = _resolve_dataset_root(cfg.dataset.repo_id, cfg.dataset.root)
            if cfg.dataset.force:
                _remove_dataset_root(dataset_root, reason="--dataset.force was set")
            dataset_root_existed_before_create = dataset_root.exists()
            try:
                dataset = LeRobotDataset.create(
                    cfg.dataset.repo_id,
                    cfg.dataset.fps,
                    root=cfg.dataset.root,
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=cfg.dataset.video,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                )
            except Exception:
                if not dataset_root_existed_before_create and dataset_root.exists():
                    _remove_dataset_root(dataset_root, reason="HIL dataset creation failed")
                raise
            created_dataset_root = dataset.root

        policy = None
        preprocessor = postprocessor = None
        if policy_cfg is not None:
            if cfg.inference_type == "rtc" and hasattr(policy_cfg, "rtc_config"):
                cfg.rtc.enabled = True
                policy_cfg.rtc_config = cfg.rtc
            policy = make_policy(policy_cfg, ds_meta=dataset.meta)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": policy_cfg.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
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
            "HIL loop rates: control_hz=%.3f, dataset_fps=%d, control_multiplier=%d, smoother_alpha=%.3f",
            control_hz,
            cfg.dataset.fps,
            obs_stride,
            cfg.smoother_alpha,
        )

        action_keys = _ordered_action_keys(dataset_features)
        sources: dict[ControlSource, ActionSource] = {}
        if teleop is not None:
            sources["teleop"] = TeleopSource("teleop", teleop, teleop_action_processor, robot_action_processor)
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
            )

        recorder = EpisodeRecorder(
            dataset=dataset,
            dataset_features=dataset_features,
            task=cfg.dataset.single_task,
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
        )
        smoother = ActionSmoother(
            alpha=cfg.smoother_alpha,
            smooth_keys=_smooth_action_keys(dataset_features),
        )
        sm = HILStateMachine(cfg.mode)
        obs_cache = ObsCache()
        last_action_to_send: dict[str, float] | None = None
        prepare_ready_announced = False

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

        def prepare_source(target: ControlSource) -> None:
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

        def activate_source(target: ControlSource) -> None:
            nonlocal last_action_to_send
            if target not in sources:
                logger.warning("Ignoring unavailable control source: %s", target)
                return
            if sm.source is not None and sm.source != target:
                sources[sm.source].on_deactivate()
            if sm.prepare_target is not None and sm.prepare_target != target:
                sources[sm.prepare_target].on_deactivate()
            handoff_action = (
                dict(last_action_to_send)
                if target in ("teleop", "correction") and last_action_to_send is not None
                else None
            )
            sources[target].on_activate(handoff_action)
            smoother.reset()
            sm.source = target
            sm.prepare_target = None
            sm.state = "recording"
            last_action_to_send = handoff_action
            log_say(f"Recording {target}", cfg.play_sounds)

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
                recorder.save()
                log_say(f"Episode {dataset.num_episodes} saved", cfg.play_sounds)
            else:
                recorder.cancel()
                log_say("No frames recorded", cfg.play_sounds)
            smoother.reset()
            last_action_to_send = None
            sm.state = "waiting"

        def stop() -> None:
            sm.stop_requested = True

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
        }

        log_say("Waiting. Press right arrow to prepare/start an episode.", cfg.play_sounds)

        with VideoEncodingManager(dataset):
            loop_i = 0
            control_interval = 1.0 / control_hz
            while not sm.stop_requested:
                loop_start = time.perf_counter()

                if inference_shutdown_event.is_set() or (engine is not None and engine.failed):
                    logger.error("Inference engine failed; aborting HIL recording.")
                    log_say("Inference engine failed; stopping HIL recording", cfg.play_sounds)
                    break

                event = events.pop_latest()
                if event is not None:
                    sm.handle(event, sm_actions)
                    if sm.stop_requested:
                        break

                is_obs_tick = loop_i % obs_stride == 0
                if is_obs_tick:
                    obs = robot_wrapper.get_observation()
                    obs_cache.raw = obs
                    obs_cache.processed = robot_observation_processor(obs)

                if is_obs_tick and sm.state == "preparing" and sm.prepare_target is not None:
                    source = sources[sm.prepare_target]
                    was_ready = source.ready
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
                        sources[sm.source].on_observation(obs_cache)
                    action_step = sources[sm.source].step(obs_cache)
                    if action_step is not None:
                        action_to_send = smoother.step(action_step.robot_action)
                        action_to_record = action_to_send
                        robot_wrapper.send_action(action_to_send)
                        last_action_to_send = action_to_send
                    elif last_action_to_send is not None:
                        robot_wrapper.send_action(last_action_to_send)
                elif sm.state == "paused" and last_action_to_send is not None:
                    robot_wrapper.send_action(last_action_to_send)

                if (
                    is_obs_tick
                    and sm.state == "recording"
                    and action_to_record is not None
                    and obs_cache.processed is not None
                ):
                    recorder.add(obs_cache.processed, action_to_record)

                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                elif sm.state == "recording":
                    logger.warning(
                        "Loop is running slower (%.1f Hz) than target control rate (%.1f Hz)",
                        1 / max(dt, 1e-6),
                        1 / control_interval,
                    )
                loop_i += 1
    finally:
        log_say("Stopping HIL recording", cfg.play_sounds, blocking=True)
        if engine is not None:
            engine.stop()
        if dataset is not None:
            dataset.finalize()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        if dataset is not None and cfg.dataset.push_to_hub:
            if dataset.num_episodes > 0:
                try:
                    dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                except Exception:
                    logger.exception("Failed to push dataset to the Hugging Face Hub.")
            else:
                logger.info("Skipping Hub push because no episodes were recorded.")
        if dataset is not None and created_dataset_root is not None and dataset.num_episodes == 0:
            try:
                _remove_dataset_root(created_dataset_root, reason="no episodes were recorded")
            except Exception:
                logger.exception("Failed to remove empty HIL dataset directory '%s'.", created_dataset_root)

    return dataset


def main():
    register_third_party_plugins()
    hil_record()


if __name__ == "__main__":
    main()
