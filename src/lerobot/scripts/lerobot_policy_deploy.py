#!/usr/bin/env python

"""Deploy a policy on a robot with RTC async inference and keyboard control."""

# from __future__ import annotations

import logging
import math
import os
import select
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Any, Literal

import torch
from deepdiff import DeepDiff

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts, dataset_to_policy_features
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.inference_engines import RTCInferenceEngine
from lerobot.inference_engines.robot_wrapper import ThreadSafeRobot
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionInterpolator, RTCConfig
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

# Import config modules so draccus can resolve --robot.type.
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

logger = logging.getLogger(__name__)


DeployState = Literal["paused", "preparing", "running", "homing"]


@dataclass
class PolicyDeployDatasetConfig:
    repo_id: str | None = None
    root: str | Path | None = None
    revision: str | None = None
    fps: int = 30
    task: str = (
        "Pick up the match in front, strike it to light it, then use it to light the small candle "
        "on the cake in front."
    )
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError("--dataset.fps must be > 0.")


@dataclass
class PolicyDeployConfig:
    robot: RobotConfig
    dataset: PolicyDeployDatasetConfig
    policy: PreTrainedConfig | None = None
    rtc: RTCConfig = field(default_factory=RTCConfig)
    rtc_queue_threshold: int = 40
    interpolation_multiplier: int = 1
    control_multiplier: int | None = 3
    control_hz: float | None = None
    smoother_alpha: float = 1.0
    policy_gripper_max_width_m: float = 0.1
    home_joints_rad: list[float] | None = None
    home_speed_rad_s: float = 0.4
    hf_hub_offline: bool = True
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Policy deployment requires --policy.path.")
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
        if self.home_joints_rad is not None and len(self.home_joints_rad) != 7:
            raise ValueError("--home_joints_rad must contain 7 joint values.")

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
        " ": "space",
        "\x1b": "esc",
        "h": "h",
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

        self._thread = Thread(target=self._run, name="policy-deploy-keyboard-listener", daemon=True)
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
            if event is not None:
                logger.info("Keyboard event: %s", event)
                self._events.push(event)


def _init_keyboard_listener(events: KeyboardEvents):
    terminal_listener = _TerminalKeyboardListener(events)
    if terminal_listener.start():
        logger.info("Keyboard: right=start, space=pause, h=home while paused, esc=exit")
        return terminal_listener

    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.right:
                events.push("right")
            elif key == keyboard.Key.space:
                events.push("space")
            elif key == keyboard.Key.esc:
                events.push("esc")
            elif hasattr(key, "char") and key.char == "h":
                events.push("h")

        pynput_listener = keyboard.Listener(on_press=on_press)
        pynput_listener.start()
        logger.info("pynput keyboard control enabled.")
        logger.info("Keyboard: right=start, space=pause, h=home while paused, esc=exit")
        return pynput_listener
    except Exception as exc:
        logger.warning("pynput keyboard control unavailable: %s", exc)

    logger.warning("Keyboard control is disabled because neither terminal stdin nor pynput is available.")
    return None


@dataclass
class ObsCache:
    raw: dict[str, Any] | None = None
    processed: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self.raw is not None and self.processed is not None


class ActionSmoother:
    """Optional EMA smoother for final robot joint targets; disabled by default."""

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


def _ordered_action_keys(features: dict) -> list[str]:
    return list(features[ACTION]["names"])


def _tensor_to_action_dict(action: torch.Tensor, keys: list[str]) -> dict[str, float]:
    action = action.squeeze().cpu()
    if len(action) != len(keys):
        raise ValueError(f"Action dim ({len(action)}) does not match action keys ({len(keys)}).")
    return {key: float(action[i]) for i, key in enumerate(keys)}


def _clamp_policy_action(action: dict[str, float], gripper_max_width_m: float) -> dict[str, float]:
    out = dict(action)
    for key, value in out.items():
        if "gripper" in key.lower():
            out[key] = float(max(0.0, min(gripper_max_width_m, value)))
    return out


def _smooth_action_keys(features: dict) -> set[str]:
    return {
        key
        for key in _ordered_action_keys(features)
        if "gripper" not in key.lower() and (key.endswith(".pos") or key.endswith(".q"))
    }


def _resolve_control_rate(dataset_fps: int, cfg: PolicyDeployConfig) -> tuple[float, int]:
    if cfg.control_multiplier is not None:
        if cfg.control_hz is not None:
            logger.warning(
                "Both control_multiplier and control_hz are set; using control_multiplier=%d.",
                cfg.control_multiplier,
            )
        multiplier = cfg.control_multiplier
    elif cfg.control_hz is not None:
        multiplier = max(1, math.floor(cfg.control_hz / dataset_fps + 0.5))
        adjusted_hz = dataset_fps * multiplier
        if not math.isclose(adjusted_hz, cfg.control_hz, rel_tol=0.0, abs_tol=1e-6):
            logger.warning(
                "Adjusted control_hz from %.3f to %.3f so observation ticks remain exactly %d FPS.",
                cfg.control_hz,
                adjusted_hz,
                dataset_fps,
            )
    else:
        multiplier = cfg.interpolation_multiplier

    return float(dataset_fps * multiplier), multiplier


def _check_metadata_compatibility(
    dataset_meta: LeRobotDatasetMetadata, robot_type: str, dataset_fps: int, runtime_features: dict
) -> None:
    fields = [
        ("robot_type", dataset_meta.robot_type, robot_type),
        ("fps", dataset_meta.fps, dataset_fps),
        ("features", dataset_meta.features, {**runtime_features, **DEFAULT_FEATURES}),
    ]
    mismatches = []
    for field, expected, present in fields:
        diff = DeepDiff(expected, present, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected dataset value {expected}, got runtime value {present}")
    if mismatches:
        raise ValueError("Dataset metadata compatibility check failed:\n" + "\n".join(mismatches))


def _check_policy_compatibility(policy_cfg: PreTrainedConfig, runtime_features: dict) -> None:
    runtime_policy_features = dataset_to_policy_features(runtime_features)
    runtime_input_features = {
        key: value for key, value in runtime_policy_features.items() if key != ACTION
    }
    runtime_output_features = {key: value for key, value in runtime_policy_features.items() if key == ACTION}

    mismatches = []
    if policy_cfg.input_features:
        diff = DeepDiff(policy_cfg.input_features, runtime_input_features)
        if diff:
            mismatches.append(f"input_features: {diff}")
    if policy_cfg.output_features:
        diff = DeepDiff(policy_cfg.output_features, runtime_output_features)
        if diff:
            mismatches.append(f"output_features: {diff}")

    action_names = getattr(policy_cfg, "action_feature_names", None)
    if action_names is not None:
        runtime_action_names = runtime_features.get(ACTION, {}).get("names")
        if list(action_names) != list(runtime_action_names or []):
            mismatches.append(
                f"action_feature_names: expected policy value {list(action_names)}, "
                f"got runtime value {list(runtime_action_names or [])}"
            )

    if mismatches:
        raise ValueError("Policy/runtime compatibility check failed:\n" + "\n".join(mismatches))


def _build_dataset_features(robot, robot_action_processor, robot_observation_processor, use_videos: bool) -> dict:
    action_features = robot_action_processor.transform_features(create_initial_features(action=robot.action_features))
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=action_features,
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )


def _build_engine(
    cfg: PolicyDeployConfig,
    policy,
    preprocessor,
    postprocessor,
    robot_wrapper: ThreadSafeRobot,
    dataset_features: dict,
    dataset_fps: int,
    shutdown_event: Event,
) -> RTCInferenceEngine:
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
        task=cfg.dataset.task,
        fps=dataset_fps,
        device=cfg.policy.device,
        rtc_queue_threshold=cfg.rtc_queue_threshold,
        shutdown_event=shutdown_event,
    )


def _load_policy_from_model_dir(policy_cfg: PreTrainedConfig):
    if policy_cfg.pretrained_path is None:
        raise ValueError("Policy deployment requires --policy.path.")
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)
    policy.to(policy_cfg.device)
    policy.eval()
    return policy


def _default_home_joints_from_robot_config(robot_config: RobotConfig, override: list[float] | None):
    if override is not None:
        return {
            "": list(override),
            "left_": list(override),
            "right_": list(override),
        }

    home: dict[str, list[float]] = {}
    if hasattr(robot_config, "home_joints_rad"):
        home[""] = list(getattr(robot_config, "home_joints_rad"))
    if hasattr(robot_config, "left_arm_config") and hasattr(robot_config.left_arm_config, "home_joints_rad"):
        home["left_"] = list(robot_config.left_arm_config.home_joints_rad)
    if hasattr(robot_config, "right_arm_config") and hasattr(robot_config.right_arm_config, "home_joints_rad"):
        home["right_"] = list(robot_config.right_arm_config.home_joints_rad)
    return home


def _joint_keys_for_prefix(action_keys: list[str], prefix: str) -> list[str]:
    keys = []
    for key in action_keys:
        if "gripper" in key.lower():
            continue
        if prefix and not key.startswith(prefix):
            continue
        unprefixed = key.removeprefix(prefix)
        if unprefixed.endswith(".pos") or unprefixed.endswith(".q"):
            keys.append(key)
    return keys[:7]


def _make_home_action(
    obs: dict[str, Any],
    action_keys: list[str],
    home_by_prefix: dict[str, list[float]],
    gripper_max_width_m: float,
) -> dict[str, float]:
    action = {key: float(obs.get(key, 0.0)) for key in action_keys}

    prefixes = [prefix for prefix in home_by_prefix if prefix] or [""]
    if "" in home_by_prefix:
        prefixes.append("")

    for prefix in dict.fromkeys(prefixes):
        joint_keys = _joint_keys_for_prefix(action_keys, prefix)
        home = home_by_prefix.get(prefix)
        if home is None or len(joint_keys) != len(home):
            continue
        for key, value in zip(joint_keys, home, strict=True):
            action[key] = float(value)

    for key in action:
        if "gripper" in key.lower():
            action[key] = float(gripper_max_width_m)
    return action


def _start_homing(
    obs: dict[str, Any],
    action_keys: list[str],
    home_by_prefix: dict[str, list[float]],
    gripper_max_width_m: float,
    home_speed_rad_s: float,
    control_interval: float,
) -> deque[dict[str, float]]:
    target = _make_home_action(obs, action_keys, home_by_prefix, gripper_max_width_m)
    start = {key: float(obs.get(key, target[key])) for key in action_keys}

    max_delta = 0.0
    for key in action_keys:
        if "gripper" in key.lower():
            continue
        max_delta = max(max_delta, abs(target[key] - start[key]))
    steps = max(1, int(math.ceil(max_delta / (home_speed_rad_s * control_interval))))

    waypoints: deque[dict[str, float]] = deque()
    for step in range(1, steps + 1):
        t = step / steps
        waypoint = {}
        for key in action_keys:
            waypoint[key] = start[key] + t * (target[key] - start[key])
        waypoints.append(waypoint)
    return waypoints


@parser.wrap()
def policy_deploy(cfg: PolicyDeployConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if cfg.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset_meta = None
    if cfg.dataset.repo_id is not None:
        dataset_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
        )
    dataset_fps = int(dataset_meta.fps if dataset_meta is not None else cfg.dataset.fps)

    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    use_videos = (
        any(ft.get("dtype") == "video" for ft in dataset_meta.features.values())
        if dataset_meta is not None
        else True
    )
    dataset_features = _build_dataset_features(
        robot, robot_action_processor, robot_observation_processor, use_videos=use_videos
    )

    shutdown_event = Event()
    listener = None
    engine = None

    try:
        policy = _load_policy_from_model_dir(cfg.policy)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        robot_wrapper = ThreadSafeRobot(robot)
        if dataset_meta is not None:
            _check_metadata_compatibility(dataset_meta, robot_wrapper.robot_type, dataset_fps, dataset_features)
        else:
            _check_policy_compatibility(cfg.policy, dataset_features)

        engine = _build_engine(
            cfg,
            policy,
            preprocessor,
            postprocessor,
            robot_wrapper,
            dataset_features,
            dataset_fps,
            shutdown_event,
        )
        engine.start()
        engine.pause()

        events = KeyboardEvents()
        listener = _init_keyboard_listener(events)

        control_hz, obs_stride = _resolve_control_rate(dataset_fps, cfg)
        control_interval = 1.0 / control_hz
        action_keys = list(getattr(cfg.policy, "action_feature_names", None) or _ordered_action_keys(dataset_features))
        interpolator = ActionInterpolator(multiplier=obs_stride)
        smoother = ActionSmoother(alpha=cfg.smoother_alpha, smooth_keys=_smooth_action_keys(dataset_features))
        obs_cache = ObsCache()
        home_by_prefix = _default_home_joints_from_robot_config(cfg.robot, cfg.home_joints_rad)
        homing_waypoints: deque[dict[str, float]] = deque()

        state: DeployState = "paused"
        stop_requested = False

        policy_param = next(policy.parameters(), None)
        logger.info(
            "Policy runtime device=%s param_device=%s param_dtype=%s compile_model=%s compile_mode=%s",
            cfg.policy.device,
            None if policy_param is None else policy_param.device,
            None if policy_param is None else policy_param.dtype,
            getattr(cfg.policy, "compile_model", None),
            getattr(cfg.policy, "compile_mode", None),
        )

        def clear_policy_state() -> None:
            engine.pause()
            engine.reset()
            interpolator.reset()
            smoother.reset()

        def prepare_policy() -> None:
            nonlocal state
            clear_policy_state()
            state = "preparing"
            engine.resume()
            log_say("Preparing policy", cfg.play_sounds)

        def pause_policy() -> None:
            nonlocal state, homing_waypoints
            clear_policy_state()
            homing_waypoints.clear()
            state = "paused"
            log_say("Paused", cfg.play_sounds)

        def begin_homing() -> None:
            nonlocal state, homing_waypoints
            if not obs_cache.ready or obs_cache.raw is None:
                logger.warning("Cannot home before the first observation is available.")
                return
            clear_policy_state()
            homing_waypoints = _start_homing(
                obs_cache.raw,
                action_keys,
                home_by_prefix,
                cfg.policy_gripper_max_width_m,
                cfg.home_speed_rad_s,
                control_interval,
            )
            state = "homing"
            log_say("Homing", cfg.play_sounds)

        log_say("Paused. Press right arrow to start policy.", cfg.play_sounds)

        loop_i = 0
        last_debug_log_s = 0.0
        debug_stats: dict[str, float] = {}
        prev_loop_start: float | None = None
        while not stop_requested:
            loop_start = time.perf_counter()
            actual_period = 0.0 if prev_loop_start is None else (loop_start - prev_loop_start)
            prev_loop_start = loop_start
            obs_read_dt = 0.0
            obs_process_dt = 0.0
            action_fetch_dt = 0.0
            action_process_dt = 0.0
            send_dt = 0.0
            action_missing = False
            interp_needed_action = False
            interp_empty = False

            if shutdown_event.is_set() or (engine is not None and engine.failed):
                logger.error("Inference engine failed; stopping policy deployment.")
                log_say("Inference engine failed; stopping", cfg.play_sounds)
                break

            event = events.pop_latest()
            if event == "esc":
                stop_requested = True
                break
            if event == "right" and state in ("paused", "preparing"):
                prepare_policy()
            elif event == "space" and state in ("preparing", "running", "homing"):
                pause_policy()
            elif event == "h" and state == "paused":
                begin_homing()

            is_obs_tick = loop_i % obs_stride == 0
            if is_obs_tick:
                obs_read_start = time.perf_counter()
                obs = robot_wrapper.get_observation()
                obs_read_dt = time.perf_counter() - obs_read_start
                obs_cache.raw = obs
                obs_process_start = time.perf_counter()
                obs_cache.processed = robot_observation_processor(obs)
                obs_process_dt = time.perf_counter() - obs_process_start

            if state == "preparing" and is_obs_tick and obs_cache.ready:
                engine.notify_observation(obs_cache.processed)
                action_fetch_start = time.perf_counter()
                action = engine.get_action(
                    build_dataset_frame(dataset_features, obs_cache.processed, prefix=OBS_STR)
                )
                action_fetch_dt = time.perf_counter() - action_fetch_start
                action_missing = action is None
                if action is not None:
                    interpolator.add(action.cpu())
                    state = "running"
                    log_say("Policy running", cfg.play_sounds)

            if state == "running":
                if is_obs_tick and obs_cache.ready:
                    engine.notify_observation(obs_cache.processed)
                    interp_needed_action = interpolator.needs_new_action()
                    if interp_needed_action:
                        action_fetch_start = time.perf_counter()
                        action = engine.get_action(
                            build_dataset_frame(dataset_features, obs_cache.processed, prefix=OBS_STR)
                        )
                        action_fetch_dt = time.perf_counter() - action_fetch_start
                        action_missing = action is None
                        if action is not None:
                            interpolator.add(action.cpu())

                interp = interpolator.get()
                interp_empty = interp is None
                if interp is not None and obs_cache.raw is not None:
                    action_process_start = time.perf_counter()
                    action_dict = _tensor_to_action_dict(interp, action_keys)
                    action_dict = _clamp_policy_action(action_dict, cfg.policy_gripper_max_width_m)
                    robot_action = robot_action_processor((action_dict, obs_cache.raw))
                    robot_action = smoother.step(robot_action)
                    action_process_dt = time.perf_counter() - action_process_start
                    send_start = time.perf_counter()
                    robot_wrapper.send_action(robot_action)
                    send_dt = time.perf_counter() - send_start

            elif state == "homing":
                if homing_waypoints:
                    waypoint = homing_waypoints.popleft()
                    send_start = time.perf_counter()
                    robot_wrapper.send_action(waypoint)
                    send_dt = time.perf_counter() - send_start
                else:
                    state = "paused"
                    log_say("Homing complete", cfg.play_sounds)

            dt = time.perf_counter() - loop_start
            measured_dt = obs_read_dt + obs_process_dt + action_fetch_dt + action_process_dt + send_dt
            unmeasured_dt = max(0.0, dt - measured_dt)
            debug_stats["loops"] = debug_stats.get("loops", 0.0) + 1.0
            debug_stats["sum_loop_dt"] = debug_stats.get("sum_loop_dt", 0.0) + dt
            debug_stats["max_loop_dt"] = max(debug_stats.get("max_loop_dt", 0.0), dt)
            debug_stats["sum_unmeasured_dt"] = debug_stats.get("sum_unmeasured_dt", 0.0) + unmeasured_dt
            debug_stats["max_unmeasured_dt"] = max(debug_stats.get("max_unmeasured_dt", 0.0), unmeasured_dt)
            if actual_period > 0:
                debug_stats["sum_actual_period"] = (
                    debug_stats.get("sum_actual_period", 0.0) + actual_period
                )
                debug_stats["max_actual_period"] = max(
                    debug_stats.get("max_actual_period", 0.0), actual_period
                )
                debug_stats["actual_period_count"] = (
                    debug_stats.get("actual_period_count", 0.0) + 1.0
                )
                period_ms = actual_period * 1e3
                if period_ms < 11.0:
                    bucket = "hist_lt11"
                elif period_ms < 15.0:
                    bucket = "hist_11_15"
                elif period_ms < 25.0:
                    bucket = "hist_15_25"
                elif period_ms < 50.0:
                    bucket = "hist_25_50"
                else:
                    bucket = "hist_gt50"
                debug_stats[bucket] = debug_stats.get(bucket, 0.0) + 1.0
            if is_obs_tick:
                debug_stats["obs_ticks"] = debug_stats.get("obs_ticks", 0.0) + 1.0
                debug_stats["sum_obs_dt"] = debug_stats.get("sum_obs_dt", 0.0) + obs_read_dt
                debug_stats["max_obs_dt"] = max(debug_stats.get("max_obs_dt", 0.0), obs_read_dt)
                debug_stats["sum_obs_process_dt"] = debug_stats.get("sum_obs_process_dt", 0.0) + obs_process_dt
                debug_stats["max_obs_process_dt"] = max(
                    debug_stats.get("max_obs_process_dt", 0.0), obs_process_dt
                )
            if action_fetch_dt > 0:
                debug_stats["action_fetches"] = debug_stats.get("action_fetches", 0.0) + 1.0
                debug_stats["sum_action_fetch_dt"] = (
                    debug_stats.get("sum_action_fetch_dt", 0.0) + action_fetch_dt
                )
                debug_stats["max_action_fetch_dt"] = max(
                    debug_stats.get("max_action_fetch_dt", 0.0), action_fetch_dt
                )
            if action_process_dt > 0:
                debug_stats["action_processes"] = debug_stats.get("action_processes", 0.0) + 1.0
                debug_stats["sum_action_process_dt"] = (
                    debug_stats.get("sum_action_process_dt", 0.0) + action_process_dt
                )
                debug_stats["max_action_process_dt"] = max(
                    debug_stats.get("max_action_process_dt", 0.0), action_process_dt
                )
            if send_dt > 0:
                debug_stats["sends"] = debug_stats.get("sends", 0.0) + 1.0
                debug_stats["sum_send_dt"] = debug_stats.get("sum_send_dt", 0.0) + send_dt
                debug_stats["max_send_dt"] = max(debug_stats.get("max_send_dt", 0.0), send_dt)
            if action_missing:
                debug_stats["action_missing"] = debug_stats.get("action_missing", 0.0) + 1.0
            if interp_empty:
                debug_stats["interp_empty"] = debug_stats.get("interp_empty", 0.0) + 1.0

            sleep_target = max(0.0, control_interval - dt)
            if sleep_target > 0:
                sleep_call_start = time.perf_counter()
                precise_sleep(sleep_target)
                sleep_actual = time.perf_counter() - sleep_call_start
                sleep_overshoot = max(0.0, sleep_actual - sleep_target)
                debug_stats["sleep_count"] = debug_stats.get("sleep_count", 0.0) + 1.0
                debug_stats["sum_sleep_target"] = debug_stats.get("sum_sleep_target", 0.0) + sleep_target
                debug_stats["sum_sleep_actual"] = debug_stats.get("sum_sleep_actual", 0.0) + sleep_actual
                debug_stats["sum_sleep_overshoot"] = (
                    debug_stats.get("sum_sleep_overshoot", 0.0) + sleep_overshoot
                )
                debug_stats["max_sleep_overshoot"] = max(
                    debug_stats.get("max_sleep_overshoot", 0.0), sleep_overshoot
                )
            else:
                debug_stats["no_sleep_count"] = debug_stats.get("no_sleep_count", 0.0) + 1.0
                if state in ("running", "homing"):
                    logger.warning(
                        "Loop is running slower (%.1f Hz) than target control rate (%.1f Hz)",
                        1 / max(dt, 1e-6),
                        1 / control_interval,
                    )

            debug_now = time.perf_counter()
            if state in ("preparing", "running", "homing") and debug_now - last_debug_log_s >= 1.0:
                rtc_debug = engine.debug_snapshot() if engine is not None else {}
                latency = rtc_debug.get("last_latency_s")
                latency_ms = None if latency is None else latency * 1e3
                rtc_timing = rtc_debug.get("last_timing_ms") or {}
                loops = max(debug_stats.get("loops", 0.0), 1.0)
                obs_ticks = max(debug_stats.get("obs_ticks", 0.0), 1.0)
                action_fetches = max(debug_stats.get("action_fetches", 0.0), 1.0)
                action_processes = max(debug_stats.get("action_processes", 0.0), 1.0)
                sends = max(debug_stats.get("sends", 0.0), 1.0)
                period_count = max(debug_stats.get("actual_period_count", 0.0), 1.0)
                sleep_count = max(debug_stats.get("sleep_count", 0.0), 1.0)
                logger.info(
                    "deploy_loop state=%s work_avg_hz=%.1f work_min_hz=%.1f actual_avg_hz=%.1f "
                    "actual_min_hz=%.1f target_hz=%.1f hist={lt11:%d,11-15:%d,15-25:%d,25-50:%d,gt50:%d} "
                    "obs_avg_ms=%.2f obs_max_ms=%.2f obs_proc_avg_ms=%.2f obs_proc_max_ms=%.2f "
                    "fetch_avg_ms=%.2f fetch_max_ms=%.2f proc_avg_ms=%.2f proc_max_ms=%.2f "
                    "send_avg_ms=%.2f send_max_ms=%.2f unmeas_avg_ms=%.2f unmeas_max_ms=%.2f "
                    "sleep_count=%d no_sleep_count=%d sleep_target_avg_ms=%.2f sleep_actual_avg_ms=%.2f "
                    "sleep_overshoot_avg_ms=%.2f sleep_overshoot_max_ms=%.2f "
                    "action_missing=%d interp_empty=%d rtc_queue=%s "
                    "rtc_last_latency_ms=%s rtc_last_delay=%s rtc_inferences=%s "
                    "rtc_phase=%s rtc_phase_age_ms=%s rtc_timing_ms=%s",
                    state,
                    loops / max(debug_stats.get("sum_loop_dt", 0.0), 1e-6),
                    1 / max(debug_stats.get("max_loop_dt", 0.0), 1e-6),
                    period_count / max(debug_stats.get("sum_actual_period", 0.0), 1e-6),
                    1 / max(debug_stats.get("max_actual_period", 0.0), 1e-6),
                    1 / control_interval,
                    int(debug_stats.get("hist_lt11", 0.0)),
                    int(debug_stats.get("hist_11_15", 0.0)),
                    int(debug_stats.get("hist_15_25", 0.0)),
                    int(debug_stats.get("hist_25_50", 0.0)),
                    int(debug_stats.get("hist_gt50", 0.0)),
                    debug_stats.get("sum_obs_dt", 0.0) / obs_ticks * 1e3,
                    debug_stats.get("max_obs_dt", 0.0) * 1e3,
                    debug_stats.get("sum_obs_process_dt", 0.0) / obs_ticks * 1e3,
                    debug_stats.get("max_obs_process_dt", 0.0) * 1e3,
                    debug_stats.get("sum_action_fetch_dt", 0.0) / action_fetches * 1e3,
                    debug_stats.get("max_action_fetch_dt", 0.0) * 1e3,
                    debug_stats.get("sum_action_process_dt", 0.0) / action_processes * 1e3,
                    debug_stats.get("max_action_process_dt", 0.0) * 1e3,
                    debug_stats.get("sum_send_dt", 0.0) / sends * 1e3,
                    debug_stats.get("max_send_dt", 0.0) * 1e3,
                    debug_stats.get("sum_unmeasured_dt", 0.0) / loops * 1e3,
                    debug_stats.get("max_unmeasured_dt", 0.0) * 1e3,
                    int(debug_stats.get("sleep_count", 0.0)),
                    int(debug_stats.get("no_sleep_count", 0.0)),
                    debug_stats.get("sum_sleep_target", 0.0) / sleep_count * 1e3,
                    debug_stats.get("sum_sleep_actual", 0.0) / sleep_count * 1e3,
                    debug_stats.get("sum_sleep_overshoot", 0.0) / sleep_count * 1e3,
                    debug_stats.get("max_sleep_overshoot", 0.0) * 1e3,
                    int(debug_stats.get("action_missing", 0.0)),
                    int(debug_stats.get("interp_empty", 0.0)),
                    rtc_debug.get("queue_size"),
                    None if latency_ms is None else round(latency_ms, 1),
                    rtc_debug.get("last_real_delay"),
                    rtc_debug.get("inference_count"),
                    rtc_debug.get("current_phase"),
                    rtc_debug.get("current_phase_duration_ms"),
                    {key: round(value, 1) for key, value in rtc_timing.items()},
                )
                last_debug_log_s = debug_now
                debug_stats.clear()
            loop_i += 1

    finally:
        log_say("Stopping policy deployment", cfg.play_sounds, blocking=True)
        if engine is not None:
            engine.stop()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()


def main():
    register_third_party_plugins()
    policy_deploy()


if __name__ == "__main__":
    main()
