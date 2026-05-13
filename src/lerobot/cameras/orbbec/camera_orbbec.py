# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import ctypes
import logging
import site
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np
from numpy.typing import NDArray

_PYORBBECSDK_IMPORT_ERROR: Exception | None = None


def _preload_orbbec_sdk_library() -> None:
    # Prefer the package-bundled Orbbec SDK over any older system / ROS copy
    # that might be injected through LD_LIBRARY_PATH.
    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        candidates = [root / "libOrbbecSDK.so", *sorted(root.glob("libOrbbecSDK.so.*"))]
        for bundled_lib in candidates:
            if bundled_lib.is_file():
                ctypes.CDLL(str(bundled_lib), mode=ctypes.RTLD_GLOBAL)
                return


try:
    _preload_orbbec_sdk_library()
    import pyorbbecsdk as ob  # type: ignore  # TODO: add type stubs for pyorbbecsdk
except Exception as e:
    _PYORBBECSDK_IMPORT_ERROR = e
    ob = None
    logging.info(f"Could not import pyorbbecsdk: {e}")

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """RGB-only Orbbec camera backed by the pyorbbecsdk v2 runtime."""

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)

        self.config = config
        self.device_identifier = config.serial_number_or_name
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.device: Any | None = None
        self.pipeline: Any | None = None
        self.pipeline_config: Any | None = None
        self.color_profile: Any | None = None
        self.serial_number: str | None = None
        self.device_name: str | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        identifier = self.serial_number or self.device_identifier or self.device_name or "default"
        return f"{self.__class__.__name__}({identifier})"

    @staticmethod
    def _require_sdk() -> None:
        if ob is None:
            raise ImportError(
                "pyorbbecsdk is not available. Install the Orbbec v2 Python package (`pyorbbecsdk2`) "
                "in the active environment."
            ) from _PYORBBECSDK_IMPORT_ERROR

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.color_profile is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._require_sdk()
        self.device = self._find_device()
        self.pipeline = ob.Pipeline(self.device) if self.device is not None else ob.Pipeline()
        self.pipeline_config = ob.Config()
        self._configure_pipeline()
        self._configure_device_properties()

        try:
            self.pipeline.start(self.pipeline_config)
        except Exception as e:
            self.pipeline = None
            self.pipeline_config = None
            raise ConnectionError(
                f"Failed to open {self}. Make sure pyorbbecsdk is installed and device permissions are configured."
            ) from e

        self._configure_capture_settings()
        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info("%s connected.", self)

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        OrbbecCamera._require_sdk()
        context = ob.Context()
        devices = context.query_devices()
        found_cameras_info: list[dict[str, Any]] = []

        for index in range(devices.get_count()):
            device = devices.get_device_by_index(index)
            device_info = device.get_device_info()
            camera_info: dict[str, Any] = {
                "name": device_info.get_name(),
                "type": "Orbbec",
                "id": device_info.get_serial_number(),
                "firmware_version": device_info.get_firmware_version(),
                "hardware_version": device_info.get_hardware_version(),
                "connection_type": device_info.get_connection_type(),
            }

            try:
                pipeline = ob.Pipeline(device)
                profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                camera_info["default_stream_profile"] = {
                    "format": color_profile.get_format().name,
                    "width": color_profile.get_width(),
                    "height": color_profile.get_height(),
                    "fps": color_profile.get_fps(),
                }
            except Exception:
                pass

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_device(self) -> Any | None:
        context = ob.Context()
        devices = context.query_devices()

        if devices.get_count() == 0:
            raise ConnectionError("No Orbbec cameras detected.")

        if not self.device_identifier:
            device = devices.get_device_by_index(0)
            info = device.get_device_info()
            self.serial_number = info.get_serial_number()
            self.device_name = info.get_name()
            return device

        serial_matches = []
        name_matches = []
        for index in range(devices.get_count()):
            device = devices.get_device_by_index(index)
            info = device.get_device_info()
            if info.get_serial_number() == self.device_identifier:
                serial_matches.append(device)
            if info.get_name() == self.device_identifier:
                name_matches.append(device)

        if len(serial_matches) == 1:
            device = serial_matches[0]
        elif len(serial_matches) > 1:
            raise ValueError(
                f"Multiple Orbbec cameras found with serial number '{self.device_identifier}'. This should not happen."
            )
        elif len(name_matches) == 1:
            device = name_matches[0]
        elif len(name_matches) > 1:
            serial_numbers = [cam.get_device_info().get_serial_number() for cam in name_matches]
            raise ValueError(
                f"Multiple Orbbec cameras found with name '{self.device_identifier}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )
        else:
            available = [
                {
                    "name": devices.get_device_by_index(index).get_device_info().get_name(),
                    "serial": devices.get_device_by_index(index).get_device_info().get_serial_number(),
                }
                for index in range(devices.get_count())
            ]
            raise ValueError(
                f"No Orbbec camera found with identifier '{self.device_identifier}'. Available devices: {available}"
            )

        info = device.get_device_info()
        self.serial_number = info.get_serial_number()
        self.device_name = info.get_name()
        return device

    def _configure_pipeline(self) -> None:
        if self.pipeline is None or self.pipeline_config is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        self.pipeline_config.disable_all_stream()

        profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)

        try:
            if self.width is not None and self.height is not None and self.fps is not None:
                self.color_profile = profile_list.get_video_stream_profile(
                    self.capture_width, self.capture_height, ob.OBFormat.RGB, self.fps
                )
            else:
                try:
                    self.color_profile = profile_list.get_video_stream_profile(0, 0, ob.OBFormat.RGB, 0)
                except Exception:
                    self.color_profile = profile_list.get_default_video_stream_profile()
        except Exception as e:
            raise RuntimeError(
                f"{self} failed to find a compatible color profile for width={self.width}, "
                f"height={self.height}, fps={self.fps}."
            ) from e

        self.pipeline_config.enable_stream(self.color_profile)

    def _property_is_supported(self, property_id: Any, permission: Any) -> bool:
        if self.device is None:
            return False

        try:
            return bool(self.device.is_property_supported(property_id, permission))
        except Exception:
            return False

    def _set_bool_property_if_supported(self, property_id: Any, value: bool, label: str) -> None:
        if self.device is None:
            return

        if not self._property_is_supported(property_id, ob.OBPermissionType.PERMISSION_WRITE):
            logger.info("%s does not support %s.", self, label)
            return

        try:
            self.device.set_bool_property(property_id, value)
        except Exception as e:
            logger.warning("Failed to set %s for %s: %s", label, self, e)

    def _set_int_property_if_supported(self, property_id: Any, value: int, label: str) -> None:
        if self.device is None:
            return

        if not self._property_is_supported(property_id, ob.OBPermissionType.PERMISSION_WRITE):
            logger.info("%s does not support %s.", self, label)
            return

        try:
            self.device.set_int_property(property_id, value)
        except Exception as e:
            logger.warning("Failed to set %s for %s: %s", label, self, e)

    def _configure_device_properties(self) -> None:
        if self.device is None:
            return

        self._set_bool_property_if_supported(
            ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            self.config.auto_exposure,
            "color auto exposure",
        )

        if not self.config.auto_exposure:
            self._set_int_property_if_supported(
                ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
                self.config.exposure,
                "color exposure",
            )

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        if self.color_profile is None:
            raise RuntimeError(f"{self}: color_profile must be initialized before use.")

        stream = self.color_profile.as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.get_fps()

        actual_width = int(round(stream.get_width()))
        actual_height = int(round(stream.get_height()))

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = actual_height, actual_width
            self.capture_width, self.capture_height = actual_width, actual_height
        else:
            self.width, self.height = actual_width, actual_height
            self.capture_width, self.capture_height = actual_width, actual_height

    def _read_from_hardware(self) -> Any:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        frame = self.pipeline.wait_for_frames(10000)
        if frame is None:
            raise RuntimeError(f"{self} read failed (frame is None).")

        return frame

    def _decode_color_frame(self, frame: Any) -> NDArray[Any]:
        color_frame = frame.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} frameset did not contain a color frame.")

        width = int(color_frame.get_width())
        height = int(color_frame.get_height())
        raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        frame_format = color_frame.get_format().name

        if frame_format == "RGB":
            image = raw.reshape((height, width, 3))
        elif frame_format == "BGR":
            image = raw.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif frame_format in ("MJPG", "MJPEG"):
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"{self} failed to decode MJPEG color frame.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif frame_format in ("YUYV", "YUY2"):
            image = raw.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)
        elif frame_format == "UYVY":
            image = raw.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
        elif frame_format == "NV12":
            image = raw.reshape((height * 3 // 2, width))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
        else:
            raise RuntimeError(f"{self} unsupported Orbbec color format: {frame_format}.")

        return image

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                color_frame = self._decode_color_frame(frame)
                processed_frame = self._postprocess_image(color_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        self.pipeline_config = None
        self.color_profile = None
        self.device = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
