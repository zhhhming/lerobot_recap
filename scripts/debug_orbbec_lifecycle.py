#!/usr/bin/env python3

"""Exercise Orbbec camera open/read/close cycles outside robot control.

This is a diagnostic script, not production robot code. It intentionally keeps
the test surface small so camera lifecycle issues can be separated from teleop,
dataset writing, and arm control.
"""

from __future__ import annotations

import argparse
import ctypes
import site
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any


DEFAULT_SERIALS = ["CP2AB530007Z", "CP2R553000EP", "CP2R553000NZ"]


def preload_orbbec_sdk_library() -> Path | None:
    """Prefer the pyorbbecsdk-bundled runtime over older ROS/system copies."""
    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        candidates = [root / "libOrbbecSDK.so", *sorted(root.glob("libOrbbecSDK.so.*"))]
        for bundled_lib in candidates:
            if bundled_lib.is_file():
                ctypes.CDLL(str(bundled_lib), mode=ctypes.RTLD_GLOBAL)
                return bundled_lib
    return None


def import_sdk():
    loaded = preload_orbbec_sdk_library()
    if loaded is not None:
        print(f"[env] preloaded {loaded}")
    import pyorbbecsdk as ob  # type: ignore

    print(f"[env] pyorbbecsdk loaded from {getattr(ob, '__file__', '<unknown>')}")
    return ob


def find_devices(ob: Any, serials: list[str]) -> list[Any]:
    ctx = ob.Context()
    device_list = ctx.query_devices()
    devices_by_serial = {}

    print(f"[sdk] found {device_list.get_count()} Orbbec device(s)")
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        serial = info.get_serial_number()
        devices_by_serial[serial] = device
        print(
            "[sdk] device "
            f"index={index} name={info.get_name()} serial={serial} "
            f"firmware={info.get_firmware_version()} connection={info.get_connection_type()}"
        )

    missing = [serial for serial in serials if serial not in devices_by_serial]
    if missing:
        raise RuntimeError(f"Missing requested serial(s): {missing}")

    return [devices_by_serial[serial] for serial in serials]


def select_color_profile(ob: Any, pipeline: Any, width: int, height: int, fps: int) -> Any:
    profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    try:
        return profile_list.get_video_stream_profile(width, height, ob.OBFormat.RGB, fps)
    except Exception as exc:
        print(f"[sdk] exact RGB {width}x{height}@{fps} unavailable: {exc}")
        profile = profile_list.get_default_video_stream_profile()
        print(
            "[sdk] using default profile "
            f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()} "
            f"{profile.get_format().name}"
        )
        return profile


def make_sdk_pipeline(ob: Any, device: Any, width: int, height: int, fps: int) -> tuple[Any, Any, str]:
    info = device.get_device_info()
    serial = info.get_serial_number()
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    config.disable_all_stream()
    color_profile = select_color_profile(ob, pipeline, width, height, fps)
    config.enable_stream(color_profile)
    return pipeline, config, serial


def frameset_has_color(frames: Any) -> bool:
    return frames is not None and frames.get_color_frame() is not None


def stop_pipelines(pipelines: list[tuple[str, Any]]) -> None:
    for serial, pipeline in reversed(pipelines):
        t0 = time.perf_counter()
        try:
            pipeline.stop()
            print(f"[sdk] stopped {serial} in {(time.perf_counter() - t0) * 1000:.1f} ms")
        except Exception as exc:
            print(f"[sdk] stop failed for {serial}: {exc}")


def run_sdk_wait(args: argparse.Namespace) -> None:
    ob = import_sdk()
    devices = find_devices(ob, args.serial)

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== sdk-wait cycle {cycle}/{args.cycles} ===")
        pipelines = [
            make_sdk_pipeline(ob, device, args.width, args.height, args.fps) for device in devices
        ]
        started: list[tuple[str, Any]] = []
        try:
            for pipeline, config, serial in pipelines:
                t0 = time.perf_counter()
                pipeline.start(config)
                started.append((serial, pipeline))
                print(f"[sdk] started {serial} in {(time.perf_counter() - t0) * 1000:.1f} ms")

            counts = defaultdict(int)
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                for serial, pipeline in started:
                    frames = pipeline.wait_for_frames(args.timeout_ms)
                    if frameset_has_color(frames):
                        counts[serial] += 1
            for serial, _ in started:
                print(f"[sdk] {serial} color frames={counts[serial]}")
        finally:
            stop_pipelines(started)
            time.sleep(args.settle_s)


def run_sdk_callback(args: argparse.Namespace) -> None:
    ob = import_sdk()
    devices = find_devices(ob, args.serial)

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== sdk-callback cycle {cycle}/{args.cycles} ===")
        pipelines = [
            make_sdk_pipeline(ob, device, args.width, args.height, args.fps) for device in devices
        ]
        queues: dict[str, Queue[Any]] = {serial: Queue(maxsize=args.queue_size) for _, _, serial in pipelines}
        started: list[tuple[str, Any]] = []

        def on_frame(frames: Any, serial: str) -> None:
            if not frameset_has_color(frames):
                return
            queue = queues[serial]
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(frames)

        try:
            for pipeline, config, serial in pipelines:
                t0 = time.perf_counter()
                pipeline.start(config, lambda frames, serial=serial: on_frame(frames, serial))
                started.append((serial, pipeline))
                print(f"[sdk] started callback {serial} in {(time.perf_counter() - t0) * 1000:.1f} ms")

            counts = defaultdict(int)
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                for serial, queue in queues.items():
                    while not queue.empty():
                        queue.get_nowait()
                        counts[serial] += 1
                time.sleep(0.005)
            for serial, _ in started:
                print(f"[sdk] {serial} callback color frames={counts[serial]}")
        finally:
            stop_pipelines(started)
            time.sleep(args.settle_s)


def run_sdk_thread(args: argparse.Namespace, *, stop_first: bool) -> None:
    ob = import_sdk()
    devices = find_devices(ob, args.serial)

    for cycle in range(1, args.cycles + 1):
        label = "sdk-thread-stop-first" if stop_first else "sdk-thread-current"
        print(f"\n=== {label} cycle {cycle}/{args.cycles} ===")
        pipelines = [
            make_sdk_pipeline(ob, device, args.width, args.height, args.fps) for device in devices
        ]
        started: list[tuple[str, Any]] = []
        stop_event = Event()
        counts = defaultdict(int)
        errors: dict[str, list[str]] = defaultdict(list)
        threads: list[tuple[str, Thread]] = []
        pipelines_stopped = False

        def read_loop(serial: str, pipeline: Any) -> None:
            while not stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(10000)
                    if frameset_has_color(frames):
                        counts[serial] += 1
                except Exception as exc:
                    errors[serial].append(str(exc))
                    if stop_event.is_set():
                        return

        try:
            for pipeline, config, serial in pipelines:
                pipeline.start(config)
                started.append((serial, pipeline))
                thread = Thread(target=read_loop, args=(serial, pipeline), name=f"orbbec-test-{serial}")
                thread.start()
                threads.append((serial, thread))
                print(f"[thread] started {serial}")

            time.sleep(args.duration_s)

            if stop_first:
                print("[thread] stopping pipelines before joining read threads")
                stop_pipelines(started)
                pipelines_stopped = True
                stop_event.set()
            else:
                print("[thread] setting stop event, joining 2s, then stopping pipelines")
                stop_event.set()

            for serial, thread in threads:
                t0 = time.perf_counter()
                thread.join(timeout=2.0)
                print(
                    f"[thread] pre-stop join {serial}: alive={thread.is_alive()} "
                    f"waited={(time.perf_counter() - t0) * 1000:.1f} ms frames={counts[serial]} "
                    f"errors={len(errors[serial])}"
                )

            if not stop_first:
                stop_pipelines(started)
                pipelines_stopped = True

            for serial, thread in threads:
                t0 = time.perf_counter()
                thread.join(timeout=3.0)
                print(
                    f"[thread] final join {serial}: alive={thread.is_alive()} "
                    f"waited={(time.perf_counter() - t0) * 1000:.1f} ms"
                )
        finally:
            stop_event.set()
            if not pipelines_stopped:
                for serial, pipeline in reversed(started):
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            time.sleep(args.settle_s)


def run_wrapper(args: argparse.Namespace) -> None:
    # Importing this module also applies LeRobot's SDK preload workaround.
    from lerobot.cameras.orbbec.camera_orbbec import OrbbecCamera
    from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== wrapper cycle {cycle}/{args.cycles} ===")
        config_kwargs = {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "warmup_s": args.warmup_s,
            "exposure": args.exposure,
        }
        if args.auto_exposure is not None:
            config_kwargs["auto_exposure"] = args.auto_exposure

        cameras = [
            OrbbecCamera(
                OrbbecCameraConfig(
                    serial_number_or_name=serial,
                    **config_kwargs,
                )
            )
            for serial in args.serial
        ]
        connected: list[OrbbecCamera] = []
        try:
            for camera in cameras:
                t0 = time.perf_counter()
                camera.connect(warmup=True)
                connected.append(camera)
                print(f"[wrapper] connected {camera} in {(time.perf_counter() - t0) * 1000:.1f} ms")

            counts = defaultdict(int)
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                for camera in connected:
                    camera.async_read(timeout_ms=args.timeout_ms)
                    counts[str(camera)] += 1
            for camera in connected:
                print(f"[wrapper] {camera} frames={counts[str(camera)]}")
        finally:
            for camera in reversed(connected):
                t0 = time.perf_counter()
                try:
                    camera.disconnect()
                    print(
                        f"[wrapper] disconnected {camera} "
                        f"in {(time.perf_counter() - t0) * 1000:.1f} ms"
                    )
                except Exception as exc:
                    print(f"[wrapper] disconnect failed for {camera}: {exc}")
            time.sleep(args.settle_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["wrapper", "sdk-wait", "sdk-callback", "sdk-thread-current", "sdk-thread-stop-first"],
        default="wrapper",
    )
    parser.add_argument("--serial", action="append", default=[], help="Camera serial. Repeat for multiple cameras.")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-s", type=int, default=1)
    exposure_group = parser.add_mutually_exclusive_group()
    exposure_group.add_argument("--auto-exposure", dest="auto_exposure", action="store_true")
    exposure_group.add_argument("--manual-exposure", dest="auto_exposure", action="store_false")
    parser.set_defaults(auto_exposure=None)
    parser.add_argument("--exposure", type=int, default=300)
    parser.add_argument("--queue-size", type=int, default=5)
    parser.add_argument("--list-only", action="store_true", help="List devices and exit without opening streams.")
    args = parser.parse_args()
    if not args.serial:
        args.serial = list(DEFAULT_SERIALS)
    return args


def main() -> int:
    args = parse_args()
    print(f"[args] {args}")
    if args.list_only:
        ob = import_sdk()
        find_devices(ob, args.serial)
    elif args.mode == "wrapper":
        run_wrapper(args)
    elif args.mode == "sdk-wait":
        run_sdk_wait(args)
    elif args.mode == "sdk-callback":
        run_sdk_callback(args)
    elif args.mode == "sdk-thread-current":
        run_sdk_thread(args, stop_first=False)
    elif args.mode == "sdk-thread-stop-first":
        run_sdk_thread(args, stop_first=True)
    else:
        raise AssertionError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
