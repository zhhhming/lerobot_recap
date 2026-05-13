#!/usr/bin/env python

"""Thread-safe robot wrapper used by asynchronous inference."""

from __future__ import annotations

from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            return self._robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def inner(self) -> Robot:
        return self._robot
