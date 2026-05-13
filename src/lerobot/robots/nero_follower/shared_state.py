import threading
import time
from dataclasses import dataclass, field


@dataclass
class TargetCmd:
    q_rad: list[float]
    gripper_m: float
    source: str  # "teleop" | "home" | "init"
    ts: float = field(default_factory=time.monotonic)


class SharedArmState:
    """Thread-safe hand-off between teleop and robot exec threads (same process).

    The teleop thread writes the latest joint + gripper target here; the robot's
    exec thread reads it at a fixed rate and forwards to pyAgxArm. Current-state
    (measured joints / gripper) is NOT stored here; the teleop reads it directly
    from the NeroArmDriver whose internal SDK read-thread keeps it fresh.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._target: TargetCmd | None = None
        self._fault: str | None = None

    def write_target(self, q_rad: list[float], gripper_m: float, source: str) -> None:
        with self._lock:
            self._target = TargetCmd(
                q_rad=list(q_rad), gripper_m=float(gripper_m), source=source, ts=time.monotonic()
            )

    def read_target(self) -> TargetCmd | None:
        with self._lock:
            return self._target

    def set_fault(self, reason: str) -> None:
        with self._lock:
            self._fault = reason

    def get_fault(self) -> str | None:
        with self._lock:
            return self._fault

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None
