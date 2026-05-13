import numpy as np


class JointSmoother:
    """EMA on joint targets + per-joint velocity clamp.

    Both mechanisms are opt-in:
      * alpha == 1.0 disables the EMA (raw target passes through)
      * max_qdot is None disables the velocity clamp
    Call reset(q_now) when re-entering active teleop or after a home to avoid
    stale internal state causing a jump on the next step.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_qdot: np.ndarray | list[float] | None = None,
        dt: float = 1.0 / 80.0,
    ) -> None:
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.max_qdot = None if max_qdot is None else np.asarray(max_qdot, dtype=float)
        self._q_ema: np.ndarray | None = None
        self._q_out: np.ndarray | None = None

    def reset(self, q_now: np.ndarray | list[float]) -> None:
        q = np.asarray(q_now, dtype=float).copy()
        self._q_ema = q
        self._q_out = q

    def step(self, q_target_raw: np.ndarray | list[float]) -> np.ndarray:
        q_raw = np.asarray(q_target_raw, dtype=float)
        if self._q_ema is None or self._q_out is None:
            self.reset(q_raw)
            return self._q_out.copy()

        q_ema = self.alpha * q_raw + (1.0 - self.alpha) * self._q_ema
        if self.max_qdot is None:
            q_out = q_ema
        else:
            step_cap = self.max_qdot * self.dt
            delta = np.clip(q_ema - self._q_out, -step_cap, step_cap)
            q_out = self._q_out + delta

        self._q_ema = q_ema
        self._q_out = q_out
        return q_out.copy()
