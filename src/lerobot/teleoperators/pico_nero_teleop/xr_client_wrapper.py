import numpy as np


class XrClient:
    """Client wrapping the xrobotoolkit_sdk for Pico controllers + headset.

    Mirrors /home/zenbot-robot/repos/xr_client.py. Kept in-tree so the package
    does not need a sys.path hack to reach that reference file. The underlying
    xrobotoolkit_sdk must be initialized exactly once per process; this class
    performs init() in __init__ and close() in close().
    """

    def __init__(self) -> None:
        import xrobotoolkit_sdk as xrt

        self._xrt = xrt
        xrt.init()

    def get_pose_by_name(self, name: str) -> np.ndarray:
        if name == "left_controller":
            return self._xrt.get_left_controller_pose()
        if name == "right_controller":
            return self._xrt.get_right_controller_pose()
        if name == "headset":
            return self._xrt.get_headset_pose()
        raise ValueError(f"Invalid pose name: {name}")

    def get_key_value_by_name(self, name: str) -> float:
        if name == "left_trigger":
            return self._xrt.get_left_trigger()
        if name == "right_trigger":
            return self._xrt.get_right_trigger()
        if name == "left_grip":
            return self._xrt.get_left_grip()
        if name == "right_grip":
            return self._xrt.get_right_grip()
        raise ValueError(f"Invalid key value name: {name}")

    def get_button_state_by_name(self, name: str) -> bool:
        if name == "A":
            return self._xrt.get_A_button()
        if name == "B":
            return self._xrt.get_B_button()
        if name == "X":
            return self._xrt.get_X_button()
        if name == "Y":
            return self._xrt.get_Y_button()
        if name == "left_menu_button":
            return self._xrt.get_left_menu_button()
        if name == "right_menu_button":
            return self._xrt.get_right_menu_button()
        raise ValueError(f"Invalid button name: {name}")

    def close(self) -> None:
        self._xrt.close()
