from dataclasses import dataclass, field

from lerobot.teleoperators.pico_nero_teleop import PicoNeroTeleopConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_pico_nero_teleop")
@dataclass(kw_only=True)
class BiPicoNeroTeleopConfig(TeleoperatorConfig):
    id: str | None = "bi_pico_nero_teleop"

    left_teleop_config: PicoNeroTeleopConfigBase = field(
        default_factory=lambda: PicoNeroTeleopConfigBase(side="left", home_button="Y")
    )
    right_teleop_config: PicoNeroTeleopConfigBase = field(
        default_factory=lambda: PicoNeroTeleopConfigBase(side="right", home_button="B")
    )
