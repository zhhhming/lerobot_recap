"""Minimal Pinocchio gravity-compensation probe for Nero + gripper.

Usage:
    python3 scripts/nero_teleop/probe_pinocchio_gravity.py --print-model
    python3 scripts/nero_teleop/probe_pinocchio_gravity.py --q-arm-rad "0,0.35,0,1.75,0,0,-0.6" --gripper-m 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pinocchio_gravity import DEFAULT_URDF, ENV_INFO, NeroPinocchioGravity, build_q_full
from _mit_test_utils import fmt_vec, parse_target_rad_arg  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-path", default=str(DEFAULT_URDF))
    ap.add_argument("--q-arm-rad", default="0,0.35,0,1.75,0,0,-0.6")
    ap.add_argument("--gripper-m", type=float, default=0.1)
    ap.add_argument("--print-model", action="store_true")
    args = ap.parse_args()

    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}. Run build_nero_urdf.sh first."
        )
    grav = NeroPinocchioGravity(urdf_path)

    q_arm = parse_target_rad_arg(args.q_arm_rad)
    q_full = build_q_full(q_arm, args.gripper_m)
    tau_g = grav.compute_full(q_arm, args.gripper_m)

    print(f"env_prefix={ENV_INFO['prefix']}")
    print(f"urdf_path={urdf_path}")
    print(f"nq={grav.model.nq} nv={grav.model.nv}")
    if args.print_model:
        print(f"joint_names={grav.model.names}")
    print(f"q_arm_rad={fmt_vec(q_arm, digits=4)}")
    print(f"gripper_m={args.gripper_m:.4f} q_full={fmt_vec(q_full, digits=4)}")
    print(f"tau_g_full={fmt_vec(tau_g, digits=4)}")
    print(f"tau_g_arm={fmt_vec(tau_g[:7], digits=4)}")
    print(f"tau_g_gripper={fmt_vec(tau_g[7:], digits=4)}")


if __name__ == "__main__":
    main()
