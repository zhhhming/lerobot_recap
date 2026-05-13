Nero URDF and mesh assets copied into `lerobot` for local teleop/runtime use.

Files:
- `urdf/nero_with_gripper_flange_description.urdf`
  Used by Placo IK in `pico_nero_teleop`.
- `urdf/nero_with_gripper_description.urdf`
  Used by Pinocchio gravity compensation in `nero_follower`.
- `meshes/`
  Meshes referenced by both URDFs via `file:///.../lerobot/src/lerobot/assets/nero/meshes/...`.

These copies were made from the local `nero` repo so teleop/runtime code does
not depend on external absolute paths outside the `lerobot` workspace.
