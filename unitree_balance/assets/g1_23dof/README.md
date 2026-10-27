# G1 with 23 actuated joints

`original.xml` and `meshes/` come from Unitree's `unitree_mujoco` repository,
pinned by commit and SHA-256 checksums in `provenance.json`. See `LICENSE`.

`g1.xml` preserves the official physical robot's body transforms, masses,
inertias, joint axes, limits, force limits and visual meshes. The six detached
dummy joints in the upstream file are SDK placeholders and are removed.
The physical robot has 12 leg joints, waist yaw, and 5 joints per arm, including
wrist roll. It has no waist roll/pitch or wrist pitch/yaw joints.

The reference `humanoid` repository's `scripts/build_g1_23dof.py` generated this
preserved training MJCF. To retain the previous
contact/control recipe, it replaces upstream mesh collisions with mjlab 1.6.0's
G1 capsule/sphere approximations, retains its seven capsules per foot and sensor
sites, and uses its position-control gains and armatures with zero passive
joint friction/damping. Wrist/hand capsules are expressed in the rigid hand
frame. Official 88 Nm hip-roll limits are respected, including action scaling.
These collision shapes are simulation approximations, not exact mesh contacts.

The `Unitree-G1-23Dof-Balance` port always selects this model. Training records
the variant and model hash; evaluation requires matching explicit 23-DoF
metadata in the checkpoint's adjacent `experiment.json`. Keep that file with
checkpoints. Missing metadata and 29-DoF checkpoints are rejected.
