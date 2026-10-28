"""Independent residual-PPO environments for the five-stage scene."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import dr
from mjlab.sensor import ContactMatch, ContactSensorCfg

from . import motion
from . import task as tk
from . import scene as sc
from .geometry import geometry_cfg
from .five_plans import CONFIG, OUT, SKILLS, POLICIES, configure_dimensions
from .five_requirements import posture_ok, upright_putdown_ok


class FiveCommand(motion.MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.skill = cfg.skill
        self.arm_ids = [i for i,n in enumerate(self.robot.joint_names)
                        if any(part in n for part in ("shoulder", "elbow", "wrist"))]
        self.setup = json.loads(CONFIG.read_text())
        self.boxe: Entity = env.scene["box"]
        self.palm_ids = [self.robot.site_names.index(s) for s in ("left_palm", "right_palm")]
        names = env.scene["hand_box"].primary_names
        self.hand_cols = [i for side in ("left", "right") for i,n in enumerate(names) if n.startswith(side)]
        assert len(self.hand_cols) == 2, names
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hold_required = torch.ones(self.num_envs, device=self.device)
        self.metrics["grip"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["box_error"] = torch.zeros(self.num_envs, device=self.device)
        if self.skill == "put-down":
            times = {name: time for name,time,_ in self.plan.phases}
            self.release_frame = round(times["clearance-step"]/env.step_dt)
            self.upright_frame = round(times["released-stand"]/env.step_dt)
            self.lower_frame = round(times["lower-arms"]/env.step_dt)
            self.turn_frame = round(times["turn-home"]/env.step_dt)
            self.release_allowed = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.turn_allowed = torch.zeros_like(self.release_allowed)
            self.release_timer = torch.zeros(self.num_envs,device=self.device)
            self.turn_timer = torch.zeros_like(self.release_timer)
        if self.skill == "carry":
            body = self.plan.sk["body"][0, :int(self.plan.sk["len"][0]), self.anchor_j]
            box = self.plan.sk["box"][0, :len(body), :3]
            self.carry_target = tk._yaw_frame(box-body[:,:3], tk.yaw_of(body[:,3:])).mean(0)

    def _resample_command(self, env_ids):
        if self.skill == "navigation":
            # Equal environment allocations keep both routes represented, even
            # when one route has much shorter episodes or fails more often.
            self.plan_ids[env_ids] = env_ids % 2
        super()._resample_command(env_ids)
        n = len(env_ids)
        b = self.world(0, "box", ids=env_ids)
        vel = self.world(0, "box_lin", ids=env_ids)
        wz = self.raw(0, "box_wz", ids=env_ids)
        z = torch.zeros(n, 2, device=self.device)
        state = torch.cat([b[:,:3], tk.yaw_quat(b[:,3]), vel, z, wz[:,None]], -1)
        self.boxe.write_root_state_to_sim(state, env_ids=env_ids)
        origins = self._env.scene.env_origins[env_ids]
        for name, pose in (("cart", self.setup["source_table_pose"]), ("shelf", self.setup["destination_table_pose"])):
            pose = torch.tensor(pose, device=self.device).expand(n,3)
            xyz = origins.clone()
            xyz[:,:2] += pose[:,:2]
            self._env.scene[name].write_mocap_pose_to_sim(torch.cat([xyz,tk.yaw_quat(pose[:,2])],-1), env_ids)
        self.phase[env_ids] = self.raw(0, "grip", ids=env_ids).long()
        self.hold_required[env_ids] = (1. + torch.rand(n, device=self.device)) if self.cfg.random_hold else 1.
        if self.skill == "put-down":
            # RSI starts beyond a gate already represent its completed posture.
            self.release_allowed[env_ids] = self.clock[env_ids,0] > self.release_frame
            self.turn_allowed[env_ids] = self.clock[env_ids,0] > self.turn_frame
            self.release_timer[env_ids] = 0.
            self.turn_timer[env_ids] = 0.
        tk._invalidate(self._env)

    def released(self):
        g = tk.geo(self._env)
        supported = self._env.scene["box_shelf"].data.found.reshape(self.num_envs,-1).sum(-1)>0
        return (supported & (g.box_v.norm(dim=-1)<.1) & (g.tilt<.15)
                & (g.touch.max(-1).values==0) & (g.gap.min(-1).values>.05))

    def arms_lowered(self):
        target = self.plan.sk["qpos"][self.plan_ids,self.turn_frame,7:]
        error = self.robot.data.joint_pos[:,self.arm_ids] - target[:,self.arm_ids]
        palms = self.robot.data.site_pos_w[:,self.palm_ids]
        low = (palms[:,:,2] < self._env.scene.env_origins[:,2,None] + self.setup["destination_top_m"]-.15).all(-1)
        return (error.abs().max(-1).values<.3) & low

    def standing(self):
        ok, errors = super().standing()
        env = self._env
        if self.skill in ("take", "carry"):
            _, actual_quat = tk._anchor(env)
            ref_quat = self.plan.sk["body"][self.plan_ids, self.lengths()-1, self.anchor_j, 3:]
            distance, heading, _, _ = errors
            ok = (distance < motion.AT[0]) & (heading.abs() < motion.AT[1])
            ok &= posture_ok(actual_quat, ref_quat, loaded=True)
        g = tk.geo(env)
        clean = tk.r_prop_contact(env) == 0
        foot = env.scene["feet_ground_contact"].data.found.reshape(self.num_envs, -1)
        ok = ok & clean & (foot > 0).all(-1)
        if self.skill in ("take", "carry"):
            ok &= g.holding() & (g.tilt < .20)
            ok &= (g.box - self.R("box")[:,:3]).norm(dim=-1) < .08
        if self.skill in ("put-down", "return", "navigation"):
            supported = env.scene["box_shelf"].data.found.reshape(self.num_envs,-1).sum(-1) > 0
            released = supported & (g.tilt < .15) & (g.box_v.norm(dim=-1) < .1)
            released &= (g.box - self.R("box")[:,:3]).norm(dim=-1) < .08
            released &= (g.touch.max(-1).values == 0) & (g.gap.min(-1).values > .05)
            if self.skill == "navigation":
                ok &= torch.where(self.plan_ids == 1, released, g.touch.max(-1).values == 0)
            else:
                ok &= released
        elif self.skill == "go-to":
            ok &= g.touch.max(-1).values == 0
        if self.skill in ("navigation", "go-to", "return", "put-down"):
            error = self.robot.data.joint_pos[:, self.arm_ids] - self.pose_joints_end()[:, self.arm_ids]
            ok &= error.abs().max(-1).values < .3
        return ok, errors

    def pose_joints_end(self):
        return self.plan.sk["qpos"][self.plan_ids, self.lengths()-1, 7:]

    def _update_command(self, env_ids):
        if env_ids is not None:
            return
        tk._invalidate(self._env)
        next_clock = self.clock[:,0]+(~self.fresh).long()
        if self.skill == "put-down":
            clean = (tk.r_prop_contact(self._env)==0) & (robot_box_contact(self._env)==0)
            released = self.released() & clean
            ready_release = released & (self.clock[:,0]>=self.release_frame)
            ready_turn = released & self.arms_lowered() & (self.clock[:,0]>=self.turn_frame)
            self.release_timer = torch.where(ready_release,self.release_timer+self._env.step_dt,0.)
            self.turn_timer = torch.where(ready_turn,self.turn_timer+self._env.step_dt,0.)
            self.release_allowed |= self.release_timer >= .2
            self.turn_allowed |= self.turn_timer >= .4
            next_clock = torch.where(self.release_allowed,next_clock,next_clock.clamp(max=self.release_frame))
            next_clock = torch.where(self.turn_allowed,next_clock,next_clock.clamp(max=self.turn_frame))
        self.clock[:,0] = torch.minimum(next_clock, self.lengths()-1)
        self.fresh[:] = False
        tk._invalidate(self._env)
        self.phase[:] = self.R("grip").long()
        ok, _ = self.standing()
        ok &= self.ended()
        self.stood = torch.where(ok, self.stood+self._env.step_dt, torch.zeros_like(self.stood))
        self.success = self.stood >= self.hold_required
        self.metrics["success"] = torch.maximum(self.metrics["success"], self.success.float())
        self.metrics["grip"][:] = tk.geo(self._env).grip
        self.metrics["box_error"][:] = (tk.geo(self._env).box - self.R("box")[:,:3]).norm(dim=-1)


@dataclass(kw_only=True)
class FiveCommandCfg(motion.MotionCommandCfg):
    skill: str
    random_hold: bool = True

    def build(self, env):
        return FiveCommand(self, env)


def obs_layout(env):
    """Table front centers, heights and final pose in the robot's yaw frame."""
    t = tk.task(env)
    r = t.robot.data
    yaw = tk.yaw_of(r.root_link_quat_w)
    origin = env.scene.env_origins
    pieces = []
    for x,y,z in ((*t.setup["source_front"][:2],t.setup["source_top_m"]),
                  (*t.setup["destination_front"][:2],t.setup["destination_top_m"])):
        world = origin + torch.tensor([x,y,z], device=env.device)
        pieces.append(tk._yaw_frame(world-r.root_link_pos_w, yaw))
    end = t.pose(-1)
    heading = tk.wrap(tk.yaw_of(end[:,3:7])-yaw)
    pieces += [tk._yaw_frame(end[:,:3]-r.root_link_pos_w,yaw), torch.sin(heading)[:,None],torch.cos(heading)[:,None]]
    return torch.cat(pieces,-1)


def r_grip_force(env):
    # Penalize excessive squeeze; support contacts remain fully physical.
    force = env.scene["hand_box"].data.force.reshape(env.num_envs,-1,3).norm(dim=-1)
    return torch.clamp(force-40., min=0.).square().mean(-1)


def r_arms_down(env):
    t = tk.task(env)
    error = t.robot.data.joint_pos[:,t.arm_ids] - t.R("qpos",1)[:,7:][:,t.arm_ids]
    return torch.exp(-error.square().mean(-1)/.2**2)


def r_carry_relative_pose(env):
    t = tk.task(env)
    pos, quat = tk._anchor(env)
    relative = tk._yaw_frame(tk.geo(env).box-pos, tk.yaw_of(quat))
    return torch.exp(-(relative-t.carry_target).square().sum(-1)/.06**2)


def r_carry_relative_velocity(env):
    t = tk.task(env)
    r = t.robot.data
    anchor = t.body_ids[t.anchor_j]
    offset = tk.geo(env).box - r.body_link_pos_w[:,anchor]
    moving_frame_v = r.body_link_lin_vel_w[:,anchor] + torch.cross(
        r.body_link_ang_vel_w[:,anchor], offset, dim=-1)
    return torch.exp(-(tk.geo(env).box_v-moving_frame_v).square().sum(-1)/.2**2)


def robot_box_contact(env):
    return env.scene["robot_box"].data.found.reshape(env.num_envs,-1).sum(-1)


def r_putdown_heading(env):
    t = tk.task(env)
    error = tk.wrap(tk.yaw_of(t.R("qpos",1)[:,3:7])-tk.yaw_of(t.robot.data.root_link_quat_w))
    return torch.exp(-error.square()/.25**2)


def r_putdown_arms(env):
    t = tk.task(env)
    return r_arms_down(env)*(t.clock[:,0]>=t.lower_frame).float()


def t_putdown_heading(env):
    t = tk.task(env)
    error = tk.wrap(tk.yaw_of(t.R("qpos",1)[:,3:7])-tk.yaw_of(t.robot.data.root_link_quat_w))
    return error.abs() > .6


def t_postrelease_collision(env):
    t = tk.task(env)
    contact = (robot_box_contact(env)>0) | (tk.r_prop_contact(env)>0)
    return (t.clock[:,0]>=t.release_frame) & contact


def putdown_posture(env):
    t = tk.task(env)
    _, quat = tk._anchor(env)
    height = t.robot.data.root_link_pos_w[:,2] - env.scene.env_origins[:,2]
    return height, quat


def t_putdown_posture(env):
    t = tk.task(env)
    height, quat = putdown_posture(env)
    return (t.clock[:,0]>=t.upright_frame) & ~upright_putdown_ok(height,quat)


def r_putdown_posture(env):
    t = tk.task(env)
    height, quat = putdown_posture(env)
    cos_tilt = (1.-2.*quat[:,1:3].square().sum(-1)).clamp(-1.,1.)
    score = torch.exp(-(height-.76).square()/.025**2 - (1.-cos_tilt)/.0025)
    return score*(t.clock[:,0]>=t.upright_frame).float()


def env_cfg(skill, num_envs=4096, play=False):
    if skill not in (*SKILLS, *POLICIES):
        raise ValueError(skill)
    folder = str(OUT/skill)
    bank = torch.load(motion.plan_path(folder), map_location="cpu", weights_only=False)
    if bank.get("config_sha256") != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError("Reference motion is stale after setup changes. Regenerate and validate it before training.")
    if not bank.get("stats", {}).get("passed", False):
        raise ValueError("The reference motion did not pass its geometry checks.")
    configure_dimensions()
    cfg = motion.env_cfg(folder, num_envs, play)
    cfg.scene = geometry_cfg("five-stage")
    cfg.scene.num_envs = num_envs
    cfg.scene.sensors = (
        ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",entity="robot"),
            secondary=ContactMatch(mode="body",pattern="terrain"),fields=("found",),reduce="netforce",num_slots=1),
        ContactSensorCfg(name="hand_box",primary=ContactMatch(mode="subtree",
            pattern=r"^(left_wrist_roll_rubber_hand|right_wrist_roll_rubber_hand)$",entity="robot"),
            secondary=ContactMatch(mode="body",pattern="box",entity="box"),fields=("found","force"),reduce="netforce",num_slots=1),
        *[ContactSensorCfg(name=f"box_{p}",primary=ContactMatch(mode="body",pattern="box",entity="box"),
            secondary=ContactMatch(mode="body",pattern=p,entity=p),fields=("found",),reduce="netforce",num_slots=1)
            for p in ("cart","shelf")],
        *[ContactSensorCfg(name=f"robot_{p}",primary=ContactMatch(mode="body",pattern=p,entity=p),
            secondary=ContactMatch(mode="subtree",pattern="pelvis",entity="robot"),fields=("found",),reduce="netforce",num_slots=1)
            for p in ("cart","shelf")])
    cfg.commands["task"] = FiveCommandCfg(plan=motion.plan_path(folder), skill=skill,
                                          rsi_noise=False, random_hold=not play)
    if skill == "put-down":
        cfg.scene.sensors += (ContactSensorCfg(name="robot_box",
            primary=ContactMatch(mode="subtree",pattern="pelvis",entity="robot"),
            secondary=ContactMatch(mode="body",pattern="box",entity="box"),
            fields=("found",),reduce="netforce",num_slots=1),)
        cfg.rewards["putdown_posture"] = RewardTermCfg(func=r_putdown_posture,weight=2.)
        cfg.terminations["postrelease_posture"] = TerminationTermCfg(func=t_putdown_posture)
        cfg.rewards["putdown_heading"] = RewardTermCfg(func=r_putdown_heading,weight=2.)
        cfg.rewards["putdown_arms"] = RewardTermCfg(func=r_putdown_arms,weight=1.)
        cfg.terminations["heading"] = TerminationTermCfg(func=t_putdown_heading)
        cfg.terminations["postrelease_collision"] = TerminationTermCfg(func=t_postrelease_collision)
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["layout"] = ObservationTermCfg(func=obs_layout)
        if skill in ("take", "carry", "put-down"):
            terms["box"] = ObservationTermCfg(func=tk.obs_box)
            terms["palms"] = ObservationTermCfg(func=tk.obs_palms)
            terms["ref_box"] = ObservationTermCfg(func=tk.obs_ref_box,params={"skill":0})
    cfg.rewards["prop_contact"] = RewardTermCfg(func=tk.r_prop_contact,weight=-2.)
    if skill in ("navigation", "go-to", "return"):
        cfg.rewards["arms_down"] = RewardTermCfg(func=r_arms_down, weight=.5)
    if skill == "carry":
        cfg.rewards["carry_relative_pose"] = RewardTermCfg(func=r_carry_relative_pose, weight=1.)
        cfg.rewards["carry_relative_velocity"] = RewardTermCfg(func=r_carry_relative_velocity, weight=.25)
    if skill in ("take", "carry", "put-down"):
        cfg.rewards["box"] = RewardTermCfg(func=tk.r_box, weight=2., params={"std":.08})
        cfg.rewards["grip"] = RewardTermCfg(func=tk.r_grip, weight=.5)
        cfg.rewards["squeeze_force"] = RewardTermCfg(func=r_grip_force,weight=-.0001)
        cfg.terminations["box_lost"] = TerminationTermCfg(func=tk.t_box_lost)
    if not play:
        cfg.events["box_mass"] = EventTermCfg(mode="startup",func=dr.pseudo_inertia,
            params={"asset_cfg":SceneEntityCfg("box"),"alpha_range":(.5*math.log(.66),.5*math.log(1.34))})
    cfg.episode_length_s += 3.
    cfg.sim.mujoco.iterations = 20
    return cfg
