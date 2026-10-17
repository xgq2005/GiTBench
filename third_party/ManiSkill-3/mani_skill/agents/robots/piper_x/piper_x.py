from copy import deepcopy

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig

@register_agent()
class PiperX(BaseAgent):
    uid = "piper_x"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/piper_x/piper_x_description/piper_x_description_d435.urdf"

    # 根据URDF更新配置
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            Link7=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Link8=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    # 定义休息位置的关键帧
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0,  # joint1
                0.0,  # joint2 (约90度)
                0.0,  # joint3
                1.0,  # joint4
                0.0,  # joint5
                0.0,  # joint6
                0.0,  # joint7 (夹爪位置，关闭)
                0.0   # joint8 (夹爪位置，关闭)
            ]),
            pose=sapien.Pose(),
        )
    )

    # 定义关节名称
    arm_joint_names = [
        "joint1",  # 基座旋转
        "joint2",  # 肩部关节
        "joint3",  # 肘部关节
        "joint4",  # 手腕旋转
        "joint5",  # 手腕俯仰
        "joint6",  # 手腕偏转
    ]

    gripper_joint_names = [
        "joint7",  # 左夹爪
        "joint8",  # 右夹爪
    ]

    # 末端执行器链接名称 - 使用夹爪基座或创建一个虚拟TCP
    ee_link_name = "piper_x_hand_tcp"  # 夹爪基座

    # 控制器参数
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # 手臂控制器配置
        # -------------------------------------------------------------------------- #
        # 关节位置PD控制器
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        # 关节增量位置PD控制器
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # 目标增量位置控制器
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # 末端执行器位置PD控制器
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path, # type: ignore
        )

        # 末端执行器位姿PD控制器
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        # 绝对位姿控制器
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-2.0,
            pos_upper=2.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        # 目标增量控制器
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # 关节速度控制器
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
        )

        # 关节位置-速度控制器
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )

        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # 夹爪控制器配置
        # -------------------------------------------------------------------------- #
        # 夹爪是平行夹爪，joint7和joint8分别控制左右夹爪
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=np.array([0.0, -0.04]),  # joint7: [0, 0.05], joint8: [-0.05, 0]
            upper=np.array([0.04, 0.0]),  # 根据URDF的limit设置
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        # 或者使用对称夹爪控制器
        gripper_pd_joint_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            # URDF joint7 upper limit is 0.05 m. The recorder stores both
            # fingers' opening, so a fully open gripper is 0.10 m total.
            upper=0.05,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"joint8": {"joint": "joint7", "multiplier": -1.0}},  # joint8与joint7反向运动
        )

        # 控制器配置字典
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_mimic
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_mimic),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_mimic),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_mimic
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_mimic),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_mimic
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_mimic
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_mimic
            ),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_mimic),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_mimic
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_mimic
            ),
        )

        # 深拷贝配置防止修改
        from copy import deepcopy as deepcopy_dict
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """初始化后设置夹爪和末端执行器链接的引用"""
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Link7"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Link8"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """检查机器人是否抓取物体"""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # 夹爪打开方向
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """检查机器人是否静止"""
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        """获取末端执行器位置"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """获取末端执行器位姿"""
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """构建抓取位姿"""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    # 使用d435相机
    @property
    def _sensor_configs(self):
        fovx = np.deg2rad(69.4)
        fovy = np.deg2rad(42.5)
        width = 640
        height = 480
        fx = (width / 2) / np.tan(fovx / 2)
        fy = (height / 2) / np.tan(fovy / 2)
        cx = width / 2
        cy = height / 2
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=width,
                height=height,
                fov=None,
                intrinsic=intrinsic,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
