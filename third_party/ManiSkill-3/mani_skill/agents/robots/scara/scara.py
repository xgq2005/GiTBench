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


@register_agent()
class Scara(BaseAgent):
    uid = "scara"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/scara/scara.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper_pad=dict(static_friction=1.5, dynamic_friction=1.0, restitution=0.1)
        ),
        link=dict(
            virtual_finger1=dict(
                material="gripper_pad", patch_radius=0.01, min_patch_radius=0.005
            ),
            virtual_finger2=dict(
                material="gripper_pad", patch_radius=0.01, min_patch_radius=0.005
            ),
            virtual_finger3=dict(
                material="gripper_pad", patch_radius=0.01, min_patch_radius=0.005
            ),
            virtual_finger4=dict(
                material="gripper_pad", patch_radius=0.01, min_patch_radius=0.005
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0,  # joint1 - 旋转关节
                0.02,  # joint2 - 棱柱关节 (默认在中间位置，范围0-0.091)
                0.0,  # joint3 - 旋转关节 (范围-2.8到0)
            ]),
            pose=sapien.Pose(),
        )
    )

    # 机械臂关节命名
    arm_joint_names = [
        "joint1",  # 旋转关节
        "joint2",  # 棱柱关节
        "joint3",  # 旋转关节
    ]

    # 夹爪关节命名
    gripper_joint_names = [
        "virtual_finger1_joint",
    ]

    # 末端执行器链接
    ee_link_name = "scara_tcp"

    # 机械臂控制器参数
    arm_stiffness = 3e2
    arm_damping = 1e2
    arm_force_limit = 100

    # 夹爪控制器参数
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Gripper - 夹爪控制器配置
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Arm - 机械臂控制器配置
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # -------------------------------------------------------------------------- #
        # Arm - 末端执行器控制器配置
        # -------------------------------------------------------------------------- #
        # 末端位置增量控制
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        # 末端位姿增量控制（SCARA通常只控制位置，但保留接口）
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

        # 末端绝对位姿控制
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.5,
            pos_upper=0.5,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Arm - 其他控制器
        # -------------------------------------------------------------------------- #
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
        )

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
        # 控制器配置字典
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        # 获取夹爪链接
        self.left_finger = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "virtual_finger1"
        )
        self.right_finger = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "virtual_finger2"
        )

        # 获取TCP链接
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5):
        """检查夹爪是否在抓取物体

        Args:
            object: 要检查的物体
            min_force: 最小接触力阈值（牛顿）
        """
        # 获取左右手指与物体的接触力
        left_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_finger, object
        )
        right_contact_forces = self.scene.get_pairwise_contact_forces(
            self.right_finger, object
        )

        # 计算总接触力大小
        left_force = torch.linalg.norm(left_contact_forces, axis=1)
        right_force = torch.linalg.norm(right_contact_forces, axis=1)

        # 判断是否在抓取（两个手指都有足够的接触力）
        is_grasping = torch.logical_and(
            left_force >= min_force,
            right_force >= min_force
        )

        return is_grasping

    def is_static(self, threshold: float = 0.2):
        """检查机械臂是否处于静止状态"""
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        """获取TCP位置"""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """获取TCP位姿"""
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, grasp_center):
        """构建夹爪抓取位姿

        Args:
            approaching: 接近方向（夹爪朝向）
            grasp_center: 抓取中心位置
        """
        # 确保接近方向是单位向量
        approaching = approaching / np.linalg.norm(approaching)

        # 创建正交坐标系
        # 选择与接近方向不共线的参考向量
        if np.abs(approaching[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])

        # 计算其他两个轴
        side = np.cross(approaching, ref)
        side = side / np.linalg.norm(side)
        up = np.cross(side, approaching)

        # 构建变换矩阵
        T = np.eye(4)
        T[:3, 0] = side
        T[:3, 1] = up
        T[:3, 2] = approaching
        T[:3, 3] = grasp_center

        return sapien.Pose(T)

    def is_empty(self, force_threshold=0.3):
        """检查夹爪是否为空（没有抓取任何物体）

        对于夹爪，我们通过检查左右手指的接触力来判断是否在抓取物体。
        如果两个手指都没有足够的接触力，则认为夹爪为空。

        Args:
            force_threshold (float, optional): 最小接触力阈值（单位：牛顿）。
                默认为0.3N。

        Returns:
            torch.Tensor: 布尔张量，表示每个环境的夹爪是否为空。
        """
        # 获取所有与手指接触的物体
        contacts = self.scene.get_contacts()
        
        # 检查是否有与左右手指的接触
        left_contacts = []
        right_contacts = []
        for contact in contacts:
            actors = contact.actors
            if self.left_finger_pad in actors:
                left_contacts.append(contact)
            if self.right_finger_pad in actors:
                right_contacts.append(contact)
        
        # 如果没有接触，夹爪为空
        if len(left_contacts) == 0 or len(right_contacts) == 0:
            batch_size = self.tcp_pos.shape[0]
            return torch.ones(batch_size, dtype=torch.bool, device=self.tcp_pos.device)
        
        # 计算左右手指的总接触力
        left_force = torch.zeros(self.tcp_pos.shape[0], device=self.tcp_pos.device)
        right_force = torch.zeros(self.tcp_pos.shape[0], device=self.tcp_pos.device)
        
        for contact in left_contacts:
            impulse = contact.impulse
            force = torch.linalg.norm(impulse, axis=1)
            left_force = torch.maximum(left_force, force)
        
        for contact in right_contacts:
            impulse = contact.impulse
            force = torch.linalg.norm(impulse, axis=1)
            right_force = torch.maximum(right_force, force)
        
        # 如果任一手指的接触力小于阈值，认为夹爪为空
        is_empty = torch.logical_or(
            left_force < force_threshold,
            right_force < force_threshold
        )
        
        return is_empty