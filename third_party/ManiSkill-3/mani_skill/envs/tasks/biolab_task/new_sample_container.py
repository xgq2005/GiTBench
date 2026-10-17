# my_pick_cube.py
from typing import Any, Union, Dict, Optional
import numpy as np
import sapien
import torch
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.piper_x import PiperX
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from gymnasium.spaces import Box, Dict as DictSpace
import gymnasium as gym
from mani_skill.envs.tasks.industrial_task.experiment_task_cfg import INDUSTRIAL_TASK_CONFIGS

BEAKER_TABLE_Z = 0.065
SOURCE_INITIAL_POSITION = [-0.2, 0.5, 0.01]
SOURCE_INITIAL_QUATERNION = [0, 0, 0, 1]


@register_env("NewSampleContainer", max_episode_steps=50)
class NewSampleContainerEnv(BaseEnv):
    """
    双臂PiperX机器人抓取立方体环境

    扩展了标准PickCube环境，支持：
    1. 使用两个piper_x机器人构建的双臂系统
    2. 可选的货架障碍物
    """

    _sample_video_link = ""
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "piper_x",
        ("piper_x", "piper_x"),
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: PiperX
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids=("piper_x", "piper_x"), robot_init_qpos_noise=0.02,
                 **kwargs):
        """
        初始化双臂抓取环境

        Args:
            robot_uids: 机器人类型，支持单臂或双臂配置
                        单臂: "piper_x"
                        双臂: ("piper_x", "piper_x")
            robot_init_qpos_noise: 机器人初始关节位置噪声
        """
        self.robot_init_qpos_noise = robot_init_qpos_noise
        cfg = INDUSTRIAL_TASK_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.base_camera_width =  cfg["base_camera_width"]
        self.base_camera_height = cfg["base_camera_height"]
        self.sensor_fovx = cfg["sensor_fovx"]
        self.sensor_fovy = cfg["sensor_fovy"]
        # 检查是否使用双臂配置
        self.is_bimanual = isinstance(robot_uids, tuple) and len(robot_uids) == 2
        if self.is_bimanual:
            self.cube_spawn_center = (0, 0.5)  # 物体生成中心
            print(f"初始化双臂系统: 使用 {robot_uids}")

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        # 如果是双臂，设置动作空间
        if self.is_bimanual:
            self._setup_bimanual_action_space()

    def _setup_bimanual_action_space(self):
        """设置双臂机器人的动作空间"""
        action_spaces = {}

        if hasattr(self.agent, 'agents'):
            for i, agent in enumerate(self.agent.agents):
                # 生成代理ID
                uid = f"{self.robot_uids[i]}-{i}" if isinstance(self.robot_uids, tuple) else f"{self.robot_uids}-{i}"
                # Reuse the active PiperX controller's absolute joint-position
                # space.  The task-specific [-1, 1] wrapper used to truncate
                # valid arm joint targets before they reached pd_joint_pos.
                action_spaces[uid] = agent.action_space

            # 创建字典动作空间
            self.action_space = DictSpace(action_spaces)
            print(f"设置双臂动作空间: {self.action_space}")

    @property
    def _default_sensor_configs(self):
        """传感器配置"""
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        fx = (self.base_camera_width / 2) / np.tan(self.sensor_fovx / 2)
        fy = (self.base_camera_height / 2) / np.tan(self.sensor_fovy / 2)
        cx = self.base_camera_width / 2
        cy = self.base_camera_height / 2
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return [CameraConfig("base_camera", pose, self.base_camera_width, self.base_camera_height, fov=None, intrinsic=intrinsic, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        """人眼渲染相机配置"""
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate=False):
        """
        加载机器人代理

        基于文档中的_load_agent方法，支持加载单臂或双臂机器人
        """
        if initial_agent_poses is None:
            cfg = INDUSTRIAL_TASK_CONFIGS["panda"]
            if self.is_bimanual:
                left_arm_pose = sapien.Pose(p=cfg["left_arm_offset"])
                right_arm_pose = sapien.Pose(p=cfg["right_arm_offset"])
                initial_agent_poses = [left_arm_pose, right_arm_pose]
            else:
                initial_agent_poses = sapien.Pose(p=cfg["single_arm_offset"])

        # 调用父类方法
        super()._load_agent(options, initial_agent_poses, build_separate)

        # 保存双臂引用
        if self.is_bimanual and hasattr(self.agent, 'agents') and len(self.agent.agents) == 2:
            self.left_arm = self.agent.agents[0]
            self.right_arm = self.agent.agents[1]

    def _load_scene(self, options: dict):
        """加载场景"""
        # 创建桌子场景
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # 烧杯 - 创建最大数量5个，在_initialize_episode中随机选择使用多少个
        self.max_num_beakers = 5
        self.beakers = []
        for i in range(self.max_num_beakers):
            builder = actors.get_actor_builder(
                self.scene,
                id="chemistry:016_beaker",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, BEAKER_TABLE_Z])
            beaker = builder.build(name=f"beaker{i+1}")
            self.beakers.append(beaker)

        builder = actors.get_actor_builder(
            self.scene,
            id="chemistry:020_wash_bottle",
        )
        builder.initial_pose = sapien.Pose(
            p=SOURCE_INITIAL_POSITION, q=SOURCE_INITIAL_QUATERNION
        )
        self.source = builder.build(name="source")

        # 目标容器从烧杯中随机选择 - 初始化为None，在_initialize_episode中随机选择
        self.target_beaker = None
        self.target_beaker_name = None
        self.target_beaker_idx = None

        self.default_yellow_pad_pos = [-0.22, -0.25, 0.003]
        self.yellow_pad = actors.build_box(
            self.scene,
            half_sizes=(0.175, 0.08, 0.003),
            color=[1, 1, 0, 1],
            name="yellow_pad",
            body_type="static",
            initial_pose=sapien.Pose(p=self.default_yellow_pad_pos),
        )

        self.default_red_pad_pos = [-0.22, 0.25, 0.003]
        self.red_pad = actors.build_box(
            self.scene,
            half_sizes=(0.175, 0.08, 0.003),
            color=[1, 0, 0, 1],
            name="red_pad",
            body_type="static",
            initial_pose=sapien.Pose(p=self.default_red_pad_pos),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每轮任务，从options接收配置参数"""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ``BaseEnv.reset`` clears velocities but does not restore poses
            # for dynamic actors. Reset the wash bottle explicitly so a
            # collision in one episode cannot leave it tipped over in the
            # next episode.
            self.source.set_pose(
                sapien.Pose(
                    p=SOURCE_INITIAL_POSITION,
                    q=SOURCE_INITIAL_QUATERNION,
                )
            )
            self.source.set_linear_velocity(torch.zeros(3, device=self.device))
            self.source.set_angular_velocity(torch.zeros(3, device=self.device))

            layout_config = options.get("layout_config", None)
            target_container = options.get("target_container", None)
            red_pad_pos = options.get("red_pad_pos", self.default_red_pad_pos)
            yellow_pad_pos = options.get("yellow_pad_pos", self.default_yellow_pad_pos)

            if red_pad_pos is not None:
                self.red_pad.set_pose(sapien.Pose(p=red_pad_pos))
            if yellow_pad_pos is not None:
                self.yellow_pad.set_pose(sapien.Pose(p=yellow_pad_pos))

            if layout_config is not None:
                containers = layout_config["containers"]
                positions = layout_config["positions"]
                self.num_beakers = len(containers)
                for i, cname in enumerate(containers):
                    pos = positions.get(cname)
                    if pos is not None:
                        self.beakers[i].set_pose(sapien.Pose(p=pos, q=[1, 0, 0, 0]))
                for i in range(len(containers), self.max_num_beakers):
                    self.beakers[i].set_pose(sapien.Pose(p=[10, 10, 10], q=[1, 0, 0, 0]))
            else:
                self.num_beakers = 3
                y_positions = [-0.1, 0, 0.1]
                for i in range(self.num_beakers):
                    self.beakers[i].set_pose(sapien.Pose(p=[0, y_positions[i], BEAKER_TABLE_Z], q=[1, 0, 0, 0]))
                for i in range(self.num_beakers, self.max_num_beakers):
                    self.beakers[i].set_pose(sapien.Pose(p=[10, 10, 10], q=[1, 0, 0, 0]))

            if target_container is not None:
                target_env_name = f"beaker{target_container.replace('C', '')}"
                target_beaker = None
                for beaker in self.beakers[:self.num_beakers]:
                    if beaker.name == target_env_name:
                        target_beaker = beaker
                        break
                if target_beaker is None:
                    target_beaker = self.beakers[0]
                self.target_beaker = target_beaker
                self.target_beaker_name = target_beaker.name
                self.target_beaker_idx = self.beakers.index(target_beaker)
            else:
                self.target_beaker_idx = 0
                self.target_beaker = self.beakers[0]
                self.target_beaker_name = self.target_beaker.name

            self.target_beaker_initial_pos = self.target_beaker.pose.sp.p
            if hasattr(self.target_beaker_initial_pos, 'cpu'):
                self.target_beaker_initial_pos = self.target_beaker_initial_pos.cpu().numpy()
            self.target_beaker_initial_pos = self.target_beaker_initial_pos.tolist()

    def _get_obs_extra(self, info: dict):
        """获取额外观察信息，修复维度问题"""
        obs = {}
        def ensure_correct_dim(tensor, target_dim=2):
            """确保张量具有正确的维度"""
            if torch.is_tensor(tensor):
                current_dim = tensor.dim()
                if current_dim < target_dim:
                    # 添加缺失的维度
                    for _ in range(target_dim - current_dim):
                        tensor = tensor.unsqueeze(0)
                elif current_dim > target_dim:
                    # 压缩多余的维度
                    for _ in range(current_dim - target_dim):
                        tensor = tensor.squeeze(0)
            return tensor

        if self.is_bimanual:
            tcp_pose_left = self.left_arm.tcp_pose.raw_pose
            tcp_pose_right = self.right_arm.tcp_pose.raw_pose
            obs["tcp_pose"] = {}
            obs["tcp_pose"]["piper_x-0"] = tcp_pose_left
            obs["tcp_pose"]["piper_x-1"] = tcp_pose_right

        else:
            tcp_pose = self.agent.tcp_pose.raw_pose
            if torch.is_tensor(tcp_pose):
                tcp_pose = ensure_correct_dim(tcp_pose, 2)
            obs["tcp_pose"] = tcp_pose

        return obs

    def evaluate(self):
        """评估任务完成情况：目标容器是否被移动超过5cm"""
        info = {}
        
        # 获取目标容器的当前位置
        target_beaker_current_pos = self.target_beaker.pose.sp.p
        if hasattr(target_beaker_current_pos, 'cpu'):
            target_beaker_current_pos = target_beaker_current_pos.cpu().numpy()
        target_beaker_current_pos = target_beaker_current_pos.tolist()
        
        # 计算目标容器当前位置与初始位置的距离
        initial_pos = np.array(self.target_beaker_initial_pos)
        current_pos = np.array(target_beaker_current_pos)
        distance = np.linalg.norm(current_pos - initial_pos)
        
        # 判断是否成功：距离大于5cm
        success_bool = distance > 0.05
        
        # 转换为PyTorch张量
        success = torch.tensor(success_bool, dtype=torch.bool, device=self.device)
        
        info['success'] = success
        info['target_beaker_name'] = self.target_beaker_name
        info['distance'] = distance
        info['target_beaker_initial_pos'] = self.target_beaker_initial_pos
        info['target_beaker_current_pos'] = target_beaker_current_pos
        
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

        return 0

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: dict
    ):
        """计算归一化密集奖励"""
        reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return reward / 5

    def _step_action(self, action):
        """
        处理动作执行

        支持双臂机器人的字典动作格式
        """
        # 如果是双臂机器人且动作不是字典，进行转换
        if self.is_bimanual and not isinstance(action, dict):
            action = self._convert_action_to_dict(action)

        # 调用父类方法
        return super()._step_action(action)

    def _convert_action_to_dict(self, action):
        """将一维动作转换为字典格式"""
        action_dict = {}

        if hasattr(self.agent, 'agents'):
            # 转换为numpy数组
            if torch.is_tensor(action):
                action_np = action.cpu().numpy()
            else:
                action_np = np.asarray(action)

            action_dim = action_np.shape[-1] if hasattr(action_np, 'shape') else len(action_np)

            for i, agent in enumerate(self.agent.agents):
                uid = f"{self.robot_uids[i]}-{i}" if isinstance(self.robot_uids, tuple) else f"{self.robot_uids}-{i}"

                if action_dim == 14:  # 两个7维动作
                    start_idx = i * 7
                    end_idx = (i + 1) * 7
                    action_dict[uid] = action_np[..., start_idx:end_idx]
                elif action_dim == 7:  # 单个动作，复制
                    action_dict[uid] = action_np
                else:  # 默认零动作
                    action_dict[uid] = np.zeros(7, dtype=np.float32)

        return action_dict
