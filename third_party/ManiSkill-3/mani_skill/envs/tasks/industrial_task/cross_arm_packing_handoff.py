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
from mani_skill.envs.tasks.industrial_task.parts import create_part_by_type


@register_env("CrossArmPackingHandoff", max_episode_steps=50)
class PackingOrderRetrievalEnv(BaseEnv):
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
                 part_types=None, num_parts=3, **kwargs):
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

        # 最后放入的零件相关属性
        self.last_placed_cube = None
        self.last_placed_cube_name = None
        self.cubes = None
        self.swap_indices = None
        self.num_parts = min(max(num_parts, 3), 5)
        if part_types is None:
            part_types = ["cube"] * self.num_parts
        self.part_types = part_types[:self.num_parts]
        while len(self.part_types) < self.num_parts:
            self.part_types.append("cube")

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
                # Keep the task space identical to PiperX's absolute
                # pd_joint_pos controller space; do not normalize arm joints.
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

        part_colors = [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
        ]
        self.cubes = []
        for i in range(self.num_parts):
            cube = create_part_by_type(
                self.scene, part_type=self.part_types[i], name=f"cube{i+1}",
                color=part_colors[i % len(part_colors)],
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
            self.cubes.append(cube)
            setattr(self, f"cube{i+1}", cube)

        # 放置隔离盒（无盖盒子）
        self.iso_box = actors.build_open_top_box(
            self.scene,
            half_sizes=(0.15, 0.15, 0.025),
            wall_thickness=0.005,
            color=[0.82, 0.71, 0.55, 1],
            name="iso_box",
            body_type="static",
            initial_pose=sapien.Pose(p=[-0.23, 0, 0.025]),
        )

        # 添加蓝色无盖盒子
        self.blue_box = actors.build_open_top_box(
            self.scene,
            half_sizes=(0.2, 0.15, 0.025),
            wall_thickness=0.005,
            color=[0.1, 0.1, 0.8, 1],
            name="blue_box",
            body_type="static",
            initial_pose=sapien.Pose(p=[-0.2, 0.3, 0.025]),
        )

        yellow_pad_pos = [-0.5, 0, 0.003]
        self.yellow_pad = actors.build_box(
            self.scene,
            half_sizes=(0.07, 0.15, 0.003),
            color=[1, 1, 0, 1],
            name="yellow_pad",
            body_type="static",
            initial_pose=sapien.Pose(p=yellow_pad_pos),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每轮任务"""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 设置物体初始位置
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size  # 在桌子上

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            
            # 初始化隔离盒位置（添加小范围位置噪声）
            iso_box_pos_x = -0.2 + np.random.uniform(-0.02, 0.02)
            iso_box_pos_y = -0.03 + np.random.uniform(-0.02, 0.02)
            iso_box_pos_z = 0.025
            self.iso_box.set_pose(sapien.Pose(p=[iso_box_pos_x, iso_box_pos_y, iso_box_pos_z]))
            
            original_positions = [
                [-0.1, 0.2],
                [-0.2, 0.2],
                [-0.3, 0.2],
                [-0.1, 0.3],
                [-0.2, 0.3],
            ]

            cube_positions = []
            for i in range(self.num_parts):
                pos_x = original_positions[i][0] + np.random.uniform(-0.02, 0.02)
                pos_y = original_positions[i][1] + np.random.uniform(-0.02, 0.02)
                pos_z = self.cube_half_size
                cube_positions.append([pos_x, pos_y, pos_z])

            for i in range(self.num_parts):
                self.cubes[i].set_pose(sapien.Pose(p=cube_positions[i]))

            self.last_placed_cube = self.cubes[-1]
            self.last_placed_cube_name = self.cubes[-1].name
            print(f"最后放入的零件: {self.last_placed_cube_name}")

            self.swap_indices = np.random.choice(self.num_parts, 2, replace=False)
            print(f"要交换的零件: {self.cubes[self.swap_indices[0]].name} 和 {self.cubes[self.swap_indices[1]].name}")

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
        """评估任务完成情况：最后放入的零件是否不在盒子中"""
        # 盒子尺寸：x方向0.2，y方向0.3
        box_half_size_x = 0.2 / 2
        box_half_size_y = 0.3 / 2
        
        # 获取盒子中心位置和最后放入的零件位置
        box_center = self.iso_box.pose.sp.p
        last_placed_cube_pos = self.last_placed_cube.pose.sp.p
        
        # 确保是 tensor 类型
        if not torch.is_tensor(box_center):
            box_center = torch.tensor(box_center, device=self.device)
        if not torch.is_tensor(last_placed_cube_pos):
            last_placed_cube_pos = torch.tensor(last_placed_cube_pos, device=self.device)
        
        # 计算最后放入的零件与盒子中心的距离
        dx = torch.abs(last_placed_cube_pos[..., 0] - box_center[..., 0])
        dy = torch.abs(last_placed_cube_pos[..., 1] - box_center[..., 1])
        
        # 检查是否在盒子范围内
        last_placed_cube_in_box = (dx <= box_half_size_x) & (dy <= box_half_size_y)
        
        # 成功条件：最后放入的零件不在盒子中
        success = ~last_placed_cube_in_box
        
        return {
            "success": success,
            "last_placed_cube_name": self.last_placed_cube_name,
            "last_placed_cube_in_box": last_placed_cube_in_box,
            "dx": dx,
            "dy": dy,
            "box_center": box_center,
            "last_placed_cube_pos": last_placed_cube_pos,
        }

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
