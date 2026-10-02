# my_pick_cube.py
from typing import Any, Union, Dict, Optional
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.piper_x import PiperX
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from gymnasium.spaces import Box, Dict as DictSpace
import gymnasium as gym
from mani_skill.envs.tasks.industrial_task.experiment_task_cfg import INDUSTRIAL_TASK_CONFIGS
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Actor

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

@register_env("PlugRestoration", max_episode_steps=50)
class PlugRestorationEnv(BaseEnv):
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
    
    _base_size = [2e-2, 1.5e-2, 1.2e-2]
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]
    _peg_gap = 7e-3
    _clearance = 1e-3
    _receptacle_size = [1e-2, 6e-2, 6e-2]
    _rack_outer_hs = [2.5e-2, 2.5e-2, 1e-2]
    _rack_hole_hs = [9e-3, 1.6e-2]

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
        self.human_cam_eye_pos = [-0.7,0,0.6]
        self.human_cam_target_pos = [-0.1,0,0]
        self.base_camera_width =  cfg["base_camera_width"]
        self.base_camera_height = cfg["base_camera_height"]
        self.sensor_fovx = cfg["sensor_fovx"]
        self.sensor_fovy = cfg["sensor_fovy"]

        # 检查是否使用双臂配置
        self.is_bimanual = isinstance(robot_uids, tuple) and len(robot_uids) == 2
        if self.is_bimanual:
            self.cube_spawn_center = (0, 0.5)  # 物体生成中心
            print(f"初始化双臂系统: 使用 {robot_uids}")

        # 被取下的插头相关属性
        self.removed_plug_idx = None
        self.removed_plug = None
        self.removed_plug_name = None

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

    def _build_charger(self, peg_size, base_size, gap, idx):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name=f"charger_{idx}")

    def _build_receptacle(self, peg_size, receptacle_size, gap, idx):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.01], q=[0.7071068, 0, 0.7071068, 0])
        return builder.build_kinematic(name=f"receptacle_{idx}")

    def _build_plug_rack(self, idx):
        builder = self.scene.create_actor_builder()
        outer_hs = self._rack_outer_hs
        hole_hs = self._rack_hole_hs
        thickness_x = outer_hs[0] - hole_hs[0]
        thickness_y = outer_hs[1] - hole_hs[1]
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#8B7355"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        cx = hole_hs[0] + thickness_x / 2
        cy = hole_hs[1] + thickness_y / 2
        walls = [
            (sapien.Pose([cx, 0, 0]), [thickness_x / 2, outer_hs[1], outer_hs[2]]),
            (sapien.Pose([-cx, 0, 0]), [thickness_x / 2, outer_hs[1], outer_hs[2]]),
            (sapien.Pose([0, cy, 0]), [hole_hs[0], thickness_y / 2, outer_hs[2]]),
            (sapien.Pose([0, -cy, 0]), [hole_hs[0], thickness_y / 2, outer_hs[2]]),
        ]
        for pose, half_size in walls:
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, outer_hs[2]])
        return builder.build_kinematic(name=f"plug_rack_{idx}")

    def _load_scene(self, options: dict):
        """加载场景"""
        # 创建桌子场景
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.num_sockets = 6
        
        # 创建num_sockets个插头
        self.pegs = []
        for i in range(self.num_sockets):
            charger = self._build_charger(
                self._peg_size,
                self._base_size,
                self._peg_gap,
                i,
            )
            self.pegs.append(charger)

        # 创建num_sockets个插座
        self.boxes = []
        for i in range(self.num_sockets):
            receptacle = self._build_receptacle(
                [
                    self._peg_size[0],
                    self._peg_size[1] + self._clearance,
                    self._peg_size[2] + self._clearance,
                ],
                self._receptacle_size,
                self._peg_gap,
                i,
            )
            self.boxes.append(receptacle)

        self.plug_racks = []
        for i in range(self.num_sockets):
            rack = self._build_plug_rack(i)
            self.plug_racks.append(rack)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每轮任务"""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 随机选择一个插座插着插头（这个插头将被实验员取下）
            self.removed_plug_idx = np.random.randint(0, self.num_sockets)
            self.removed_plug = self.pegs[self.removed_plug_idx]
            self.removed_plug_name = self.removed_plug.name
            print(f"被取下的插头: {self.removed_plug_name} (索引: {self.removed_plug_idx})")
            
            # 生成插座位置（沿y轴均匀分布）
            y_spacing = 0.2
            y_start = (self.num_sockets - 1) * y_spacing / 2
            
            for i in range(self.num_sockets):
                y_pos = y_start - i * y_spacing
                box_pos = torch.zeros((b, 3))
                box_pos[:, 0] = -0.3
                box_pos[:, 1] = y_pos
                box_pos[:, 2] = 0.01

                # 插座板平放在桌面上，旋转+90°绕y轴使孔朝上
                box_quat = torch.tensor([0.7071068, 0, 0.7071068, 0], device=self.device).unsqueeze(0).expand(b, -1)
                box_pose = Pose.create_from_pq(box_pos, box_quat)
                self.boxes[i].set_pose(box_pose)

            for i in range(self.num_sockets):
                y_pos = y_start - i * y_spacing
                rack_pos = torch.zeros((b, 3))
                rack_pos[:, 0] = -0.1
                rack_pos[:, 1] = y_pos
                rack_pos[:, 2] = self._rack_outer_hs[2]
                self.plug_racks[i].set_pose(Pose.create_from_pq(rack_pos))

            # 设置插头的位置
            for i in range(self.num_sockets):
                y_pos = y_start - i * y_spacing
                peg_pos = torch.zeros((b, 3))
                
                if i == self.removed_plug_idx:
                    # 这个插头已经插入插座中（插座平放，插头从上方插入）
                    peg_pos[:, 0] = -0.3
                    peg_pos[:, 1] = y_pos
                    peg_pos[:, 2] = 0.02
                    # 插头旋转+90°绕y轴，使插脚朝下插入插座（local +X -> world -Z）
                    quat = torch.tensor([0.7071068, 0, 0.7071068, 0], device=self.device).unsqueeze(0).expand(b, -1)
                else:
                    # 其他插头竖直插入plug_rack（插脚朝下）
                    peg_pos[:, 0] = -0.1
                    peg_pos[:, 1] = y_pos
                    peg_pos[:, 2] = self._rack_outer_hs[2] * 2
                    quat = torch.tensor([0.7071068, 0, 0.7071068, 0], device=self.device).unsqueeze(0).expand(b, -1)
                
                peg_pose = Pose.create_from_pq(peg_pos, quat)
                self.pegs[i].set_pose(peg_pose)

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
        """评估任务完成情况：被取下的插头是否插回原来的插座"""
        removed_plug_pose = self.removed_plug.pose.sp
        target_socket = self.boxes[self.removed_plug_idx]
        socket_pose = target_socket.pose.sp

        # 将插头位置变换到插座local坐标系
        # 插座旋转+90°绕y轴：local -X = world +Z（孔朝上），local +X = world -Z
        plug_in_socket_local = socket_pose.inv() * removed_plug_pose
        local_p = plug_in_socket_local.p
        if hasattr(local_p, 'cpu'):
            local_p = local_p.cpu().numpy()
        local_p = np.asarray(local_p).flatten()[:3]

        # local X方向：负值=在孔上方（world +Z），正值=在孔下方
        # 插头应插入孔内：local X在[-peg_size[0], 0]范围
        # local YZ：应在孔的范围内
        xy_dist = np.sqrt(local_p[1]**2 + local_p[2]**2)
        insert_depth = -local_p[0]

        # 判断条件：XY对准 + 插入深度足够
        xy_aligned = xy_dist < 0.015
        depth_ok = insert_depth > 0.002
        plug_in_socket = xy_aligned and depth_ok

        distance = np.linalg.norm(local_p)

        return {
            "success": torch.tensor(plug_in_socket, device=self.device),
            "removed_plug_name": self.removed_plug_name,
            "plug_in_socket": torch.tensor(plug_in_socket, device=self.device),
            "distance": torch.tensor(distance, device=self.device),
            "xy_dist": torch.tensor(xy_dist, device=self.device),
            "insert_depth": torch.tensor(insert_depth, device=self.device),
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
