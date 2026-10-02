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
from mani_skill.envs.tasks.gitbench_paths import project_asset_path


@register_env("SkippedBin", max_episode_steps=50)
class SkippedBinEnv(BaseEnv):
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
        self.base_camera_height =  cfg["base_camera_height"]
        self.sensor_fovx = cfg["sensor_fovx"]
        self.sensor_fovy = cfg["sensor_fovy"]
        # 检查是否使用双臂配置
        self.is_bimanual = isinstance(robot_uids, tuple) and len(robot_uids) == 2
        if self.is_bimanual:
            self.cube_spawn_center = (0, 0.5)  # 物体生成中心
            print(f"初始化双臂系统: 使用 {robot_uids}")

        # 初始化跳过盒子和打开顺序相关属性
        self.skipped_box = None
        self.skipped_box_name = None
        self.open_order = []
        self.lid_joint_name = "lid_joint"  # 盖子关节名称

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
        return [CameraConfig("base_camera", pose, self.base_camera_width, self.base_camera_height, fov=None,
                             intrinsic=intrinsic, near=0.01, far=100)]

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
        self.box = []
        loader = self.scene.create_urdf_loader()
        urdf_path = project_asset_path("industry", "models", "box_with_lid.urdf")
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        builder = articulation_builders[0]
        
        for link_builder in builder.link_builders:
            if hasattr(link_builder, 'joint_record') and link_builder.joint_record is not None:
                if 'lid_joint' in link_builder.joint_record.name:
                    link_builder.joint_record.limits = [0, 2]

        self.max_num_boxes = 5
        for i in range(self.max_num_boxes):
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.box.append(builder.build(name=f"box{i+1}", fix_root_link=True))


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
            
            self.num_boxes = 3
            
            y_positions = [-0.3, -0.15, 0, 0.15, 0.3]
            
            for i in range(self.num_boxes):
                box = self.box[i]
                pos_x = -0.2
                pos_y = y_positions[i]
                pos_z = 0.05
                box.set_pose(sapien.Pose(p=[pos_x, pos_y, pos_z]))
            
            # 将多余的盒子移出场景（隐藏）
            for i in range(self.num_boxes, self.max_num_boxes):
                box = self.box[i]
                box.set_pose(sapien.Pose(p=[10, 10, 10], q=[1, 0, 0, 0]))  # 移到远处
            
            # 随机选择一个盒子作为被跳过的盒子
            self.skipped_idx = np.random.randint(0, self.num_boxes)
            self.skipped_box = self.box[self.skipped_idx]
            self.skipped_box_name = self.skipped_box.name
            all_indices = list(range(self.num_boxes))
            all_indices.remove(self.skipped_idx)
            self.open_order = np.random.permutation(all_indices).tolist()
            print(f"打开顺序: {[self.box[i].name for i in self.open_order]}")
            
            # 初始化所有盒子的盖子为关闭状态
            for box in self.box:
                self.close_lid(box)

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
        """评估任务完成情况：被跳过的盒子的盖子是否打开"""
        success = self.is_lid_open(self.skipped_box)
        success = torch.tensor([success], dtype=torch.bool, device=self.device)

        return {
            "success": success,
            "skipped_box_name": self.skipped_box_name,
            "skipped_box_lid_open": success,
        }
    
    def get_lid_joint(self, box):
        """
        获取盒子的盖子关节
        
        Args:
            box: 盒子对象
            
        Returns:
            盖子关节对象，如果找不到则返回None
        """
        if hasattr(box, 'active_joints'):
            for joint in box.active_joints:
                if self.lid_joint_name in joint.name:
                    return joint
        return None
    
    def open_lid(self, box):
        """
        打开盒子的盖子（直接设置关节位置，用于实验员模拟）
        
        Args:
            box: 盒子对象
        """
        lid_joint = self.get_lid_joint(box)
        if lid_joint is not None:
            lid_joint.set_drive_properties(stiffness=100, damping=10, force_limit=100, mode="force")
            qpos = box.get_qpos()
            if hasattr(qpos, 'clone'):
                qpos = qpos.clone()
            else:
                qpos = np.array(qpos, copy=True).flatten()
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy().flatten()
            qpos = np.asarray(qpos).flatten()
            for i, joint in enumerate(box.active_joints):
                if self.lid_joint_name in joint.name:
                    qpos[i] = 2
                    break
            box.set_qpos(qpos)
            lid_joint.set_drive_target(2)
        else:
            print(f"警告: 找不到盒子 {box.name} 的盖子关节")
    
    def close_lid(self, box):
        """
        关闭盒子的盖子（直接设置关节位置，用于实验员模拟）
        
        Args:
            box: 盒子对象
        """
        lid_joint = self.get_lid_joint(box)
        if lid_joint is not None:
            lid_joint.set_drive_properties(stiffness=100, damping=10, force_limit=100, mode="force")
            qpos = box.get_qpos()
            if hasattr(qpos, 'clone'):
                qpos = qpos.clone()
            else:
                qpos = np.array(qpos, copy=True).flatten()
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy().flatten()
            qpos = np.asarray(qpos).flatten()
            for i, joint in enumerate(box.active_joints):
                if self.lid_joint_name in joint.name:
                    qpos[i] = 0
                    break
            box.set_qpos(qpos)
            lid_joint.set_drive_target(0)
        else:
            print(f"警告: 找不到盒子 {box.name} 的盖子关节")
    
    def release_lid(self, box):
        """
        释放盖子关节的驱动力，使机器人可以物理拨开盖子
        
        Args:
            box: 盒子对象
        """
        lid_joint = self.get_lid_joint(box)
        if lid_joint is not None:
            lid_joint.set_drive_properties(stiffness=0, damping=5, force_limit=100, mode="force")
        else:
            print(f"警告: 找不到盒子 {box.name} 的盖子关节")
    
    def is_lid_open(self, box, threshold=1.3963):
        """
        检查盒子的盖子是否打开
        
        Args:
            box: 盒子对象
            threshold: 判断打开的阈值（弧度）
            
        Returns:
            bool: 盖子是否打开
        """
        lid_joint = self.get_lid_joint(box)
        if lid_joint is not None:
            qpos = lid_joint.qpos
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy()
            return abs(qpos) > threshold
        return False

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
