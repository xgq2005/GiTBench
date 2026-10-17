"""Biolab L4 evaluation task."""
import sapien
import torch
import numpy as np
import os
from typing import Tuple, Dict, Any
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.piper_x import PiperX
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils import randomization
from mani_skill.envs.tasks.custom_task.biolab.biolab_scene_cfg import BIOLAB_SCENE_CONFIGS
from mani_skill.envs.tasks.gitbench_paths import project_asset_path

# 生物实验室化学模型基础路径宏定义
BIOLAB_CHEMISTRY_MODEL_DIR = str(project_asset_path("chemistry", "models"))
normal_x=10
@register_env("Scene_L4_01", max_episode_steps=200)
class Scene_L4_01(BaseEnv):
    """
    生物实验室搅拌计数记忆任务环境
    
    任务：机器人需要记忆搅拌次数，使用玻璃棒搅拌烧杯中的液体
    """
    
    SUPPORTED_ROBOTS = [("piper_x", "piper_x")]
    agent: MultiAgent[Tuple[PiperX, PiperX]]
    
    # 预定义的烧杯位置数组（10个坐标组，每个坐标组包含3个烧杯的位置）
    PREDEFINED_BEAKER_POSITIONS = [
        # 坐标组 0
        [
            [-0.15, 0.25, 0.07],    # beaker3
            [-0.15, 0.125, 0.07],   # beaker2
            [-0.15, 0, 0.07],  # beaker1
            [-0.15, -0.125, 0.07]    # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 1
        [
            [-0.05, 0.115, 0.07],
            [-0.169, 0.00, 0.07],
            [-0.1, -0.139, 0.07],
            [-0.15, -0.25, 0.07]   # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 2
        [
            [-0.05, 0.115, 0.07],
            [-0.169, 0.06, 0.07],
            [-0.15, -0.1, 0.07],
            [-0.15, -0.25, 0.07]   # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 3
        [
            [-0.1, 0.2, 0.07],     # beaker4 (与其他烧杯距离 > 0.08)
            [-0.169, 0.06, 0.07],
            [-0.15, -0.1, 0.07],
            [-0.05, -0.18, 0.07]
        ],
        # 坐标组 4
        [
            [-0.1, 0.15, 0.07],
            [-0.05, 0, 0.07],
            [-0.15, -0.12, 0.07],
            [-0.15, -0.25, 0.07]     # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 5
        [
            [-0.1, 0.25, 0.07],
            [-0.15, 0.1, 0.07],
            [-0.15, -0.05, 0.07],
            [-0.1, -0.22, 0.07]   # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 6
        [
            [-0.2, 0.25, 0.07],
            [-0.05, 0.1, 0.07],
            [-0.15, -0.05, 0.07],
            [-0.2, -0.22, 0.07]   # beaker4 (与其他烧杯距离 > 0.08)
        ],
        # 坐标组 7
        [
            [-0.15, 0.25, 0.07],
            [-0.05, 0.1, 0.07],
            [-0.15, -0.05, 0.07],
            [-0.05, -0.22, 0.07]   # beaker4 (与其他烧杯距离 > 0.08)
        ],
    ]

    def __init__(
        self, 
        *args, 
        robot_uids=("piper_x", "piper_x"), 
        robot_init_qpos_noise=None,
        beaker_position_index=None,  # 新增参数：指定使用哪个坐标组
        scene_name="default",
        **kwargs
    ):
        cfg = BIOLAB_SCENE_CONFIGS.get(scene_name, BIOLAB_SCENE_CONFIGS["default"])
        if robot_init_qpos_noise is None:
            robot_init_qpos_noise = cfg["robot_init_qpos_noise"]
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.beaker_position_index = beaker_position_index
        # 统一加载所有配置参数到 self
        self.goal_thresh = cfg["goal_thresh"]
        self.base_camera_width = cfg["base_camera_width"]
        self.base_camera_height = cfg["base_camera_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.sensor_fovx = cfg["sensor_fovx"]
        self.sensor_fovy = cfg["sensor_fovy"]
        self.left_robot_pose = cfg["left_robot_pose"]
        self.right_robot_pose = cfg["right_robot_pose"]
        self._cfg = cfg
        self.stir_count = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        left_pose = sapien.Pose(p=self._cfg["left_robot_pose"])
        right_pose = sapien.Pose(p=self._cfg["right_robot_pose"])
        super()._load_agent(options, [left_pose, right_pose])

    def set_actor_color(self, actor, rgba):
        """
        修改 actor 所有视觉材质的颜色，支持多材质模型。
        rgba: [r, g, b, a]，每个分量 0~1
        """
        if hasattr(actor, '_objs'):
            for entity in actor._objs:
                if hasattr(entity, 'components'):
                    for comp in entity.components:
                        comp_type = type(comp).__name__
                        if 'RenderBodyComponent' in comp_type or hasattr(comp, 'render_shapes'):
                            for shape in comp.render_shapes:
                                if hasattr(shape, 'get_materials'):
                                    materials = shape.get_materials()
                                    for mat in materials:
                                        if mat is not None:
                                            mat.set_base_color(rgba)
                                elif hasattr(shape, 'material'):
                                    mat = shape.material
                                    if mat is not None:
                                        mat.set_base_color(rgba)
            return

        if hasattr(actor, 'get_visual_bodies'):
            for visual in actor.get_visual_bodies():
                for shape in visual.render_shapes:
                    if hasattr(shape, 'get_materials'):
                        for mat in shape.get_materials():
                            if mat:
                                mat.set_base_color(rgba)
                    elif hasattr(shape, 'material') and shape.material:
                        shape.material.set_base_color(rgba)
            return

        print(f"[WARN] Cannot find render components for {actor.name}")

    # 定义搅拌计数记忆任务物体配置
    BIOLAB_OBJECTS = [
        # 托盘
        {
            "name": "tray",
            "type": "robocasa_actor",
            "path": "objects/objaverse/tray/tray_1",
            "pos": [-0.35, 0, 0.01],
            "rot_deg": [90, 0, 0],
            "scale": [1.8, 1.5, 1],
            "color": [1, 1, 1, 1],
            "static": True
        },
        # 烧杯1
        {
            "name": "beaker1",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "016_beaker"),
            "collision_file": "collision.ply",
            "visual_file": "textured.obj",
            "pos": [normal_x, 0.2, 0.05],
            "rot_deg": [0, 0, 0],
            "scale": [1.5, 1.5, 1.0],
            "density": 10,
            "static": False
        },
        # 烧杯2
        {
            "name": "beaker2",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "016_beaker"),
            "collision_file": "collision.ply",
            "visual_file": "textured.obj",
            "pos": [normal_x, 0, 0.05],
            "rot_deg": [0, 0, 0],
            "scale": [1.5, 1.5, 1.0],
            "density": 10,
            "static": False
        },
        # 烧杯3
        {
            "name": "beaker3",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "016_beaker"),
            "collision_file": "collision.ply",
            "visual_file": "textured.obj",
            "pos": [normal_x, -0.2, 0.05],
            "rot_deg": [0, 0, 0],
            "scale": [1.5, 1.5, 1.0],
            "density": 10,
            "static": False
        },
        # 烧杯4
        {
            "name": "beaker4",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "016_beaker"),
            "collision_file": "collision.ply",
            "visual_file": "textured.obj",
            "pos": [normal_x-0.1, 0, 0.05],
            "rot_deg": [0, 0, 0],
            "scale": [1.5, 1.5, 1.0],
            "density": 10,
            "static": False
        },
        # 透明细长圆柱体（替代玻璃棒）
        {
            "name": "glass_rod1",
            "type": "cylinder",
            "radius": 0.005,
            "half_length": 0.075,
            "pos": [normal_x, 0.2, 0.15],
            "rot_deg": [0, 90, 0],
            "color": [0.8, 0.9, 1.0, 0.8],
            "density": 10,
            "static": True
        },
        # # 玻璃棒2
        # {
        #     "name": "glass_rod2",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "019_glass_rod"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": [0, 0, 0.05],
        #     "rot_deg": [0, 0, 0],
        #     "scale": [1.0, 1.0, 1.0],
        #     "density": 10
        # },
        # # 玻璃棒3
        # {
        #     "name": "glass_rod3",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "019_glass_rod"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": [0, -0.2, 0.05],
        #     "rot_deg": [0, 0, 0],
        #     "scale": [1.0, 1.0, 1.0],
        #     "density": 10
        # }
    ]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        self.biolab_objects = {}
        
        for obj_cfg in self.BIOLAB_OBJECTS:
            try:
                if obj_cfg["type"] == "robocasa_actor":
                    robocasa_root = str(project_asset_path("scene_datasets", "robocasa_dataset", "assets"))
                    model_path = os.path.join(robocasa_root, obj_cfg["path"], "model.xml")
                    
                    if not os.path.exists(model_path):
                        print(f"警告: 模型不存在 {model_path}")
                        continue
                    
                    loader = self.scene.create_mjcf_loader()
                    loader.visual_groups = [0, 1, 2, 3, 4, 5]
                    if obj_cfg.get("scale"):
                        loader.scale = obj_cfg["scale"]
                    
                    builders = loader.parse(model_path)
                    
                    if builders["actor_builders"]:
                        builder = builders["actor_builders"][0]
                    else:
                        print(f"警告: {obj_cfg['path']} 期望 actor 但未找到")
                        continue
                    
                    rot_rad = [np.deg2rad(r) for r in obj_cfg["rot_deg"]]
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rot_rad)
                    quat = r.as_quat()
                    pose = sapien.Pose(p=obj_cfg["pos"], q=quat)
                    builder.initial_pose = pose
                    
                    if obj_cfg.get("static"):
                        obj = builder.build_static(name=obj_cfg["name"])
                    else:
                        obj = builder.build(name=obj_cfg["name"])
                    
                    self.biolab_objects[obj_cfg["name"]] = obj
                    
                    if obj_cfg.get("color"):
                        self.set_actor_color(obj, obj_cfg["color"])
                    
                    print(f"加载成功: {obj_cfg['name']}, 类型: RoboCasa Actor")
                    
                elif obj_cfg["type"] == "cylinder":
                    builder = self.scene.create_actor_builder()
                    
                    radius = obj_cfg["radius"]
                    half_length = obj_cfg["half_length"]
                    
                    builder.add_capsule_collision(
                        radius=radius,
                        half_length=half_length,
                        density=obj_cfg["density"]
                    )
                    
                    viz_mat = sapien.render.RenderMaterial(
                        base_color=obj_cfg["color"],
                        roughness=0.3,
                        specular=0.5
                    )
                    
                    builder.add_capsule_visual(
                        radius=radius,
                        half_length=half_length,
                        material=viz_mat
                    )
                    
                    rot_rad = [np.deg2rad(r) for r in obj_cfg["rot_deg"]]
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rot_rad)
                    quat = r.as_quat()
                    pose = sapien.Pose(p=obj_cfg["pos"], q=quat)
                    builder.initial_pose = pose
                    
                    if obj_cfg.get("static", False):
                        obj = builder.build_kinematic(name=obj_cfg["name"])
                    else:
                        obj = builder.build(name=obj_cfg["name"])
                    self.biolab_objects[obj_cfg["name"]] = obj
                    
                    print(f"加载成功: {obj_cfg['name']}, 类型: 透明圆柱体, 质量: {obj.mass if hasattr(obj, 'mass') else 'N/A'}")
                    
                else:
                    collision_path = os.path.join(obj_cfg["model_path"], obj_cfg["collision_file"])
                    visual_path = os.path.join(obj_cfg["model_path"], obj_cfg["visual_file"])
                    
                    if not os.path.exists(collision_path):
                        print(f"警告: 碰撞文件不存在 {collision_path}")
                        continue
                    if not os.path.exists(visual_path):
                        print(f"警告: 视觉文件不存在 {visual_path}")
                        continue
                    
                    builder = self.scene.create_actor_builder()
                    builder.add_multiple_convex_collisions_from_file(
                        filename=collision_path,
                        scale=obj_cfg["scale"],
                        density=obj_cfg["density"]
                    )
                    builder.add_visual_from_file(
                        filename=visual_path,
                        scale=obj_cfg["scale"]
                    )
                    
                    rot_rad = [np.deg2rad(r) for r in obj_cfg["rot_deg"]]
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('xyz', rot_rad)
                    quat = r.as_quat()
                    pose = sapien.Pose(p=obj_cfg["pos"], q=quat)
                    builder.initial_pose = pose
                    
                    if obj_cfg.get("static", False):
                        obj = builder.build_kinematic(name=obj_cfg["name"])
                    else:
                        obj = builder.build(name=obj_cfg["name"])
                    self.biolab_objects[obj_cfg["name"]] = obj
                    
                    print(f"加载成功: {obj_cfg['name']}, 类型: {type(obj)}, 质量: {obj.mass if hasattr(obj, 'mass') else 'N/A'}")
                
            except Exception as e:
                print(f"加载物体 {obj_cfg['name']} 时出错: {e}")
                continue
        
        # 创建目标位置指示器（已禁用）
        # self.goal_indicator = actors.build_sphere(
        #     self.scene,
        #     radius=0.02,
        #     color=[0, 1, 0, 0.5],
        #     name="goal_indicator",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(p=[0.3, 0.3, 0.02])
        # )

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.stir_count = 0
            # ========== 在此之后重新设置你期望的初始姿态 ==========
            agent: MultiAgent = self.agent
            
            # 设置你期望的初始关节角度
            qpos = np.array([
                0.0,   # joint 0
                0.0,   # joint 1
                0.0,   # joint 2
                0.95,   # joint 3
                0.0,   # joint 4
                0.0,   # joint 5
                0,  # gripper left (joint7) - 闭合
                0  # gripper right (joint8) - 闭合
            ])
            qpos = np.tile(qpos, (b, 1))
            
            # 应用初始关节角度和基座位置
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(sapien.Pose(self._cfg["left_robot_pose"]))  # 左机器人
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(sapien.Pose(self._cfg["right_robot_pose"]))  # 右机器人
            
            # 重置控制器，使目标位置与当前位置同步
            for arm_idx in [0, 1]:
                arm_agent = agent.agents[arm_idx]
                if hasattr(arm_agent, 'controller'):
                    arm_agent.controller.reset()
            
            # 设置烧杯位置
            beaker_positions = {}
            beaker_names = ["beaker1", "beaker2", "beaker3", "beaker4"]
            
            # 使用预定义位置或随机位置
            if self.beaker_position_index is not None:
                # 使用指定的预定义坐标组
                pos_index = self.beaker_position_index % len(self.PREDEFINED_BEAKER_POSITIONS)
                predefined_positions = self.PREDEFINED_BEAKER_POSITIONS[pos_index]
                print(f"使用预定义烧杯位置组 #{pos_index}")
                
                for i, name in enumerate(beaker_names):
                    if name in self.biolab_objects:
                        xyz = torch.zeros((b, 3))
                        xyz[:, 0] = predefined_positions[i][0]
                        xyz[:, 1] = predefined_positions[i][1]
                        xyz[:, 2] = predefined_positions[i][2]
                        self.biolab_objects[name].set_pose(Pose.create_from_pq(xyz))
                        beaker_positions[name] = xyz
            else:
                # 随机化烧杯位置，确保间隔至少0.1
                for i, name in enumerate(beaker_names):
                    if name in self.biolab_objects:
                        xyz = torch.zeros((b, 3))
                        
                        for env_i in range(b):
                            max_attempts = 100
                            for attempt in range(max_attempts):
                                # x限制在（-0.2,0.1）之间
                                x = torch.rand(1).item() * 0.25 - 0.2
                                # y限制在(-0.3,0.3)之间
                                y = torch.rand(1).item() * 0.6 - 0.3
                                
                                # 检查与已有烧杯的间隔
                                valid = True
                                for prev_name in beaker_names[:i]:
                                    if prev_name in beaker_positions:
                                        prev_pos = beaker_positions[prev_name][env_i]
                                        distance = torch.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                                        if distance < 0.15:
                                            valid = False
                                            break
                                
                                if valid:
                                    xyz[env_i, 0] = x
                                    xyz[env_i, 1] = y
                                    break
                        
                        xyz[:, 2] = 0.07
                        self.biolab_objects[name].set_pose(Pose.create_from_pq(xyz))
                        beaker_positions[name] = xyz
            
            # # 根据烧杯位置设置对应的玻璃棒位置（从烧杯上方落下）
            # for i, (beaker_name, rod_name) in enumerate(zip(["beaker1", "beaker2", "beaker3"])):
            #     if beaker_name in beaker_positions and rod_name in self.biolab_objects:
            #         beaker_pos = beaker_positions[beaker_name]
                    
            #         # 玻璃棒位置：在烧杯中心上方，让它自然落下
            #         rod_xyz = torch.zeros((b, 3))
            #         rod_xyz[:, 0] = beaker_pos[:, 0]  # X位置与烧杯相同
            #         rod_xyz[:, 1] = beaker_pos[:, 1]+0.03  # Y位置与烧杯相同
            #         rod_xyz[:, 2] = beaker_pos[:, 2] + 0.2  # Z位置在烧杯上方20cm，让它落下
                    
            #         from scipy.spatial.transform import Rotation as R
            #         r = R.from_euler('xyz', [0, 90, 0])
            #         rod_quat = r.as_quat()
                    
            #         # 为每个环境创建相同的旋转
            #         rod_quat_tensor = torch.tensor(rod_quat, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(b, 1)
                    
            #         # 设置玻璃棒位置和姿态
            #         rod_pose = Pose.create_from_pq(rod_xyz, rod_quat_tensor)
            #         self.biolab_objects[rod_name].set_pose(rod_pose)
            
            # 随机化目标位置（已禁用，因为goal_indicator已移除）
            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, 0] = torch.rand(b) * 0.3 + 0.1
            # goal_xyz[:, 1] = torch.rand(b) * 0.3 + 0.1
            # goal_xyz[:, 2] = 0.07
            # self.goal_indicator.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self):
        return {"success": torch.ones(self.num_envs, dtype=bool)}

    def _get_obs_extra(self, info):
        obs = {}
        if self.obs_mode_struct.use_state:
            for name, obj in self.biolab_objects.items():
                obs[f"{name}_pose"] = obj.pose.raw_pose
            obs["stir_count"] = torch.tensor([self.stir_count], dtype=torch.float32)
            # obs["goal_pos"] = self.goal_indicator.pose.p  # 已移除goal_indicator
        return obs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return [CameraConfig("human_cam", pose, self.base_camera_width, self.base_camera_height, self.sensor_fovy, 0.01, 100)]
        # pose = sapien_utils.look_at(self.sensor_cam_eye_pos, self.sensor_cam_target_pos)
        # return [CameraConfig("sensor_cam", pose, self.base_camera_width, self.base_camera_height, self.sensor_fovy, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, self.sensor_fovy, 0.01, 100)
