"""Biolab L5 evaluation task."""
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
from mani_skill.envs.tasks.custom_task.biolab.biolab_scene_cfg import BIOLAB_SCENE_CONFIGS
from mani_skill.envs.tasks.gitbench_paths import project_asset_path

# 化学模型基础路径（用于 sample_tube）
BIOLAB_CHEMISTRY_MODEL_DIR = str(project_asset_path("chemistry", "models"))
# cube_rack 放置位置
cube_rack_pos = [-0.21, 0, 0.05]
cube_rack01_pos = [cube_rack_pos[0]-0.135, cube_rack_pos[1]+0.2, cube_rack_pos[2]+0.01]
cube_rack02_pos = [cube_rack_pos[0]-0.135, cube_rack_pos[1]-0.2, cube_rack_pos[2]+0.01]
cube_racks_pos = [cube_rack_pos,cube_rack01_pos, cube_rack02_pos]
distance = 0.0275
distance02=0.04625
height=0.1
cube_place_pos = [
    [cube_racks_pos[0][0], cube_racks_pos[0][1]+distance*4, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]+distance*3, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]+distance*2, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]+distance, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1], height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]-distance, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]-distance*2, height], 
    [cube_racks_pos[0][0], cube_racks_pos[0][1]-distance*3, height],
    [cube_racks_pos[0][0], cube_racks_pos[0][1]-distance*4, height],    
]
cube_place02_pos = [
    [cube_racks_pos[1][0]+distance02, cube_racks_pos[1][1], height],
    [cube_racks_pos[1][0], cube_racks_pos[1][1], height],
    [cube_racks_pos[1][0]-distance02, cube_racks_pos[1][1], height],
]
cube_place03_pos = [
    [cube_racks_pos[2][0]+distance02, cube_racks_pos[2][1], height],
    [cube_racks_pos[2][0], cube_racks_pos[2][1], height],
    [cube_racks_pos[2][0]-distance02, cube_racks_pos[2][1], height],
]
cube_origin_place =[
    cube_place_pos[0],
    cube_place_pos[2],
    cube_place_pos[4],
    cube_place_pos[6],
    cube_place_pos[8], 
]
yingshe=[0,2,4,6,8]
control_cube_static = False
tube_scale=[0.3, 0.3, 0.68]
# ============================================================
# 样本架布局：16组，每组5个编号（对应样本架5个槽位）
# 位置固定，试管编号按不同排列组合
# 下标是样本编号
# 含的值是槽位
# ============================================================
LAYOUT_8 = [
    [0, 1, 2, 3, 4],  # 0: 顺序
    [4, 3, 2, 1, 0],  # 1: 逆序
    [0, 2, 4, 1, 3],  # 2
    [3, 1, 4, 2, 0],  # 3
    [1, 0, 3, 4, 2],  # 4
    [2, 4, 0, 3, 1],  # 5
    [4, 0, 1, 2, 3],  # 6
    [3, 4, 1, 0, 2],  # 7
]

# ============================================================
# 放置架 repeat_id：4组，每组3个编号（对应放置架3个槽位）
# ============================================================
REPEAT_4 = [
    [-0.4, 0.0, 0.3],
    [-0.3, 0.0, 0.3],
    [-0.3, 0.1, 0.25],
    [-0.4, 0.1, 0.25]
]
rack_density = 800
tube_density = 1000
@register_env("Scene_L5_02", max_episode_steps=200)
class Scene_L5_02(BaseEnv):
    """
    Biolab scene with cube rack (L5_02)
    """
    
    BIOLAB_CHEMISTRY_MODEL_DIR = str(project_asset_path("chemistry", "models"))
    
    SUPPORTED_ROBOTS = [("piper_x", "piper_x")]
    agent: MultiAgent[Tuple[PiperX, PiperX]]

    def __init__(
        self, 
        *args, 
        robot_uids=("piper_x", "piper_x"), 
        robot_init_qpos_noise=None,
        scene_name="default",
        **kwargs
    ):
        cfg = BIOLAB_SCENE_CONFIGS.get(scene_name, BIOLAB_SCENE_CONFIGS["default"])
        if robot_init_qpos_noise is None:
            robot_init_qpos_noise = cfg["robot_init_qpos_noise"]
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        left_pose = sapien.Pose(p=self._cfg["left_robot_pose"])
        right_pose = sapien.Pose(p=self._cfg["right_robot_pose"])
        super()._load_agent(options, [left_pose, right_pose], build_separate=False)

    # 物体配置列表
    OBJECTS = [
        # cube_rack - 主架子模型（.glb）
        {
            "name": "tube_rack",
            "type": "actor",
            "model_path": str(project_asset_path("custom", "tube04.glb")),
            "pos": cube_racks_pos[0],
            "rot_deg": [90, 0, 90],
            "scale": [0.32, 0.77, 0.32],
            "density": rack_density,
            "static": True,
            "color": [0.8, 0.3, 0.3, 1.0]  # ← 红
        },
        {
            "name": "tube_rack02",
            "type": "actor",
            "model_path": str(project_asset_path("custom", "tube05.glb")),
            "pos": cube_racks_pos[1],  
            "rot_deg": [0, 0, 90],
            "scale": [0.35, 0.97, 0.35],
            "density": rack_density,
            "static": True,
            "color": [0.3, 0.95, 0.8, 1.0]  # ← 黄色
        },
        {
            "name": "tube_rack03",
            "type": "actor",
            "model_path": str(project_asset_path("custom", "tube05.glb")),
            "pos": cube_racks_pos[2],  
            "rot_deg": [0, 0, 90],
            "scale": [0.35, 0.97, 0.35],
            "density": rack_density,
            "static": True,
            "color": [0.3, 0.3, 0.8, 1.0]  # ← 黄色
        },
        # sample_tube_1 - 放置在 cube_rack 上的试管
        {
            "name": "sample_tube_0",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
            "collision_file": "tube_with_cube3.ply",
            "visual_file": "textured.obj",
            "pos": [cube_place_pos[yingshe[0]][0], cube_place_pos[yingshe[0]][1], cube_place_pos[yingshe[0]][2]+0.0],
            "rot_deg": [0, 0, 0],
            "scale": tube_scale,
            "density": tube_density,
            "static": False
        },
                # sample_tube_1 - 放置在 cube_rack 上的试管
        {
            "name": "sample_tube_1",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
            "collision_file": "tube_with_cube3.ply",
            "visual_file": "textured.obj",
            "pos": [cube_place_pos[yingshe[1]][0], cube_place_pos[yingshe[1]][1], cube_place_pos[yingshe[1]][2]+0.0],
            "rot_deg": [0, 0, 0],
            "scale": tube_scale,
            "density": tube_density,
            "static": control_cube_static
        },
                # sample_tube_1 - 放置在 cube_rack 上的试管
        {
            "name": "sample_tube_2",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
            "collision_file": "tube_with_cube3.ply",
            "visual_file": "textured.obj",
            "pos": [cube_place_pos[yingshe[2]][0], cube_place_pos[yingshe[2]][1], cube_place_pos[yingshe[2]][2]+0.0],
            "rot_deg": [0, 0, 0],
            "scale": tube_scale,
            "density": tube_density,
            "static": control_cube_static
        },
                # sample_tube_1 - 放置在 cube_rack 上的试管
        {
            "name": "sample_tube_3",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
            "collision_file": "tube_with_cube3.ply",
            "visual_file": "textured.obj",
            "pos": [cube_place_pos[yingshe[3]][0], cube_place_pos[yingshe[3]][1], cube_place_pos[yingshe[3]][2]+0.0],
            "rot_deg": [0, 0, 0],
            "scale": tube_scale,
            "density": tube_density,
            "static": control_cube_static
        },
        {
            "name": "sample_tube_4",
            "type": "actor",
            "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
            "collision_file": "tube_with_cube3.ply",
            "visual_file": "textured.obj",
            "pos": [cube_place_pos[yingshe[4]][0], cube_place_pos[yingshe[4]][1], cube_place_pos[yingshe[4]][2]+0.0],
            "rot_deg": [0, 0, 0],
            "scale": tube_scale,
            "density": tube_density,
            "static": control_cube_static
        },
        # {
        #     "name": "sample_tube_9",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
        #     "collision_file": "tube_with_cube.ply",
        #     "visual_file": "tube_with_cube.obj",
        #     "pos": [cube_place02_pos[1][0], cube_place02_pos[1][1], cube_place02_pos[1][2]+0.0],
        #     "rot_deg": [0, 0, 0],
        #     "scale": tube_scale,
        #     "density": 1000,
        #     "static": control_cube_static
        # },
        # {
        #     "name": "sample_tube_10",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": [cube_place02_pos[1][0], cube_place02_pos[1][1], cube_place02_pos[1][2]+0.03],
        #     "rot_deg": [0, 0, 0],
        #     "scale": tube_scale,
        #     "density": 1000,
        #     "static": control_cube_static
        # },
        # {
        #     "name": "sample_tube_11",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": [cube_place02_pos[2][0], cube_place02_pos[2][1], cube_place02_pos[2][2]+0],
        #     "rot_deg": [0, 0, 0],
        #     "scale": tube_scale,
        #     "density": 1000,
        #     "static": control_cube_static
        # },
        # {
        #     "name": "sample_tube_12",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": cube_place02_pos[3],
        #     "rot_deg": [0, 0, 0],
        #     "scale": tube_scale,
        #     "density": 1000,
        #     "static": control_cube_static
        # },
        # {
        #     "name": "sample_tube_13",
        #     "type": "actor",
        #     "model_path": os.path.join(BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
        #     "collision_file": "collision.ply",
        #     "visual_file": "textured.obj",
        #     "pos": cube_place03_pos[4],
        #     "rot_deg": [0, 0, 0],
        #     "scale": [0.8, 0.8, 0.9],
        #     "density": 1000,
        #     "static": control_cube_static
        # },
    ]
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        self.objects = {}
        
        # 创建高摩擦系数的物理材质，用于试管
        high_friction_material = sapien.physx.PhysxMaterial(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0
        )
        
        for obj_cfg in self.OBJECTS:
            try:
                model_path = obj_cfg["model_path"]
                builder = self.scene.create_actor_builder()
                
                # 如果是 .glb 文件，直接加载
                if model_path.endswith('.glb'):
                    builder.add_visual_from_file(
                        filename=model_path,
                        scale=obj_cfg.get("scale", [1, 1, 1])
                    )
                    builder.add_nonconvex_collision_from_file(
                        filename=model_path,
                        scale=obj_cfg.get("scale", [1, 1, 1])
                    )
                else:
                    # 如果是目录模型（如 test_tube），使用 collision/visual 文件
                    collision_path = os.path.join(model_path, obj_cfg["collision_file"])
                    visual_path = os.path.join(model_path, obj_cfg["visual_file"])
                    
                    if not os.path.exists(collision_path):
                        print(f"警告: 碰撞文件不存在 {collision_path}")
                        continue
                    if not os.path.exists(visual_path):
                        print(f"警告: 视觉文件不存在 {visual_path}")
                        continue
                    
                    if "sample_tube" in obj_cfg["name"]:
                        builder.add_multiple_convex_collisions_from_file(
                            filename=collision_path,
                            scale=obj_cfg["scale"],
                            density=obj_cfg["density"],
                            material=high_friction_material
                        )
                    else:
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
                
                self.objects[obj_cfg["name"]] = obj
                # if "color" in obj_cfg:
                #     for vb in obj.get_visual_bodies():
                #         for rs in vb.get_render_shapes():
                #             if hasattr(rs, 'material') and rs.material:
                #                 rs.material.set_base_color(obj_cfg["color"])
                # 如果是试管，降低质量使其更容易被抓取（原密度1000，改为0.05kg）
                # 同时将质心向下偏移5cm，使试管更稳定
                if "sample_tube" in obj_cfg["name"]:
                    obj.set_mass(0.015)
                    if hasattr(obj, "set_cmass_local_pose"):
                        obj.set_cmass_local_pose(Pose(p=[0, 0, -0.5]))  # 局部坐标系中向下偏移5cm
                print(f"加载成功: {obj_cfg['name']}")
                
            except Exception as e:
                print(f"加载物体 {obj_cfg['name']} 时出错: {e}")
                continue

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            agent: MultiAgent = self.agent
            qpos = np.array([
                0.0, 0.0, 0.0, 0.95, 0.0, 0.0,
                0, 0
            ])
            qpos = np.tile(qpos, (b, 1))
            
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(sapien.Pose(self._cfg["left_robot_pose"]))
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(sapien.Pose(self._cfg["right_robot_pose"]))
            
            for arm_idx in [0, 1]:
                arm_agent = agent.agents[arm_idx]
                if hasattr(arm_agent, 'controller'):
                    arm_agent.controller.reset()

    def evaluate(self):
        return {"success": torch.ones(self.num_envs, dtype=bool)}

    def _get_obs_extra(self, info):
        obs = {}
        if self.obs_mode_struct.use_state:
            for name, obj in self.objects.items():
                obs[f"{name}_pose"] = obj.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return [CameraConfig("human_cam", pose, self.base_camera_width, self.base_camera_height, self.sensor_fovy, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, self.sensor_fovy, 0.01, 100)
