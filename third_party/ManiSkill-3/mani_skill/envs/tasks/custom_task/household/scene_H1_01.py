import sapien
import torch
import numpy as np
import os
import sapien
from typing import Tuple, Dict, Any
from mani_skill.utils.building import MJCFLoader   # 在文件开头添加导入
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.piper_x import PiperX
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.custom_task.biolab.biolab_scene_cfg import BIOLAB_SCENE_CONFIGS
from mani_skill.envs.tasks.gitbench_paths import project_asset_path
from transforms3d.euler import euler2quat
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as Rot

@register_env("SceneH1_01Env", max_episode_steps=200)
class SceneH1_01Env(BaseEnv):
    SUPPORTED_ROBOTS = [("piper_x", "piper_x")]
    agent: MultiAgent[Tuple[PiperX, PiperX]]
    cube_half_size = 0
    goal_thresh = 0

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
       

    # 在环境类中定义配置列表，每个元素是 (模型相对路径, 位姿)
    ROBOCASA_OBJECTS = [
        # ==================== 普通物体 (Actor) ====================

        # {"type": "actor", "path": "objects/objaverse/plate/plate_1",    "pos": [-0.25, -0.12, 0.12], "rot_deg": [90, 90, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/plate/plate_1",    "pos": [-0.25, 0.09, 0.12], "rot_deg": [90, 90, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/plate/plate_1",    "pos": [-0.25, -0.02, 0.12], "rot_deg": [90, 90, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/plate/plate_1",    "pos": [-0.25, -0.17, 0.12], "rot_deg": [90, 90, 0], "scale": [1, 1, 1], "static": False},
        
        
    ]
    # 6 组碟子初始化位姿（每组 3 个，对应 plate1/plate2/plate3）
    PREDEFINED_PLATE_POSITIONS = [
        # 组 0：1 6
        [
            {"pos": [-0.25, 0.15, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.09, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 1：1 7
        [
            {"pos": [-0.25, 0.15, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.12, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 2：1 8
        [
            {"pos": [-0.25, 0.15, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.17, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 3：2 6
        [
            {"pos": [-0.25, 0.12, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.09, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 4：3 7
        [
            {"pos": [-0.25, 0.09, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.12, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 5：3 8
        [
            {"pos": [-0.25, 0.09, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.17, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 6：1 6 8
        [
            {"pos": [-0.25, 0.15, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.09, 0.12], "rot_deg": [90, 90, 0]},   # plate2
            {"pos": [-0.25, -0.17, 0.12], "rot_deg": [90, 90, 0]},   # plate3
        ],
        # 组 7：2 7
        [
            {"pos": [-0.25, 0.12, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.12, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
        # 组 8：2 8
        [
            {"pos": [-0.25, 0.12, 0.12], "rot_deg": [90, 90, 0]},     # plate1
            {"pos": [-0.25, -0.17, 0.12], "rot_deg": [90, 90, 0]},   # plate2
        ],
    ]
    # 6 组目标位置（每组一个 [x, y, z]，对应架子上的槽位）
    PREDEFINED_TARGET_POSITIONS = [
        [-0.08,  0.13, 0.12],   # 组 0：0
        [-0.08, 0.06, 0.12],   # 组 1：1
        [-0.08, 0.02, 0.12],   # 组 2：2
        [-0.08, -0.02, 0.12],   # 组 3：3
        [-0.08, -0.06, 0.12],   # 组 4：4
        [-0.08, -0.13, 0.12],   # 组 5：5
    ]
    
    def set_actor_color(self, actor, rgba):
        """
        修改 actor 所有视觉材质的颜色，支持多材质模型。
        rgba: [r, g, b, a]，每个分量 0~1
        """
        # 处理 ManiSkill 的 Actor 包装（支持 batch 环境）
        if hasattr(actor, '_objs'):
            for entity in actor._objs:
                if hasattr(entity, 'components'):
                    for comp in entity.components:
                        comp_type = type(comp).__name__
                        if 'RenderBodyComponent' in comp_type or hasattr(comp, 'render_shapes'):
                            for shape in comp.render_shapes:
                                # 如果 shape 有多个材质，会有一个 get_materials() 或 iterate over material slots
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

        # 如果直接是 sapien.Entity / sapien.Actor
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
    def set_articulation_color(self, articulation, rgba):
        """为关节物体的每个连杆设置颜色"""
        if hasattr(articulation, '_objs'):
            sapien_artic = articulation._objs[0]  # 取第一个环境的底层
            for link in sapien_artic.get_links():
                self.set_actor_color(link, rgba)
        else:
            for link in articulation.get_links():
                self.set_actor_color(link, rgba)
    
    def _load_scene(self, options: dict):

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        # 添加挡板（手动构建，避免 build_cube 的 bug）
        builder = self.scene.create_actor_builder()
        half_thick = 0.0   # X 方向半长（厚度）
        half_width = 0.195    # Y 方向半长（宽度）
        half_height = 0.2  # Z 方向半长（高度）

        # 添加碰撞体
        builder.add_box_collision(half_size=[half_thick, half_width, half_height])

        # 添加视觉体（灰色半透明，也可以改成不透明）
        material = sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1.0])
        builder.add_box_visual(half_size=[half_thick, half_width, half_height], material=material)

        # 设置初始位姿（放在桌面上，底部贴地）
        builder.initial_pose = sapien.Pose(p=[-0.35, -0.2, 0.15])  # x=0.2, y=0, z=0.15 使底部在 0

        # 构建静态物体
        self.baffle = builder.build_static(name="baffle")    # 2. 添加挡板（放在桌子面上）
        self.cube = actors.build_cube(
            self.scene,
            half_size=0,
            color=[0,0,0,1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )
        # 3. 加载架子 GLB 模型
        # 3. 加载架子 GLB 模型（带旋转和缩放）
        jiazi_path = str(project_asset_path("custom", "jiazi.glb"))
        jiazi3_path = str(project_asset_path("custom", "jiazi3.glb"))
        if os.path.exists(jiazi_path):
            jiazi3_builder = self.scene.create_actor_builder()
            
            # 旋转修正：如果模型朝向不对，调整这里的角度
            rot_correction = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi / 2))  # 绕 X 轴转 90°
            scale_factor = (0.5, 0.45, 0.55)  # 均匀放大 1.5 倍
            
            jiazi3_builder.add_visual_from_file(
                jiazi3_path, 
                pose=rot_correction, 
                scale=scale_factor,
                material=sapien.render.RenderMaterial(base_color=[0.3, 0.5, 0.8, 1.0])  # 蓝色
            )
            jiazi3_builder.add_nonconvex_collision_from_file(
                jiazi3_path, 
                pose=rot_correction, 
                scale=scale_factor
            )
            
            jiazi3_builder.initial_pose = sapien.Pose(p=[-0.2, 0.0, 0.02])  # 整体位置
            self.jiazi3 = jiazi3_builder.build_static(name="jiazi3")
            print(f"架子模型加载成功: {jiazi3_path}")
            
            # 第二个架子，x 偏移 +0.2
            jiazi_builder2 = self.scene.create_actor_builder()
            jiazi_builder2.add_visual_from_file(
                jiazi_path, 
                pose=rot_correction, 
                scale=scale_factor,
                material=sapien.render.RenderMaterial(base_color=[1, 0.8, 0.2, 1.0])  # 橙色
            )
            jiazi_builder2.add_nonconvex_collision_from_file(
                jiazi_path, 
                pose=rot_correction, 
                scale=scale_factor
            )
            jiazi_builder2.initial_pose = sapien.Pose(p=[0, 0.0, 0.02])  # x 偏移 +0.2
            self.jiazi2 = jiazi_builder2.build_static(name="jiazi2")
            print(f"第二个架子加载成功: x+0.2")
        else:
            print(f"警告: 架子模型不存在 {jiazi_path}")
            # 存储加载的 RoboCasa 物体，以便后续引用（如随机化位置）
        self.robocasa_objects = []
        # RoboCasa 资产根目录（根据你的实际路径调整）
        robocasa_root = str(project_asset_path("scene_datasets", "robocasa_dataset", "assets"))
        for idx, obj_cfg in enumerate(self.ROBOCASA_OBJECTS):
            # 构建完整模型路径
            if obj_cfg.get("file"):
                model_path = os.path.join(robocasa_root, obj_cfg["path"], obj_cfg["file"])
            else:
                model_path = os.path.join(robocasa_root, obj_cfg["path"], "model.xml")
            
            if not os.path.exists(model_path):
                print(f"警告: 模型不存在 {model_path}")
                continue

            loader = self.scene.create_mjcf_loader()
            loader.visual_groups = [0, 1, 2, 3, 4, 5]
            if obj_cfg.get("scale"):
                loader.scale = obj_cfg["scale"]

            builders = loader.parse(model_path)

            if obj_cfg["type"] == "actor":
                if builders["actor_builders"]:
                    builder = builders["actor_builders"][0]
                else:
                    print(f"警告: {obj_cfg['path']} 期望 actor 但未找到")
                    continue
            elif obj_cfg["type"] == "articulation":
                if builders["articulation_builders"]:
                    builder = builders["articulation_builders"][0]
                else:
                    print(f"警告: {obj_cfg['path']} 期望 articulation 但未找到")
                    continue
            else:
                print(f"未知类型: {obj_cfg['type']}")
                continue

            # 旋转转换
            rot_rad = [np.deg2rad(r) for r in obj_cfg["rot_deg"]]
            r = R.from_euler('xyz', rot_rad)
            quat = r.as_quat()
            pose = sapien.Pose(p=obj_cfg["pos"], q=quat)
            builder.initial_pose = pose

            # 生成唯一名称
            unique_name = f"robocasa_{idx}_{obj_cfg['path'].replace('/', '_')}"
            if obj_cfg.get("file"):
                unique_name += f"_{obj_cfg['file'].replace('.xml', '')}"
            
            # 根据配置决定 body 类型：kinematic > static > dynamic
            if obj_cfg.get("kinematic", False):
                obj = builder.build_kinematic(name=unique_name)
            elif obj_cfg.get("static", False):
                obj = builder.build_static(name=unique_name)
            else:
                obj = builder.build_dynamic(name=unique_name)
            # 假设你想把某个盘子改为红色（配置中已有 color 字段）
            if "color" in obj_cfg:
                self.set_actor_color(obj, obj_cfg["color"])
            
            # 保存盘子引用供后续任务使用
            if "plate" in obj_cfg["path"]:
                print(f"  [scene] 碟子加载 idx={idx}, pos={obj_cfg['pos']}")
                if idx == 0:
                    self.plate1 = obj  # 右侧红色盘子
                    self.red_plate = obj
                    print(f"  [scene] -> 赋值给 plate1")
                elif idx == 1:
                    self.plate2 = obj  # 左侧红色盘子
                    print(f"  [scene] -> 赋值给 plate2")
                elif idx == 2:
                    self.plate3 = obj  # 中间红色盘子
                    print(f"  [scene] -> 赋值给 plate3")
            
            self.robocasa_objects.append(obj)
            print(f"加载成功: {obj_cfg['path']} -> {unique_name}")
        # 1. 定义 YCB 模型及其初始位姿
        ycb_objects = [
            # {"id": "004_sugar_box",        "pose": sapien.Pose(p=[0.5, -0.3, 0.95])},
            # {"id": "005_tomato_soup_can",  "pose": sapien.Pose(p=[0.5, 0.0, 0.95])},
            # {"id": "006_mustard_bottle",   "pose": sapien.Pose(p=[0.5, 0.3, 0.95])},
            # {"id": "017_orange",   "pose": sapien.Pose(p=[0.5, 0.5, 0.95])},
            # {"id": "018_plum",   "pose": sapien.Pose(p=[0.5, 0.7, 0.95])},
            # {"id": "019_pitcher_base",   "pose": sapien.Pose(p=[0.5, 0.9, 0.95])},
            # {"id": "021_bleach_cleanser",   "pose": sapien.Pose(p=[0.5, 1.3, 0.95])}
        ]
        # 2. 循环加载，并保存到列表
        self.ycb_objects = []  # 存储所有物体
        for obj in ycb_objects:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj['id']}")
            builder.initial_pose = obj["pose"]
            actor = builder.build(name=f"ycb_{obj['id']}")
            self.ycb_objects.append(actor)

        self.goal = actors.build_sphere(
            self.scene,
            radius=0,
            color=[0,1,0,0.5],
            name="goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose()
        )
    
        # ========== 定义任务序列 ==========
        tasks = [
            {
                "name": "static",
                "subgoal_segment": "static",
                "phase_type": "warmup",
            },
            {
                "name": "reset",
                "subgoal_segment": "reset",
                "phase_type": "reset",
            },
            {
                "name": "pick up the red plate",
                "subgoal_segment": "pick up the red plate at <>",
                "choice_label": "pick red plate",
                "phase_type": "eval",
            },
        ]
        self.task_list = tasks
    
    def _is_static_complete(self):
        return torch.tensor([True])
    
    def _is_reset_complete(self):
        return torch.tensor([True])
        
    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            # 随机化立方体位置（在桌子上方）
            xyz = torch.zeros((b,3))
            xyz[:,0] = torch.rand(b) * 0.6 - 0.3
            xyz[:,1] = torch.rand(b) * 0.4 - 0.2
            xyz[:,2] = 0.02
            self.cube.set_pose(Pose.create_from_pq(xyz))
            # 随机化目标位置
            goal_xyz = torch.zeros((b,3))
            goal_xyz[:,0] = torch.rand(b) * 0.6 - 0.3
            goal_xyz[:,1] = torch.rand(b) * 0.4 + 0.2
            goal_xyz[:,2] = 0.02
            self.goal.set_pose(Pose.create_from_pq(goal_xyz))
            
            # 重置机器人到初始姿态（与 L4 一致）
            agent: MultiAgent = self.agent
            qpos = np.array([
                0.0,   # joint1
                0.0,   # joint2
                0.0,   # joint3
                0.95,  # joint4
                0.0,   # joint5
                0.0,   # joint6
                0,     # gripper left (joint7)
                0      # gripper right (joint8)
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

    def evaluate(self, solve_complete_eval=False):
        tasks = [
            {
                "name": "static",
                "subgoal_segment": "static",
                "phase_type": "warmup",
            },
            {
                "name": "reset",
                "subgoal_segment": "reset",
                "phase_type": "reset",
            },
            {
                "name": "pick up the red plate",
                "subgoal_segment": "pick up the red plate",
                "choice_label": "pick red plate",
                "phase_type": "eval",
            },
        ]
        self.task_list = tasks
        
        dist = torch.linalg.norm(self.cube.pose.p - self.goal.pose.p, axis=1)
        is_success = dist < 0.03
        return {"task_list": tasks, "success": is_success, "fail": torch.tensor([False])}

    def _get_obs_extra(self, info):
        obs = {}
        if self.obs_mode_struct.use_state:
            obs.update(cube_pose=self.cube.pose.raw_pose, goal_pos=self.goal.pose.p)
        return obs

    def compute_dense_reward(self, obs, action, info):
        dist = torch.linalg.norm(self.cube.pose.p - self.goal.pose.p, axis=1)
        return -dist

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 1.0

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
