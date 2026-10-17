import sapien
import torch
import numpy as np
import os
import sapien
from typing import Tuple, Dict, Any
from mani_skill.utils.building import MJCFLoader
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
import glob
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as Rot

@register_env("SceneH3_04Env", max_episode_steps=200)
class SceneH3_04Env(BaseEnv):
    SUPPORTED_ROBOTS = [("piper_x", "piper_x")]
    agent: MultiAgent[Tuple[PiperX, PiperX]]
    # cube_half_size = 0
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
        # ========== 组 0：左列竖直 上→中→下 ==========
        {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [5, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [5, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [5, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [5, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 1：右列竖直 上→中→下 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 2：左上→右上→右中 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 3：左中→右中→右下 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 4：左下→右下→右中 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 5：右上→左上→左中 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 6：左上→右上→右中→左中 (4个碗) ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 7：左中→左下→右下 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},

        # # ========== 组 8：左上→左中→右中 ==========
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        # {"type": "actor", "path": "objects/objaverse/bowl/bowl_4",    "pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0], "scale": [1, 1, 1], "static": False},
        {
            "name": "tray",
            "type": "actor",
            "path": "objects/objaverse/tray/tray_1",
            "pos": [-0.35, 0, 0.006],
            "rot_deg": [90, 0, 0],
            "scale": [1.3, 1.1, 0.5],
            "color": [1, 1, 1, 1],
            "static": True
        },
    ]
    # 9 组碗初始化位姿（每组 2~3 个，对应 bowl1/bowl2/bowl3）
    PREDEFINED_BOWL_POSITIONS = [
        # 组 0：1 6 (左列竖直: 上→中→下)
        [
            {"pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0]},      # bowl1
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl2
            {"pos": [-0.2, -0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl3
        ],
        # 组 1：1 7 (右列竖直: 上→中→下)
        [
            {"pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl1
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl2
            {"pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0]},    # bowl3
        ],
        # 组 2：1 8 (左上→右上→右中)
        [
            {"pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0]},      # bowl1
            {"pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl2
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl3
        ],
        # 组 3：2 6 (左中→右中→右下)
        [
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl1
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl2
            {"pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0]},    # bowl3
        ],
        # 组 4：3 7 (左下→右下→右中)
        [
            {"pos": [-0.2, -0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl1
            {"pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0]},    # bowl2
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl3
        ],
        # 组 5：3 8 (右上→左上→左中)
        [
            {"pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl1
            {"pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0]},      # bowl2
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl3
        ],
        # 组 6：1 6 8 (左上→右上→右中→左中, 4个碗)
        [
            {"pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0]},      # bowl1
            {"pos": [-0.05, 0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl2
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl3
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl4
        ],
        # 组 7：2 7 (左中→左下→右下)
        [
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl1
            {"pos": [-0.2, -0.2, 0.05], "rot_deg": [0, 0, 0]},     # bowl2
            {"pos": [-0.05, -0.2, 0.05], "rot_deg": [0, 0, 0]},    # bowl3
        ],
        # 组 8：2 8 (左上→左中→右中)
        [
            {"pos": [-0.2, 0.2, 0.05], "rot_deg": [0, 0, 0]},      # bowl1
            {"pos": [-0.2, 0, 0.05], "rot_deg": [0, 0, 0]},        # bowl2
            {"pos": [-0.05, 0, 0.05], "rot_deg": [0, 0, 0]},       # bowl3
            {"pos": [-0.15, -0.2, 0.05], "rot_deg": [0, 0, 0]},       # bowl4
            
        ],
    ]
    # 目标位置（每组一个 [x, y, z]，对应架子上的槽位）
    PREDEFINED_TARGET_POSITIONS = [
        [-0.298, 0.298, 0.046],
        [-0.298, 0.298, 0.1],
        [-0.298, 0.298, 0.146],
        [-0.298, -0.302, 0.046],
        [-0.298, -0.302, 0.1],
        [-0.298, -0.302, 0.146],
    ]

    def set_actor_color(self, actor, rgba):
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

    def set_articulation_color(self, articulation, rgba):
        if hasattr(articulation, '_objs'):
            sapien_artic = articulation._objs[0]
            for link in sapien_artic.get_links():
                self.set_actor_color(link, rgba)
        else:
            for link in articulation.get_links():
                self.set_actor_color(link, rgba)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        # # 添加挡板
        # builder = self.scene.create_actor_builder()
        # half_thick = 0.0
        # half_width = 0.195
        # half_height = 0.2
        # builder.add_box_collision(half_size=[half_thick, half_width, half_height])
        # material = sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1.0])
        # builder.add_box_visual(half_size=[half_thick, half_width, half_height], material=material)
        # builder.initial_pose = sapien.Pose(p=[-0.35, -0.2, 0.15])
        # self.baffle = builder.build_static(name="baffle")

        # self.cube = actors.build_cube(
        #     self.scene, half_size=0, color=[0,0,0,1],
        #     name="cube", initial_pose=sapien.Pose(p=[0, 0, 0.02])
        # )
        self.robocasa_objects = []
        robocasa_root = str(project_asset_path("scene_datasets", "robocasa_dataset", "assets"))
        for idx, obj_cfg in enumerate(self.ROBOCASA_OBJECTS):
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

            rot_rad = [np.deg2rad(r) for r in obj_cfg["rot_deg"]]
            r = R.from_euler('xyz', rot_rad)
            quat = r.as_quat()
            pose = sapien.Pose(p=obj_cfg["pos"], q=quat)
            builder.initial_pose = pose

            unique_name = f"robocasa_{idx}_{obj_cfg['path'].replace('/', '_')}"
            if obj_cfg.get("file"):
                unique_name += f"_{obj_cfg['file'].replace('.xml', '')}"

            if obj_cfg.get("kinematic", False):
                obj = builder.build_kinematic(name=unique_name)
            elif obj_cfg.get("static", False):
                obj = builder.build_static(name=unique_name)
            else:
                obj = builder.build_dynamic(name=unique_name)
            if "color" in obj_cfg:
                self.set_actor_color(obj, obj_cfg["color"])

            # 保存碗引用供后续任务使用
            if "bowl" in obj_cfg["path"]:
                print(f"  [scene] 碗加载 idx={idx}, pos={obj_cfg['pos']}")
                if idx == 0:
                    self.bowl1 = obj
                    self.red_bowl = obj
                    print(f"  [scene] -> 赋值给 bowl1")
                elif idx == 1:
                    self.bowl2 = obj
                    print(f"  [scene] -> 赋值给 bowl2")
                elif idx == 2:
                    self.bowl3 = obj
                    print(f"  [scene] -> 赋值给 bowl3")

            # 保存托盘引用
            if "tray" in obj_cfg.get("name", "") or "tray" in obj_cfg["path"]:
                self.tray = obj
                print(f"  [scene] -> 赋值给 self.tray")

            self.robocasa_objects.append(obj)
            print(f"加载成功: {obj_cfg['path']} -> {unique_name}")

        ycb_objects = []
        self.ycb_objects = []
        for obj in ycb_objects:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{obj['id']}")
            builder.initial_pose = obj["pose"]
            actor = builder.build(name=f"ycb_{obj['id']}")
            self.ycb_objects.append(actor)

        self.goal = actors.build_sphere(
            self.scene, radius=0, color=[0,1,0,0.5],
            name="goal", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose()
        )

        # ========== 定义任务序列 ==========
        tasks = [
            {
                "name": "static", "subgoal_segment": "static",
                "phase_type": "warmup",
            },
            {
                "name": "reset", "subgoal_segment": "reset",
                "phase_type": "reset",
            },
            {
                "name": "pick up the red bowl",
                "subgoal_segment": "pick up the red bowl at <>",
                "choice_label": "pick red bowl",
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
            agent: MultiAgent = self.agent
            qpos = np.array([0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0, 0])
            qpos = np.tile(qpos, (b, 1))
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(sapien.Pose(self._cfg["left_robot_pose"]))
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(sapien.Pose(self._cfg["right_robot_pose"]))
            for arm_idx in [0, 1]:
                arm_agent = agent.agents[arm_idx]
                if hasattr(arm_agent, 'controller'):
                    arm_agent.controller.reset()

    def _is_bowl_picked(self, color="red"):
        return torch.tensor([True])

    def _check_pick_failure(self):
        return torch.tensor([False])

    def evaluate(self, solve_complete_eval=False):
        tasks = [
            {
                "name": "static", "subgoal_segment": "static",
                "phase_type": "warmup",
            },
            {
                "name": "reset", "subgoal_segment": "reset",
                "phase_type": "reset",
            },
            {
                "name": "pick up the red bowl",
                "subgoal_segment": "pick up the red bowl",
                "choice_label": "pick red bowl",
                "phase_type": "eval",
            },
        ]
        self.task_list = tasks
        return {"task_list": tasks, "success": torch.tensor([True]), "fail": torch.tensor([False])}

    def _get_obs_extra(self, info):
        obs = {}
        return obs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 1.0

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return [CameraConfig("human_cam", pose, self.base_camera_width, self.base_camera_height, self.sensor_fovy, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.human_cam_eye_pos, self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, self.sensor_fovy, 0.01, 100)
