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
from mani_skill.utils.structs.types import SceneConfig, SimConfig
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

# ===== 架子位置（模块级常量，修改此处即可整体移动） =====
_BLUE_RACK_POS = [-0.1, 0.25, 0.02]      # 蓝色架子 (jiazi3)
_ORANGE_RACK_POS = [0, 0.0, 0.02]       # 橙色架子 (jiazi2)
_SHELF_POS = [-0.05, -0.2, 0.03]       # 桌子 (guizi)

# ===== 碟子/目标位置偏移量（相对于对应架子中心） =====
_PLATE_X_OFFSET = -0.1    # 碟子 x = 蓝架.x + 此值 = -0.25
_PLATE_Y_OFFSET = +0.05      # 碟子 y = 蓝架.y + 此值 = 0.25
_PLATE_Z_OFFSET = 0.10     # 碟子 z = 蓝架.z + 此值 = 0.12
_TARGET_X_OFFSET = 0.15   # 目标 x = 橙架.x + 此值 = -0.08
_TARGET_Z_OFFSET = 0.10    # 目标 z = 橙架.z + 此值 = 0.12

# ===== 架子姿态（旋转、缩放） =====
# 架子旋转使用 transforms3d.euler2quat('sxyz') 约定
_RACK_ROT_DEG = [90, 0, 0]         # 架子 GLB 模型旋转（欧拉角 deg）
_RACK_SCALE = (0.5, 0.45, 0.55)     # 架子缩放
_SHELF_SCALE = (0.1, 0.175, 0.12)     # 桌子缩放

import glob

# ===== 碟子旋转（以架子姿态为基准） =====
_PLATE_ROT_DEG = [90, 0, -90]        # 碟子旋转（欧拉角 deg）
_PLATE_ROT_DEG_02 = [90, 0, 0]        # 碟子旋转（欧拉角 deg）
plate_scale=[0.15,0.15,0.5]
bili=1
plate_collision_scale=[plate_scale[0]*bili,plate_scale[1]*bili,plate_scale[2]*bili]
# 自动检测碰撞体文件数量
_collision_dir = str(project_asset_path("scene_datasets", "robocasa_dataset", "assets", "objects", "objaverse", "plate", "plate_7", "collision", "new_coll01"))
_collision_files = sorted(glob.glob(f"{_collision_dir}/*.obj"))
_num_collision_parts = len(_collision_files)
print(f"[scene] 检测到 {_num_collision_parts} 个碰撞体文件")
# 各槽位的 y 偏移（相对于架子中心 y=0）
# 槽位编号: p1=0.15, p2=0.12, p3=0.09, p4=-0.09, p5=-0.12, p6=-0.17
plate_static=False
def _make_plate_pos(y_offsets):
    """根据 y 偏移列表生成 plate 位置列表"""
    return [
        {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET,
                 _BLUE_RACK_POS[1] + y,
                 _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
         "rot_deg": _PLATE_ROT_DEG}
        for y in y_offsets
    ]

rack_dis_01=-0.02
rack_dis_02=0.09
rack_dis_03=0.18

# 数据编号
# --layout-id	layout_id 列（已更新为 layout_01~09）	提取数字部分，并减1：layout_01 → 0, layout_02 → 1, ... layout_09 → 8
# --pass-id	从 history_events_to_execute_zh 中提取“取出 BowlX”	Bowl1 → 0, Bowl2 → 1, Bowl3 → 2, Bowl4 → 3
# --place-01	repeat_id 列（如 repeat_01）	提取数字部分，减1：repeat_01 → 0, repeat_02 → 1, repeat_03 → 2
# --grasp-id	target_object_ids 列（如 Bowl1）	同 pass-id 转换规则
# --mode
TASK_SEQUENCES_240 = [
    (0, 0, 0, 0, 0, 0),
    (1, 3, 1, 1, 1, 1),
    (2, 6, 2, 2, 2, 0),
    (3, 0, 0, 0, 1, 1),
    (4, 3, 1, 1, 1, 0),
    (5, 6, 2, 2, 2, 1),
    (6, 0, 0, 0, 0, 0),
    (7, 3, 1, 1, 2, 1),
    (8, 6, 2, 2, 2, 0),
    (9, 0, 0, 0, 0, 1),
    (10, 3, 1, 1, 1, 0),
    (11, 6, 2, 2, 0, 1),
    (12, 0, 0, 0, 0, 0),
    (13, 3, 1, 1, 1, 1),
    (14, 6, 2, 2, 2, 0),
    (15, 0, 0, 0, 1, 1),
    (16, 3, 1, 1, 1, 0),
    (17, 6, 2, 2, 2, 1),
    (18, 0, 0, 0, 0, 0),
    (19, 3, 1, 1, 2, 1),
    (20, 6, 2, 2, 2, 0),
    (21, 0, 0, 0, 0, 1),
    (22, 3, 1, 1, 1, 0),
    (23, 6, 2, 2, 0, 1),
    (24, 0, 0, 0, 0, 0),
    (25, 3, 1, 1, 1, 1),
    (26, 6, 2, 2, 2, 0),
    (27, 0, 0, 0, 1, 1),
    (28, 3, 1, 1, 1, 0),
    (29, 6, 2, 2, 2, 1),
    (30, 0, 0, 0, 0, 0),
    (31, 3, 1, 1, 2, 1),
    (32, 6, 2, 2, 2, 0),
    (33, 0, 0, 0, 0, 1),
    (34, 3, 1, 1, 1, 0),
    (35, 6, 2, 2, 0, 1),
    (36, 0, 0, 0, 0, 0),
    (37, 3, 1, 1, 1, 1),
    (38, 6, 2, 2, 2, 0),
    (39, 0, 0, 0, 1, 1),
    (40, 3, 1, 1, 1, 0),
    (41, 6, 2, 2, 2, 1),
    (42, 0, 0, 0, 0, 0),
    (43, 3, 1, 1, 2, 1),
    (44, 6, 2, 2, 2, 0),
    (45, 0, 0, 0, 0, 1),
    (46, 3, 1, 1, 1, 0),
    (47, 6, 2, 2, 0, 1),
    (48, 0, 0, 0, 0, 0),
    (49, 3, 1, 1, 1, 1),
    (50, 6, 2, 2, 2, 0),
    (51, 0, 0, 0, 1, 1),
    (52, 3, 1, 1, 1, 0),
    (53, 6, 2, 2, 2, 1),
    (54, 0, 0, 0, 0, 0),
    (55, 3, 1, 1, 2, 1),
    (56, 6, 2, 2, 2, 0),
    (57, 0, 0, 0, 0, 1),
    (58, 3, 1, 1, 1, 0),
    (59, 6, 2, 2, 0, 1),
    (60, 0, 0, 0, 0, 0),
    (61, 3, 1, 1, 1, 1),
    (62, 6, 2, 2, 2, 0),
    (63, 0, 0, 0, 1, 1),
    (64, 3, 1, 1, 1, 0),
    (65, 6, 2, 2, 2, 1),
    (66, 0, 0, 0, 0, 0),
    (67, 3, 1, 1, 2, 1),
    (68, 6, 2, 2, 2, 0),
    (69, 0, 0, 0, 0, 1),
    (70, 3, 1, 1, 1, 0),
    (71, 6, 2, 2, 0, 1),
    (72, 0, 0, 0, 0, 0),
    (73, 3, 1, 1, 1, 1),
    (74, 6, 2, 2, 2, 0),
    (75, 0, 0, 0, 1, 1),
    (76, 3, 1, 1, 1, 0),
    (77, 6, 2, 2, 2, 1),
    (78, 0, 0, 0, 0, 0),
    (79, 3, 1, 1, 2, 1),
    (80, 0, 0, 0, 0, 0),
    (81, 4, 1, 1, 1, 0),
    (82, 4, 1, 2, 1, 1),
    (83, 7, 2, 0, 2, 1),
    (84, 7, 2, 1, 2, 0),
    (85, 1, 0, 2, 0, 0),
    (86, 1, 0, 0, 0, 0),
    (87, 4, 1, 1, 1, 0),
    (88, 4, 1, 2, 1, 1),
    (89, 7, 2, 0, 2, 1),
    (90, 7, 2, 1, 2, 0),
    (91, 1, 0, 2, 0, 0),
    (92, 1, 0, 0, 0, 0),
    (93, 4, 1, 1, 1, 0),
    (94, 4, 1, 2, 1, 1),
    (95, 7, 2, 0, 2, 1),
    (96, 7, 2, 1, 2, 0),
    (97, 1, 0, 2, 0, 0),
    (98, 1, 0, 0, 0, 0),
    (99, 4, 1, 1, 1, 0),
    (100, 4, 1, 2, 1, 1),
    (101, 7, 2, 0, 2, 1),
    (102, 7, 2, 1, 2, 0),
    (103, 1, 0, 2, 0, 0),
    (104, 1, 0, 0, 0, 0),
    (105, 4, 1, 1, 1, 0),
    (106, 4, 1, 2, 1, 1),
    (107, 7, 2, 0, 2, 1),
    (108, 7, 2, 1, 2, 0),
    (109, 1, 0, 2, 0, 0),
    (110, 1, 0, 0, 0, 0),
    (111, 4, 1, 1, 1, 0),
    (112, 4, 1, 2, 1, 1),
    (113, 7, 2, 0, 2, 1),
    (114, 7, 2, 1, 2, 0),
    (115, 1, 0, 2, 0, 0),
    (116, 1, 0, 0, 0, 0),
    (117, 4, 1, 1, 1, 0),
    (118, 4, 1, 2, 1, 1),
    (119, 7, 2, 0, 2, 1),
    (120, 7, 2, 1, 2, 0),
    (121, 1, 0, 2, 0, 0),
    (122, 1, 0, 0, 0, 0),
    (123, 4, 1, 1, 1, 0),
    (124, 4, 1, 2, 1, 1),
    (125, 7, 2, 0, 2, 1),
    (126, 7, 2, 1, 2, 0),
    (127, 1, 0, 2, 0, 0),
    (128, 1, 0, 0, 0, 0),
    (129, 4, 1, 1, 1, 0),
    (130, 4, 1, 2, 1, 1),
    (131, 7, 2, 0, 2, 1),
    (132, 7, 2, 1, 2, 0),
    (133, 1, 0, 2, 0, 0),
    (134, 1, 0, 0, 0, 0),
    (135, 4, 1, 1, 1, 0),
    (136, 4, 1, 2, 1, 1),
    (137, 7, 2, 0, 2, 1),
    (138, 7, 2, 1, 2, 0),
    (139, 1, 0, 2, 0, 0),
    (140, 1, 0, 0, 0, 0),
    (141, 4, 1, 1, 1, 0),
    (142, 4, 1, 2, 1, 1),
    (143, 7, 2, 0, 2, 1),
    (144, 7, 2, 1, 2, 0),
    (145, 1, 0, 2, 0, 0),
    (146, 1, 0, 0, 0, 0),
    (147, 4, 1, 1, 1, 0),
    (148, 4, 1, 2, 1, 1),
    (149, 7, 2, 0, 2, 1),
    (150, 7, 2, 1, 2, 0),
    (151, 1, 0, 2, 0, 0),
    (152, 1, 0, 0, 0, 0),
    (153, 4, 1, 1, 1, 0),
    (154, 4, 1, 2, 1, 1),
    (155, 7, 2, 0, 2, 1),
    (156, 7, 2, 1, 2, 0),
    (157, 1, 0, 2, 0, 0),
    (158, 1, 0, 0, 0, 0),
    (159, 4, 1, 1, 1, 0),
    (160, 1, 0, 0, 0, 0),
    (161, 4, 1, 1, 1, 1),
    (162, 8, 2, 2, 2, 0),
    (163, 1, 0, 0, 1, 1),
    (164, 5, 1, 1, 1, 0),
    (165, 8, 2, 2, 2, 1),
    (166, 2, 0, 0, 0, 0),
    (167, 5, 1, 1, 2, 1),
    (168, 8, 2, 2, 2, 0),
    (169, 2, 0, 0, 0, 1),
    (170, 2, 0, 0, 0, 0),
    (171, 5, 1, 1, 1, 0),
    (172, 5, 1, 2, 1, 1),
    (173, 8, 2, 0, 2, 1),
    (174, 8, 2, 1, 2, 0),
    (175, 2, 0, 2, 0, 0),
    (176, 2, 0, 0, 0, 0),
    (177, 5, 1, 1, 1, 0),
    (178, 5, 1, 2, 1, 1),
    (179, 8, 2, 0, 2, 1),
    (180, 2, 0, 0, 0, 0),
    (181, 5, 1, 1, 1, 1),
    (182, 8, 2, 2, 2, 0),
    (183, 2, 0, 0, 1, 1),
    (184, 5, 1, 1, 1, 0),
    (185, 8, 2, 2, 2, 1),
    (186, 2, 0, 0, 0, 0),
    (187, 5, 1, 1, 2, 1),
    (188, 8, 2, 2, 2, 0),
    (189, 2, 0, 0, 0, 1),
    (190, 5, 1, 1, 1, 0),
    (191, 8, 2, 2, 0, 1),
    (192, 2, 0, 0, 0, 0),
    (193, 5, 1, 1, 1, 1),
    (194, 8, 2, 2, 2, 0),
    (195, 2, 0, 0, 1, 1),
    (196, 5, 1, 1, 1, 0),
    (197, 8, 2, 2, 2, 1),
    (198, 2, 0, 0, 0, 0),
    (199, 5, 1, 1, 2, 1),
    (200, 2, 0, 0, 0, 0),
    (201, 5, 1, 1, 1, 1),
    (202, 8, 2, 2, 2, 0),
    (203, 8, 3, 0, 0, 1),
    (204, 5, 1, 1, 1, 0),
    (205, 8, 2, 2, 2, 1),
    (206, 2, 0, 0, 0, 0),
    (207, 5, 1, 1, 2, 1),
    (208, 2, 0, 2, 0, 0),
    (209, 2, 0, 0, 0, 1),
    (210, 2, 0, 0, 0, 0),
    (211, 5, 1, 1, 1, 0),
    (212, 5, 1, 2, 1, 1),
    (213, 8, 2, 0, 2, 1),
    (214, 8, 2, 1, 2, 0),
    (215, 2, 0, 2, 0, 0),
    (216, 2, 0, 0, 0, 0),
    (217, 5, 1, 1, 1, 0),
    (218, 5, 1, 2, 1, 1),
    (219, 8, 2, 0, 2, 1),
    (220, 8, 2, 1, 2, 0),
    (221, 2, 0, 2, 0, 0),
    (222, 2, 0, 0, 0, 0),
    (223, 5, 1, 1, 1, 0),
    (224, 5, 1, 2, 1, 1),
    (225, 8, 2, 0, 2, 1),
    (226, 8, 2, 1, 2, 0),
    (227, 2, 0, 2, 0, 0),
    (228, 2, 0, 0, 0, 0),
    (229, 5, 1, 1, 1, 0),
    (230, 5, 1, 2, 1, 1),
    (231, 8, 2, 0, 2, 1),
    (232, 8, 2, 1, 2, 0),
    (233, 2, 0, 2, 0, 0),
    (234, 2, 0, 0, 0, 0),
    (235, 5, 1, 1, 1, 0),
    (236, 5, 1, 2, 1, 1),
    (237, 8, 2, 0, 2, 1),
    (238, 8, 2, 1, 2, 0),
    (239, 2, 0, 2, 0, 0),
]
PREDEFINED_PLATE_POSITIONS = [
        # ── 组 0: p1, p3, p5 (3 个碟子) ──
        [
             {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
             {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 1: p1, p2, p4, p6 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
              {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 2: p1, p3, p6 (3 个碟子) ──
        [
             {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.3, -0.2, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
             {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 3: p2, p3, p5 (3 个碟子) ──
        [
            {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.4, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 4: p1, p2, p3, p4 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 5: p1, p4, p5, p6 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.3, -0.2, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
        ],
        # ── 组 6: p2, p4, p6 (3 个碟子) ──
        [
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            
        ],
        # ── 组 7: p3, p4, p5, p6 (4 个碟子) ──
        [
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 8: p1, p2, p5, p6 (4 个碟子) ──
        [
            {"pos": [-0.3, -0.2, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
    ]

@register_env("SceneH5_02Env", max_episode_steps=200)
class SceneH5_02Env(BaseEnv):
    SUPPORTED_ROBOTS = [("piper_x", "piper_x")]
    agent: MultiAgent[Tuple[PiperX, PiperX]]
    cube_half_size = 0
    goal_thresh = 0
    PREDEFINED_PLATE_POSITIONS = [
        # ── 组 0: p1, p3, p5 (3 个碟子) ──
        [
             {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
             {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 1: p1, p2, p4, p6 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
              {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 2: p1, p3, p6 (3 个碟子) ──
        [
             {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            # [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],
            {"pos": [-0.33, -0.2, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
             {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 3: p2, p3, p5 (3 个碟子) ──
        [
            {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 4: p1, p2, p3, p4 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 5: p1, p4, p5, p6 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.33, -0.2, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
        ],
        # ── 组 6: p2, p4, p6 (3 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
        ],
        # ── 组 7: p3, p4, p5, p6 (4 个碟子) ──
        [
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            #  {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
        # ── 组 8: p1, p2, p5, p6 (4 个碟子) ──
        [
            {"pos": [-0.35, -0.15, 0.04], "rot_deg": _PLATE_ROT_DEG_02},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_02, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos": [_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_03, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
            {"pos":[_BLUE_RACK_POS[0] + _PLATE_X_OFFSET+rack_dis_01, _BLUE_RACK_POS[1] + _PLATE_Y_OFFSET, _BLUE_RACK_POS[2] + _PLATE_Z_OFFSET],"rot_deg": _PLATE_ROT_DEG},  # p5
        ],
    ]
    layout_id=0
    def __init__(
        self, 
        *args, 
        robot_uids=("piper_x", "piper_x"), 
        robot_init_qpos_noise=None,
        scene_name="default",
        layout_id=None,  # 新增这一行
        **kwargs
    ):
        # 如果传入了 layout_id，使用传入的值；否则使用类属性默认值
        if layout_id is not None:
            self.layout_id = layout_id
        else:
            self.layout_id = 0
        print(f"  [debug] SceneH5_02Env 初始化: layout_id = {self.layout_id}")
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
       

    # ===== 架子位置（修改此处即可整体移动） =====
    BLUE_RACK_POS = _BLUE_RACK_POS      # 蓝色架子 (jiazi3)
    ORANGE_RACK_POS = _ORANGE_RACK_POS   # 橙色架子 (jiazi2)

    # ===== 碟子/目标位置偏移量（相对于对应架子中心） =====
    PLATE_X_OFFSET = _PLATE_X_OFFSET
    PLATE_Z_OFFSET = _PLATE_Z_OFFSET
    TARGET_X_OFFSET = _TARGET_X_OFFSET
    TARGET_Z_OFFSET = _TARGET_Z_OFFSET

    @property
    def ROBOCASA_OBJECTS(self):
        """根据当前 layout 的 PREDEFINED_PLATE_POSITIONS 动态生成碟子列表"""
        plates = []
        layout_plates = PREDEFINED_PLATE_POSITIONS[self.layout_id]
        
        for idx, plate_cfg in enumerate(layout_plates):
            # 支持两种格式：列表 [x,y,z] 或字典 {"pos": [...], "rot_deg": ...}
            if isinstance(plate_cfg, dict):
                pos = plate_cfg.get("pos", [0, 0, 0])
                rot_deg = plate_cfg.get("rot_deg", [0, 0, 0])
            else:
                pos = plate_cfg
                rot_deg = [0, 0, 0]
            
            plates.append({
                "type": "actor",
                "model_path": str(project_asset_path("scene_datasets", "robocasa_dataset", "assets", "objects", "objaverse", "plate", "plate_7")),
                "visual_file": "visual/new.obj",
                "collision_parts": [
                    {"file": f"collision/new_coll01/output_{i:03d}.obj", "scale": plate_collision_scale}
                    for i in range(_num_collision_parts - 1)
                ],
                "pos": pos,
                "rot_deg": rot_deg,
                "scale": plate_scale,
                "static": plate_static,
            })
        
        return plates

    # ===== 9 组碟子放置位姿（世界坐标），每组 3-4 个碟子 =====
    # 槽位编号: p1=0.15, p2=0.12, p3=0.09, p4=-0.09, p5=-0.12, p6=-0.17 (y 偏移)
    # 所有坐标已计算出世界坐标（基于 _BLUE_RACK_POS=[-0.05, 0.25, 0.02]）
    # x = -0.05 + (-0.07) = -0.12, z = 0.02 + 0.10 = 0.12
    # y = 0.25 + y_offset

    # # ===== 6 组目标位置（相对于橙色架子） =====
    # PREDEFINED_TARGET_POSITIONS = [
    #     [_ORANGE_RACK_POS[0] + _TARGET_X_OFFSET, y, _ORANGE_RACK_POS[2] + _TARGET_Z_OFFSET]
    #     for y in [0.13, 0.06, 0.02, -0.02, -0.06, -0.13]
    # ]
    
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
        self.cube = actors.build_cube(
            self.scene,
            half_size=0,
            color=[0,0,0,1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )
        # 3. 加载架子 GLB 模型
        jiazi_path = str(project_asset_path("custom", "jiazi5.glb"))
        if os.path.exists(jiazi_path):
            jiazi_builder = self.scene.create_actor_builder()
            
            # 旋转修正：引用模块级常量
            rot_correction = sapien.Pose(q=euler2quat(
                np.deg2rad(_RACK_ROT_DEG[0]),
                np.deg2rad(_RACK_ROT_DEG[1]),
                np.deg2rad(_RACK_ROT_DEG[2])
            ))
            scale_factor = _RACK_SCALE
            
            jiazi_builder.add_visual_from_file(
                jiazi_path, 
                pose=rot_correction, 
                scale=scale_factor,
                material=sapien.render.RenderMaterial(base_color=[0.3, 0.5, 0.8, 1.0])  # 蓝色
            )
            jiazi_builder.add_nonconvex_collision_from_file(
                jiazi_path, 
                pose=rot_correction, 
                scale=scale_factor
            )
            
            jiazi_builder.initial_pose = sapien.Pose(p=_BLUE_RACK_POS)  # 整体位置
            self.jiazi = jiazi_builder.build_static(name="jiazi")
            print(f"架子模型加载成功: {jiazi_path}")
            
            # # 第二个架子，x 偏移 +0.2
            # jiazi_builder2 = self.scene.create_actor_builder()
            # jiazi_builder2.add_visual_from_file(
            #     jiazi_path, 
            #     pose=rot_correction, 
            #     scale=scale_factor,
            #     material=sapien.render.RenderMaterial(base_color=[1, 0.8, 0.2, 1.0])  # 橙色
            # )
            # jiazi_builder2.add_nonconvex_collision_from_file(
            #     jiazi_path, 
            #     pose=rot_correction, 
            #     scale=scale_factor
            # )
            # jiazi_builder2.initial_pose = sapien.Pose(p=[0, 0.0, 0.02])  # x 偏移 +0.2
            # self.jiazi2 = jiazi_builder2.build_static(name="jiazi2")
            # print(f"第二个架子加载成功: x+0.2")
        else:
            print(f"警告: 架子模型不存在 {jiazi_path}")
            # 存储加载的 RoboCasa 物体，以便后续引用（如随机化位置）

        guizi_path = str(project_asset_path("custom", "guizi.glb"))
        if os.path.exists(guizi_path):
            guizi_builder = self.scene.create_actor_builder()
            
            # 旋转修正：引用模块级常量
            rot_correction = sapien.Pose(q=euler2quat(
                np.deg2rad(_RACK_ROT_DEG[0]),
                np.deg2rad(_RACK_ROT_DEG[1]),
                np.deg2rad(_RACK_ROT_DEG[2])
            ))
            scale_factor = _SHELF_SCALE
            
            guizi_builder.add_visual_from_file(
                guizi_path, 
                pose=rot_correction, 
                scale=scale_factor,
                material=sapien.render.RenderMaterial(base_color=[1, 0.8, 0.2, 1.0])  # 橙色
            )
            guizi_builder.add_nonconvex_collision_from_file(
                guizi_path, 
                pose=rot_correction, 
                scale=scale_factor
            )
            
            guizi_builder.initial_pose = sapien.Pose(p=_SHELF_POS)  # 整体位置
            self.guizi = guizi_builder.build_static(name="guizi")
            print(f"柜子模型加载成功: {guizi_path}")

        self.robocasa_objects = []
        # RoboCasa 资产根目录（根据你的实际路径调整）
        robocasa_root = str(project_asset_path("scene_datasets", "robocasa_dataset", "assets"))
        for idx, obj_cfg in enumerate(self.ROBOCASA_OBJECTS):
            # 检查是否使用新的加载方式（model_path + visual_file + collision_file/collision_parts）
            has_model_path = obj_cfg.get("model_path")
            has_visual = obj_cfg.get("visual_file")
            has_collision = obj_cfg.get("collision_file") or obj_cfg.get("collision_parts")
            
            if has_model_path and has_visual and has_collision:
                # 使用类似 L5_02 的加载方式
                builder = self.scene.create_actor_builder()
                
                visual_path = os.path.join(obj_cfg["model_path"], obj_cfg["visual_file"])
                
                if not os.path.exists(visual_path):
                    print(f"警告: 视觉文件不存在 {visual_path}")
                
                # 添加碰撞体
                if obj_cfg.get("collision_parts"):
                    # 组合碰撞体：多个凸形组合
                    for part_cfg in obj_cfg["collision_parts"]:
                        part_path = os.path.join(obj_cfg["model_path"], part_cfg["file"])
                        builder.add_multiple_convex_collisions_from_file(
                            filename=part_path,
                            scale=part_cfg.get("scale", [1, 1, 1])
                        )
                    print(f"  [scene] 使用组合碰撞体: {len(obj_cfg['collision_parts'])} 个部分")
                elif obj_cfg.get("collision_file"):
                    # 单个碰撞体
                    collision_path = os.path.join(obj_cfg["model_path"], obj_cfg["collision_file"])
                    if not os.path.exists(collision_path):
                        print(f"警告: 碰撞文件不存在 {collision_path}")
                    builder.add_nonconvex_collision_from_file(
                        filename=collision_path,
                        scale=obj_cfg.get("scale", [1, 1, 1])
                    )
                    print(f"  [scene] 碰撞体加载完成")
                
                # 添加视觉
                print(f"  [debug] 视觉路径: {visual_path}")
                print(f"  [debug] 视觉文件存在: {os.path.exists(visual_path)}")
                builder.add_visual_from_file(
                    filename=visual_path,
                    scale=obj_cfg.get("scale", [1, 1, 1])
                )
                print(f"  [scene] 视觉加载完成")
            else:
                # 使用原来的 MJCF loader 方式
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
            # 如果指定了自定义碰撞体文件，加载它（暂时禁用）
            # if obj_cfg.get("collision_file"):
            #     collision_path = os.path.join(robocasa_root, obj_cfg["collision_file"])
            #     if os.path.exists(collision_path):
            #         builder.add_multiple_convex_collisions_from_file(
            #             filename=collision_path,
            #             scale=obj_cfg.get("scale", [1, 1, 1])
            #         )
            #         print(f"  [碰撞体] 已加载自定义碰撞体: {collision_path}")
            #     else:
            #         print(f"警告: 碰撞文件不存在 {collision_path}")
            # 生成唯一名称
            if obj_cfg.get("model_path"):
                # 从 model_path 中提取目录名作为标识
                model_dir = os.path.basename(obj_cfg["model_path"])
                unique_name = f"robocasa_{idx}_{model_dir}"
            else:
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
            obj_path = obj_cfg.get("model_path", obj_cfg.get("path", ""))
            if "plate" in obj_path:
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
            display_path = obj_cfg.get("model_path", obj_cfg.get("path", "unknown"))
            print(f"加载成功: {display_path} -> {unique_name}")
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
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                rest_offset=0,
                solver_position_iterations=30,   # 增大位置求解迭代次数，提高接触稳定性
                solver_velocity_iterations=4,    # 增大速度求解迭代次数
                enable_friction_every_iteration=True,  # 每轮迭代都计算摩擦
                enable_tgs=True,                 # 启用 Temporal Gauss-Seidel 求解器
            ),
        )

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
