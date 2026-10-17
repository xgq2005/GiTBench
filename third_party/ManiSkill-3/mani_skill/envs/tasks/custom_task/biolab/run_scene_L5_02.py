#!/usr/bin/env python3
"""
Scene_L5_02 数据采集脚本

功能:
  1. 创建 Scene_L5_02 环境，加载试管和架子
  2. 给定试管配置，用机械臂抓取试管，提起，放到指定位置
  3. 录制机械臂状态、相机图像并保存到 HDF5

用法:
  python run_scene_L5_02.py [--tube-idx 0] [--target-pos x y z] [--target-rot r p y] [--output-name NAME]

示例:
  python run_scene_L5_02.py \
      --tube-idx 0 \
      --target-pos -0.15 0.0 0.15 \
      --target-rot 90 90 0 \
      --output-name tube_placement_001
"""

import os
import sys
import argparse
import shutil
import gymnasium as gym
import numpy as np
import torch
import sapien
import time
import threading
import queue
import gc
from pathlib import Path
from datetime import datetime
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation as R

# 导入 biolab 场景
import mani_skill.envs.tasks.custom_task.biolab.Scene_L5_02

# 导入数据记录器和运动规划工具
from mani_skill.envs.tasks.custom_task.motion_planning_utils import (
    find_reachable_pose,
    combine_quaternions,
    compute_grasp_pose,
    move_to_position_only,
)
from mani_skill.envs.tasks.custom_task.motion_planning_utils_02 import move_with_solver, PiperXArmMotionPlanningSolver
from mani_skill.envs.tasks.custom_task.biolab.Scene_L5_02 import (
    Scene_L5_02,
    cube_origin_place,
    cube_place_pos,
    cube_place02_pos,
    cube_place03_pos,
    LAYOUT_8,
    REPEAT_4,
    tube_scale,
    control_cube_static
)

# ========== 左手（机械臂0）提起试管后的交接姿态 ==========
# 定义：欧拉角（roll, pitch, yaw） → 计算为四元数 [x, y, z, w]
# LIFT_TUBE_ROT_DEG = [0, 0, 0]
# _LIFT_TUBE_ROT_RAD = [np.deg2rad(r) for r in LIFT_TUBE_ROT_DEG]
# lift_tube_quat = R.from_euler('xyz', _LIFT_TUBE_ROT_RAD).as_quat().astype(np.float32)
LIFT_TUBE_ROT_DEG = [90, 0, 0]  # 绕X轴旋转90°，和默认夹取姿态一致
_LIFT_TUBE_ROT_RAD = [np.deg2rad(r) for r in LIFT_TUBE_ROT_DEG]
lift_tube_quat = R.from_euler('xyz', _LIFT_TUBE_ROT_RAD).as_quat().astype(np.float32)
right_pass_rot_deg=[0,0,-45]
MAX_Z_HEIGHT = 0.3
place_height = 0.275
NUM_DESCENT_STEPS = 5
GRIP_FRICTION = 10.0  # 夹具与物体间的摩擦系数（建议范围: 0.5-10.0）
# ========== 全局夹爪抓取参数 ==========
GRIPPER_STIFFNESS = 100000   # 夹爪刚度（提高可增加摩擦力）
GRIPPER_DAMPING = 100000      # 夹爪阻尼（提高可减少晃动）
GRIPPER_FORCE_LIMIT =50000  # 夹爪最大力（提高可防止滑落）
GRIPPER_CLOSE_POS = -1.2    # 夹爪闭合位置（更负=更闭合）
GRIPPER_CLOSE_STEPS = 50   # 夹爪闭合步数（增加可让抓取更稳定）
STABILIZE_WAIT_STEPS = 50  # 抓取后稳定等待步数
MOVE_MAX_TRAJECTORY_STEPS = 250  # move_to_position_only 最大轨迹步数
# ========== 调试可视化函数 ==========
_debug_grasp_visuals = {}  # 缓存已创建的视觉标记，避免重复创建
_ee_frame_visual = None  # 末端坐标系可视化对象
check_height = -0.03  # 检查高度误差
# 交接
right_height = 0.06
grasp_height=0.038
STABILIZE_GRASP_STEPS = 50  # 右臂稳定夹住等待步数（增加等待时间）
# ========== 步骤4: 分步小幅度提起（每次0.04m），共提0.2m ==========
STEP_LIFT_HEIGHT = 0.05  # 每步提起高度
NUM_LIFT_STEPS = 4      # 提起步数（0.04m × 5 = 0.2m）
def debug_show_grasp_pose(env, position, quat_scipy, name="grasp_debug", wait_key=True):
    """
    在场景中显示目标抓取位姿的可视化标记（夹爪模型 + 坐标轴）

    Args:
        env: 环境
        position: 位置 [x, y, z]
        quat_scipy: 四元数 [x, y, z, w]（scipy 格式）
        name: 标记名称（用于缓存，避免重复创建）
        wait_key: 是否等待按键 [c] 继续
    """
    import sapien
    from transforms3d import quaternions as tq

    scene = env.unwrapped.scene
    
    # 如果标记已存在，直接更新位姿
    if name in _debug_grasp_visuals:
        marker = _debug_grasp_visuals[name]
        # scipy [x,y,z,w] → sapien [w,x,y,z]
        sapien_q = [float(quat_scipy[3]), float(quat_scipy[0]), float(quat_scipy[1]), float(quat_scipy[2])]
        marker.set_pose(sapien.Pose(p=[float(position[0]), float(position[1]), float(position[2])], q=sapien_q))
    else:
        # 创建夹爪形状的可视化标记
        builder = scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose())
        
        # 中心球体
        builder.add_sphere_visual(
            radius=0.015,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.8])
        )
        # 夹爪主体（绿色方块）
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.06]),
            half_size=[0.01, 0.01, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        # 夹爪手指（两个蓝色/红色方块）
        finger_width = 0.04
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, finger_width + 0.01, -0.04]),
            half_size=[0.008, 0.01, 0.015],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, -finger_width - 0.01, -0.04]),
            half_size=[0.008, 0.01, 0.015],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        # 坐标轴：X-红, Y-绿, Z-蓝（沿三个方向伸出的小棒）
        axis_len = 0.06
        axis_thick = 0.003
        for axis, color in [
            ([axis_len, 0, 0], [1, 0, 0, 0.8]),
            ([0, axis_len, 0], [0, 1, 0, 0.8]),
            ([0, 0, axis_len], [0, 0, 1, 0.8]),
        ]:
            builder.add_box_visual(
                pose=sapien.Pose(p=[a/2 for a in axis]),
                half_size=[abs(a)/2 if a != 0 else axis_thick for a in axis],
                material=sapien.render.RenderMaterial(base_color=color),
            )
        
        # scipy [x,y,z,w] → sapien [w,x,y,z]
        sapien_q = [float(quat_scipy[3]), float(quat_scipy[0]), float(quat_scipy[1]), float(quat_scipy[2])]
        marker = builder.build_kinematic(name=name)
        marker.set_pose(sapien.Pose(p=[float(position[0]), float(position[1]), float(position[2])], q=sapien_q))
        _debug_grasp_visuals[name] = marker
    
    # 渲染并等待用户查看
    env.unwrapped.render_human()
    if wait_key:
        print(f"  🔍 [{name}] 按 [c] 继续查看下一个，按 [q] 跳过等待...")
        viewer = env.unwrapped._viewer
        if viewer is not None:
            import time
            start_wait = time.time()
            while time.time() - start_wait < 30:  # 最多等30秒
                env.unwrapped.render_human()
                if viewer.window.key_down('c'):
                    break
                if viewer.window.key_down('q'):
                    print("  ⏭ 跳过后续等待")
                    break
                time.sleep(0.02)


def debug_clear_visuals(env):
    """清理所有调试可视化标记"""
    global _debug_grasp_visuals, _ee_frame_visual
    scene = env.unwrapped.scene
    for name, marker in _debug_grasp_visuals.items():
        if marker is not None:
            scene.remove_actor(marker)
    _debug_grasp_visuals = {}
    if _ee_frame_visual is not None:
        try:
            scene.remove_actor(_ee_frame_visual)
        except:
            pass
        _ee_frame_visual = None


def ensure_ee_frame_visual(scene):
    """创建/获取末端坐标系可视化（红X绿Y蓝Z），持续跟随末端移动"""
    global _ee_frame_visual
    if _ee_frame_visual is not None:
        try:
            _ = _ee_frame_visual.pose
            return
        except:
            _ee_frame_visual = None
    
    builder = scene.create_actor_builder()
    builder.set_initial_pose(sapien.Pose())
    axis_len = 0.07
    width = 0.003
    # X - 红
    builder.add_box_visual(
        pose=sapien.Pose(p=[axis_len/2, 0, 0]),
        half_size=[axis_len/2, width, width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
    )
    # Y - 绿
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, axis_len/2, 0]),
        half_size=[width, axis_len/2, width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
    )
    # Z - 蓝
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, axis_len/2]),
        half_size=[width, width, axis_len/2],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1])
    )
    # 中心球体（白色）
    builder.add_sphere_visual(
        radius=0.006,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 0.9])
    )
    _ee_frame_visual = builder.build_kinematic(name="ee_frame_visual")
    print("  [坐标显示] ✅ 末端坐标系已创建（红X 绿Y 蓝Z）")

# ========== 全局运动规划器（懒初始化，复用实例） ==========
_g_solvers = [None, None]  # 左臂[0], 右臂[1]

def _get_solver(env, robot_idx):
    """懒初始化/获取全局规划器实例，复用避免重复加载 URDF"""
    global _g_solvers
    if _g_solvers[robot_idx] is not None:
        return _g_solvers[robot_idx]
    arm_names = ["left", "right"]
    solver = PiperXArmMotionPlanningSolver(
        env, debug=False, vis=False, arm=arm_names[robot_idx], print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5,
    )
    _g_solvers[robot_idx] = solver
    return solver

def move_to_position_only(
    env, robot_idx, target_pos, steps=400, record_step=None,
    gripper_open=True, target_quat=None,
    allow_adaptive_search=True, limit_search_range=False,
    max_trajectory_steps=None,
    wrist_flip=False,):
    """
    简单封装，调用 motion_planning_utils.py 的原始 move_to_position_only。
    步数超限时返回 False 并触发外部重试
    """
    global MOVE_MAX_TRAJECTORY_STEPS
    if max_trajectory_steps is None:
        max_trajectory_steps = MOVE_MAX_TRAJECTORY_STEPS
    
    from mani_skill.envs.tasks.custom_task.motion_planning_utils import move_to_position_only as _orig_move
    from mani_skill.envs.tasks.custom_task.motion_planning_utils import TrajectoryStepsExceededError
    try:
        return _orig_move(
            env, robot_idx=robot_idx, target_pos=target_pos,
            steps=steps, record_step=record_step,
            gripper_open=gripper_open, target_quat=target_quat,
            allow_adaptive_search=allow_adaptive_search,
            limit_search_range=limit_search_range,
            max_trajectory_steps=max_trajectory_steps,
            wrist_flip=wrist_flip,
        )
    except TrajectoryStepsExceededError as e:
        print(f"  ⚠️ 轨迹步数超限: {e}")
        print(f"  → 任务将失败并触发重试机制")
        return False


def lift_and_return(env, robot_idx, lift_height=0.15, steps=100, record_step=None):
    """
    上升至当前位置正上方 +lift_height 处，然后返回原位置
    """
    from scipy.spatial.transform import Rotation as R
    agent = env.unwrapped.agent
    robot = agent.agents[robot_idx]
    
    tcp_pose = robot.tcp_pose
    p = tcp_pose.p
    if len(p.shape) == 2:
        p = p[0]
    original_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    
    q = tcp_pose.q
    if len(q.shape) == 2:
        q = q[0]
    original_quat = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    
    lift_pos = original_pos.copy()
    lift_pos[2] += lift_height
    
    print(f"    → 上升至 [{lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f}]")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=lift_pos,
        target_quat=original_quat,
        steps=steps, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"    ⚠️ 上升失败")
        return False
    
    print(f"    → 返回原位置 [{original_pos[0]:.3f}, {original_pos[1]:.3f}, {original_pos[2]:.3f}]")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=original_pos,
        target_quat=original_quat,
        steps=steps, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"    ⚠️ 返回失败")
        return False
    
    return True


from mani_skill.envs.tasks.custom_task.household.scene_H5_02 import (
    _SHELF_POS,
    _PLATE_X_OFFSET as _TUBE_X_OFFSET,
    _PLATE_Y_OFFSET as _TUBE_Y_OFFSET,
    _PLATE_Z_OFFSET as _TUBE_Z_OFFSET,
)
tube_on_shelf_pose=[_SHELF_POS[0], _SHELF_POS[1] + 0.12, _SHELF_POS[2] + 0.05]
tube_on_table_pose=[-0.4, 0, 0.05]
# tube_on_table_pose=[-0.4, -0.15, 0.05]
final_pos = np.array(tube_on_table_pose, dtype=np.float32)

# ===== 3 个放置目标位置（place_01 参数选择） =====
TUBE_PLACE_TARGETS_3 = [
    [-0.35, 0.0, 0.05],    # 目标 0
    [-0.35, 0.0, 0.05],   # 目标 1
    [-0.35, 0.0, 0.05],     # 目标 2
]

# ===== 任务参数组合序列（240 组，待用户填入） =====
# 每组: (layout_id, pass_id, place_01, grasp_id, mode)
#   layout_id: 布局索引 (0-8)，对应 PREDEFINED_TUBE_POSITIONS 的 9 组
#   pass_id:   左臂传递给右臂的试管编号（在 layout 内的索引）
#   place_01:  右臂放置目标位置索引 (0-2)，对应 TUBE_PLACE_TARGETS_3
#   grasp_id:  左臂抓取放置的试管编号（在 layout 内的索引）
#   mode:      0=左臂放回原位, 1=就近放到 shelf
# 示例:
# TASK_SEQUENCES_240 = [
#     (0, 0, 0, 0, 0),   # layout 0, 传递试管0, 目标0, 抓取试管0, 放回原位
#     (1, 1, 2, 0, 1),   # layout 1, 传递试管1, 目标2, 抓取试管0, 放到shelf
#     ...
# ]
TASK_SEQUENCES_240 = []  # 将由用户填入

# ===== shelf 位置（用于 mode=1 的放置目标） =====
SHELF_PLACE_POS = np.array([_SHELF_POS[0], _SHELF_POS[1], _SHELF_POS[2]], dtype=np.float32)
# SHELF_PLACE_POS = np.array([_SHELF_POS[0], _SHELF_POS[1] + 0.12, _SHELF_POS[2] + 0.05], dtype=np.float32)
def get_tube_object(env, tube_idx=0):
    """根据索引获取试管对象

    Args:
        env: 环境
        tube_idx: 试管索引 (0: sample_tube_0, 1: sample_tube_1, 2: sample_tube_2, 3: sample_tube_3)

    Returns:
        试管对象或 None
    """
    env_unwrapped = env.unwrapped

    # Scene_L5_02 中物体存储在 self.objects 字典中
    tube_names = ['sample_tube_0', 'sample_tube_1', 'sample_tube_2', 'sample_tube_3','sample_tube_4']

    # 调试：打印所有试管的位姿
    if hasattr(env_unwrapped, 'objects'):
        for name in tube_names:
            if name in env_unwrapped.objects:
                obj = env_unwrapped.objects[name]
                pos = obj.pose.p
                if hasattr(pos, 'cpu'):
                    pos = pos.cpu().numpy()
                print(f"  [debug] {name} pos: {pos}")

    # 从 objects 字典中查找
    if 0 <= tube_idx < len(tube_names):
        name = tube_names[tube_idx]
        if hasattr(env_unwrapped, 'objects') and name in env_unwrapped.objects:
            return env_unwrapped.objects[name]

    # 回退：从 robocasa_objects 中查找
    if hasattr(env_unwrapped, 'robocasa_objects'):
        tube_objects = []
        for obj in env_unwrapped.robocasa_objects:
            if 'tube' in obj.name:
                tube_objects.append(obj)
        if tube_idx < len(tube_objects):
            return tube_objects[tube_idx]
    return None


def print_tube_pose(env, tube_idx=0, prefix=""):
    """输出试管当前完整位姿信息（位置 + 四元数 + 欧拉角 + 轴线方向）"""
    pos, quat = get_tube_pose(env, tube_idx)
    if pos is None or quat is None:
        print(f"{prefix}试管{tube_idx}: 无法获取位姿")
        return
    euler = R.from_quat(quat).as_euler('xyz', degrees=True)
    rot_matrix = R.from_quat(quat).as_matrix()
    axis_z = rot_matrix @ np.array([0, 0, 1])
    print(f"{prefix}[试管{tube_idx} 位姿]")
    print(f"{prefix}  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"{prefix}  四元数: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"{prefix}  欧拉角 (绕世界轴 xyz 内旋): [roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}]°")
    print(f"{prefix}  局部Z轴（试管轴线）方向: [{axis_z[0]:.3f}, {axis_z[1]:.3f}, {axis_z[2]:.3f}]")
    print(f"{prefix}  竖直时轴线应为 [0, 0, 1]")


def get_tube_pose(env, tube_idx=0):
    """获取试管的完整位姿（位置 + 姿态四元数）

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        (pos, quat) 元组，或 (None, None)
    """
    tube = get_tube_object(env, tube_idx)
    if tube is None:
        return None, None

    pose = tube.pose
    pos = pose.p
    q = pose.q
    if len(pos.shape) == 2:
        pos = pos[0]
    if len(q.shape) == 2:
        q = q[0]
    pos_np = np.array(pos.cpu().numpy() if hasattr(pos, 'cpu') else pos, dtype=np.float32)
    q_np = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    return pos_np, q_np


def get_tube_cmass_pose(env, tube_idx=0):
    """获取试管质心在世界坐标系中的位置

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        (pos, quat) 元组，质心的世界坐标位置和姿态，或 (None, None)
    """
    tube = get_tube_object(env, tube_idx)
    if tube is None:
        return None, None

    # 质心在世界坐标系中的位姿 = actor位姿 * 局部质心偏移
    cmass_pose = tube.pose * tube.cmass_local_pose
    pos = cmass_pose.p
    q = cmass_pose.q
    if len(pos.shape) == 2:
        pos = pos[0]
    if len(q.shape) == 2:
        q = q[0]
    pos_np = np.array(pos.cpu().numpy() if hasattr(pos, 'cpu') else pos, dtype=np.float32)
    q_np = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    return pos_np, q_np


def get_tube_end_pose(env, tube_idx=0):
    """获取试管末端（底部最低点）在世界坐标系中的位置

    通过碰撞网格找到z值最小的顶点，即为试管底部位置。

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        (pos, quat) 元组，底部位置和试管姿态，或 (None, None)
    """
    tube = get_tube_object(env, tube_idx)
    if tube is None:
        return None, None

    mesh = tube.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return None, None

    vertices = mesh.vertices
    # 找到z值最小的顶点（底部最低点）
    min_z = np.min(vertices[:, 2])
    bottom_vertices = vertices[np.abs(vertices[:, 2] - min_z) < 0.001]
    end_pos = np.mean(bottom_vertices, axis=0)  # 底部中心

    # 姿态用试管自身姿态
    q = tube.pose.q
    if len(q.shape) == 2:
        q = q[0]
    q_np = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)

    return end_pos.astype(np.float32), q_np


def check_tube_rack_contact(env, tube_idx):
    """检查试管与架子之间的接触力
    
    Args:
        env: 环境
        tube_idx: 试管索引
        
    Returns:
        float: 接触冲量总和
    """
    from mani_skill.utils import sapien_utils
    
    total_impulse = 0.0
    if tube_idx is None:
        return total_impulse
        
    try:
        all_contacts = env.unwrapped.scene.get_contacts()
        tube_actor = get_tube_object(env, tube_idx)
        
        if tube_actor is not None and len(all_contacts) > 0:
            tube_entity = tube_actor._bodies[0].entity if hasattr(tube_actor, '_bodies') else None
            if tube_entity is not None:
                rack_names = ['tube_rack', 'tube_rack02', 'tube_rack03']
                for rack_name in rack_names:
                    try:
                        if hasattr(env.unwrapped, 'objects') and rack_name in env.unwrapped.objects:
                            rack_actor = env.unwrapped.objects[rack_name]
                            rack_entity = rack_actor._bodies[0].entity if hasattr(rack_actor, '_bodies') else None
                            if rack_entity is not None:
                                pair_impulse = sapien_utils.get_pairwise_contact_impulse(
                                    all_contacts, tube_entity, rack_entity
                                )
                                total_impulse += np.linalg.norm(pair_impulse)
                    except:
                        pass
    except:
        pass
    
    return total_impulse


def get_tube_position(env, tube_idx=0):
    """获取试管当前位置

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        位置 numpy 数组 [x, y, z] 或 None
    """
    pos, _ = get_tube_pose(env, tube_idx)
    return pos
    pos_np = np.array(pos.cpu().numpy() if hasattr(pos, 'cpu') else pos, dtype=np.float32)
    return pos_np


def get_tube_quat(env, tube_idx=0):
    """获取试管当前姿态四元数

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        四元数 numpy 数组 [w, x, y, z] 或 None
    """
    tube = get_tube_object(env, tube_idx)
    if tube is None:
        return None

    pose = tube.pose
    q = pose.q
    if len(q.shape) == 2:
        q = q[0]
    q_np = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    return q_np


def compute_grasp_quat_from_tube(env, tube_idx, base_rot_deg=[135, 0, -40]):
    """根据试管姿态计算机械臂夹取姿态四元数

    对于水平试管（在桌面上）：
    - 夹爪从侧面接近，与试管边沿平行（约45°倾斜）
    - base_rot_deg: [roll=135°, pitch=0°, yaw=接近方向]
      roll=135° 使夹爪Z轴指向斜下方45°，Y轴与垂直方向成45°（平行于边沿）

    对于垂直试管（在架子上）：
    - 夹爪垂直于试管表面接近
    - base_rot_deg: yaw分量用于绕法兰盘旋转调整

    Args:
        env: 环境
        tube_idx: 试管索引
        base_rot_deg: 基准欧拉角 [roll, pitch, yaw]（度）

    Returns:
        夹取姿态四元数 [x, y, z, w]（scipy 格式）
    """
    # 获取试管当前四元数 [w, x, y, z]
    tube_q_wxyz = get_tube_quat(env, tube_idx)
    if tube_q_wxyz is None:
        rot_rad = [np.deg2rad(r) for r in base_rot_deg]
        return R.from_euler('xyz', rot_rad).as_quat().astype(np.float32)

    # 转换为 scipy 格式 [x, y, z, w]
    tube_quat_scipy = np.array([
        tube_q_wxyz[1], tube_q_wxyz[2], tube_q_wxyz[3], tube_q_wxyz[0]
    ], dtype=np.float32)
    tube_rot = R.from_quat(tube_quat_scipy)

    # 试管表面法线（试管局部坐标系的 z 轴在世界坐标系中的方向）
    tube_normal = tube_rot.apply([0, 0, 1])
    tube_normal = tube_normal / np.linalg.norm(tube_normal)

    # 判断试管放置方式：水平（桌面）还是垂直（架子）
    # 如果法线与世界Z轴夹角 < 30°，认为是水平放置
    cos_up = np.dot(tube_normal, [0, 0, 1])

    if cos_up > np.cos(np.deg2rad(30)):
        # ===== 水平试管（桌面上）：从侧面斜夹边沿 =====
        # 试管边沿约45°斜角，夹爪需要倾斜以平行于边沿
        # roll=135°: 夹爪Z轴指向斜下方45°，Y轴与垂直成45°（平行边沿）
        roll_deg = base_rot_deg[0] if len(base_rot_deg) >= 1 else 135
        yaw_deg = base_rot_deg[2] if len(base_rot_deg) >= 3 else 0

        grasp_rot = R.from_euler('xyz', [0, 180, 0], degrees=True)
    else:
        # ===== 垂直试管（架子上）：夹爪垂直于试管表面接近 =====
        ee_default = np.array([0, 0, 1], dtype=np.float64)
        target_dir = -tube_normal.astype(np.float64)

        cos_angle = np.clip(np.dot(ee_default, target_dir), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle > 1e-6:
            axis = np.cross(ee_default, target_dir)
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-6:
                axis = axis / axis_len
                align_rot = R.from_rotvec(axis * angle)
            else:
                align_rot = R.from_euler('xyz', [np.pi, 0, 0])
        else:
            align_rot = R.from_quat([0, 0, 0, 1])

        yaw_deg = base_rot_deg[2] if len(base_rot_deg) >= 3 else 0
        flange_rot = R.from_euler('xyz', [0, 0, yaw_deg], degrees=True)
        # 绕世界坐标系x轴旋转-45°（左乘，在世界坐标系下先倾斜）
        world_x_rot = R.from_euler('xyz', [-45, 0, 0], degrees=True)
        # 关节6额外旋转90°（绕夹爪Z轴）
        joint6_rot = R.from_euler('xyz', [0, 0, 90], degrees=True)
        # 绕当前局部Y轴（绿色棒）旋转90°
        local_y_rot = R.from_euler('xyz', [0, 90, 0], degrees=True)
        # 再绕当前局部Z轴旋转180°
        extra_z_rot = R.from_euler('xyz', [0, 0, 180], degrees=True)
        # 再绕当前局部X轴（红色棒）旋转45°
        extra_x_rot = R.from_euler('xyz', [45, 0, 0], degrees=True)
        grasp_rot = world_x_rot * align_rot * flange_rot * joint6_rot * local_y_rot * extra_z_rot * extra_x_rot

    return grasp_rot.as_quat().astype(np.float32)  # [x, y, z, w]

def find_closest_robot_to_tube(env, tube_idx=0):
    """判断哪个机械臂离指定试管最近

    Args:
        env: 环境
        tube_idx: 试管索引

    Returns:
        (robot_idx, distance) 机械臂索引和距离，如果失败返回 (None, None)
    """
    tube_pos = get_tube_position(env, tube_idx)
    if tube_pos is None:
        return None, None

    agent = env.unwrapped.agent
    min_distance = float('inf')
    closest_robot_idx = None

    for robot_idx, robot in enumerate(agent.agents):
        ee_pose = robot.tcp_pose
        ee_pos = ee_pose.p
        if len(ee_pos.shape) == 2:
            ee_pos = ee_pos[0]

        ee_pos_np = np.array(ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else ee_pos, dtype=np.float32)
        distance = np.linalg.norm(ee_pos_np - tube_pos)

        print(f"  机械臂{robot_idx} 末端位置: [{ee_pos_np[0]:.3f}, {ee_pos_np[1]:.3f}, {ee_pos_np[2]:.3f}], 距离试管: {distance:.3f}m")

        if distance < min_distance:
            min_distance = distance
            closest_robot_idx = robot_idx

    print(f"  最近的机械臂: 机械臂{closest_robot_idx}, 距离: {min_distance:.3f}m")
    return closest_robot_idx, min_distance

def pass_to(env, tube_idx=0, robot_idx=0, target_pos=None, target_quat=None,
            record_step=None, recorder=None, waypoints=None, grasp_rot_deg=None):
    """左臂（0号机械臂）抓取指定试管并递送到交接位置（不释放夹具，等待另一机械臂来接）

    流程:
      1. 移动到试管旁边
      2. 下降到抓取位置
      3. 闭合夹爪抓取试管
      4. 提起试管（使用 lift_tube_quat 姿态）
      5. 移动到交接位置上方（保持抓取，等待交接）

    Args:
        env: 环境
        tube_idx: 试管索引 (0: sample_tube_1, 1: sample_tube_2, 2: sample_tube_3)
        robot_idx: 机械臂索引，固定为 0（左臂）
        target_pos: 交接位置 [x, y, z]
        target_quat: 交接姿态四元数 [x, y, z, w]（推荐使用 lift_tube_quat）
        grasp_rot_deg: 夹取姿态欧拉角 [roll, pitch, yaw]（度），默认水平夹取
        record_step: 录制函数
        recorder: 数据记录器，用于设置 subtask 标签

    Returns:
        (success, robot_idx, hold_pos, hold_quat, hold_qpos)
        hold_pos/hold_quat/hold_qpos 是机械臂抓取试管后的保持位置，用于另一机械臂来接
    """
    # 获取试管位置
    tube_pos = get_tube_position(env, tube_idx)
    if tube_pos is None:
        print(f"错误: 未找到试管 {tube_idx}")
        return (False, robot_idx, None, None, None)

    # 固定使用 robot_idx（默认0-左臂），不自动选择
    agent = env.unwrapped.agent
    robot = agent.agents[robot_idx]

    # # 初始化末端坐标系可视化（已关闭）
    # ensure_ee_frame_visual(env.unwrapped.scene)
    # if _ee_frame_visual is not None:
    #     _ee_frame_visual.set_pose(robot.tcp_pose)

    # 保存初始位置
    current_ee_pose = robot.tcp_pose
    current_p = current_ee_pose.p
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    current_p_np = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)

    initial_pos = current_p_np.copy()
    current_q = current_ee_pose.q
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    initial_quat = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)
    initial_qpos = robot.robot.get_qpos()[0].cpu().numpy().copy()

    # 设置默认目标位置
    if target_pos is None:
        target_pos = np.array([-0.05, -0.065, 0.12], dtype=np.float32)
    else:
        target_pos = np.array(target_pos, dtype=np.float32)

    arm_name = "left arm"  # 左臂（0号机械臂）固定执行

    def _get_tube_rack_color(pos_x, pos_y):
        if pos_y > 0.1 or pos_y < -0.1:
            return "yellow"
        else:
            return "red"

    source_tube_rack_color = _get_tube_rack_color(tube_pos[0], tube_pos[1])

    print(f"\n========== 左臂（0号机械臂）抓取试管{tube_idx} 并递送 ==========")
    print(f"  试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}]")
    print(f"  交接位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    if target_quat is not None:
        print(f"  交接姿态 (四元数): [{target_quat[0]:.6f}, {target_quat[1]:.6f}, {target_quat[2]:.6f}, {target_quat[3]:.6f}]")

    # 夹取姿态：基于试管位置相对于左臂的角度自动计算yaw（以X轴为0度）
    if grasp_rot_deg is not None:
        # 获取左臂基座位置
        robot_base = robot.robot.pose.p
        if hasattr(robot_base, 'cpu'):
            robot_base = robot_base.cpu().numpy()
        robot_base = np.array(robot_base).flatten()

        # 计算从机械臂到试管的方向角（以X轴为0度，Y轴正方向为正）
        dx = tube_pos[0] - robot_base[0]
        dy = tube_pos[1] - robot_base[1]
        auto_yaw = np.rad2deg(np.arctan2(dy, dx))

        # 使用用户指定的roll和pitch，自动计算yaw
        roll_deg = grasp_rot_deg[0]
        pitch_deg = grasp_rot_deg[1]
        auto_yaw = grasp_rot_deg[2]
        print(f"  左臂基座位置: [{robot_base[0]:.3f}, {robot_base[1]:.3f}, {robot_base[2]:.3f}]")
        print(f"  自动计算yaw: {auto_yaw:.1f}° (从机械臂到试管的方向角，X轴为0°)")

        rot_rad = [np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(auto_yaw)]
        grasp_quat = R.from_euler('xyz', rot_rad).as_quat().astype(np.float32)
        print(f"  夹取姿态 (欧拉角 [{roll_deg:.0f}, {pitch_deg:.0f}, {auto_yaw:.1f}]° -> 四元数): [{grasp_quat[0]:.4f}, {grasp_quat[1]:.4f}, {grasp_quat[2]:.4f}, {grasp_quat[3]:.4f}]")
    else:
        grasp_quat = np.array([0.0, 0.7071, 0.0, 0.7071], dtype=np.float32)  # 默认水平夹取
        # 优先尝试：从负X方向接近（夹爪z轴指向正X）
    
    fixed_z_axis = np.array([1.0, 0.0, 0.0])  # 固定：从正X方向指向试管
    fixed_x_axis = np.array([0.0, 0.0, -1.0])  # 夹爪闭合方向竖直向下
    # if tube_pos[1] < -0.04:
    #     pitch = np.deg2rad(45)
    # else:
    #     pitch = np.deg2rad(60)
    pitch = np.deg2rad(45)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    z_axis = Ry @ fixed_z_axis
    x_axis = Ry @ fixed_x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    left_matrix_quat = R.from_matrix(rot_mat).as_quat()
    grasp_quat=left_matrix_quat
    # 获取试管当前姿态并计算其z轴朝向
    tube_actor = get_tube_object(env, tube_idx)
    if tube_actor is not None:
        tube_pose = tube_actor.pose
        tube_q = tube_pose.q
        if len(tube_q.shape) == 2:
            tube_q = tube_q[0]
        tube_q = np.array(tube_q.cpu().numpy() if hasattr(tube_q, 'cpu') else tube_q)
        tube_R = R.from_quat([tube_q[1], tube_q[2], tube_q[3], tube_q[0]])
        tube_rot_mat = tube_R.as_matrix()
        tube_z_axis = tube_rot_mat[:, 2]
        print(f"    试管当前z轴朝向: [{tube_z_axis[0]:.4f}, {tube_z_axis[1]:.4f}, {tube_z_axis[2]:.4f}]")
    else:
        tube_z_axis = None
        print(f"    试管当前z轴朝向: 无法获取")

    # 计算机械臂夹取姿态的z轴朝向
    gripper_R = R.from_quat([grasp_quat[0], grasp_quat[1], grasp_quat[2], grasp_quat[3]])
    gripper_rot_mat = gripper_R.as_matrix()
    gripper_z_axis = gripper_rot_mat[:, 2]
    print(f"    夹具末端z轴朝向: [{gripper_z_axis[0]:.4f}, {gripper_z_axis[1]:.4f}, {gripper_z_axis[2]:.4f}]")
    if tube_z_axis is not None:
        dot_product = np.dot(tube_z_axis, gripper_z_axis)
        print(f"    点积（验证垂直性）: {dot_product:.4f} (应为0表示垂直)")

    # 计算安全升起高度（用于步骤0和 return_arm 归位时先升后降）
    RAISE_HEIGHT_OFFSET = 0.2
    raised_pos = initial_pos.copy()
    raised_pos[2] += RAISE_HEIGHT_OFFSET
    raised_pos[2] = min(raised_pos[2], MAX_Z_HEIGHT)

    # ========== 步骤0: 原地升起机械臂至安全高度 ==========
    print(f"\n  步骤0: 原地升起机械臂至安全高度（+{RAISE_HEIGHT_OFFSET*100:.0f}cm）...")
    if recorder:
        recorder.set_subtask(f"left arm approaches for the tube (on red rack)")
    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=raised_pos,
                                 target_quat=initial_quat, steps=30, record_step=record_step,
                                 gripper_open=True, allow_adaptive_search=True,
                                 limit_search_range=False)
    if not success:
        print(f"  ❌ 原地升起失败")
        return (False, robot_idx, None, None, None)
    print(f"  ✅ 原地升起成功")
    raised_pos[1]=0
    if recorder:
        recorder.set_subtask(f"left arm approaches for the tube (on red rack)")
    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=raised_pos,
                                 target_quat=initial_quat, steps=30, record_step=record_step,
                                 gripper_open=True, allow_adaptive_search=True,
                                 limit_search_range=False)
    if not success:
        print(f"  ❌ 升起后偏移到中心失败")
        return (False, robot_idx, None, None, None)
    print(f"  ✅ 升起后偏移到中心成功")
    # 接近位置：从机械臂方向（负X方向）靠近试管，X偏移量要小
    # 左机械臂基座在 (-0.5, -0.2)，试管在 (-0.103, 0.212)
    # 偏移量过大（0.15m）会导致步骤2需要移动太远，IK不可达
    # 改为0.05m，使步骤2只需移动很小距离
    grasp_approach_pos = tube_pos.copy()
    grasp_approach_pos[2] += grasp_height+0.1     # 略高于试管8cm，避免碰撞
    grasp_approach_pos[2] = min(grasp_approach_pos[2], MAX_Z_HEIGHT)
    grasp_approach_pos[0] -= 0.05      # 从X方向靠近（仅偏移5cm，步骤2更容易到达）


    # if waypoints is not None and len(waypoints) > 0:
    #     print(f"\n  步骤1a: 依次经过 {len(waypoints)} 个过程点...")
    #     for wp_idx, wp in enumerate(waypoints):
    #         wp_pos = np.array(wp[0], dtype=np.float32)
    #         wp_quat = np.array(wp[1], dtype=np.float32) if len(wp) > 1 else grasp_quat
    #         if recorder:
    #             recorder.set_subtask(f"{arm_name} moves to waypoint {wp_idx + 1}")
    #         print(f"    过程点 {wp_idx + 1}: pos={wp_pos}")
    #         success = move_to_position_only(env, robot_idx=robot_idx, target_pos=wp_pos,
    #                                          target_quat=wp_quat, steps=30, record_step=record_step,
    #                                          gripper_open=True, allow_adaptive_search=True,
    #                                          limit_search_range=True)
    #         if not success:
    #             print(f"    ❌ 移动到过程点 {wp_idx + 1} 失败")
    #             return (False, robot_idx, None, None, None)
    #     print(f"  ✅ 所有过程点到达成功")

    print(f"\n  步骤1b: 下降到试管{tube_idx}旁边...")
    if recorder:
        recorder.set_subtask(f"left arm approaches for the tube (on red rack)")
    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=grasp_approach_pos,
                                     target_quat=grasp_quat, steps=30, record_step=record_step,
                                     gripper_open=True, allow_adaptive_search=True,
                                     limit_search_range=True)
    if not success:
        print(f"  ❌ 移动到试管旁边失败")
        return (False, robot_idx, None, None, None)
    print(f"  ✅ 移动到试管旁边成功")

    # ========== 步骤2: 下降到抓取位置 ==========
    # 抓取位置：从试管位置小偏移，使IK更易解
    # 左机械臂够不到试管精确位置（Y=0.212离基座Y=-0.2太远），
    # 需要Y方向向机械臂侧偏移
    grasp_pos = tube_pos.copy()
    grasp_pos[2] += grasp_height     # 略高于试管中心3cm，保证IK可达又不至于抓空
    grasp_pos[2] = min(grasp_pos[2], MAX_Z_HEIGHT)
    grasp_pos[0] += 0.0     # X方向微调
    # Y方向向机械臂基座侧偏移（左机械臂Y=-0.2，试管Y=0.212）
    # if tube_pos[1] > 0:
    #     grasp_pos[1] -= 0.03  # 正Y试管向负Y偏移3cm，靠近左机械臂
    # else:
    #     grasp_pos[1] += 0.03  # 负Y试管向正Y偏移3cm
    

    print(f"\n  步骤2: 下降到抓取位置...")
    if recorder:
        recorder.set_subtask(f"left arm approaches for the tube (on red rack)")
    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=grasp_pos,
                                     target_quat=grasp_quat, steps=30, record_step=record_step,
                                     gripper_open=True, allow_adaptive_search=True,
                                     limit_search_range=True)
    if not success:
        print(f"  ❌ 下降到抓取位置失败")
        return (False, robot_idx, None, None, None)
    print(f"  ✅ 下降到抓取位置成功")

    # ========== 步骤3: 闭合夹爪抓取试管 ==========
    print(f"\n  步骤3: 闭合夹爪抓取试管...")
    if recorder:
        recorder.set_subtask(f"left arm picks up the tube (from red rack)")

    # 增加夹爪闭合力度：临时提高 force_limit 和 stiffness
    old_stiffness = None
    old_force_limit = None
    try:
        # 临时提高夹爪抓取力
        joints = robot.robot.get_active_joints()
        gripper_joints = [j for j in joints if 'joint7' in j.name or 'joint8' in j.name]
        if gripper_joints:
            for j in gripper_joints:
                print(f"  {j.name}: stiffness={j.get_stiffness()}, force_limit={j.get_force_limit()}")
            old_stiffness = [j.get_stiffness() for j in gripper_joints]
            old_force_limit = [j.get_force_limit() for j in gripper_joints]
            for j in gripper_joints:
                j.set_drive_properties(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE_LIMIT)
            print(f"  设置后: stiffness={GRIPPER_STIFFNESS}, damping={GRIPPER_DAMPING}, force_limit={GRIPPER_FORCE_LIMIT}")
    except Exception as e:
        print(f"  [info] 无法调整夹爪参数: {e}")

    GRIPPER_OPEN_POS = 1.0   # 夹爪张开位置
    
    # ========== 直接闭合夹爪 ==========
    print(f"  直接闭合夹爪...")
    for i in range(30):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        
        other_robot_idx = 1 - robot_idx
        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action:
            other_robot = agent.agents[other_robot_idx]
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])  # 保持张开

        if record_step:
            record_step(action)
        else:
            env.step(action)

    print(f"  ✅ 夹爪闭合完成")
    # ========== 步骤3d: 等待夹爪完全稳定 ==========
    print(f"  步骤3d: 等待夹爪稳定...")
    for i in range(50):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        
        other_robot_idx = 1 - robot_idx
        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action:
            other_robot = agent.agents[other_robot_idx]
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])

        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 夹爪稳定等待完成")
    
    # ========== 步骤4: 分步小幅度提起（每次0.04m），共提0.2m ==========
    # STEP_LIFT_HEIGHT = 0.04  # 每步提起高度
    # NUM_LIFT_STEPS = 5      # 提起步数（0.04m × 5 = 0.2m）
    
    actual_pose = robot.tcp_pose
    actual_q = actual_pose.q
    if len(actual_q.shape) == 2:
        actual_q = actual_q[0]
    lift_quat = np.array(actual_q.cpu().numpy() if hasattr(actual_q, 'cpu') else actual_q, dtype=np.float32)
    
    current_lift_pos = grasp_pos.copy()
    print(f"\n  步骤4: 分步提起试管（每次0.02m，共{NUM_LIFT_STEPS}步，总共0.2m）...")
    if recorder:
        recorder.set_subtask(f"left arm picks up the tube (from red rack)")
    
    for lift_step in range(NUM_LIFT_STEPS):
        # 每次提起前重置夹爪参数
        try:
            joints = robot.robot.get_active_joints()
            gripper_joints = [j for j in joints if 'joint7' in j.name or 'joint8' in j.name]
            if gripper_joints:
                for j in gripper_joints:
                    j.set_drive_properties(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE_LIMIT)
        except Exception as e:
            pass
        
        # 读取当前试管位置作为基准
        if tube_idx is not None:
            tube_pos, _ = get_tube_pose(env, tube_idx)
            if tube_pos is not None:
                current_lift_pos[0] = tube_pos[0]
                current_lift_pos[1] = tube_pos[1]
        
        # 高度逐级增加
        current_lift_pos[2] += STEP_LIFT_HEIGHT
        current_lift_pos[2] = min(current_lift_pos[2], MAX_Z_HEIGHT)
        target_z = current_lift_pos[2]
        
        # ===== 提起步骤a: 在当前高度读取试管姿态，调整使竖直 =====
        # if tube_idx is not None:
        #     tube_lift_pos, tube_lift_quat = get_tube_end_pose(env, tube_idx)
        #     if tube_lift_quat is not None:
        #         tube_rot = R.from_quat([tube_lift_quat[1], tube_lift_quat[2], 
        #                             tube_lift_quat[3], tube_lift_quat[0]])
        #         tube_z_world = tube_rot.apply([0, 0, 1])
        #         world_z = np.array([0, 0, 1])
        #         R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
        #         ee_pose = robot.tcp_pose
        #         ee_q = ee_pose.q
        #         if len(ee_q.shape) == 2:
        #             ee_q = ee_q[0]
        #         gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
        #         gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
        #         gripper_rot_new = R_map * gripper_rot
        #         adjusted_quat_scipy = gripper_rot_new.as_quat()
        #         adjusted_lift_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
        #                                     adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
        #     else:
        #         adjusted_lift_quat = lift_quat
        # else:
        adjusted_lift_quat = lift_quat
        
        # 读取当前位置
        ee_pose_before = robot.tcp_pose
        p_before = ee_pose_before.p
        if len(p_before.shape) == 2:
            p_before = p_before[0]
        curr_pos_before = np.array(p_before.cpu().numpy() if hasattr(p_before, 'cpu') else p_before)
        
        # 保持当前高度，调整姿态
        adjust_pos = curr_pos_before.copy()
        adjust_pos[2] = curr_pos_before[2]  # 保持当前高度
        
        # print(f"    提起步骤a: 调整姿态（Z={adjust_pos[2]:.3f}m保持）...")
        # move_to_position_only(
        #     env, robot_idx=robot_idx,
        #     target_pos=adjust_pos,
        #     target_quat=adjusted_lift_quat,
        #     steps=50, record_step=record_step,
        #     gripper_open=False,
        #     allow_adaptive_search=True, limit_search_range=True,
        # )
        # 读取当前实际姿态
        ee_pose_now = robot.tcp_pose
        q_now = ee_pose_now.q
        if len(q_now.shape) == 2:
            q_now = q_now[0]
        current_quat = np.array(q_now.cpu().numpy() if hasattr(q_now, 'cpu') else q_now, dtype=np.float32)
        adjusted_lift_quat = current_quat
        # 读取当前试管位置作为基准
        if tube_idx is not None:
            tube_pos, _ = get_tube_pose(env, tube_idx)
            if tube_pos is not None:
                current_lift_pos[0] = tube_pos[0]  # X从试管读取
                current_lift_pos[1] = tube_pos[1]  # Y从试管读取

        # 高度逐级增加
        current_lift_pos[2] += STEP_LIFT_HEIGHT  # Z逐级增加
        # ===== 提起步骤b: 上升到目标高度 =====
        print(f"    提起步骤b: 上升 {curr_pos_before[2]:.3f}m → {target_z:.3f}m")
        success = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=current_lift_pos,
            target_quat=adjusted_lift_quat,
            steps=100, record_step=record_step,
            gripper_open=False,
            allow_adaptive_search=True,
            limit_search_range=True,
        )
        if not success:
            print(f"  ⚠️ 步骤{lift_step + 1}提升失败...")
            # 可以继续尝试或者使用备用姿态
        # ========== 保持左臂静止 ==========
        # if other_robot is not None:
        #     for _ in range(steps):
        #         agent = env.unwrapped._agent
        #         action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        #         # 设置右臂目标
        #         right_uid = f"piper_x-{robot_idx}"
        #         if right_uid in action:
        #             current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
        #             arm_qpos = current_qpos[:6]
        #             action[right_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        #         # 保持左臂当前姿态
        #         left_uid = f"piper_x-{other_robot_idx}"
        #         if left_uid in action:
        #             left_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
        #             left_arm_qpos = left_qpos[:6]
        #             left_gripper = left_qpos[6] if len(left_qpos) > 6 else -1.0
        #             action[left_uid] = np.concatenate([left_arm_qpos, [left_gripper]])
        #         if record_step:
        #             record_step(action)
        #         else:
        #             env.step(action)
        if not success:
            print(f"  ⚠️ 步骤{lift_step + 1}提升失败，尝试基于位置的yaw计算...")
            # 备用姿态计算逻辑（与之前相同）
            robot_base_pos = robot.robot.pose.p
            if len(robot_base_pos.shape) == 2:
                robot_base_pos = robot_base_pos[0]
            robot_base_np = np.array(robot_base_pos.cpu().numpy() if hasattr(robot_base_pos, 'cpu') else robot_base_pos, dtype=np.float32)
            dx = grasp_pos[0] - robot_base_np[0]
            dy = grasp_pos[1] - robot_base_np[1]
            yaw_angle = np.arctan2(dy, dx)
            z_axis = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0.0])
            x_axis = np.array([0.0, 0.0, -1.0])
            y_axis = np.cross(z_axis, x_axis)
            rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
            lift_quat = R.from_matrix(rot_matrix).as_quat()
            success = move_to_position_only(env, robot_idx=robot_idx, target_pos=current_lift_pos,
                                             target_quat=lift_quat, steps=30, record_step=record_step,
                                             gripper_open=False, allow_adaptive_search=True,
                                             limit_search_range=False)
            if not success:
                print(f"  ❌ 步骤{lift_step + 1}最终提升失败")
                return (False, robot_idx, None, None, None)
    
    lift_pos = current_lift_pos.copy()
    print(f"  ✅ 分步提起完成，最终高度: {lift_pos[2]:.3f}m")
    # if recorder:
    #     recorder.set_subtask(f"left arm moves the tube")
    lift_pos01 = lift_pos.copy()
    lift_pos01[0]=target_pos[0]
    # success = move_to_position_only(env, robot_idx=robot_idx, target_pos=lift_pos01,
    #                                  target_quat=lift_quat, steps=150, record_step=record_step,
    #                              gripper_open=False, allow_adaptive_search=True,
    #                              limit_search_range=False)
    # if not success:
    #     print(f"  ❌ 最终提升失败")
    #     return (False, robot_idx, None, None, None)
    # print(f"  ✅ 最终提升成功，当前高度: {lift_pos[2]:.3f}m")
    lift_pos = lift_pos01.copy()
    
    # z_axis = np.array([0, 1, 0])  # 右臂: z轴指向负Y
    # x_axis = np.array([0, 0, -1])
    # y_axis = np.cross(z_axis, x_axis)
    # rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    # matrix_base_quat = R.from_matrix(rot_mat).as_quat()
    # success = move_to_position_only(env, robot_idx=robot_idx, target_pos=lift_pos01,
    #                                  target_quat=matrix_base_quat, steps=150, record_step=record_step,
    #                              gripper_open=False, allow_adaptive_search=True,
    #                              limit_search_range=False)
    # if not success:
    #     print(f"  ❌ 空间变换失败")
    #     return (False, robot_idx, None, None, None)
    # print(f"  ✅ 空间变换成功，当前高度: {lift_pos[2]:.3f}m")
    # 第二步：在空中转换到交接姿态（此时试管已脱离试管架，不会碰撞）
    # if target_quat is not None:
    #     lift_rot_pos = lift_pos01.copy()
    #     print(f"\n  步骤4b: 在空中转换到交接姿态...")
    #     success = move_to_position_only(env, robot_idx=robot_idx, target_pos=lift_rot_pos,
    #                                      target_quat=target_quat, steps=50, record_step=record_step,
    #                                      gripper_open=False, allow_adaptive_search=True,
    #                                      limit_search_range=True)
    #     if not success:
    #         print(f"  ⚠️ 转换姿态失败，使用夹取姿态继续")
    #     else:
    #         print(f"  ✅ 已转换到交接姿态")
    #         lift_quat = target_quat

    # lift2_pos = lift_pos.copy()
    # lift2_pos[1] -= 0.1

    # print(f"\n  步骤4-5: 从侧面出去..防止碰到其他试管...")
    # if recorder:
    #     recorder.set_subtask(f"left arm picks up the tube (from {source_tube_rack_color} rack)")
    # success = move_to_position_only(env, robot_idx=robot_idx, target_pos=lift2_pos,
    #                                  target_quat=lift_quat, steps=50, record_step=record_step,
    #                                  gripper_open=False, allow_adaptive_search=True,
    #                                  limit_search_range=False)
    # if not success:
    #     print(f"  ❌ 提起试管失败")
    #     return (False, robot_idx, None, None, None)
    # print(f"  ✅ 从侧面出去成功，防止碰到其他试管，当前高度: {lift2_pos[2]:.3f}m")
    # ========== 步骤5: 移动到交接位置上方（保持抓取，等待交接） ==========
    target_hold_pos = target_pos.copy()
    target_hold_pos[0] += 0 # 往前推试管，方便右臂交接
    target_hold_pos[2] += 0.05  # 交接位置上方保持
    target_hold_pos[2] = min(target_hold_pos[2], MAX_Z_HEIGHT)  # ← 新增
    move_quat = target_quat if target_quat is not None else grasp_quat

    # 步骤5a: 先水平移动到交接位置上方（只改变X和Y，保持当前高度）
    # 从试管位置到交接位置水平距离约25cm，拆成两步让IK更容易
    current_ee_pos = env.unwrapped.agent.agents[robot_idx].tcp_pose.p
    if hasattr(current_ee_pos, 'cpu'):
        current_ee_pos = current_ee_pos.cpu().numpy()
    current_ee_pos = np.array(current_ee_pos).flatten()

    step5a_pos = target_hold_pos.copy()
    step5a_pos[2] = current_ee_pos[2]  # 保持当前高度，只移动X和Y

    print(f"\n  步骤5a: 水平移动到交接位置上方（保持高度）...")
    print(f"    水平目标: [{step5a_pos[0]:.3f}, {step5a_pos[1]:.3f}, {step5a_pos[2]:.3f}]")
    # if recorder:
    #     recorder.set_subtask("left arm moves the tube.")
    # success = move_to_position_only(env, robot_idx=robot_idx, target_pos=step5a_pos,
    #                                  target_quat=move_quat, steps=80, record_step=record_step,
    #                                  gripper_open=False, allow_adaptive_search=True,
    #                                  limit_search_range=True)
    # if not success:
    #     print(f"  ❌ 水平移动到交接位置上方失败")
    #     return (False, robot_idx, None, None, None)
    # print(f"  ✅ 水平移动到交接位置上方成功")
    # 步骤5a2: 保持高度移动到目标X
    curr = env.unwrapped.agent.agents[robot_idx].tcp_pose.p
    if hasattr(curr, 'cpu'):
        curr = curr.cpu().numpy()
    curr = np.array(curr).flatten()
    step5a2_pos = curr.copy()
    step5a2_pos[0] = target_hold_pos[0] - 0.1
    step5a2_pos[2] = curr[2]
    step5a2_pos[2] = min(step5a2_pos[2], MAX_Z_HEIGHT)
    
    # print(f"\n  步骤5a2: 保持高度移动到目标X位置...")
    # print(f"    目标: [{step5a2_pos[0]:.3f}, {step5a2_pos[1]:.3f}, {step5a2_pos[2]:.3f}]")
    # if recorder:
    #     recorder.set_subtask("left arm moves the tube.")
    # success = move_to_position_only(
    #     env, robot_idx=robot_idx, target_pos=step5a2_pos,
    #     target_quat=move_quat, steps=50, record_step=record_step,
    #     gripper_open=False, allow_adaptive_search=True,
    #     limit_search_range=True
    # )
    # if success:
    #     print(f"  ✅ 步骤5a2成功")
    # else:
    #     print(f"  ⚠️ 步骤5a2失败，继续...")

    # 步骤5b: 下降到交接高度（只改变Z）
    print(f"\n  步骤5b: 下降到交接高度...")
    print(f"    下降目标: [{target_hold_pos[0]:.3f}, {target_hold_pos[1]:.3f}, {target_hold_pos[2]:.3f}]")
    if recorder:
        recorder.set_subtask("left arm passes the tube to the right arm")
    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=target_hold_pos,
                                     target_quat=move_quat, steps=50, record_step=record_step,
                                     gripper_open=False, allow_adaptive_search=True,
                                     limit_search_range=True)
    if not success:
        print(f"  ❌ 下降到交接高度失败")
        return (False, robot_idx, None, None, None)
    print(f"  ✅ 已到达交接位置，保持夹爪闭合等待另一机械臂")

    print(f"\n========== 左臂（0号机械臂）递送完成（保持抓取等待交接） ==========")
    return True, robot_idx, initial_pos, initial_quat, initial_qpos

def right_pass(
    env, tube_idx,
    robot_idx,
    initial_ee_pos, initial_ee_quat, initial_qpos,
    recorder, record_step,):
    """
    右臂交接流程：读取TCP位姿 → 接近 → 抓取 → 左臂释放 → 左臂返回

    不含右臂放置/返回逻辑（步骤7在 main() 中根据 right_place 决定）

    Args:
        env: 环境
        tube_idx: 试管索引
        robot_idx: pass_to 返回的机械臂索引（0=左臂抓管，1=右臂抓管）
        initial_ee_pos: 左臂初始末端位置（用于返回）
        initial_ee_quat: 左臂初始末端姿态
        initial_qpos: 左臂初始关节位置
        recorder: 录制器
        record_step: 步骤录制函数

    Returns:
        tuple: (success, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat)
               - success: 交接是否成功
               - right_initial_pos/right_initial_quat/right_initial_qpos: 右臂初始位姿（用于步骤7返回）
    """
    print(f"\n{'=' * 60}")
    print(f"✅ pass_to 成功，开始交接流程（右臂接取试管）...")
    print(f"{'=' * 60}")

    left_arm_idx = robot_idx          # 左臂（抓着试管的臂）
    right_arm_idx = 1 - robot_idx     # 右臂（来接的臂）

    agent = env.unwrapped.agent

    # 保存右臂初始姿态
    right_robot = agent.agents[right_arm_idx]
    right_initial_ee_pose = right_robot.tcp_pose
    right_initial_p = right_initial_ee_pose.p
    if len(right_initial_p.shape) == 2:
        right_initial_p = right_initial_p[0]
    right_initial_pos = np.array(right_initial_p.cpu().numpy() if hasattr(right_initial_p, 'cpu') else right_initial_p, dtype=np.float32)
    right_initial_q = right_initial_ee_pose.q
    if len(right_initial_q.shape) == 2:
        right_initial_q = right_initial_q[0]
    right_initial_quat = np.array(right_initial_q.cpu().numpy() if hasattr(right_initial_q, 'cpu') else right_initial_q, dtype=np.float32)
    right_initial_qpos = right_robot.robot.get_qpos()[0].cpu().numpy().copy()
    print(f"  保存右臂初始位置: {right_initial_pos}")
    print(f"  保存右臂初始姿态: {right_initial_quat}")

    # 读取试管当前位置和姿态
    tube_current_pos, tube_current_quat = get_tube_pose(env, tube_idx)
    if tube_current_pos is None:
        print("  ❌ 无法获取试管位置，跳过交接")
        return False, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    tube_current_euler = R.from_quat(tube_current_quat).as_euler('xyz', degrees=True)
    print(f"  试管当前位置: [{tube_current_pos[0]:.3f}, {tube_current_pos[1]:.3f}, {tube_current_pos[2]:.3f}]")
    print(f"  试管当前姿态 (欧拉角): [{tube_current_euler[0]:.1f}, {tube_current_euler[1]:.1f}, {tube_current_euler[2]:.1f}]°")
    print(f"  试管当前姿态 (四元数): [{tube_current_quat[0]:.4f}, {tube_current_quat[1]:.4f}, {tube_current_quat[2]:.4f}, {tube_current_quat[3]:.4f}]")

    left_robot = agent.agents[left_arm_idx]

    # ========== 步骤0: 读取右臂当前TCP位姿作为交接姿态 ==========
    right_ee_pose = right_robot.tcp_pose
    right_p = right_ee_pose.p
    right_q = right_ee_pose.q
    if len(right_p.shape) == 2:
        right_p = right_p[0]
    if len(right_q.shape) == 2:
        right_q = right_q[0]
    right_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
    right_quat = np.array(right_q.cpu().numpy() if hasattr(right_q, 'cpu') else right_q, dtype=np.float32)
    print(f"\n  步骤0: 读取右臂当前TCP位姿作为交接姿态...")
    print(f"    位置: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
    print(f"    四元数: [{right_quat[0]:.4f}, {right_quat[1]:.4f}, {right_quat[2]:.4f}, {right_quat[3]:.4f}]")
    recorder.set_subtask("left arm passes the tube to the right arm")

    w_curr, x_curr, y_curr, z_curr = right_quat
    right_grasp_quat = np.array([w_curr, x_curr, y_curr, z_curr], dtype=np.float32)

    # 绕世界Y轴-45° + 自身X轴60°
    right_quat_scipy = np.array([x_curr, y_curr, z_curr, w_curr], dtype=np.float32)
    r_curr = R.from_quat(right_quat_scipy)
    angle_deg = -45.0
    r_rot = R.from_euler('y', np.deg2rad(angle_deg))
    r_new = r_rot * r_curr
    right_grasp_quat_scipy = r_new.as_quat()

    r_curr_after_y = R.from_quat(right_grasp_quat_scipy)
    r_rot_x = R.from_euler('x', np.deg2rad(60))
    r_new_combined = r_curr_after_y * r_rot_x
    right_grasp_quat_rotated_scipy = r_new_combined.as_quat()

    x_new, y_new, z_new, w_new = right_grasp_quat_rotated_scipy
    right_grasp_quat_rotated = np.array([w_new, x_new, y_new, z_new], dtype=np.float32)
    print(f"    原始TCP四元数 (sapien): [{w_curr:.4f}, {x_curr:.4f}, {y_curr:.4f}, {z_curr:.4f}]")
    print(f"    绕世界Y轴-45°+自身X轴60°后四元数: [{w_new:.4f}, {x_new:.4f}, {y_new:.4f}, {z_new:.4f}]")

    # 右臂交接姿态（基轴定义）
    z_axis = np.array([0, 1, 0])
    x_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    right_handover_quat = R.from_matrix(rot_mat).as_quat()

    # ========== 步骤0b: 右臂原地上升到试管高度 ==========
    right_raise_pos = right_pos.copy()
    right_raise_pos[2] = tube_current_pos[2]
    print(f"\n  步骤0b: 右臂原地上升到试管高度...")
    print(f"    当前位置: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
    print(f"    目标高度: [{right_raise_pos[0]:.3f}, {right_raise_pos[1]:.3f}, {right_raise_pos[2]:.3f}]")
    recorder.set_subtask("left arm passes the tube to the right arm")
    success_raise = move_to_position_only(
        env, robot_idx=right_arm_idx, target_pos=right_raise_pos,
        target_quat=right_grasp_quat, gripper_open=True, steps=30,
        record_step=record_step,
        allow_adaptive_search=True, limit_search_range=False,
    )
    if not success_raise:
        print(f"  ❌ 右臂上升失败")
        return False, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    print(f"  ✅ 右臂已上升到试管高度")

    # ========== 步骤1: 右臂张开夹具，从右侧接近试管 ==========
    left_ee_pose = left_robot.tcp_pose
    left_p = left_ee_pose.p
    if len(left_p.shape) == 2:
        left_p = left_p[0]
    left_z = float(np.array(left_p.cpu().numpy() if hasattr(left_p, 'cpu') else left_p)[2])
    
    right_approach_pos = tube_current_pos.copy()
    right_approach_pos[0] -= 0
    right_approach_pos[1] -= 0.05
    right_approach_pos[2] = left_z - right_height  # 基于左臂末端高度下降0.05
    print(f"\n  步骤1: 右臂从右侧接近试管...")
    print(f"    左臂末端Z: {left_z:.3f}m")
    print(f"    试管位置: [{tube_current_pos[0]:.3f}, {tube_current_pos[1]:.3f}, {tube_current_pos[2]:.3f}]")
    print(f"    接近位置: [{right_approach_pos[0]:.3f}, {right_approach_pos[1]:.3f}, {right_approach_pos[2]:.3f}]")
    recorder.set_subtask("left arm passes the tube to the right arm")
    success_step = move_to_position_only(
        env, robot_idx=right_arm_idx, target_pos=right_approach_pos,
        target_quat=right_handover_quat, gripper_open=True, steps=50,
        record_step=record_step,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success_step:
        print(f"  ❌ 右臂接近试管失败")
        return False, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    print(f"  ✅ 右臂已接近试管")

    # ========== 步骤2: 右臂移动到抓取位置 ==========
    right_grasp_pos = tube_current_pos.copy()
    right_grasp_pos[0] += 0
    right_grasp_pos[1] += 0.02
    right_grasp_pos[2] = left_z - right_height
    print(f"\n  步骤2: 右臂移动到抓取位置...")
    print(f"    抓取位置: [{right_grasp_pos[0]:.3f}, {right_grasp_pos[1]:.3f}, {right_grasp_pos[2]:.3f}]")
    recorder.set_subtask("left arm passes the tube to the right arm")
    success_grasp = move_to_position_only(
        env, robot_idx=right_arm_idx,
        target_pos=right_grasp_pos,
        target_quat=right_handover_quat,
        steps=40, record_step=record_step,
        gripper_open=True, allow_adaptive_search=True,
        limit_search_range=True,
    )
    if not success_grasp:
        print(f"  ❌ 右臂移动到抓取位置失败")
        return False, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    print(f"  ✅ 右臂已到达抓取位置")

    # ========== 步骤3: 右臂闭合夹具接住试管 ==========
    print(f"\n  步骤3: 右臂闭合夹具接住试管...")
    recorder.set_subtask("left arm passes the tube to the right arm")
    try:
        joints_r = right_robot.robot.get_active_joints()
        gripper_joints_r = [j for j in joints_r if 'joint7' in j.name or 'joint8' in j.name]
        if gripper_joints_r:
            for j in gripper_joints_r:
                j.set_drive_properties(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE_LIMIT)
    except Exception as e:
        print(f"  [info] 无法调整右臂夹爪参数: {e}")

    GRIPPER_OPEN_POS = 1.0   # 夹爪张开位置
    print(f"  [debug] 步骤3开始: GRIPPER_CLOSE_STEPS={GRIPPER_CLOSE_STEPS}, GRIPPER_CLOSE_POS={GRIPPER_CLOSE_POS}")
    for i in range(GRIPPER_CLOSE_STEPS):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        right_robot_uid = f"piper_x-{right_arm_idx}"
        if right_robot_uid in action:
            current_qpos = right_robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            progress = (i + 1) / GRIPPER_CLOSE_STEPS
            gripper_pos = GRIPPER_OPEN_POS + (GRIPPER_CLOSE_POS - GRIPPER_OPEN_POS) * progress
            action[right_robot_uid] = np.concatenate([arm_qpos, [gripper_pos]])
        left_robot_uid = f"piper_x-{left_arm_idx}"
        if left_robot_uid in action:
            left_qpos = left_robot.robot.get_qpos()[0].cpu().numpy()
            left_arm_qpos = left_qpos[:6]
            action[left_robot_uid] = np.concatenate([left_arm_qpos, [GRIPPER_CLOSE_POS]])
        if record_step:
            record_step(action)
        else:
            env.step(action)
        if i % 10 == 0 or i == GRIPPER_CLOSE_STEPS - 1:
                print(f"  [debug] 步骤3 #{i}: 右臂gripper_pos={gripper_pos:.3f}")
    print(f"  ✅ 右臂已接住试管")

    # ========== 步骤3b: 右臂稳定等待 ==========
    print(f"\n  步骤3b: 右臂稳定等待 {STABILIZE_GRASP_STEPS} 帧...")
    for i in range(STABILIZE_GRASP_STEPS):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        right_robot_uid = f"piper_x-{right_arm_idx}"
        if right_robot_uid in action:
            current_qpos = right_robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            action[right_robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        left_robot_uid = f"piper_x-{left_arm_idx}"
        if left_robot_uid in action:
            left_qpos = left_robot.robot.get_qpos()[0].cpu().numpy()
            left_arm_qpos = left_qpos[:6]
            action[left_robot_uid] = np.concatenate([left_arm_qpos, [GRIPPER_CLOSE_POS]])
        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 右臂已稳定夹住")

    # ========== 步骤4: 左臂张开夹具释放试管 ==========
    print(f"\n  步骤4: 左臂张开夹具释放试管...")
    
    # 短暂等待确认右臂抓稳
    print(f"  等待确认右臂抓稳...")
    for i in range(10):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        left_robot_uid = f"piper_x-{left_arm_idx}"
        if left_robot_uid in action:
            left_qpos = left_robot.robot.get_qpos()[0].cpu().numpy()
            left_arm_qpos = left_qpos[:6]
            action[left_robot_uid] = np.concatenate([left_arm_qpos, [GRIPPER_CLOSE_POS]])
        right_robot_uid = f"piper_x-{right_arm_idx}"
        if right_robot_uid in action:
            right_qpos = right_robot.robot.get_qpos()[0].cpu().numpy()
            right_arm_qpos = right_qpos[:6]
            action[right_robot_uid] = np.concatenate([right_arm_qpos, [GRIPPER_CLOSE_POS]])
        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  开始释放...")
    
    recorder.set_subtask("left arm passes the tube to the right arm")
    for i in range(30):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        left_robot_uid = f"piper_x-{left_arm_idx}"
        if left_robot_uid in action:
            current_qpos = left_robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            progress = (i + 1) / 30
            left_gripper_pos = GRIPPER_CLOSE_POS + (GRIPPER_OPEN_POS - GRIPPER_CLOSE_POS) * progress
            action[left_robot_uid] = np.concatenate([arm_qpos, [left_gripper_pos]])
            if i % 10 == 0 or i == 29:
                print(f"  [debug] 步骤4 #{i}: 左臂gripper_pos={left_gripper_pos:.3f}")
        right_robot_uid = f"piper_x-{right_arm_idx}"
        if right_robot_uid in action:
            right_qpos = right_robot.robot.get_qpos()[0].cpu().numpy()
            right_arm_qpos = right_qpos[:6]
            action[right_robot_uid] = np.concatenate([right_arm_qpos, [GRIPPER_CLOSE_POS]])
        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 左臂已释放试管，试管现由右臂持有")

    # ========== 步骤5: 左臂后撤 ==========
    print("\n  步骤5: 左臂后撤...")
    left_ee_pose = left_robot.tcp_pose
    left_p = left_ee_pose.p
    if len(left_p.shape) == 2:
        left_p = left_p[0]
    left_curr_pos = np.array(left_p.cpu().numpy() if hasattr(left_p, 'cpu') else left_p, dtype=np.float32)
    retreat_pos = left_curr_pos.copy()
    retreat_pos[0] -= 0
    retreat_pos[1] += 0.1
    retreat_pos[2] += 0.2
    retreat_pos[2] = min(retreat_pos[2], MAX_Z_HEIGHT)
    left_curr_quat = left_robot.tcp_pose.q
    if len(left_curr_quat.shape) == 2:
        left_curr_quat = left_curr_quat[0]
    left_curr_quat = np.array(left_curr_quat.cpu().numpy() if hasattr(left_curr_quat, 'cpu') else left_curr_quat, dtype=np.float32)
    
    move_to_position_only(
        env, robot_idx=left_arm_idx,
        target_pos=retreat_pos,
        target_quat=left_curr_quat,  # 保持当前姿态后撤
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True,
    )

    # ========== 步骤6: 左臂返回初始位置 ==========
    print(f"\n  步骤6: 左臂返回初始位置...")
    recorder.set_subtask("left arm retracts")
    return_arm_to_initial_pose(
        env=env,
        robot_idx=left_arm_idx,
        target_pos=initial_ee_pos,
        target_quat=initial_ee_quat,
        initial_qpos=initial_qpos,
        record_step=record_step,
    )

    return True, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
def right_place_procedure(
    env, tube_idx,
    right_arm_idx, left_arm_idx,
    right_robot, left_robot,
    right_handover_quat,
    right_initial_pos, right_initial_quat, right_initial_qpos,
    right_place_idx,
    recorder, record_step,):
    """
    右臂放置试管到指定位置

    步骤: 7a(移动到上方) → 姿态补偿 → 7b(下降插入) → 7c(释放) → 7d(后撤)
    """
    place_target = np.array(cube_place02_pos[right_place_idx], dtype=np.float32)
    # 根据位置确定目标架子颜色
    if place_target[1] > 0.1 or place_target[1] < -0.1:
        target_color = "yellow"
    else:
        target_color = "red"
    print(f"\n{'=' * 60}")
    print(f"  步骤7: 右臂放置试管到位置 {right_place_idx}...")
    print(f"    目标位置: [{place_target[0]:.3f}, {place_target[1]:.3f}, {place_target[2]:.3f}]")
    print(f"    目标架子颜色: {target_color}")
    print(f"{'=' * 60}")

    # 7a: 移动到目标位置上方
    above_pos = place_target.copy()
    above_pos[2] = place_height
    above_pos[2] = min(above_pos[2], MAX_Z_HEIGHT)
    print(f"\n  步骤7a: 右臂移动到目标上方...")
    recorder.set_subtask(f"right arm moves the tube.")
    success_above = move_to_position_only(
        env, robot_idx=right_arm_idx,
        target_pos=above_pos,
        target_quat=right_handover_quat,
        steps=50, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=False,
    )

    if success_above:
        # ========== 步骤7a-1: 先切换到矩阵基座姿态 ==========
        if right_arm_idx == 0:
            z_axis = np.array([0, -1, 0])
        else:
            z_axis = np.array([0, 1, 0])
        x_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        rot_mat = np.column_stack([x_axis, y_axis, z_axis])
        matrix_base_quat = R.from_matrix(rot_mat).as_quat()
        print(f"\n  步骤7a-1: 切换到矩阵基座姿态（中间态）...")
        print(f"    矩阵基座四元数: [{matrix_base_quat[0]:.4f}, {matrix_base_quat[1]:.4f}, {matrix_base_quat[2]:.4f}, {matrix_base_quat[3]:.4f}]")
        recorder.set_subtask(f"right arm places the tube (on yellow rack)")
        right_ee_pose = right_robot.tcp_pose
        right_p = right_ee_pose.p
        if len(right_p.shape) == 2:
            right_p = right_p[0]
        curr_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
        move_to_position_only(
            env, robot_idx=right_arm_idx,
            target_pos=curr_pos,  # 保持当前位置不变
            target_quat=matrix_base_quat,
            steps=60, record_step=record_step,
            gripper_open=False,
            allow_adaptive_search=True, limit_search_range=True,
        )

        # 读取试管当前姿态，补偿pitch角
        if tube_idx is not None:
            tube_place_pos, tube_place_quat = get_tube_end_pose(env, tube_idx)
        else:
            tube_place_pos = None
        if tube_place_pos is not None:
            tube_place_euler = R.from_quat(tube_place_quat).as_euler('xyz', degrees=True)
            pitch_angle = tube_place_euler[1]
            roll_angle = tube_place_euler[0]
            print(f"\n  试管当前姿态 (欧拉角): [{tube_place_euler[0]:.1f}, {tube_place_euler[1]:.1f}, {tube_place_euler[2]:.1f}]°")
            print(f"  补偿roll角: {-roll_angle:.1f}°")
            print(f"  补偿pitch角: {-pitch_angle:.1f}°")

            theta = np.deg2rad(pitch_angle)  # 补偿角度（弧度）
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            if right_arm_idx == 0:
                z_axis = np.array([0, -1, 0])
            else:
                z_axis = np.array([0, 1, 0])
            x_axis = np.array([-sin_t, 0, -cos_t])
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])
            adjusted_quat = R.from_matrix(rot_mat).as_quat()

            # ========== 步骤7a-2: 先调整姿态（原地旋转，不移动位置） ==========
            print(f"\n  步骤7a-2: 原地调整姿态补偿...")
            print(f"    使用调整后姿态: 四元数=[{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
            recorder.set_subtask(f"right arm places the tube (on yellow rack)")
            right_ee_pose = right_robot.tcp_pose
            right_p = right_ee_pose.p
            if len(right_p.shape) == 2:
                right_p = right_p[0]
            curr_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
            move_to_position_only(
                env, robot_idx=right_arm_idx,
                target_pos=curr_pos,  # 保持当前位置不变
                target_quat=adjusted_quat,
                steps=80, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )

            # ========== 步骤7a-3: 姿态调整后，重新读取试管位置，调整TCP位置对齐 ==========
            tube_place_pos2, tube_place_quat2 = get_tube_end_pose(env, tube_idx)
            if tube_place_pos2 is not None:
                print(f"\n  步骤7a-3: 姿态调整后重新读取试管位置，调整TCP位置...")
                right_ee_pose = right_robot.tcp_pose
                right_p = right_ee_pose.p
                if len(right_p.shape) == 2:
                    right_p = right_p[0]
                curr_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
                tube_offset = place_target - tube_place_pos2
                adjusted_target = curr_pos + tube_offset
                adjusted_target[2] = curr_pos[2]
                print(f"    右臂TCP当前位置: [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
                print(f"    试管位置: [{tube_place_pos2[0]:.3f}, {tube_place_pos2[1]:.3f}, {tube_place_pos2[2]:.3f}]")
                print(f"    目标放置位置: [{place_target[0]:.3f}, {place_target[1]:.3f}, {place_target[2]:.3f}]")
                print(f"    调整后目标位置: [{adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f}]")
                recorder.set_subtask(f"right arm places the tube (on yellow rack)")
                adjusted_target[1] -= 0.0
                move_to_position_only(
                    env, robot_idx=right_arm_idx,
                    target_pos=adjusted_target,
                    target_quat=adjusted_quat,
                    steps=50, record_step=record_step,
                    gripper_open=False,
                    allow_adaptive_search=True, limit_search_range=True,
                )
            else:
                print(f"  ⚠️ 姿态调整后无法读取试管位置，使用原始偏移")
                right_ee_pose = right_robot.tcp_pose
                right_p = right_ee_pose.p
                if len(right_p.shape) == 2:
                    right_p = right_p[0]
                curr_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
                tube_offset = place_target - tube_place_pos
                adjusted_target = curr_pos + tube_offset
                adjusted_target[2] = curr_pos[2]
        else:
            print(f"  ⚠️ 无法读取试管姿态，使用原始姿态和位置")
            adjusted_quat = right_handover_quat
            adjusted_target = place_target
        adjusted_target[2] = place_target[2]
        # 7b: 圆形搜索下降到插入位置
        print(f"\n  步骤7b: 右臂圆形搜索放置位置...")
        print(f"    调整后姿态: 四元数=[{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
        recorder.set_subtask(f"right arm places the tube (on yellow rack)")
        # ========== 步骤6: 分步下降并实时调整 ==========
        print(f"\n  步骤6: right arm 分步下降放置（先调整位置再下降）...")

        # 获取当前位置作为下降起点
        ee_pose = right_robot.tcp_pose
        p = ee_pose.p
        if len(p.shape) == 2:
            p = p[0]
        current_pos_before_descent = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
        START_Z = current_pos_before_descent[2]
        TARGET_Z = adjusted_target[2]
        Z_STEP = (START_Z - TARGET_Z) / NUM_DESCENT_STEPS
        print(f"    下降起点高度: {START_Z:.3f}m, 目标高度: {TARGET_Z:.3f}m")
        # ========== 步骤6: 分步下降并实时调整 ==========
        # 先定义 end_pos（目标放置位置）
        pos_idx = right_place_idx if right_place_idx is not None else 0
        end_pos = np.array(cube_place02_pos[pos_idx], dtype=np.float32)

        print(f"\n  步骤6: right arm 分步下降放置（先调整位置再下降）...")
        for descent_step in range(NUM_DESCENT_STEPS):
            current_z = START_Z - descent_step * Z_STEP  # 当前高度
            target_z = START_Z - (descent_step + 1) * Z_STEP  # 目标高度
            print(f"\n  下降步骤 {descent_step + 1}/{NUM_DESCENT_STEPS}: {current_z:.3f}m → {target_z:.3f}m")
            
            # ===== 步骤6a: 在当前高度原地调整位置 =====
            tube_place_pos_temp, tube_place_quat_temp = get_tube_end_pose(env, tube_idx)
            
            # 调整姿态
            if tube_place_quat_temp is not None:
                tube_rot = R.from_quat([tube_place_quat_temp[1], tube_place_quat_temp[2], 
                                      tube_place_quat_temp[3], tube_place_quat_temp[0]])
                tube_z_world = tube_rot.apply([0, 0, 1])
                world_z = np.array([0, 0, 1])
                R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
                ee_pose = right_robot.tcp_pose
                ee_q = ee_pose.q
                if len(ee_q.shape) == 2:
                    ee_q = ee_q[0]
                gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
                gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
                gripper_rot_new = R_map * gripper_rot
                adjusted_quat_scipy = gripper_rot_new.as_quat()
                adjusted_quat_temp = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
                                               adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
            else:
                adjusted_quat_temp = adjusted_quat
            
            # 调整位置（Z保持当前高度不变）
            if tube_place_pos_temp is not None:
                ee_pose = right_robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_temp = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                adjusted_pos = curr_pos_temp + tube_offset
                adjusted_pos[2] = current_z  # 保持当前高度
                print(f"    步骤6a: 调整位置 [{curr_pos_temp[0]:.3f}, {curr_pos_temp[1]:.3f}] → [{adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}] (Z={current_z:.3f}m不变)")
                print(f"      试管位置: [{tube_place_pos_temp[0]:.3f}, {tube_place_pos_temp[1]:.3f}, {tube_place_pos_temp[2]:.3f}]")
                print(f"      目标位置(end_pos): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
                print(f"      偏移量: [{tube_offset[0]:.3f}, {tube_offset[1]:.3f}, {tube_offset[2]:.3f}]")
            else:
                adjusted_pos = adjusted_target.copy()
                adjusted_pos[2] = current_z
            
            # 原地调整位置（Z不变）
            success_adjust = move_to_position_only(
                env, robot_idx=right_arm_idx,
                target_pos=adjusted_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            if not success_adjust:
                print(f"    ⚠️ 步骤6a调整失败，继续...")
            
            # ===== 步骤6b: 下降到目标高度 =====
            if tube_place_pos_temp is not None:
                ee_pose = right_robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_after = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                descend_pos = curr_pos_after + tube_offset
                descend_pos[2] = target_z  # 下降到目标高度
                print(f"    步骤6b: 下降 {curr_pos_after[2]:.3f}m → {target_z:.3f}m")
            else:
                descend_pos = adjusted_pos.copy()
                descend_pos[2] = target_z
            
            success_descend = move_to_position_only(
                env, robot_idx=right_arm_idx,
                target_pos=descend_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            # ===== 高度检测：试管末端是否已到达目标位置 =====
            tube_check_pos, _ = get_tube_end_pose(env, tube_idx)
            if tube_check_pos is not None:
                tube_check_pos = np.array(tube_check_pos, dtype=np.float32)
                # 如果试管末端高度 < 目标位置高度，说明试管已放到目标位置
                if tube_check_pos[2] < end_pos[2] + check_height:   # 允许1cm误差
                    print(f"    ✅ 试管已到达目标高度，停止下降")
                    break
            if not success_descend:
                print(f"  ⚠️ 步骤6b下降失败，继续...")

        # 最终放置（降速：steps从50改为80）
        print(f"\n  最终放置...")
        recorder.set_subtask(f"right arm places the tube (on yellow rack)")
        # ========== 步骤7c: 张开夹具停顿，让试管滑入 ==========
    print(f"\n  步骤7c: 张开夹具停顿...")
    for i in range(50):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        right_robot_uid = f"piper_x-{right_arm_idx}"
        if right_robot_uid in action:
            right_qpos = right_robot.robot.get_qpos()[0].cpu().numpy()
            right_arm_qpos = right_qpos[:6]
            action[right_robot_uid] = np.concatenate([right_arm_qpos, [1.0]])  # 张开夹具
        
        left_robot_uid = f"piper_x-{left_arm_idx}"
        if left_robot_uid in action:
            left_qpos = left_robot.robot.get_qpos()[0].cpu().numpy()
            left_arm_qpos = left_qpos[:6]
            action[left_robot_uid] = np.concatenate([left_arm_qpos, [1.0]])

        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 张开夹具停顿完成")
    #     success_insert, final_pos, final_quat = try_place_with_circular_search(
    #         env=env,
    #         robot_idx=right_arm_idx,
    #         robot=right_robot,
    #         target_pos=adjusted_target,
    #         target_quat=adjusted_quat,
    #         record_step=record_step,
    #         gripper_open_after=True,
    #         other_robot_idx=left_arm_idx,
    #         other_robot=left_robot,
    #         tube_idx=tube_idx,
    #     )

    #     if success_insert:
    #         print(f"  ✅ 右臂圆形搜索放置成功")
    #     else:
    #         print(f"  ❌ 右臂下降插入失败")
    
    # # 7d: 后撤到初始位置
    print(f"\n  步骤7d: 右臂后撤到初始位置上方...")
    recorder.set_subtask("right arm retracts")
    right_initial_pos[2]+=0.2
    right_initial_pos[2] = min(right_initial_pos[2], MAX_Z_HEIGHT)
        # 步骤7d-1: 保持当前姿态移动到(-0.2, 0, 当前高度)
    print(f"    步骤7d-1: 移动到(-0.2, 0, 当前高度)...")
    right_ee_pose = right_robot.tcp_pose
    right_p = right_ee_pose.p
    if len(right_p.shape) == 2:
        right_p = right_p[0]
    right_curr_pos = np.array(right_p.cpu().numpy() if hasattr(right_p, 'cpu') else right_p, dtype=np.float32)
    right_curr_quat = right_robot.tcp_pose.q
    if len(right_curr_quat.shape) == 2:
        right_curr_quat = right_curr_quat[0]
    right_curr_quat = np.array(right_curr_quat.cpu().numpy() if hasattr(right_curr_quat, 'cpu') else right_curr_quat, dtype=np.float32)
    pre_retreat_pos = np.array([-0.3, 0.0, right_curr_pos[2]], dtype=np.float32)
    move_to_position_only(
        env, robot_idx=right_arm_idx,
        target_pos=pre_retreat_pos,
        target_quat=right_curr_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True,
    )

    return_arm_to_initial_pose(
        env=env,
        robot_idx=right_arm_idx,
        target_pos=right_initial_pos,
        target_quat=right_initial_quat,
        initial_qpos=right_initial_qpos,
        record_step=record_step,
    )
def grasp_and_place_02(
    env, tube_idx,
    robot_idx, other_robot_idx,
    robot, other_robot,
    start_idx, end_idx,
    recorder, record_step,
    right_handover_quat=None,
    repeat_id=None,):
    """
    左臂专用夹取放置函数，从左侧接近目标，使用左臂矩阵基座姿态

    与 grasp_and_place 的区别：
      1. 接近方向：pos[1] += 0.05（从左侧接近）
      2. 夹取姿态：z_axis=[0,-1,0], x_axis=[0,0,-1]（左臂矩阵基座）
      3. 放置补偿：z_axis=[0,-1,0]
    """
    arm_name = "left arm" if robot_idx == 0 else "right arm"

    def _get_tube_rack_color(pos_x, pos_y):
        if pos_y > 0.1 or pos_y < -0.1:
            return "yellow"
        else:
            return "red"

    # 解析起始位置
    if 0 <= start_idx <= 4:
        tube_start_pos, tube_start_quat = get_tube_pose(env, start_idx)
        if tube_start_pos is not None:
            start_pos = np.array(tube_start_pos, dtype=np.float32)
            start_label = f"试管{start_idx}当前位置"
            print(f"  {start_label}: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        else:
            print(f"  ❌ 无法读取试管{start_idx}位姿，回退到预设位置")
            start_pos = np.array(cube_place_pos[start_idx + 2], dtype=np.float32)
            start_label = f"cube_place_pos[{start_idx + 2}](回退)"

    source_color = _get_tube_rack_color(start_pos[0], start_pos[1])

    # 解析目标位置
    if 5 <= end_idx <= 7:
        pos_idx = end_idx - 5
        if pos_idx >= len(cube_place02_pos):
            print(f"❌ end_idx={end_idx} 对应的 cube_place02_pos 索引 {pos_idx} 超出范围 (len={len(cube_place02_pos)})")
            return False
        end_pos = np.array(cube_place02_pos[pos_idx], dtype=np.float32)
        end_label = f"cube_place02_pos[{pos_idx}]"
    elif 8 <= end_idx <= 10:
        pos_idx = end_idx - 8
        if pos_idx >= len(cube_place03_pos):
            print(f"❌ end_idx={end_idx} 对应的 cube_place03_pos 索引 {pos_idx} 超出范围 (len={len(cube_place03_pos)})")
            return False
        end_pos = np.array(cube_place03_pos[pos_idx], dtype=np.float32)
        end_label = f"cube_place03_pos[{pos_idx}]"
    else:
        print(f"❌ end_idx={end_idx} 无效，必须为 5-12")
        return False

    target_color = _get_tube_rack_color(end_pos[0], end_pos[1])

    # 获取试管当前姿态并计算其z轴朝向
    tube_actor = get_tube_object(env, start_idx)
    if tube_actor is not None:
        tube_pose = tube_actor.pose
        tube_q = tube_pose.q
        if len(tube_q.shape) == 2:
            tube_q = tube_q[0]
        tube_q = np.array(tube_q.cpu().numpy() if hasattr(tube_q, 'cpu') else tube_q)
        tube_R = R.from_quat([tube_q[1], tube_q[2], tube_q[3], tube_q[0]])
        tube_rot_mat = tube_R.as_matrix()
        tube_z_axis = tube_rot_mat[:, 2]
    else:
        tube_z_axis = None

    # 夹取姿态：z轴指向正X方向，x轴竖直向下，绕Y轴旋转45度
    fixed_z_axis = np.array([1.0, 0.0, 0.0])  # 从正X方向指向试管
    fixed_x_axis = np.array([0.0, 0.0, -1.0])  # 夹爪闭合方向竖直向下
    
    if start_pos[0] < -0.35:
        pitch = np.deg2rad(50)
    else:
        pitch = np.deg2rad(52)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    z_axis = Ry @ fixed_z_axis
    x_axis = Ry @ fixed_x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    left_matrix_quat = R.from_matrix(rot_mat).as_quat()

    print(f"\n{'=' * 60}")
    print(f"  {arm_name} 夹取试管 (start_idx={start_idx}) 并放置到目标位置 (end_idx={end_idx})")
    print(f"    夹取位置 ({start_label}): [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    print(f"    放置位置 ({end_label}): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    if tube_z_axis is not None:
        print(f"    试管当前z轴朝向: [{tube_z_axis[0]:.4f}, {tube_z_axis[1]:.4f}, {tube_z_axis[2]:.4f}]")
    print(f"    夹具末端z轴朝向: [{z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f}]")
    if tube_z_axis is not None:
        dot_product = np.dot(tube_z_axis, z_axis)
        print(f"    点积（验证垂直性）: {dot_product:.4f} (应为0表示垂直)")
    print(f"    夹取姿态四元数 [x,y,z,w]: {left_matrix_quat}")
    print(f"{'=' * 60}")

    # 保存初始位姿
    current_ee_pose = robot.tcp_pose
    current_p = current_ee_pose.p
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    initial_pos = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q = current_ee_pose.q
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    initial_quat = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)
    initial_qpos = robot.robot.get_qpos()[0].cpu().numpy().copy()
    
    # ========== 步骤0: 原地升起机械臂至安全高度 ==========
    RAISE_HEIGHT_OFFSET = 0.20
    raised_pos = initial_pos.copy()
    raised_pos[2] += RAISE_HEIGHT_OFFSET
    raised_pos[2] = min(raised_pos[2], MAX_Z_HEIGHT)
    print(f"\n  步骤0: {arm_name} 原地升起至安全高度（+{RAISE_HEIGHT_OFFSET*100:.0f}cm）...")
    recorder.set_subtask(f"left arm approaches for the tube (on yellow rack)")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=raised_pos,
        target_quat=initial_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=False,
    )
    if not success:
        print(f"  ❌ 原地升起失败")
        return False
    # initial_pos = raised_pos.copy()
    print(f"  ✅ 原地升起成功")

    # ========== 步骤1: 读取两个夹爪位置，调整使试管中心在中间 ==========
    tube_pos_1, _ = get_tube_pose(env, start_idx)
    if tube_pos_1 is not None:
        tube_pos_1 = np.array(tube_pos_1, dtype=np.float32)
    else:
        tube_pos_1 = start_pos.copy()
    
    current_ee_pose = robot.tcp_pose
    current_ee_pos = np.array(current_ee_pose.p.cpu().numpy() if hasattr(current_ee_pose.p, 'cpu') else current_ee_pose.p)
    if len(current_ee_pos.shape) == 2:
        current_ee_pos = current_ee_pos[0]
    current_ee_pos = current_ee_pos.astype(np.float32)
    
    finger_spacing = 0.035  # 每个手指距中心4cm
    
    ee_quat = np.array(current_ee_pose.q.cpu().numpy() if hasattr(current_ee_pose.q, 'cpu') else current_ee_pose.q)
    if len(ee_quat.shape) == 2:
        ee_quat = ee_quat[0]
    ee_R = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
    
    local_finger1 = np.array([0, finger_spacing, 0])
    local_finger2 = np.array([0, -finger_spacing, 0])
    
    finger1_pos = current_ee_pos + ee_R.as_matrix() @ local_finger1
    finger2_pos = current_ee_pos + ee_R.as_matrix() @ local_finger2
    gripper_center = (finger1_pos + finger2_pos) / 2
    
    print(f"\n  步骤1: 对准试管中心...")
    print(f"    夹爪手指1: [{finger1_pos[0]:.3f}, {finger1_pos[1]:.3f}]")
    print(f"    夹爪手指2: [{finger2_pos[0]:.3f}, {finger2_pos[1]:.3f}]")
    print(f"    夹爪中心: [{gripper_center[0]:.3f}, {gripper_center[1]:.3f}]")
    print(f"    试管位置: [{tube_pos_1[0]:.3f}, {tube_pos_1[1]:.3f}]")
    
    offset = tube_pos_1 - gripper_center
    offset[2] = 0
    
    # 需要改成：
    # 先计算yaw角
    yaw = np.arctan2(z_axis[1], z_axis[0])
    # 基于yaw角计算接近位置
    target_pos = tube_pos_1.copy()
    target_pos[0] -= 0.08 * np.cos(yaw)  # 减号：旁边
    target_pos[1] -= 0.08 * np.sin(yaw)
    target_pos[2] += 0.1
    target_pos[2] = min(target_pos[2], MAX_Z_HEIGHT)
    
    print(f"    调整偏移: [{offset[0]:.3f}, {offset[1]:.3f}]")
    print(f"    目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    recorder.set_subtask(f"left arm approaches for the tube (on yellow rack)")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=target_pos,
        target_quat=left_matrix_quat,
        steps=100, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 对准失败")
        return False
    print(f"  ✅ 对准成功")

    # ========== 步骤2: 下降到试管中心向上0.03m处 ==========
    tube_end_pos, _ = get_tube_pose(env, start_idx)
    if tube_end_pos is not None:
        tube_end_pos = np.array(tube_end_pos, dtype=np.float32)
    else:
        tube_end_pos = start_pos.copy()
    
    # 需要改成：
    yaw = np.arctan2(z_axis[1], z_axis[0])
    grasp_pos = tube_end_pos.copy()
    grasp_pos[0] += 0.01 * np.cos(yaw)   # 加号：目标位置
    grasp_pos[1] += 0.01 * np.sin(yaw)
    grasp_pos[2] += grasp_height
    
    print(f"\n  步骤2: 下降到抓取位置...")
    print(f"    目标位置: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    recorder.set_subtask(f"left arm approaches for the tube (on yellow rack)")
    
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=grasp_pos,
        target_quat=left_matrix_quat,
        steps=100, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ⚠️ 下降失败，继续尝试...")
    
    if not success:
        print(f"  ❌ 下降到夹取位置失败")
        return False
    print(f"  ✅ 下降到夹取位置成功")

    # ========== 步骤3: 闭合夹爪抓取试管 ==========
    print(f"\n  步骤3: {arm_name} 闭合夹爪抓取试管...")
    recorder.set_subtask(f"left arm picks up the tube (from yellow rack)")
    
    # 增加夹爪闭合力度
    old_stiffness = None
    old_force_limit = None
    try:
        joints = robot.robot.get_active_joints()
        gripper_joints = [j for j in joints if 'joint7' in j.name or 'joint8' in j.name]
        if gripper_joints:
            old_stiffness = [j.get_stiffness() for j in gripper_joints]
            old_force_limit = [j.get_force_limit() for j in gripper_joints]
            for j in gripper_joints:
                j.set_drive_properties(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE_LIMIT)
    except Exception as e:
        print(f"  [info] 无法调整夹爪参数: {e}")

    GRIPPER_OPEN_POS = 1.0   # 夹爪张开位置
    for i in range(GRIPPER_CLOSE_STEPS):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            progress = (i + 1) / GRIPPER_CLOSE_STEPS
            gripper_pos = GRIPPER_OPEN_POS + (GRIPPER_CLOSE_POS - GRIPPER_OPEN_POS) * progress
            action[robot_uid] = np.concatenate([arm_qpos, [gripper_pos]])
        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action:
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            other_progress = (i + 1) / GRIPPER_CLOSE_STEPS
            other_gripper_pos = GRIPPER_OPEN_POS + (GRIPPER_CLOSE_POS - GRIPPER_OPEN_POS) * other_progress
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [other_gripper_pos]])
        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 夹爪闭合完成")
    
    # ========== 抓取后稳定等待 - 检测夹爪闭合状态 ==========
    print(f"  步骤3b: 等待夹爪闭合...")
    GRIPPER_CLOSED_THRESHOLD = -1.5  # 夹爪闭合阈值
    max_wait_steps = 60  # 最大等待步数
    gripper_closed = False
    
    for i in range(max_wait_steps):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            gripper_pos = current_qpos[6] if len(current_qpos) > 6 else GRIPPER_CLOSE_POS
            
            if gripper_pos <= GRIPPER_CLOSED_THRESHOLD:
                gripper_closed = True
            
            action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])

        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action:
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])

        if record_step:
            record_step(action)
        else:
            env.step(action)
        
        if gripper_closed and i >= 10:
            print(f"  ✅ 夹爪已闭合 (位置: {gripper_pos:.3f})，等待 {i+1} 步后继续")
            break
    
    if not gripper_closed:
        print(f"  ⚠️ 夹爪未完全闭合 (位置: {gripper_pos:.3f})，但继续执行")
    print(f"  ✅ 稳定等待完成")

    # ========== 步骤4: 分步小幅度提起（每次0.04m），共提0.2m ==========
    # STEP_LIFT_HEIGHT = 0.05  # 每步提起高度
    # NUM_LIFT_STEPS = 5      # 提起步数（0.04m × 5 = 0.2m）
    
    ee_pose = robot.tcp_pose
    q = ee_pose.q
    if len(q.shape) == 2:
        q = q[0]
    lift_quat = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    
    # 使用实际的当前末端位置作为提起起点，而不是 grasp_pos
    actual_ee_pose = robot.tcp_pose
    actual_ee_p = actual_ee_pose.p
    if len(actual_ee_p.shape) == 2:
        actual_ee_p = actual_ee_p[0]
    current_lift_pos = np.array(actual_ee_p.cpu().numpy() if hasattr(actual_ee_p, 'cpu') else actual_ee_p, dtype=np.float32)
    print(f"  [提起起点] 实际末端: [{current_lift_pos[0]:.3f}, {current_lift_pos[1]:.3f}, {current_lift_pos[2]:.3f}]")
    
    print(f"\n  步骤4: {arm_name} 分步提起试管（每次0.02m，共{NUM_LIFT_STEPS}步，总共0.2m）...")
    recorder.set_subtask(f"left arm picks up the tube (from yellow rack)")
    
    for lift_step in range(NUM_LIFT_STEPS):
        current_lift_pos[2] += STEP_LIFT_HEIGHT
        current_lift_pos[2] = min(current_lift_pos[2], MAX_Z_HEIGHT)
        target_z = current_lift_pos[2]
        
        # ===== 提起步骤a: 在当前高度读取试管姿态，调整使竖直 =====
        # if tube_idx is not None:
        #     tube_lift_pos, tube_lift_quat = get_tube_end_pose(env, tube_idx)
        #     if tube_lift_quat is not None:
        #         tube_rot = R.from_quat([tube_lift_quat[1], tube_lift_quat[2], 
        #                             tube_lift_quat[3], tube_lift_quat[0]])
        #         tube_z_world = tube_rot.apply([0, 0, 1])
        #         world_z = np.array([0, 0, 1])
        #         R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
        #         ee_pose = robot.tcp_pose
        #         ee_q = ee_pose.q
        #         if len(ee_q.shape) == 2:
        #             ee_q = ee_q[0]
        #         gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
        #         gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
        #         gripper_rot_new = R_map * gripper_rot
        #         adjusted_quat_scipy = gripper_rot_new.as_quat()
        #         adjusted_lift_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
        #                                     adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
        #     else:
        #         adjusted_lift_quat = lift_quat
        # else:
        # 读取当前实际姿态
        ee_pose_now = robot.tcp_pose
        q_now = ee_pose_now.q
        if len(q_now.shape) == 2:
            q_now = q_now[0]
        current_quat = np.array(q_now.cpu().numpy() if hasattr(q_now, 'cpu') else q_now, dtype=np.float32)
        adjusted_lift_quat = current_quat
        
        # # 读取当前位置
        # ee_pose_before = robot.tcp_pose
        # p_before = ee_pose_before.p
        # if len(p_before.shape) == 2:
        #     p_before = p_before[0]
        # curr_pos_before = np.array(p_before.cpu().numpy() if hasattr(p_before, 'cpu') else p_before)
        
        # # 保持当前高度，调整姿态
        # adjust_pos = curr_pos_before.copy()
        # adjust_pos[2] = curr_pos_before[2]  # 保持当前高度
        
        # print(f"    提起步骤a: 调整姿态（Z={adjust_pos[2]:.3f}m保持）...")
        # move_to_position_only(
        #     env, robot_idx=robot_idx,
        #     target_pos=adjust_pos,
        #     target_quat=adjusted_lift_quat,
        #     steps=50, record_step=record_step,
        #     gripper_open=False,
        #     allow_adaptive_search=True, limit_search_range=True,
        # )
        # 读取当前试管位置作为基准
        if tube_idx is not None:
            tube_pos, _ = get_tube_pose(env, tube_idx)
            if tube_pos is not None:
                current_lift_pos[0] = tube_pos[0]  # X从试管读取
                current_lift_pos[1] = tube_pos[1]  # Y从试管读取

        # 高度逐级增加
        pos_before_lift = current_lift_pos[2]  # 保存提升前的高度
        current_lift_pos[2] += STEP_LIFT_HEIGHT  # Z逐级增加
        target_z = current_lift_pos[2]
        # ===== 提起步骤b: 上升到目标高度 =====
        print(f"    提起步骤b: 上升 {pos_before_lift:.3f}m → {target_z:.3f}m")
        success = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=current_lift_pos,
            target_quat=adjusted_lift_quat,
            steps=100, record_step=record_step,
            gripper_open=False,
            allow_adaptive_search=True,
            limit_search_range=True,
        )
        if not success:
            print(f"  ⚠️ 步骤{lift_step + 1}提升失败，尝试基于当前位置调整yaw...")
            # 保持当前tcp姿态的roll和pitch，只调整yaw
            current_ee = robot.tcp_pose
            current_ee_p = current_ee.p
            if len(current_ee_p.shape) == 2:
                current_ee_p = current_ee_p[0]
            current_ee_np = np.array(current_ee_p.cpu().numpy() if hasattr(current_ee_p, 'cpu') else current_ee_p, dtype=np.float32)
            
            # 计算从当前位置到目标位置的yaw
            dx = current_lift_pos[0] - current_ee_np[0]
            dy = current_lift_pos[1] - current_ee_np[1]
            yaw_angle = np.arctan2(dy, dx)
            z_axis = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0.0])
            x_axis = np.array([0.0, 0.0, -1.0])
            y_axis = np.cross(z_axis, x_axis)
            rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
            lift_quat = R.from_matrix(rot_matrix).as_quat()
            success = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=current_lift_pos,
                target_quat=lift_quat,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            if not success:
                print(f"  ❌ 步骤{lift_step + 1}最终提升失败")
                return False
    
    lift_pos = current_lift_pos.copy()
    print(f"  ✅ 分步提起完成，最终高度: {lift_pos[2]:.3f}m")

    # ========== 调整夹爪姿态使试管竖直 ==========
    print(f"\n  调整夹爪姿态使试管竖直...")
    try:
        from scipy.spatial.transform import Rotation as R_scipy
        tube_actor = get_tube_object(env, tube_idx)
        if tube_actor is not None:
            # 获取当前试管姿态 (SAPIEN: [w, x, y, z])
            tube_pose = tube_actor.pose
            tube_q = tube_pose.q
            if len(tube_q.shape) == 2:
                tube_q = tube_q[0]
            tube_q = np.array(tube_q.cpu().numpy() if hasattr(tube_q, 'cpu') else tube_q)

            # 获取当前夹爪姿态 (SAPIEN: [w, x, y, z])
            ee_pose = robot.tcp_pose
            gripper_q = ee_pose.q
            if len(gripper_q.shape) == 2:
                gripper_q = gripper_q[0]
            gripper_q = np.array(gripper_q.cpu().numpy() if hasattr(gripper_q, 'cpu') else gripper_q)

            # 转换为 SciPy 格式 [x, y, z, w]
            tube_rot = R_scipy.from_quat([tube_q[1], tube_q[2], tube_q[3], tube_q[0]])
            gripper_rot = R_scipy.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])

            # 相对姿态: tube = gripper * relative
            relative_rot = gripper_rot.inv() * tube_rot

            # 目标：试管竖直（单位四元数，即无旋转）
            vertical_rot = R_scipy.from_quat([0, 0, 0, 1])

            # 新夹爪姿态: vertical = gripper_new * relative
            gripper_rot_new = vertical_rot * relative_rot.inv()

            # 转换回 SAPIEN [w, x, y, z]
            new_q_scipy = gripper_rot_new.as_quat()  # [x, y, z, w]
            new_quat = np.array([new_q_scipy[3], new_q_scipy[0], new_q_scipy[1], new_q_scipy[2]])

            # 获取当前位置（保持位置不变，只旋转）
            ee_pos = robot.tcp_pose.p
            if len(ee_pos.shape) == 2:
                ee_pos = ee_pos[0]
            current_pos = np.array(ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else ee_pos, dtype=np.float32)

            # 平滑旋转到新姿态（试管会跟着转，自然变竖直）
            move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=current_pos,
                target_quat=new_quat,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            print(f"    ✓ 夹爪姿态已调整，试管应已竖直")
    except Exception as e:
        print(f"    ⚠️ 调整夹爪姿态失败: {e}")

    # ========== 步骤4a: 移动到中间点（使用 REPEAT_4 或默认值） ==========
    if repeat_id is not None and 0 <= repeat_id < len(REPEAT_4):
        mid_pos = np.array(REPEAT_4[repeat_id], dtype=np.float32)
        print(f"  使用 REPEAT_4[{repeat_id}] = {REPEAT_4[repeat_id]} 作为中间点")
    else:
        mid_pos = np.array([-0.4, 0.0, 0.25], dtype=np.float32)
    print(f"\n  步骤4a: {arm_name} 移动到中间点 [{mid_pos[0]:.3f}, {mid_pos[1]:.3f}, {mid_pos[2]:.3f}]...")
    recorder.set_subtask(f"left arm moves the tube.")
    ee_pose = robot.tcp_pose
    current_quat = ee_pose.q
    if len(current_quat.shape) == 2:
        current_quat = current_quat[0]
    current_quat = np.array(
        current_quat.cpu().numpy() if hasattr(current_quat, "cpu") else current_quat,
        dtype=np.float32,
    )
    if robot_idx == 0:
        z_axis = np.array([0, -1, 0])
    else:
        z_axis = np.array([0, 1, 0])
    x_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    matrix_base_quat = R.from_matrix(rot_mat).as_quat()
    print(f"\n  步骤5a-1: 切换到矩阵基座姿态（中间态）...")
    print(
        f"    矩阵基座四元数: [{matrix_base_quat[0]:.4f}, "
        f"{matrix_base_quat[1]:.4f}, {matrix_base_quat[2]:.4f}, "
        f"{matrix_base_quat[3]:.4f}]"
    )
    move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=mid_pos,
        target_quat=matrix_base_quat,
        steps=100, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=False,
    )
    print(f"  ✅ 到达中间点")

    # ========== 步骤5a: 读取试管姿态，补偿角度和位置（原地调整） ==========
    if tube_idx is not None:
        tube_place_pos, tube_place_quat = get_tube_pose(env, tube_idx)
    else:
        tube_place_pos = None
    # # ========== 步骤5a-1: 先切换到左臂矩阵基座姿态 ==========
    # print(f"\n  步骤5a-1: 切换到左臂矩阵基座姿态（中间态）...")
    # print(f"    矩阵基座四元数: [{left_matrix_quat[0]:.4f}, {left_matrix_quat[1]:.4f}, {left_matrix_quat[2]:.4f}, {left_matrix_quat[3]:.4f}]")
    # recorder.set_subtask(f"{arm_name} switches to matrix base pose")
    # ee_pose = robot.tcp_pose
    # p = ee_pose.p
    # if len(p.shape) == 2:
    #     p = p[0]
    # curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    # move_to_position_only(
    #     env, robot_idx=robot_idx,
    #     target_pos=curr_pos,
    #     target_quat=left_matrix_quat,
    #     steps=60, record_step=record_step,
    #     gripper_open=False,
    #     allow_adaptive_search=True, limit_search_range=True,
    # )

    # ========== 步骤5: 移动到目标位置上方 ==========
    # above_pos[0] = lift_pos[0] - 0.2
    # above_pos[1] = lift_pos[1]
    # above_pos[2] += 0.3
    # print(f"\n  步骤5: {arm_name} 移动到目标上方...")
    # recorder.set_subtask(f"{arm_name} moves above placement")
    # success_above = move_to_position_only(
    #     env, robot_idx=robot_idx,
    #     target_pos=above_pos,
    #     target_quat=left_matrix_quat,
    #     steps=50, record_step=record_step,
    #     gripper_open=False,
    #     allow_adaptive_search=True, limit_search_range=False,
    # )
    above_pos = end_pos.copy()
    above_pos[2] += 0.16
    above_pos[2] = min(above_pos[2], MAX_Z_HEIGHT)
    recorder.set_subtask(f"left arm places the tube (on yellow rack)")
    ee_pose = robot.tcp_pose
    current_quat = ee_pose.q
    if len(current_quat.shape) == 2:
        current_quat = current_quat[0]
    current_quat = np.array(
        current_quat.cpu().numpy() if hasattr(current_quat, "cpu") else current_quat,
        dtype=np.float32,
    )
    success_above = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_pos,
        target_quat=matrix_base_quat,
        steps=100, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=False,
    )

    if success_above:
        # 读取试管当前姿态，补偿pitch角
        if tube_idx is not None:
            tube_place_pos, tube_place_quat = get_tube_pose(env, tube_idx)
        else:
            tube_place_pos = None
        if tube_place_pos is not None:
            # 修正格式转换（SAPIEN [w,x,y,z] → SciPy [x,y,z,w]）
            tube_rot = R.from_quat([tube_place_quat[1], tube_place_quat[2], tube_place_quat[3], tube_place_quat[0]])
            tube_z_world = tube_rot.apply([0, 0, 1])

            world_z = np.array([0, 0, 1])
            cos_angle = np.dot(tube_z_world, world_z)
            angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            print(f"  试管z轴朝向: {tube_z_world}")
            print(f"  与世界z轴的夹角: {angle_deg:.2f}°")

            # 用 align_vectors 自动计算最优旋转
            R_map, _ = R.align_vectors([world_z], [tube_z_world])

            # 获取当前夹爪姿态
            ee_pose = robot.tcp_pose
            ee_q = ee_pose.q
            if len(ee_q.shape) == 2:
                ee_q = ee_q[0]
            gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
            gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])

            # 新夹爪姿态
            gripper_rot_new = R_map * gripper_rot
            adjusted_quat_scipy = gripper_rot_new.as_quat()
            adjusted_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], adjusted_quat_scipy[1], adjusted_quat_scipy[2]])

            print(f"    调整后四元数: [{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")

            # ========== 步骤5a-2: 先调整姿态（原地旋转，不移动位置） ==========
            print(f"\n  步骤5a-2: 原地调整姿态补偿...")
            print(f"    使用调整后姿态: 四元数=[{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
            recorder.set_subtask(f"left arm places the tube (on yellow rack)")
            ee_pose = robot.tcp_pose
            p = ee_pose.p
            if len(p.shape) == 2:
                p = p[0]
            curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
            move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=curr_pos,
                target_quat=adjusted_quat,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=False,
            )

            # ========== 步骤5a-3: 姿态调整后，重新读取试管位置，调整TCP位置对齐 ==========
            tube_place_pos2, tube_place_quat2 = get_tube_end_pose(env, tube_idx)
            tube_origin_pos2, _ = get_tube_pose(env, tube_idx)  # 同时读取网格原点位置用于对比
            # 调试：直接读取cmass_local_pose确认设置值
            tube_obj = get_tube_object(env, tube_idx)
            cmass_local = tube_obj.cmass_local_pose.p
            if len(cmass_local.shape) == 2:
                cmass_local = cmass_local[0]
            cmass_local_np = np.array(cmass_local.cpu().numpy() if hasattr(cmass_local, 'cpu') else cmass_local, dtype=np.float32)
            print(f"  [调试] cmass_local_pose = [{cmass_local_np[0]:.4f}, {cmass_local_np[1]:.4f}, {cmass_local_np[2]:.4f}]")
            if tube_place_pos2 is not None:
                print(f"\n  步骤5a-3: 姿态调整后重新读取试管位置，调整TCP位置...")
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos2
                adjusted_target = curr_pos + tube_offset
                adjusted_target[2] = curr_pos[2]
                print(f"    {arm_name}TCP当前位置: [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
                print(f"    试管质心位置: [{tube_place_pos2[0]:.3f}, {tube_place_pos2[1]:.3f}, {tube_place_pos2[2]:.3f}]")
                if tube_origin_pos2 is not None:
                    print(f"    试管原点位置: [{tube_origin_pos2[0]:.3f}, {tube_origin_pos2[1]:.3f}, {tube_origin_pos2[2]:.3f}]")
                print(f"    试管目标位置(end_pos): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
                print(f"    调整后目标位置: [{adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f}]")
                recorder.set_subtask(f"left arm places the tube (on yellow rack)")
                adjusted_target[1] += 0
                ee_pose = robot.tcp_pose
                q = ee_pose.q
                if len(q.shape) == 2:
                    q = q[0]
                current_quat_now = np.array(
                    q.cpu().numpy() if hasattr(q, "cpu") else q,
                    dtype=np.float32,
                )
                move_to_position_only(
                    env, robot_idx=robot_idx,
                    target_pos=adjusted_target,
                    target_quat=current_quat_now,
                    steps=100, record_step=record_step,
                    gripper_open=False,
                    allow_adaptive_search=True, limit_search_range=False,
                )
            else:
                print(f"  ⚠️ 姿态调整后无法读取试管位置，使用原始偏移")
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos
                adjusted_target = curr_pos + tube_offset
                adjusted_target[2] = curr_pos[2]
        else:
            print(f"  ⚠️ 无法读取试管姿态，使用原始姿态和位置")
            adjusted_quat = left_matrix_quat
            adjusted_target = end_pos
        adjusted_target[2] = end_pos[2]
        # ========== 步骤6: 分步下降并实时调整 ==========
        print(f"\n  步骤6: left arm 分步下降放置（先调整位置再下降）...")

        # 获取当前位置作为下降起点
        ee_pose = robot.tcp_pose
        p = ee_pose.p
        if len(p.shape) == 2:
            p = p[0]
        current_pos_before_descent = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
        START_Z = current_pos_before_descent[2]
        TARGET_Z = adjusted_target[2]
        Z_STEP = (START_Z - TARGET_Z) / NUM_DESCENT_STEPS
        print(f"    下降起点高度: {START_Z:.3f}m, 目标高度: {TARGET_Z:.3f}m")

        for descent_step in range(NUM_DESCENT_STEPS):
            current_z = START_Z - descent_step * Z_STEP  # 当前高度
            target_z = START_Z - (descent_step + 1) * Z_STEP  # 目标高度
            print(f"\n  下降步骤 {descent_step + 1}/{NUM_DESCENT_STEPS}: {current_z:.3f}m → {target_z:.3f}m")
            
            # ===== 步骤6a: 在当前高度原地调整位置 =====
            tube_place_pos_temp, tube_place_quat_temp = get_tube_end_pose(env, tube_idx)
            
            # 调整姿态
            if tube_place_quat_temp is not None:
                tube_rot = R.from_quat([tube_place_quat_temp[1], tube_place_quat_temp[2], 
                                    tube_place_quat_temp[3], tube_place_quat_temp[0]])
                tube_z_world = tube_rot.apply([0, 0, 1])
                world_z = np.array([0, 0, 1])
                R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
                ee_pose = robot.tcp_pose
                ee_q = ee_pose.q
                if len(ee_q.shape) == 2:
                    ee_q = ee_q[0]
                gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
                gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
                gripper_rot_new = R_map * gripper_rot
                adjusted_quat_scipy = gripper_rot_new.as_quat()
                adjusted_quat_temp = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
                                            adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
            else:
                adjusted_quat_temp = adjusted_quat
            
            # 调整位置（Z保持当前高度不变）
            if tube_place_pos_temp is not None:
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_temp = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                adjusted_pos = curr_pos_temp + tube_offset
                adjusted_pos[2] = current_z  # 保持当前高度
                print(f"    步骤6a: 调整位置 [{curr_pos_temp[0]:.3f}, {curr_pos_temp[1]:.3f}] → [{adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}] (Z={current_z:.3f}m不变)")
                print(f"      试管位置: [{tube_place_pos_temp[0]:.3f}, {tube_place_pos_temp[1]:.3f}, {tube_place_pos_temp[2]:.3f}]")
                print(f"      目标位置(end_pos): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
                print(f"      偏移量: [{tube_offset[0]:.3f}, {tube_offset[1]:.3f}, {tube_offset[2]:.3f}]")
            else:
                adjusted_pos = adjusted_target.copy()
                adjusted_pos[2] = current_z
            
            # 原地调整位置（Z不变）
            success_adjust = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=adjusted_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            if not success_adjust:
                print(f"    ⚠️ 步骤6a调整失败，继续...")
            tube_place_pos_temp, tube_place_quat_temp = get_tube_end_pose(env, tube_idx)
            # ===== 步骤6b: 下降到目标高度 =====
            if tube_place_pos_temp is not None:
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_after = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                descend_pos = curr_pos_after + tube_offset
                descend_pos[2] = target_z  # 下降到目标高度
                print(f"    步骤6b: 下降 {curr_pos_after[2]:.3f}m → {target_z:.3f}m")
            else:
                descend_pos = adjusted_pos.copy()
                descend_pos[2] = target_z
            
            success_descend = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=descend_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            # ===== 高度检测：试管末端是否已到达目标位置 =====
            tube_check_pos, _ = get_tube_end_pose(env, tube_idx)
            if tube_check_pos is not None:
                tube_check_pos = np.array(tube_check_pos, dtype=np.float32)
                # 如果试管末端高度 < 目标位置高度，说明试管已放到目标位置
                if tube_check_pos[2] < end_pos[2] + check_height:  # 允许1cm误差
                    print(f"    ✅ 试管已到达目标高度，停止下降")
                    break
            
            if not success_descend:
                print(f"  ⚠️ 步骤6b下降失败，继续...")
                continue

        # 最终放置
        print(f"\n  最终放置...")
        # success_insert, final_pos, final_quat = try_place_with_circular_search(
        #     env=env,
        #     robot_idx=robot_idx,
        #     robot=robot,
        #     target_pos=adjusted_target,
        #     target_quat=adjusted_quat,
        #     record_step=record_step,
        #     gripper_open_after=True,
        #     other_robot_idx=other_robot_idx,
        #     other_robot=other_robot,
        #     tube_idx=tube_idx,
        # )
        # if success_insert:
        #     print(f"  ✅ {arm_name} 圆形搜索放置成功")
        # else:
        #     print(f"  ❌ 圆形搜索放置失败")
        #     return False
    else:
        print(f"  ❌ 移动到目标上方失败")
        return False

    # ========== 步骤8: 缓慢后撤到初始位置 ==========
    print(f"\n  步骤8: {arm_name} 缓慢后撤到初始位置上方...")
    recorder.set_subtask(f"left arm retracts")
    # 先缓慢抬升一段
    ee_pose = robot.tcp_pose
    p = ee_pose.p
    if len(p.shape) == 2:
        p = p[0]
    curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    lift_up_pos = curr_pos.copy()
    lift_up_pos[2] += 0.15
    lift_up_pos[2] = min(lift_up_pos[2], MAX_Z_HEIGHT)
    move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=lift_up_pos,
        steps=50, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    initial_pos_raised = initial_pos.copy()
    initial_pos_raised[2] += 0.15
    initial_pos_raised[2] = min(initial_pos_raised[2], MAX_Z_HEIGHT)
    return_arm_to_initial_pose(
        env=env,
        robot_idx=robot_idx,
        target_pos=initial_pos_raised,
        target_quat=initial_quat,
        initial_qpos=initial_qpos,
        record_step=record_step,
    )
    initial_pos = initial_pos_raised.copy()
    return_arm_to_initial_pose(
        env=env,
        robot_idx=robot_idx,
        target_pos=initial_pos,
        target_quat=initial_quat,
        initial_qpos=initial_qpos,
        record_step=record_step,
    )
    print(f"  ✅ {arm_name} 已返回初始位置")
    return True
def grasp_and_place(
    env, tube_idx,
    robot_idx, other_robot_idx,
    robot, other_robot,
    start_idx, end_idx,
    right_handover_quat,
    recorder, record_step,):
    """
    指定机械臂从起始位置夹取试管，放置到目标位置

    起始位置 start_idx 映射:
      0-4 → 直接读取试管0-4的当前位置
      5-8 → cube_place02_pos[0-3]（右臂放置的中间位置）
    目标位置 end_idx 映射:
      5-8 → cube_place02_pos[0-3]（右臂放置的中间位置）
      9-12 → cube_place03_pos[0-3]（最终放置位置）

    夹取逻辑参考 pass_to 前半部分，放置逻辑参考 right_place_procedure 后半部分

    流程:
      1. 移动到目标位置上方
      2. 下降到夹取位置
      3. 闭合夹爪抓取
      4. 提起试管
      5. 移动到放置位置上方
      6. 下降到插入位置
      7. 张开夹具释放
      8. 后撤到初始位置
    """
    arm_name = "left arm" if robot_idx == 0 else "right arm"

    # # 解析起始位置
    # if 0 <= start_idx <= 4:
    #     start_pos = np.array(cube_place_pos[start_idx * 2], dtype=np.float32)
    #     start_label = f"cube_place_pos[{start_idx * 2}]"
    # elif 5 <= start_idx <= 8:
    #     start_pos = np.array(cube_place02_pos[start_idx - 5], dtype=np.float32)
    #     start_label = f"cube_place02_pos[{start_idx - 5}]"
    # else:
    #     print(f"❌ start_idx={start_idx} 无效，必须为 0-8")
    #     return False
    # 解析起始位置
    if 0 <= start_idx <= 4:
        # 直接读取试管当前位姿（更准确，因为试管可能被移动过）
        tube_start_pos, tube_start_quat = get_tube_pose(env, start_idx)
        if tube_start_pos is not None:
            start_pos = np.array(tube_start_pos, dtype=np.float32)
            start_label = f"试管{start_idx}当前位置"
            print(f"  {start_label}: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        else:
            print(f"  ❌ 无法读取试管{start_idx}位姿，回退到预设位置")
            start_pos = np.array(cube_place_pos[start_idx + 2], dtype=np.float32)
            start_label = f"cube_place_pos[{start_idx + 2}](回退)"

    # 解析目标位置
    if 5 <= end_idx <= 8:
        pos_idx = end_idx - 5
        if pos_idx >= len(cube_place02_pos):
            print(f"❌ end_idx={end_idx} 对应的 cube_place02_pos 索引 {pos_idx} 超出范围 (len={len(cube_place02_pos)})")
            return False
        end_pos = np.array(cube_place02_pos[pos_idx], dtype=np.float32)
        end_label = f"cube_place02_pos[{pos_idx}]"
    elif 9 <= end_idx <= 12:
        pos_idx = end_idx - 9
        if pos_idx >= len(cube_place03_pos):
            print(f"❌ end_idx={end_idx} 对应的 cube_place03_pos 索引 {pos_idx} 超出范围 (len={len(cube_place03_pos)})")
            return False
        end_pos = np.array(cube_place03_pos[pos_idx], dtype=np.float32)
        end_label = f"cube_place03_pos[{pos_idx}]"
    else:
        print(f"❌ end_idx={end_idx} 无效，必须为 5-12")
        return False

    # 根据位置 y 坐标确定架子颜色
    def _get_tube_rack_color(pos_x, pos_y):
        if pos_y > 0.1 or pos_y < -0.1:
            return "yellow"
        else:
            return "red"
    source_tube_rack_color = _get_tube_rack_color(start_pos[0], start_pos[1])
    target_tube_rack_color = _get_tube_rack_color(end_pos[0], end_pos[1])

    # 获取试管当前姿态并计算其z轴朝向
    tube_actor = get_tube_object(env, start_idx)
    if tube_actor is not None:
        tube_pose = tube_actor.pose
        tube_q = tube_pose.q
        if len(tube_q.shape) == 2:
            tube_q = tube_q[0]
        tube_q = np.array(tube_q.cpu().numpy() if hasattr(tube_q, 'cpu') else tube_q)
        tube_R = R.from_quat([tube_q[1], tube_q[2], tube_q[3], tube_q[0]])
        tube_rot_mat = tube_R.as_matrix()
        tube_z_axis = tube_rot_mat[:, 2]
    else:
        tube_z_axis = None

    # 计算机械臂夹取姿态的z轴朝向
    gripper_R = R.from_quat([right_handover_quat[0], right_handover_quat[1], right_handover_quat[2], right_handover_quat[3]])
    gripper_rot_mat = gripper_R.as_matrix()
    gripper_z_axis = gripper_rot_mat[:, 2]

    print(f"\n{'=' * 60}")
    print(f"  {arm_name} 夹取试管 (start_idx={start_idx}) 并放置到目标位置 (end_idx={end_idx})")
    print(f"    夹取位置 ({start_label}): [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    print(f"    放置位置 ({end_label}): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    if tube_z_axis is not None:
        print(f"    试管当前z轴朝向: [{tube_z_axis[0]:.4f}, {tube_z_axis[1]:.4f}, {tube_z_axis[2]:.4f}]")
    print(f"    夹具末端z轴朝向: [{gripper_z_axis[0]:.4f}, {gripper_z_axis[1]:.4f}, {gripper_z_axis[2]:.4f}]")
    if tube_z_axis is not None:
        dot_product = np.dot(tube_z_axis, gripper_z_axis)
        print(f"    点积（验证垂直性）: {dot_product:.4f} (应为0表示垂直)")
    print(f"{'=' * 60}")

    # 保存初始位姿
    current_ee_pose = robot.tcp_pose
    current_p = current_ee_pose.p
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    initial_pos = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q = current_ee_pose.q
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    initial_quat = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)
    initial_qpos = robot.robot.get_qpos()[0].cpu().numpy().copy()
    
    # ========== 步骤0: 原地升起机械臂至安全高度 ==========
    RAISE_HEIGHT_OFFSET = 0.25
    raised_pos = initial_pos.copy()
    raised_pos[2] += RAISE_HEIGHT_OFFSET
    raised_pos[2] = min(raised_pos[2], MAX_Z_HEIGHT)
    print(f"\n  步骤0: {arm_name} 原地升起至安全高度（+{RAISE_HEIGHT_OFFSET*100:.0f}cm）...")
    recorder.set_subtask(f"right arm approaches for the tube (on red rack)")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=raised_pos,
        target_quat=initial_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=False,
    )
    if not success:
        print(f"  ❌ 原地升起失败")
        return False
    print(f"  ✅ 原地升起成功")

    # ========== 动态计算抓取姿态（45度倾斜）==========
    # 策略：先尝试固定方向（负X轴），失败后再使用动态yaw角
    robot_base_pos = robot.robot.pose.p
    if len(robot_base_pos.shape) == 2:
        robot_base_pos = robot_base_pos[0]
    robot_base_np = np.array(robot_base_pos.cpu().numpy() if hasattr(robot_base_pos, 'cpu') else robot_base_pos, dtype=np.float32)
    
    # 优先尝试：从负X方向接近（夹爪z轴指向正X）
    fixed_z_axis = np.array([1.0, 0.0, 0.0])  # 固定：从正X方向指向试管
    fixed_x_axis = np.array([0.0, 0.0, -1.0])  # 夹爪闭合方向竖直向下
    
    pitch = np.deg2rad(40)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    fixed_z_axis = Ry @ fixed_z_axis
    fixed_x_axis = Ry @ fixed_x_axis
    
    y_axis = np.cross(fixed_z_axis, fixed_x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    rot_matrix = np.column_stack([fixed_x_axis, y_axis, fixed_z_axis])
    initial_grasp_quat = R.from_matrix(rot_matrix).as_quat()
    
    print(f"  [抓取姿态] 优先尝试固定方向: z_axis=[{fixed_z_axis[0]:.3f}, {fixed_z_axis[1]:.3f}, {fixed_z_axis[2]:.3f}]")
    
    # 需要改成：
    # 先计算yaw角（从 initial_grasp_quat 提取）
    yaw = np.arctan2(fixed_z_axis[1], fixed_z_axis[0])
    print(f"  [yaw角] {np.rad2deg(yaw):.1f}°")
    # 然后用yaw角计算
    above_pos = start_pos.copy()
    above_pos[0] -= 0.1 * np.cos(yaw)  # 减号
    # above_pos[1] -= 0.1 * np.sin(yaw)
    above_pos[1] -= 0
    above_pos[2] += grasp_height+0.1
    above_pos[2] = min(above_pos[2], MAX_Z_HEIGHT)
    print(f"    目标位置: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}]")
    
    # 先用固定方向尝试步骤1
    success_step1 = True
    print(f"\n  步骤1: {arm_name} 移动到夹取位置上方（固定方向）...")
    success_step1 = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_pos,
        target_quat=initial_grasp_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    
    # 如果固定方向失败，改用动态yaw角
    if not success_step1:
        print(f"  ⚠️ 固定方向失败，改用动态yaw角...")
        dx = start_pos[0] - robot_base_np[0]
        dy = start_pos[1] - robot_base_np[1]
        yaw_angle = np.arctan2(dy, dx)
        print(f"  [动态抓取姿态] 基座→试管 yaw={np.rad2deg(yaw_angle):.1f}°")
        
        dynamic_z_axis = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0.0])
        dynamic_x_axis = np.array([0.0, 0.0, -1.0])
        
        dynamic_z_axis = Ry @ dynamic_z_axis
        dynamic_x_axis = Ry @ dynamic_x_axis
        
        y_axis = np.cross(dynamic_z_axis, dynamic_x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rot_matrix = np.column_stack([dynamic_x_axis, y_axis, dynamic_z_axis])
        initial_grasp_quat = R.from_matrix(rot_matrix).as_quat()
        
        print(f"  [抓取姿态] 动态方向: z_axis=[{dynamic_z_axis[0]:.3f}, {dynamic_z_axis[1]:.3f}, {dynamic_z_axis[2]:.3f}]")
        
        # 再次尝试步骤1
        print(f"\n  步骤1: {arm_name} 移动到夹取位置上方（动态yaw角）...")
        success_step1 = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=above_pos,
            target_quat=initial_grasp_quat,
            steps=30, record_step=record_step,
            gripper_open=True,
            allow_adaptive_search=True, limit_search_range=True,
        )
    
    if not success_step1:
        print(f"  ❌ 移动到夹取位置上方失败")
        return False
    print(f"  ✅ 移动到夹取位置上方成功")
    dynamic_grasp_quat = initial_grasp_quat  # 保存用于步骤2
    if 'dynamic_z_axis' in locals():
        step2_yaw = np.arctan2(dynamic_z_axis[1], dynamic_z_axis[0])
    else:
        step2_yaw = yaw
    # ========== 步骤2: 下降到抓取位置（实时读取试管位置）===========
    tube_center_pos, _ = get_tube_pose(env, start_idx)
    if tube_center_pos is not None:
        if len(tube_center_pos.shape) == 2 and tube_center_pos.shape[0] == 1:
            tube_center_pos = tube_center_pos[0]
        grasp_pos = tube_center_pos.copy()
        # 需要改成：
        grasp_pos[0] += 0.01 * np.cos(yaw)  # 加号
        grasp_pos[1] += 0
        grasp_pos[2] += grasp_height  # Z方向：试管上方3cm
        grasp_pos[2] = min(grasp_pos[2], MAX_Z_HEIGHT)
        print(f"    试管位置: [{tube_center_pos[0]:.3f}, {tube_center_pos[1]:.3f}, {tube_center_pos[2]:.3f}]")
        print(f"    目标位置: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    else:
        grasp_pos = start_pos.copy()
        grasp_pos[0] += 0.01
        grasp_pos[1] += 0.0
        grasp_pos[2] += grasp_height
        print(f"    ⚠️ 无法读取试管位置，使用预设位置")
    
    print(f"\n  步骤2: {arm_name} 下降到抓取位置...")
    recorder.set_subtask(f"right arm approaches for the tube (on red rack)")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=grasp_pos,
        target_quat=dynamic_grasp_quat,
        steps=100, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 下降到抓取位置失败")
        return False
    print(f"  ✅ 下降到抓取位置成功")

    # ========== 步骤3: 闭合夹爪抓取试管 ==========
    print(f"\n  步骤3: {arm_name} 闭合夹爪抓取试管...")
    recorder.set_subtask(f"right arm picks up the tube (from red rack)")
    
    # 增加夹爪闭合力度：临时提高 force_limit 和 stiffness
    old_stiffness = None
    old_force_limit = None
    try:
        joints = robot.robot.get_active_joints()
        gripper_joints = [j for j in joints if 'joint7' in j.name or 'joint8' in j.name]
        if gripper_joints:
            old_stiffness = [j.get_stiffness() for j in gripper_joints]
            old_force_limit = [j.get_force_limit() for j in gripper_joints]
            for j in gripper_joints:
                j.set_drive_properties(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE_LIMIT)
    except Exception as e:
        print(f"  [info] 无法调整夹爪参数: {e}")

    # ========== 直接闭合夹爪 ==========
    print(f"  直接闭合夹爪...")
    for i in range(30):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        
        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action:
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])  # 保持张开

        if record_step:
            record_step(action)
        else:
            env.step(action)

    print(f"  ✅ 夹爪闭合完成")
    # ========== 步骤3c: 保持当前位置和姿态，让夹爪稳定 ==========
    # print(f"  步骤3c: 保持当前姿态稳定夹爪...")
    # ee_pose = robot.tcp_pose
    # p = ee_pose.p
    # if len(p.shape) == 2:
    #     p = p[0]
    # current_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    # q = ee_pose.q
    # if len(q.shape) == 2:
    #     q = q[0]
    # current_quat = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    
    # success_stabilize = move_to_position_only(
    #     env, robot_idx=robot_idx,
    #     target_pos=current_pos,
    #     target_quat=current_quat,
    #     steps=200, record_step=record_step,
    #     gripper_open=False,
    #     allow_adaptive_search=True,
    #     limit_search_range=True,
    # )
    # if not success_stabilize:
    #     print(f"  ⚠️ 稳定步骤失败，继续...")
    # ========== 步骤3c: 手动保持夹爪闭合稳定 ==========
    # print(f"  步骤3c: 保持夹爪闭合稳定...")
    # for i in range(50):
    #     action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
    #     robot_uid = f"piper_x-{robot_idx}"
    #     if robot_uid in action:
    #         current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
    #         arm_qpos = current_qpos[:6]
    #         action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])  # 使用 -1.8
        
    #     other_robot_uid = f"piper_x-{other_robot_idx}"
    #     if other_robot_uid in action:
    #         other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
    #         other_arm_qpos = other_qpos[:6]
    #         action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])

    #     if record_step:
    #         record_step(action)
    #     else:
    #         env.step(action)
    
    print(f"  ✅ 夹爪稳定完成")   
    print(f"  步骤3d: 等待夹爪稳定...")
    for i in range(50):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        robot_uid = f"piper_x-{robot_idx}"
        if robot_uid in action:
            current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = current_qpos[:6]
            action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])
        
        other_robot_idx = 1 - robot_idx
        other_robot_uid = f"piper_x-{other_robot_idx}"
        if other_robot_uid in action and other_robot is not None:
            other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
            other_arm_qpos = other_qpos[:6]
            action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])

        if record_step:
            record_step(action)
        else:
            env.step(action)
    print(f"  ✅ 夹爪稳定等待完成") 
    # print(f"  ✅ 夹爪稳定完成")    
    # # ========== 抓取后稳定等待 ==========
    # print(f"  步骤3b: 等待稳定...")
    # for i in range(STABILIZE_WAIT_STEPS):
    #     action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
    #     robot_uid = f"piper_x-{robot_idx}"
    #     if robot_uid in action:
    #         current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
    #         arm_qpos = current_qpos[:6]
    #         action[robot_uid] = np.concatenate([arm_qpos, [GRIPPER_CLOSE_POS]])

    #     other_robot_uid = f"piper_x-{other_robot_idx}"
    #     if other_robot_uid in action:
    #         other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
    #         other_arm_qpos = other_qpos[:6]
    #         action[other_robot_uid] = np.concatenate([other_arm_qpos, [1.0]])

    #     if record_step:
    #         record_step(action)
    #     else:
    #         env.step(action)
    # print(f"  ✅ 稳定等待完成")

    
    ee_pose = robot.tcp_pose
    q = ee_pose.q
    if len(q.shape) == 2:
        q = q[0]
    lift_quat = np.array(q.cpu().numpy() if hasattr(q, 'cpu') else q, dtype=np.float32)
    
    # ========== 步骤4: 分步小幅度提起（每次0.02m），共提0.2m ==========
    # 使用实际的当前末端位置作为提起起点，而不是 grasp_pos
    actual_ee_pose = robot.tcp_pose
    actual_ee_p = actual_ee_pose.p
    if len(actual_ee_p.shape) == 2:
        actual_ee_p = actual_ee_p[0]
    current_lift_pos = np.array(actual_ee_p.cpu().numpy() if hasattr(actual_ee_p, 'cpu') else actual_ee_p, dtype=np.float32)
    print(f"  [提起起点] 实际末端位置: [{current_lift_pos[0]:.3f}, {current_lift_pos[1]:.3f}, {current_lift_pos[2]:.3f}]")
    
    print(f"\n  步骤4: {arm_name} 分步提起试管（每次0.02m，共{NUM_LIFT_STEPS}步，总共0.2m）...")
    recorder.set_subtask(f"right arm picks up the tube (from red rack)")
    
    for lift_step in range(NUM_LIFT_STEPS):
        current_lift_pos[2] += STEP_LIFT_HEIGHT
        current_lift_pos[2] = min(current_lift_pos[2], MAX_Z_HEIGHT)
        target_z = current_lift_pos[2]
        
        # ===== 提起步骤a: 在当前高度读取试管姿态，调整使竖直 =====
        # if tube_idx is not None:
        #     tube_lift_pos, tube_lift_quat = get_tube_end_pose(env, tube_idx)
        #     if tube_lift_quat is not None:
        #         tube_rot = R.from_quat([tube_lift_quat[1], tube_lift_quat[2], 
        #                             tube_lift_quat[3], tube_lift_quat[0]])
        #         tube_z_world = tube_rot.apply([0, 0, 1])
        #         world_z = np.array([0, 0, 1])
        #         R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
        #         ee_pose = robot.tcp_pose
        #         ee_q = ee_pose.q
        #         if len(ee_q.shape) == 2:
        #             ee_q = ee_q[0]
        #         gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
        #         gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
        #         gripper_rot_new = R_map * gripper_rot
        #         adjusted_quat_scipy = gripper_rot_new.as_quat()
        #         adjusted_lift_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
        #                                     adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
        #     else:
        #         adjusted_lift_quat = lift_quat
        # else:
        adjusted_lift_quat = lift_quat
        
        # 读取当前位置
        ee_pose_before = robot.tcp_pose
        p_before = ee_pose_before.p
        if len(p_before.shape) == 2:
            p_before = p_before[0]
        curr_pos_before = np.array(p_before.cpu().numpy() if hasattr(p_before, 'cpu') else p_before)
        
        # 保持当前高度，调整姿态
        adjust_pos = curr_pos_before.copy()
        adjust_pos[2] = curr_pos_before[2]  # 保持当前高度
        
        # print(f"    提起步骤a: 调整姿态（Z={adjust_pos[2]:.3f}m保持）...")
        # move_to_position_only(
        #     env, robot_idx=robot_idx,
        #     target_pos=adjust_pos,
        #     target_quat=adjusted_lift_quat,
        #     steps=50, record_step=record_step,
        #     gripper_open=False,
        #     allow_adaptive_search=True, limit_search_range=True,
        # )
        
        # ===== 提起步骤b: 上升到目标高度 =====
        print(f"    提起步骤b: 上升 {curr_pos_before[2]:.3f}m → {target_z:.3f}m")
        success = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=current_lift_pos,
            target_quat=adjusted_lift_quat,
            steps=100, record_step=record_step,
            gripper_open=False,
            allow_adaptive_search=True,
            limit_search_range=True,
        )
        if not success:
            print(f"  ⚠️ 步骤{lift_step + 1}提升失败，尝试基于位置的yaw计算...")
            # 备用姿态计算
            robot_base_pos = robot.robot.pose.p
            if len(robot_base_pos.shape) == 2:
                robot_base_pos = robot_base_pos[0]
            robot_base_np = np.array(robot_base_pos.cpu().numpy() if hasattr(robot_base_pos, 'cpu') else robot_base_pos, dtype=np.float32)
            dx = grasp_pos[0] - robot_base_np[0]
            dy = grasp_pos[1] - robot_base_np[1]
            yaw_angle = np.arctan2(dy, dx)
            z_axis = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0.0])
            x_axis = np.array([0.0, 0.0, -1.0])
            y_axis = np.cross(z_axis, x_axis)
            rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
            lift_quat = R.from_matrix(rot_matrix).as_quat()
            success = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=current_lift_pos,
                target_quat=lift_quat,
                steps=30, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True,
                limit_search_range=True,
            )
            if not success:
                print(f"  ❌ 步骤{lift_step + 1}最终提升失败")
                return False
    
    lift_pos = current_lift_pos.copy()
    print(f"  ✅ 分步提起完成，最终高度: {lift_pos[2]:.3f}m")

    # # ========== 步骤50: 移动到目标位置上方 ==========
    # lift_pos[0]-=0.15
    # lift_pos[2] = min(lift_pos[2], MAX_Z_HEIGHT)
    # print(f"\n  步骤5_0: {arm_name} 移动到目标上方...")
    # recorder.set_subtask(f"right arm moves the tube.")
    # success_above = move_to_position_only(
    #     env, robot_idx=robot_idx,
    #     target_pos=lift_pos,
    #     target_quat=lift_quat,
    #     steps=100, record_step=record_step,
    #     gripper_open=False,
    #     allow_adaptive_search=True, limit_search_range=False,
    # )
    # 新步骤：先移动到[-0.4, 0.0, 0.3]，保持当前姿态
    ee_pose_before_5a = robot.tcp_pose
    p_before = ee_pose_before_5a.p
    q_before = ee_pose_before_5a.q
    if len(p_before.shape) == 2:
        p_before = p_before[0]
    if len(q_before.shape) == 2:
        q_before = q_before[0]
    pos_before = np.array(p_before.cpu().numpy() if hasattr(p_before, 'cpu') else p_before)
    quat_before = np.array(q_before.cpu().numpy() if hasattr(q_before, 'cpu') else q_before)
    
    intermediate_pos = np.array([-0.3, 0.0, 0.3], dtype=np.float32)
    print(f"\n  步骤5_intermediate: 移动到 [-0.3, 0.0, 0.3]，保持姿态...")
    print(f"    当前位置: [{pos_before[0]:.3f}, {pos_before[1]:.3f}, {pos_before[2]:.3f}]")
    print(f"    目标位置: [{intermediate_pos[0]:.3f}, {intermediate_pos[1]:.3f}, {intermediate_pos[2]:.3f}]")
    recorder.set_subtask(f"right arm moves the tube.")
    move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=intermediate_pos,
        target_quat=quat_before,
        steps=80, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=True,
    )
    # ========== 步骤5a: 读取试管姿态，补偿角度和位置（原地调整） ==========
    if tube_idx is not None:
        tube_place_pos, tube_place_quat = get_tube_end_pose(env, tube_idx)
    else:
        tube_place_pos = None
    #    ========== 步骤5a-1: 先切换到矩阵基座姿态（同 right_pass 的 right_handover_quat） ==========

    if robot_idx == 0:
        z_axis = np.array([0, -1, 0])
    else:
        z_axis = np.array([0, 1, 0])
    x_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    matrix_base_quat = R.from_matrix(rot_mat).as_quat()
    print(f"\n  步骤5a-1: 切换到矩阵基座姿态（中间态）...")
    print(f"    矩阵基座四元数: [{matrix_base_quat[0]:.4f}, {matrix_base_quat[1]:.4f}, {matrix_base_quat[2]:.4f}, {matrix_base_quat[3]:.4f}]")
    recorder.set_subtask(f"right arm moves the tube.")
    ee_pose = robot.tcp_pose
    p = ee_pose.p
    if len(p.shape) == 2:
        p = p[0]
    curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=curr_pos,  # 保持当前位置不变
        target_quat=matrix_base_quat,
        steps=100, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=True,
    )

    above_pos = end_pos.copy()
    above_pos[2] += 0.2
    above_pos[2] = min(above_pos[2], MAX_Z_HEIGHT)
    recorder.set_subtask(f"right arm moves the tube.")
    success_above = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_pos,
        target_quat=matrix_base_quat,
        steps=100, record_step=record_step,
        gripper_open=False,
        allow_adaptive_search=True, limit_search_range=False,
    )
            
    if success_above:
        # 读取试管当前姿态，补偿pitch角（同 right_place_procedure 步骤7a-2）
        if tube_idx is not None:
            tube_place_pos, tube_place_quat = get_tube_pose(env, tube_idx)
        else:
            tube_place_pos = None
        if tube_place_pos is not None:
            tube_rot = R.from_quat([tube_place_quat[1], tube_place_quat[2], tube_place_quat[3], tube_place_quat[0]])
            tube_z_world = tube_rot.apply([0, 0, 1])

            world_z = np.array([0, 0, 1])
            cos_angle = np.dot(tube_z_world, world_z)
            angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            print(f"  试管z轴朝向: {tube_z_world}")
            print(f"  与世界z轴的夹角: {angle_deg:.2f}°")

            # 用 align_vectors 自动计算最优旋转
            R_map, _ = R.align_vectors([world_z], [tube_z_world])

            # 获取当前夹爪姿态
            ee_pose = robot.tcp_pose
            ee_q = ee_pose.q
            if len(ee_q.shape) == 2:
                ee_q = ee_q[0]
            gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
            gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])

            # 新夹爪姿态
            gripper_rot_new = R_map * gripper_rot
            adjusted_quat_scipy = gripper_rot_new.as_quat()
            adjusted_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], adjusted_quat_scipy[1], adjusted_quat_scipy[2]])

            print(f"    调整后四元数: [{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
        else:
            print(f"  ⚠️ 无法读取试管姿态，使用矩阵基座姿态")
            z_axis = np.array([0, 1, 0])
            x_axis = np.array([0, 0, -1])
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])
            adjusted_quat = R.from_matrix(rot_mat).as_quat()

        # ========== 步骤5a-2: 切换到矩阵基座姿态 ==========
        print(f"\n  步骤5a-2: 切换到矩阵基座姿态...")
        print(f"    矩阵基座四元数: [{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
        recorder.set_subtask(f"right arm places the tube (on yellow rack)")
        ee_pose = robot.tcp_pose
        p = ee_pose.p
        if len(p.shape) == 2:
            p = p[0]
        curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
        move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=curr_pos,  # 保持当前位置不变
            target_quat=adjusted_quat,
            steps=100, record_step=record_step,
            gripper_open=False,
            allow_adaptive_search=True, limit_search_range=True,
        )

        # ========== 步骤5a-3: 读取试管位置，调整TCP位置对齐 ==========
        tube_place_pos2, tube_place_quat2 = get_tube_end_pose(env, tube_idx)
        if tube_place_pos2 is not None:
            print(f"\n  步骤5a-3: 读取试管位置，调整TCP位置...")
            ee_pose = robot.tcp_pose
            p = ee_pose.p
            if len(p.shape) == 2:
                p = p[0]
            curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
            tube_offset = end_pos - tube_place_pos2
            adjusted_target = curr_pos + tube_offset
            adjusted_target[2] = place_height
            print(f"    {arm_name}TCP当前位置: [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
            print(f"    试管位置: [{tube_place_pos2[0]:.3f}, {tube_place_pos2[1]:.3f}, {tube_place_pos2[2]:.3f}]")
            print(f"    试管目标位置(end_pos): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
            print(f"    调整后目标位置: [{adjusted_target[0]:.3f}, {adjusted_target[1]:.3f}, {adjusted_target[2]:.3f}]")
            recorder.set_subtask(f"right arm places the tube (on yellow rack)")
            adjusted_target[1] -= 0
            move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=adjusted_target,
                target_quat=adjusted_quat,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
        else:
            print(f"  ⚠️ 无法读取试管位置，使用原始目标位置")
            adjusted_target = end_pos
        adjusted_target[2] = end_pos[2]
        # ========== 步骤6: 分步下降并实时调整 ==========
        print(f"\n  步骤6: right arm 分步下降放置（先调整位置再下降）...")

        # 获取当前位置作为下降起点
        ee_pose = robot.tcp_pose
        p = ee_pose.p
        if len(p.shape) == 2:
            p = p[0]
        current_pos_before_descent = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
        START_Z = current_pos_before_descent[2]
        TARGET_Z = adjusted_target[2]
        Z_STEP = (START_Z - TARGET_Z) / NUM_DESCENT_STEPS
        print(f"    下降起点高度: {START_Z:.3f}m, 目标高度: {TARGET_Z:.3f}m")

        for descent_step in range(NUM_DESCENT_STEPS):
            current_z = START_Z - descent_step * Z_STEP  # 当前高度
            target_z = START_Z - (descent_step + 1) * Z_STEP  # 目标高度
            print(f"\n  下降步骤 {descent_step + 1}/{NUM_DESCENT_STEPS}: {current_z:.3f}m → {target_z:.3f}m")
            
            # ===== 步骤6a: 在当前高度原地调整位置 =====
            tube_place_pos_temp, tube_place_quat_temp = get_tube_end_pose(env, tube_idx)
            
            # 调整姿态
            if tube_place_quat_temp is not None:
                tube_rot = R.from_quat([tube_place_quat_temp[1], tube_place_quat_temp[2], 
                                    tube_place_quat_temp[3], tube_place_quat_temp[0]])
                tube_z_world = tube_rot.apply([0, 0, 1])
                world_z = np.array([0, 0, 1])
                R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
                ee_pose = robot.tcp_pose
                ee_q = ee_pose.q
                if len(ee_q.shape) == 2:
                    ee_q = ee_q[0]
                gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
                gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
                gripper_rot_new = R_map * gripper_rot
                adjusted_quat_scipy = gripper_rot_new.as_quat()
                adjusted_quat_temp = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], 
                                            adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
            else:
                adjusted_quat_temp = adjusted_quat
            
            # 调整位置（Z保持当前高度不变）
            if tube_place_pos_temp is not None:
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_temp = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                adjusted_pos = curr_pos_temp + tube_offset
                adjusted_pos[2] = current_z  # 保持当前高度
                print(f"    步骤6a: 调整位置 [{curr_pos_temp[0]:.3f}, {curr_pos_temp[1]:.3f}] → [{adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}] (Z={current_z:.3f}m不变)")
                print(f"      试管位置: [{tube_place_pos_temp[0]:.3f}, {tube_place_pos_temp[1]:.3f}, {tube_place_pos_temp[2]:.3f}]")
                print(f"      目标位置(end_pos): [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
                print(f"      偏移量: [{tube_offset[0]:.3f}, {tube_offset[1]:.3f}, {tube_offset[2]:.3f}]")
            else:
                adjusted_pos = adjusted_target.copy()
                adjusted_pos[2] = current_z
            
            # 原地调整位置（Z不变）
            success_adjust = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=adjusted_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            if not success_adjust:
                print(f"    ⚠️ 步骤6a调整失败，继续...")
            
            # ===== 步骤6b: 下降到目标高度 =====
            if tube_place_pos_temp is not None:
                ee_pose = robot.tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2:
                    p = p[0]
                curr_pos_after = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                tube_offset = end_pos - tube_place_pos_temp
                descend_pos = curr_pos_after + tube_offset
                descend_pos[2] = target_z  # 下降到目标高度
                print(f"    步骤6b: 下降 {curr_pos_after[2]:.3f}m → {target_z:.3f}m")
            else:
                descend_pos = adjusted_pos.copy()
                descend_pos[2] = target_z
            
            success_descend = move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=descend_pos,
                target_quat=adjusted_quat_temp,
                steps=100, record_step=record_step,
                gripper_open=False,
                allow_adaptive_search=True, limit_search_range=True,
            )
            # ===== 高度检测：试管末端是否已到达目标位置 =====
            tube_check_pos, _ = get_tube_end_pose(env, tube_idx)
            if tube_check_pos is not None:
                tube_check_pos = np.array(tube_check_pos, dtype=np.float32)
                # 如果试管末端高度 < 目标位置高度，说明试管已放到目标位置
                if tube_check_pos[2] < end_pos[2] + check_height:   # 允许1cm误差
                    print(f"    ✅ 试管已到达目标高度，停止下降")
                    break
            if not success_descend:
                print(f"  ⚠️ 步骤6b下降失败，继续...")

        # 最终放置
        print(f"\n  最终放置...")
        print(f"\n  步骤7c: 张开夹具停顿...")
        for i in range(50):
            action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
            robot_uid = f"piper_x-{robot_idx}"
            if robot_uid in action:
                qpos = robot.robot.get_qpos()[0].cpu().numpy()
                arm_qpos = qpos[:6]
                action[robot_uid] = np.concatenate([arm_qpos, [1.0]])  # 张开夹具
            
            other_uid = f"piper_x-{other_robot_idx}"
            if other_uid in action:
                other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
                other_arm_qpos = other_qpos[:6]
                action[other_uid] = np.concatenate([other_arm_qpos, [1.0]])
            if record_step:
                record_step(action)
            else:
                env.step(action)
        print(f"  ✅ 张开夹具停顿完成")
        # recorder.set_subtask(f"right arm places the tube (on yellow rack)")
        # success_insert, final_pos, final_quat = try_place_with_circular_search(
        #     env=env,
        #     robot_idx=robot_idx,
        #     robot=robot,
        #     target_pos=adjusted_target,
        #     target_quat=adjusted_quat,
        #     record_step=record_step,
        #     gripper_open_after=True,
        #     other_robot_idx=other_robot_idx,
        #     other_robot=other_robot,
        #     tube_idx=tube_idx,
        # )

        # if success_insert:
        #     print(f"  ✅ right arm 圆形搜索放置成功")
        # else:
        #     print(f"  ❌ 圆形搜索放置失败")
        #     return False
    else:
        print(f"  ❌ 移动到目标上方失败")
        return False
        # 步骤8-0: 保持当前姿态移动到(-0.2, 0, 当前高度)
    print(f"    步骤8-0: 移动到(-0.2, 0, 当前高度)...")
    ee_pose = robot.tcp_pose
    p = ee_pose.p
    if len(p.shape) == 2:
        p = p[0]
    curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
    curr_quat = robot.tcp_pose.q
    if len(curr_quat.shape) == 2:
        curr_quat = curr_quat[0]
    curr_quat = np.array(curr_quat.cpu().numpy() if hasattr(curr_quat, 'cpu') else curr_quat, dtype=np.float32)
    pre_retreat_pos = np.array([-0.3, 0.0, curr_pos[2]+0.1], dtype=np.float32)
    move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=pre_retreat_pos,
        target_quat=curr_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True,
    )

    print(f"  步骤8b: right arm 下降到初始位置...")
    return_arm_to_initial_pose(
        env=env,
        robot_idx=robot_idx,
        target_pos=initial_pos,
        target_quat=initial_quat,
        initial_qpos=initial_qpos,
        record_step=record_step,
    )
    print(f"  ✅ right arm 已返回初始位置")
    return True


def try_place_with_circular_search(
    env, robot_idx, robot,
    target_pos, target_quat,
    circle_radius=0.01, num_circle_points=12,
    descend_step=0.05, force_threshold=0.4,
    record_step=None, gripper_open_after=True,
    other_robot_idx=None, other_robot=None,
    approach_height=0.1,
    tube_idx=None,
):
    """
    在目标位置周围进行圆形搜索，找到无碰撞的放置位置

    流程：
    1. 在 target_xy 为中心、circle_radius 为半径的圆上生成 num_circle_points 个点（含中心点）
    2. 对每个点：
       a. 移动到该点上方 approach_height 处
       b. 逐级下降（每次 0.03m），每步检查接触力
       c. 如有接触（接触力 > force_threshold）→ 提起来 → 移动到下一个点
       d. 如无接触 → 继续下降
       e. 下降到距离目标位置 < 0.1m → 成功
    3. 如果所有点都失败 → 返回失败

    Returns:
        tuple: (success, final_pos, final_quat)
    """
    arm_name = f"机械臂{robot_idx}"

    # 计算夹爪到试管末端的偏移量，让试管末端以 target_pos 为圆心移动
    gripper_to_tube_offset = np.array([0.0, 0.0, 0.0])
    if tube_idx is not None:
        try:
            # 使用 get_tube_end_pose 获取试管末端（底部）位置
            tube_end_pos, tube_end_quat = get_tube_end_pose(env, tube_idx)
            if tube_end_pos is not None:
                # 获取夹爪末端位置
                ee_pose = robot.tcp_pose
                gripper_pos = ee_pose.p
                if hasattr(gripper_pos, 'cpu'):
                    gripper_pos = gripper_pos.cpu().numpy()
                if len(gripper_pos.shape) == 2:
                    gripper_pos = gripper_pos[0]
                gripper_pos = np.array(gripper_pos.ravel())
                
                # 计算偏移量：夹爪到试管末端的偏移
                gripper_to_tube_offset = tube_end_pos - gripper_pos
                print(f"    [debug] 夹爪位置: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
                print(f"    [debug] 试管末端位置: [{tube_end_pos[0]:.3f}, {tube_end_pos[1]:.3f}, {tube_end_pos[2]:.3f}]")
                print(f"    [debug] 偏移量: [{gripper_to_tube_offset[0]:.3f}, {gripper_to_tube_offset[1]:.3f}, {gripper_to_tube_offset[2]:.3f}]")
            else:
                print(f"    [debug] 无法获取试管末端位置，使用默认偏移")
        except Exception as e:
            print(f"    [debug] 计算偏移量失败: {e}")

    # 生成圆上的试探点（含中心点）- 目标是让试管底部以 target_pos 为圆心
    # 夹爪需要移动到 probe_point + gripper_to_tube_offset 的位置
    probe_points = [np.array([target_pos[0], target_pos[1], target_pos[2]])]  # 中心点
    for i in range(num_circle_points):
        angle = 2 * np.pi * i / num_circle_points
        px = target_pos[0] + circle_radius * np.cos(angle)
        py = target_pos[1] + circle_radius * np.sin(angle)
        probe_points.append(np.array([px, py, target_pos[2]]))

    print(f"\n{'=' * 60}")
    print(f"  {arm_name} 圆形搜索放置位置（半径={circle_radius}m，{len(probe_points)}个试探点）")
    print(f"  目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"{'=' * 60}")

    for pt_idx, probe_pt in enumerate(probe_points):
        # 计算夹爪需要到达的位置（probe_point + 偏移量）
        # 这样试管底部会以 target_pos 为圆心移动
        gripper_target = probe_pt + gripper_to_tube_offset
        
        # 该点上方 approach_height 处
        above_pt = gripper_target.copy()
        above_pt[2] += approach_height

        print(f"\n  试探点 {pt_idx + 1}/{len(probe_points)}: "
              f"[{probe_pt[0]:.3f}, {probe_pt[1]:.3f}, {probe_pt[2]:.3f}]")

        # 移动到该点上方放
        print(f"    移动到上方 {approach_height*100:.0f}cm...")
        success = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=above_pt,
            target_quat=target_quat,
            steps=20, record_step=record_step,
            gripper_open=not gripper_open_after,
            allow_adaptive_search=True, limit_search_range=True,
        )
        if not success:
            print(f"    ⚠️ 无法到达该点上方，跳过")
            continue

        contact_detected = False
        
        # # 逐级下降，每步读取试管末端位置并调整
        # num_steps = int(approach_height / descend_step)
        # contact_detected = False
        # current_above_pt = above_pt.copy()  # 当前夹爪目标位置（会被调整）
        
        # for step_i in range(1, num_steps + 1):
        # 直接下降到目标位置上方
        success = move_to_position_only(
            env, robot_idx=robot_idx,
            target_pos=above_pt,
            target_quat=target_quat,
            steps=10, record_step=record_step,
            gripper_open=not gripper_open_after,
            allow_adaptive_search=True, limit_search_range=True,
        )

        # 检查接触力（仅试管与架子之间的碰撞，排除夹爪夹持力）
        total_impulse = 0.0

        # 获取场景中所有接触
        all_contacts = env.unwrapped.scene.get_contacts()

        # 1. 检查试管与架子之间的接触（使用 get_pairwise_contact_impulse 精确过滤）
        if tube_idx is not None:
            try:
                tube_actor = get_tube_object(env, tube_idx)
                if tube_actor is not None and len(all_contacts) > 0:
                    tube_entity = tube_actor._bodies[0].entity if hasattr(tube_actor, '_bodies') else None
                    if tube_entity is not None:
                        # 检查试管与所有架子之间的接触
                        rack_names = ['tube_rack', 'tube_rack02', 'tube_rack03']
                        for rack_name in rack_names:
                            try:
                                if hasattr(env.unwrapped, 'objects') and rack_name in env.unwrapped.objects:
                                    rack_actor = env.unwrapped.objects[rack_name]
                                    rack_entity = rack_actor._bodies[0].entity if hasattr(rack_actor, '_bodies') else None
                                    if rack_entity is not None:
                                        pair_impulse = sapien_utils.get_pairwise_contact_impulse(
                                            all_contacts, tube_entity, rack_entity
                                        )
                                        impulse_norm = np.linalg.norm(pair_impulse)
                                        if impulse_norm > 0.001:
                                            print(f"    [debug] 试管与{rack_name}接触冲量: {impulse_norm:.6f}")
                                        total_impulse += impulse_norm
                            except:
                                pass
            except:
                pass

        # 有接触力：碰到架子其他部位了，没插进去，提起来移动到下一个位置
        if total_impulse > force_threshold:
            print(f"    ⚠️ 检测到接触力 {total_impulse:.4f} > {force_threshold}，提起来移动到下一个位置")
            contact_detected = True
            # 不 break，而是让下面的 contact_detected 处理块执行 continue

        # 无接触力：检查是否距离目标位置 < 0.1
        if tube_idx is not None:
            try:
                tube_actor = get_tube_object(env, tube_idx)
                if tube_actor is not None:
                    tube_pos = tube_actor.pose.p
                    if hasattr(tube_pos, 'cpu'):
                        tube_pos = tube_pos.cpu().numpy()
                    tube_pos = np.array(tube_pos.ravel())
                    dist_to_target = np.linalg.norm(tube_pos - target_pos)
                    print(f"    试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}], "
                            f"距离目标: {dist_to_target:.4f}m")
                    if dist_to_target < 0.1:
                        print(f"    ✅ 距离目标位置 {dist_to_target:.4f}m < 0.1m，放置成功！")
                        if gripper_open_after:
                            print(f"    张开夹爪释放试管...")
                            for i in range(30):
                                action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                                robot_uid = f"piper_x-{robot_idx}"
                                if robot_uid in action:
                                    current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
                                    arm_qpos = current_qpos[:6]
                                    action[robot_uid] = np.concatenate([arm_qpos, [1.0]])
                                if other_robot_idx is not None and other_robot is not None:
                                    other_uid = f"piper_x-{other_robot_idx}"
                                    if other_uid in action:
                                        other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
                                        other_arm_qpos = other_qpos[:6]
                                        action[other_uid] = np.concatenate([other_arm_qpos, [1.0]])
                                if record_step:
                                    record_step(action)
                                else:
                                    env.step(action)
                        print(f"  ✅ 试探点 {pt_idx + 1} 放置成功（距离目标 < 0.1m）！")
                        return (True, probe_pt, target_quat)
            except:
                pass

        if contact_detected:
            # 检测到接触后，完全提起到安全高度，再移动到下一个试探点
            # 避免试管在架子表面摩擦拖动
            LIFT_HEIGHT_AFTER_CONTACT = 0.15  # 提起到安全高度
            current_ee_pose = robot.tcp_pose
            p = current_ee_pose.p
            if len(p.shape) == 2:
                p = p[0]
            curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
            
            # 抬起到安全高度
            lift_pos = curr_pos.copy()
            lift_pos[2] += LIFT_HEIGHT_AFTER_CONTACT
            lift_pos[2] = min(lift_pos[2], MAX_Z_HEIGHT)
            print(f"    提起到安全高度 {LIFT_HEIGHT_AFTER_CONTACT*100:.0f}cm，避免摩擦...")
            move_to_position_only(
                env, robot_idx=robot_idx,
                target_pos=lift_pos,
                target_quat=target_quat,
                steps=20, record_step=record_step,
                gripper_open=not gripper_open_after,
                allow_adaptive_search=True, limit_search_range=True,
            )
            continue

    # 所有点都失败
    print(f"\n  ❌ 所有 {len(probe_points)} 个试探点均失败，无法放置")
    return (False, target_pos, target_quat)

def fake_place(env, tube_idx, robot, robot_idx=1, other_robot_idx=0, other_robot=None,
               record_step=None, approach_height=0.2):
    """
    右臂移动到试管上方再返回原位
    
    Args:
        env: 环境
        tube_idx: 试管序号 (0-4)
        robot_idx: 机械臂索引 (默认1=右臂)
        other_robot_idx: 另一机械臂索引
        other_robot: 另一机械臂对象
        record_step: 录制函数
        approach_height: 试管上方高度
    """
    arm_name = f"机械臂{robot_idx}"
    
    # 获取试管位置
    tube_pos = get_tube_position(env, tube_idx)
    if tube_pos is None:
        print(f"  ❌ 无法获取试管{tube_idx}位置")
        return False
    
    # 获取当前TCP位姿（用于返回原位）
    current_ee_pose = robot.tcp_pose
    current_p = current_ee_pose.p
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    current_pos = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q = current_ee_pose.q
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    current_quat = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)
    
    # 计算试管上方位置
    above_pos = np.array([tube_pos[0], tube_pos[1], tube_pos[2] + approach_height], dtype=np.float32)
    
    print(f"\n{'=' * 60}")
    print(f"  {arm_name} fake_place: 移动到试管{tube_idx}上方再返回")
    print(f"  试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}]")
    print(f"  上方位置: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}]")
    print(f"{'=' * 60}")
    
    # 计算原位上空0.2m
    above_current_pos = current_pos.copy()
    above_current_pos[2] += 0.2
    
    # 1. 原地上升0.2m
    print(f"\n  原地上升0.2m...")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_current_pos,
        target_quat=current_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 原地上升失败")
        return False
    
    # 2. 移动到试管上方
    print(f"\n  移动到试管{tube_idx}上方...")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_pos,
        target_quat=current_quat,
        steps=50, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 移动到试管上方失败")
        return False
    
    # 3. 返回原位上空0.2m
    print(f"\n  返回原位上空0.2m...")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=above_current_pos,
        target_quat=current_quat,
        steps=50, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 返回原位上空失败")
        return False
    
    # 4. 下降回原位
    print(f"\n  下降回原位...")
    success = move_to_position_only(
        env, robot_idx=robot_idx,
        target_pos=current_pos,
        target_quat=current_quat,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True,
    )
    if not success:
        print(f"  ❌ 下降回原位失败")
        return False
    
    print(f"  ✅ {arm_name} fake_place 完成")
    return True
def fake_pass(env, robot_idx=0, target_pos=None, target_quat=None,
              record_step=None, recorder=None):
    """
    左臂和右臂直接在交接点进行空交接（不夹试管）
    
    左臂移动到交接点，右臂也移动到交接点做交接动作，然后返回
    
    Returns:
        (success, robot_idx, initial_ee_pos, initial_ee_quat, initial_qpos,
         right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat)
    """
    print(f"\n{'=' * 60}")
    print(f"  fake_pass: 空交接（不夹试管）")
    print(f"{'=' * 60}")
    
    agent = env.unwrapped.agent
    left_arm_idx = robot_idx
    right_arm_idx = 1 - robot_idx
    
    # 默认交接点 [-0.4, 0, 0.25]
    if target_pos is None:
        target_pos = np.array([-0.4, 0, 0.25], dtype=np.float32)
    else:
        target_pos = np.array(target_pos, dtype=np.float32)
    
    left_robot = agent.agents[left_arm_idx]
    # 保存左臂初始位姿
    current_ee_pose = left_robot.tcp_pose
    current_p = current_ee_pose.p
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    initial_pos = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q = current_ee_pose.q
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    initial_quat = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)
    initial_qpos = left_robot.robot.get_qpos()[0].cpu().numpy().copy()
    
    # 左臂初始上升0.2m
    left_above_initial = initial_pos.copy()
    left_above_initial[2] += 0.2
    print(f"\n  左臂原地上升0.2m...")
    success = move_to_position_only(
        env, robot_idx=left_arm_idx, target_pos=left_above_initial,
        steps=30, record_step=record_step, gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, None, None, None, None
    
    # 左臂移动到交接点上方
    above_pos = target_pos.copy()
    above_pos[2] += 0.0
    print(f"\n  左臂移动到交接点上方...")
    success = move_to_position_only(
        env, robot_idx=left_arm_idx, target_pos=above_pos,
        steps=50, record_step=record_step, gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, None, None, None, None
    
    print(f"  左臂下降到交接点（等待右臂到来）...")
    success = move_to_position_only(
        env, robot_idx=left_arm_idx, target_pos=target_pos,
        target_quat=target_quat, steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, None, None, None, None
    
    # 右臂部分
    right_robot = agent.agents[right_arm_idx]
    # 保存右臂初始位姿
    right_initial_ee_pose = right_robot.tcp_pose
    right_initial_p = right_initial_ee_pose.p
    if len(right_initial_p.shape) == 2:
        right_initial_p = right_initial_p[0]
    right_initial_pos = np.array(right_initial_p.cpu().numpy() if hasattr(right_initial_p, 'cpu') else right_initial_p, dtype=np.float32)
    right_initial_q = right_initial_ee_pose.q
    if len(right_initial_q.shape) == 2:
        right_initial_q = right_initial_q[0]
    right_initial_quat = np.array(right_initial_q.cpu().numpy() if hasattr(right_initial_q, 'cpu') else right_initial_q, dtype=np.float32)
    right_initial_qpos = right_robot.robot.get_qpos()[0].cpu().numpy().copy()
    
    # 右臂初始上升（若当前高度低于0.2m则上升0.2m，否则跳过）
    right_above_initial = right_initial_pos.copy()
    if right_initial_pos[2] < 0.2:
        right_above_initial[2] += 0.2
        print(f"\n  右臂原地上升0.2m...")
        success = move_to_position_only(
            env, robot_idx=right_arm_idx, target_pos=right_above_initial,
            steps=30, record_step=record_step, gripper_open=True,
            allow_adaptive_search=True, limit_search_range=True)
        if not success:
            return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, None
    else:
        print(f"\n  右臂当前高度 {right_initial_pos[2]:.3f}m >= 0.2m，跳过上升")
    # 右臂接近交接点
    print(f"\n  右臂接近交接点...")
    right_above = target_pos.copy()
    right_above[2] -= 0.05
    success = move_to_position_only(
        env, robot_idx=right_arm_idx, target_pos=right_above,
        steps=50, record_step=record_step, gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, None
    
    # 右臂到达交接点
    # 右臂交接姿态（基轴定义）
    z_axis = np.array([0, 1, 0])
    x_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    right_handover_quat = R.from_matrix(rot_mat).as_quat()
    success = move_to_position_only(
        env, robot_idx=right_arm_idx, target_pos=target_pos,
        target_quat=right_handover_quat, steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    
    print(f"  ✅ 右臂已到达交接点，左臂归位")
    
    # 左臂返回原地上空0.2m
    print(f"\n  左臂返回原地上空0.2m...")
    success = move_to_position_only(
        env, robot_idx=left_arm_idx, target_pos=left_above_initial,
        target_quat=initial_quat, steps=50, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    
    # 左臂下降回原位
    print(f"  左臂下降回原位...")
    success = move_to_position_only(
        env, robot_idx=left_arm_idx, target_pos=initial_pos,
        target_quat=initial_quat, steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    
    print(f"  ✅ 左臂归位完成，右臂归位")
    
    # 右臂返回原地上空
    if right_initial_pos[2] < 0.2:
        print(f"\n  右臂返回原地上空0.2m...")
        success = move_to_position_only(
            env, robot_idx=right_arm_idx, target_pos=right_above_initial,
            steps=50, record_step=record_step,
            gripper_open=True,
            allow_adaptive_search=True, limit_search_range=True)
        if not success:
            return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat
    else:
        print(f"\n  右臂当前高度 {right_initial_pos[2]:.3f}m >= 0.2m，跳过返回上升") 

    # 右臂下降回原位（不约束姿态，让IK有更多自由度）
    print(f"  右臂下降回原位...")
    success = move_to_position_only(
        env, robot_idx=right_arm_idx, target_pos=right_initial_pos,
        steps=30, record_step=record_step,
        gripper_open=True,
        allow_adaptive_search=True, limit_search_range=True)
    if not success:
        return False, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat   
    print(f"  ✅ fake_pass 空交接完成")
    return True, robot_idx, initial_pos, initial_quat, initial_qpos, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat

def swap_tubes(env, record_step=None):
    """交换两个试管的位置（带抛物线轨迹动画）"""
    if hasattr(env.unwrapped, 'sample_tube_1') and hasattr(env.unwrapped, 'sample_tube_2'):
        sample_tube_1 = env.unwrapped.sample_tube_1
        sample_tube_2 = env.unwrapped.sample_tube_2

        pose1 = sample_tube_1.pose
        pose2 = sample_tube_2.pose

        pos1 = pose1.p
        pos2 = pose2.p
        q1 = pose1.q
        q2 = pose2.q

        if len(pos1.shape) == 2:
            pos1 = pos1[0]
        if len(pos2.shape) == 2:
            pos2 = pos2[0]
        if len(q1.shape) == 2:
            q1 = q1[0]
        if len(q2.shape) == 2:
            q2 = q2[0]

        pos1_np = np.array(pos1.cpu().numpy() if hasattr(pos1, 'cpu') else pos1, dtype=np.float32)
        pos2_np = np.array(pos2.cpu().numpy() if hasattr(pos2, 'cpu') else pos2, dtype=np.float32)
        q1_np = np.array(q1.cpu().numpy() if hasattr(q1, 'cpu') else q1, dtype=np.float32)
        q2_np = np.array(q2.cpu().numpy() if hasattr(q2, 'cpu') else q2, dtype=np.float32)

        steps = 20
        for i in range(steps + 1):
            t = i / steps

            new_pos1 = pos1_np * (1 - t) + pos2_np * t
            new_pos2 = pos2_np * (1 - t) + pos1_np * t

            lift_height = 0.2 * np.sin(t * np.pi)
            new_pos1[2] += lift_height
            new_pos2[2] += lift_height

            new_pose1 = sapien.Pose(p=new_pos1, q=q1_np)
            new_pose2 = sapien.Pose(p=new_pos2, q=q2_np)

            sample_tube_1.set_pose(new_pose1)
            sample_tube_2.set_pose(new_pose2)

            env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=False)
            env.unwrapped.capture_sensor_data()

            if record_step:
                sensor_data = {}
                for sensor_name, sensor in env.unwrapped.scene.sensors.items():
                    sensor_data[sensor_name] = sensor.get_obs()

                env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)
                render_images = env.unwrapped.scene.get_human_render_camera_images()

                mock_obs = {'sensor_data': sensor_data}
                for cam_name, cam_data in render_images.items():
                    if isinstance(cam_data, torch.Tensor):
                        mock_obs[cam_name] = cam_data.cpu().numpy().copy()
                    elif isinstance(cam_data, np.ndarray):
                        mock_obs[cam_name] = cam_data.copy()
                    elif isinstance(cam_data, dict) and 'rgb' in cam_data:
                        rgb = cam_data['rgb']
                        if isinstance(rgb, torch.Tensor):
                            cam_data['rgb'] = rgb.cpu().numpy().copy()
                        else:
                            cam_data['rgb'] = rgb.copy()
                        mock_obs[cam_name] = cam_data
                    else:
                        mock_obs[cam_name] = cam_data

                action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                record_step(action, mock_obs=mock_obs)
            else:
                action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                env.step(action)

            if i % 5 == 0:
                print(f"  交换进度: {i}/{steps}")

        print("✅ 试管位置已交换")
    else:
        print("⚠️ 未找到试管对象，跳过交换")


def move_baffle(env, move_to_far=True, record_step=None):
    """移开挡板到远处（不出现在画面中）"""
    if hasattr(env.unwrapped, 'baffle'):
        baffle = env.unwrapped.baffle

        current_pose = baffle.pose
        current_pos = current_pose.p
        current_q = current_pose.q

        if len(current_pos.shape) == 2:
            current_pos = current_pos[0]
        if len(current_q.shape) == 2:
            current_q = current_q[0]

        current_pos_np = np.array(current_pos.cpu().numpy() if hasattr(current_pos, 'cpu') else current_pos, dtype=np.float32)
        current_q_np = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)

        if move_to_far:
            target_pos = np.array([0.0, -10.0, 0.0], dtype=np.float32)

            steps = 50
            for i in range(steps + 1):
                t = i / steps

                new_pos = current_pos_np * (1 - t) + target_pos * t
                lift_height = 0.3 * np.sin(t * np.pi)
                new_pos[2] += lift_height

                new_pose = sapien.Pose(p=new_pos, q=current_q_np)
                baffle.set_pose(new_pose)

                env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=False)
                env.unwrapped.capture_sensor_data()

                if record_step:
                    sensor_data = {}
                    for sensor_name, sensor in env.unwrapped.scene.sensors.items():
                        sensor_data[sensor_name] = sensor.get_obs()

                    env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)
                    render_images = env.unwrapped.scene.get_human_render_camera_images()

                    mock_obs = {'sensor_data': sensor_data}
                    for cam_name, cam_data in render_images.items():
                        if isinstance(cam_data, torch.Tensor):
                            mock_obs[cam_name] = cam_data.cpu().numpy().copy()
                        elif isinstance(cam_data, np.ndarray):
                            mock_obs[cam_name] = cam_data.copy()
                        elif isinstance(cam_data, dict) and 'rgb' in cam_data:
                            rgb = cam_data['rgb']
                            if isinstance(rgb, torch.Tensor):
                                cam_data['rgb'] = rgb.cpu().numpy().copy()
                            else:
                                cam_data['rgb'] = rgb.copy()
                            mock_obs[cam_name] = cam_data
                        else:
                            mock_obs[cam_name] = cam_data

                    action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                    record_step(action, mock_obs=mock_obs)
                else:
                    action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                    env.step(action)

                if i % 10 == 0:
                    print(f"  移动进度: {i}/{steps}")

            print("✅ 挡板已移动到远处")
        else:
            current_pose = baffle.pose
            pos = current_pose.p
            if len(pos.shape) == 2:
                pos = pos[0]

            pos_np = np.array(pos.cpu().numpy() if hasattr(pos, 'cpu') else pos, dtype=np.float32)
            q_np = np.array(current_pose.q.cpu().numpy() if hasattr(current_pose.q, 'cpu') else current_pose.q, dtype=np.float32)
            if len(q_np.shape) == 2:
                q_np = q_np[0]

            new_pose = sapien.Pose(p=pos_np + np.array([0.3, 0.0, 0.0], dtype=np.float32), q=q_np)
            baffle.set_pose(new_pose)
            print("✅ 挡板已移动，偏移量: 0.3m")
    else:
        print("⚠️ 未找到挡板对象，跳过")


def return_arm_to_initial_pose(env, robot_idx, target_pos, target_quat, initial_qpos,
                                record_step=None):
    """将机械臂返回初始位置

    Args:
        env: 环境
        robot_idx: 机械臂索引
        target_pos: 初始末端位置 [x, y, z]
        target_quat: 初始末端姿态四元数 [x, y, z, w]
        initial_qpos: 初始关节位置
        record_step: 录制函数

    Returns:
        是否成功
    """
    print(f"\n========== 机械臂{robot_idx} 返回初始位置 ==========")

    agent = env.unwrapped.agent
    robot = agent.agents[robot_idx]

    target_pos = np.array(target_pos, dtype=np.float32)
    return_pos = target_pos.copy()
    return_pos[2] += 0.05  # 略高于初始位置

    print(f"  目标位置: {return_pos}")
    print(f"  目标四元数: {target_quat}")

    success = move_to_position_only(env, robot_idx=robot_idx, target_pos=return_pos,
                                     target_quat=target_quat, steps=80,
                                     record_step=record_step,
                                     gripper_open=True, allow_adaptive_search=True,
                                     limit_search_range=True)

    if success:
        print(f"  ✅ 机械臂已通过运动规划返回初始位置")
    else:
        print(f"  ⚠️ 运动规划失败，使用关节插值控制")
        # 备用方案：关节位置插值
        for i in range(150):
            action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
            robot_uid = f"piper_x-{robot_idx}"
            if robot_uid in action:
                t = min(1.0, (i + 1) / 50.0)
                current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
                interpolated_qpos = current_qpos * (1 - t) + initial_qpos[:len(current_qpos)] * t
                action[robot_uid] = interpolated_qpos[:len(action[robot_uid])]

            other_robot_idx = 1 - robot_idx
            other_robot_uid = f"piper_x-{other_robot_idx}"
            if other_robot_uid in action:
                other_robot = agent.agents[other_robot_idx]
                other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
                other_arm_qpos = other_qpos[:6]
                other_gripper_pos = -1.0
                action[other_robot_uid] = np.concatenate([other_arm_qpos, [other_gripper_pos]])

            if record_step:
                record_step(action)
            else:
                env.step(action)

        print(f"  ✅ 机械臂已通过关节控制返回初始位置")

    print(f"========== 返回完成 ==========")
    return True


def move_baffle_to_y(env, target_y, record_step=None):
    """移动挡板到指定的 y 位置

    Args:
        env: 环境
        target_y: 目标 y 坐标
        record_step: 录制函数
    """
    if hasattr(env.unwrapped, 'baffle'):
        baffle = env.unwrapped.baffle

        current_pose = baffle.pose
        current_pos = current_pose.p
        current_q = current_pose.q

        if len(current_pos.shape) == 2:
            current_pos = current_pos[0]
        if len(current_q.shape) == 2:
            current_q = current_q[0]

        current_pos_np = np.array(current_pos.cpu().numpy() if hasattr(current_pos, 'cpu') else current_pos, dtype=np.float32)
        current_q_np = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)

        target_pos = current_pos_np.copy()
        target_pos[1] = target_y

        steps = 50
        for i in range(steps + 1):
            t = i / steps
            new_pos = current_pos_np * (1 - t) + target_pos * t
            new_pose = sapien.Pose(p=new_pos, q=current_q_np)
            baffle.set_pose(new_pose)

            env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=False)
            env.unwrapped.capture_sensor_data()

            if record_step:
                sensor_data = {}
                for sensor_name, sensor in env.unwrapped.scene.sensors.items():
                    sensor_data[sensor_name] = sensor.get_obs()

                env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)
                render_images = env.unwrapped.scene.get_human_render_camera_images()

                mock_obs = {'sensor_data': sensor_data}
                for cam_name, cam_data in render_images.items():
                    if isinstance(cam_data, torch.Tensor):
                        mock_obs[cam_name] = cam_data.cpu().numpy().copy()
                    elif isinstance(cam_data, np.ndarray):
                        mock_obs[cam_name] = cam_data.copy()
                    elif isinstance(cam_data, dict) and 'rgb' in cam_data:
                        rgb = cam_data['rgb']
                        if isinstance(rgb, torch.Tensor):
                            cam_data['rgb'] = rgb.cpu().numpy().copy()
                        else:
                            cam_data['rgb'] = rgb.copy()
                        mock_obs[cam_name] = cam_data
                    else:
                        mock_obs[cam_name] = cam_data

                action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                record_step(action, mock_obs=mock_obs)
            else:
                action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
                env.step(action)

            if i % 10 == 0:
                print(f"  挡板移动进度: {i}/{steps}")

        print(f"✅ 挡板已移动到 y={target_y:.2f}")
    else:
        print("⚠️ 未找到挡板对象，跳过")


def check_tube_placement_accuracy(env, tube_idx, target_pos):
    """检查试管当前位置与目标位置的距离（XY平面）

    Returns:
        (success: bool, distance: float)
    """
    tube = get_tube_object(env, tube_idx)
    if tube is None:
        return False, float('inf')
    tube_pose = tube.pose.p
    if hasattr(tube_pose, 'cpu'):
        tube_pose = tube_pose.cpu().numpy()
    tube_pos = np.array(tube_pose).flatten()
    target_np = np.array(target_pos, dtype=np.float32).flatten()
    xy_dist = np.linalg.norm(tube_pos[:2] - target_np[:2])
    print(f"  试管位置: [{tube_pos[0]:.4f}, {tube_pos[1]:.4f}, {tube_pos[2]:.4f}]")
    print(f"  目标位置: [{target_np[0]:.4f}, {target_np[1]:.4f}, {target_np[2]:.4f}]")
    print(f"  XY距离: {xy_dist:.4f} (阈值: 0.05)")
    return xy_dist <= 0.05, xy_dist


def get_next_episode_number(base_dir):
    """自动检测下一个可用的 episode 序号"""
    hdf5_dir = os.path.join(base_dir, "hdf5")
    os.makedirs(hdf5_dir, exist_ok=True)
    existing = [f for f in os.listdir(hdf5_dir) if f.startswith("house1_episode") and f.endswith(".hdf5")]
    if not existing:
        return 0
    numbers = []
    for f in existing:
        try:
            num = int(f.replace("house1_episode", "").replace(".hdf5", ""))
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 0


def improve_grip_physics(env):
    """提高夹爪与试管之间的摩擦力，降低试管质量，防止抓取滑落"""
    import sapien
    env_unwrapped = env.unwrapped
    agent = env_unwrapped.agent

    # 1. 提高夹爪碰撞面的摩擦力
    for robot in agent.agents:
        try:
            for link in robot.robot.get_links():
                for shape in link.get_collision_shapes():
                    try:
                        mat = shape.physical_material
                        mat.static_friction = GRIP_FRICTION
                        mat.dynamic_friction = GRIP_FRICTION
                        mat.restitution = 0.0
                    except:
                        pass
        except:
            pass

    # 2. 降低试管质量并提高摩擦力
    for name in ['sample_tube_1', 'sample_tube_2', 'sample_tube_3']:
        if hasattr(env_unwrapped, name):
            tube = getattr(env_unwrapped, name)
            try:
                # 降低质量到原来的1/5
                for link in tube.get_links():
                    link.mass = 0.01
                    for shape in link.get_collision_shapes():
                        try:
                            mat = shape.physical_material
                            mat.static_friction = GRIP_FRICTION
                            mat.dynamic_friction = GRIP_FRICTION
                            mat.restitution = 0.01
                        except:
                            pass
            except Exception as e:
                print(f"  [info] 无法修改{name}物理属性: {e}")
    print("  [物理] 已提高夹爪摩擦力(2.0) + 降低试管质量(0.02)")


TASK_SEQUENCES = [
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 0
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 1
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 2
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 3
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 4
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 5
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 6
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 7
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 8
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 9
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 10
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 11
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 12
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 13
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 14
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 15
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 16
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 17
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 18
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 19
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 20
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 21
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 22
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 23
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 24
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 25
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 26
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 27
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 28
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 29
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 30
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 31
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 32
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 33
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 34
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 35
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 36
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 37
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 38
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 39
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 40
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 41
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 42
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 43
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 44
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 45
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 46
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 47
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 48
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 49
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 50
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 51
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 52
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 53
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 54
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 55
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 56
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 57
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 58
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 59
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 60
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 61
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 62
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 63
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 64
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 65
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 66
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 67
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 68
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 69
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 70
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 71
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 72
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 73
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 74
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 75
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 76
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 77
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 78
    },
    {
        'data_name_prefix': 'train-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 79
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 80
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 81
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 82
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 83
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 84
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 85
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 86
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 87
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 88
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 89
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 90
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 91
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 92
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 93
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 94
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 95
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 96
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 97
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 98
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 99
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 100
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 101
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 102
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 103
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 104
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 105
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 106
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 107
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 108
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 109
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 110
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 111
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 112
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 113
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 114
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 115
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 116
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 117
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 118
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 119
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 120
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 121
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 122
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 123
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 124
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 125
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 126
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 127
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 128
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 129
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 130
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 131
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 132
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 133
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 134
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 135
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 136
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 137
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 138
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 139
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 140
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 141
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 142
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 143
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 144
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 145
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 146
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 147
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 148
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 149
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 150
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 151
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 152
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 153
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 154
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 155
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 156
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 157
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 7,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 158
    },
    {
        'data_name_prefix': 'train-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 7,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 159
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 160
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 161
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 162
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 163
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 164
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 165
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 3,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 166
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 167
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 168
    },
    {
        'data_name_prefix': 'val-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 169
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 170
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 171
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 172
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 173
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 174
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 175
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 176
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 2,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 177
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 3,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 178
    },
    {
        'data_name_prefix': 'val-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 1,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 179
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 180
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 181
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 182
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 183
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 184
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 185
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 186
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 187
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 188
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 189
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 1,
        '序号': 190
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 191
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD3'},
        ],
        'target_object': 3,
        '序号': 192
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
        ],
        'target_object': 4,
        '序号': 193
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 1,
        '序号': 194
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 195
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 196
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD2'},
        ],
        'target_object': 4,
        '序号': 197
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD3'},
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
        ],
        'target_object': 1,
        '序号': 198
    },
    {
        'data_name_prefix': 'test-ID',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD1'},
        ],
        'target_object': 2,
        '序号': 199
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 200
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T5', 'location': 'DD3'},
        ],
        'target_object': 4,
        '序号': 201
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD1'},
            {'type': 'handoff', 'object': 'T5', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T5', 'arm': 'right', 'location': 'D3'},
        ],
        'target_object': 5,
        '序号': 202
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 203
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD3'},
        ],
        'target_object': 2,
        '序号': 204
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 205
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that was moved into the inspection area last into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'distractor_place', 'object': 'T5', 'location': 'DD2'},
            {'type': 'handoff', 'object': 'T4', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T4', 'arm': 'right', 'location': 'D1'},
        ],
        'target_object': 4,
        '序号': 206
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T5', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T5', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T1', 'location': 'DD3'},
        ],
        'target_object': 5,
        '序号': 207
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D3'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 208
    },
    {
        'data_name_prefix': 'test-OOD',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T2', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T2', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T3', 'location': 'DD2'},
        ],
        'target_object': 2,
        '序号': 209
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 210
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 211
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 212
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 213
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 214
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 215
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 216
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 217
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 218
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 219
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 220
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 221
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 222
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 223
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 224
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 7,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 225
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 226
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 0,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 227
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 228
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 1,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 229
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 230
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 2,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 231
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 232
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 3,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 233
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 234
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm just received into the isolation rack.',
        'layout': 4,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 235
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD2'},
        ],
        'target_object': 3,
        '序号': 236
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the left arm handed to the right arm into the isolation rack.',
        'layout': 5,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T4', 'location': 'DD1'},
        ],
        'target_object': 1,
        '序号': 237
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T1', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T1', 'arm': 'right', 'location': 'D1'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD2'},
        ],
        'target_object': 1,
        '序号': 238
    },
    {
        'data_name_prefix': 'test-CF',
        'instruction': 'Using the left arm, place the sample tube that the right arm put on the inspection station into the isolation rack.',
        'layout': 6,
        'repeat_id': 4,
        'sequence': [
            {'type': 'handoff', 'object': 'T3', 'from_arm': 'left', 'to_arm': 'right'},
            {'type': 'place', 'object': 'T3', 'arm': 'right', 'location': 'D2'},
            {'type': 'distractor_place', 'object': 'T2', 'location': 'DD1'},
        ],
        'target_object': 3,
        '序号': 239
    },
]

def main(tube_idx=0, target_pos=None, target_rot_deg=None, output_name=None, grasp_rot_deg=None, calibrate=False, tube_pos_idx=None, sequence=None, repeat_id=None, calib_original_y=None, task_idx=None,
        layout_id=None, pass_id=None, place_01=None, grasp_id=None, mode=None, retry_attempt=0,
        right_place=None, grasp_start=None, grasp_end=None, grasp_arm=None,
        grasp_start02=None, grasp_end02=None, fake_place_idx=None, do_fake_pass=False):
    """
    主函数：创建环境，执行试管抓取放置任务，录制数据

    Args:
        tube_idx: 试管索引 (0: sample_tube_1, 1: sample_tube_2)
        target_pos: 目标位置 [x, y, z]
        target_rot_deg: 目标姿态欧拉角 [roll, pitch, yaw]（度）
        output_name: 输出名称，用于命名保存目录
        grasp_rot_deg: 夹取姿态欧拉角 [roll, pitch, yaw]（度）
        calibrate: 自动微调模式。放置后检测XY距离，若>0.05则自动微调target-pos y值重试
        tube_pos_idx: 预定义试管位姿组索引，使用 PREDEFINED_TUBE_POSITIONS 中的一组
        sequence: 任务序列，每个元素为 dict，支持 type='cover'/'swap'/'remove_occluder'
        task_idx: 任务序列索引，传入后直接使用该索引作为 episode 编号（遵循 TASK_SEQUENCES 顺序）
        layout_id: 布局索引 (0-8)，覆盖 tube_pos_idx
        pass_id: 左臂传递给右臂的试管索引，覆盖 tube_idx
        place_01: 右臂放置目标索引 (0-2)，覆盖 target_pos
        grasp_id: 左臂抓取放置的试管索引
        mode: 0=放回原位(左臂), 1=放到shelf(就近臂)
    """
    # The benchmark imports the control functions in this module without
    # collector I/O. Keep HDF5/video-only dependencies on the CLI path.
    import cv2
    from mani_skill.envs.tasks.custom_task.biolab.data_recorder import BioLabDataRecorder

    # 声明使用全局变量 final_pos（默认=桌面放置位），place_01 有效时覆盖它
    global final_pos

    # 新参数映射：如果传入了新参数，覆盖旧参数
    if layout_id is not None:
        tube_pos_idx = layout_id
    if pass_id is not None:
        tube_idx = pass_id
    if place_01 is not None:
        if 0 <= place_01 < len(TUBE_PLACE_TARGETS_3):
            target_pos = np.array(TUBE_PLACE_TARGETS_3[place_01], dtype=np.float32)
            final_pos = np.array(TUBE_PLACE_TARGETS_3[place_01], dtype=np.float32)
            print(f"  使用 TUBE_PLACE_TARGETS_3[{place_01}]: target_pos={target_pos.tolist()}, final_pos={final_pos.tolist()}")
        else:
            print(f"  ⚠️ place_01={place_01} 超出范围 (0-{len(TUBE_PLACE_TARGETS_3)-1})，使用默认 target_pos")
    if mode is not None:
        print(f"  模式: {'放回原位' if mode == 0 else '放到 shelf'}")
    
    base_dir = "/media/ningkang/win_data02/a_datasets/L5"
    # 如果传入了 task_idx，直接用它作为 episode 编号（遵循 TASK_SEQUENCES 顺序）
    if task_idx is not None:
        episode_idx = task_idx
    else:
        # 兼容旧调用：自动检测下一个可用的 episode 序号
        episode_idx = get_next_episode_number(base_dir)
    episode_name = f"bio5_episode{episode_idx}"
    print(f"  Episode: {episode_name}")

    # 创建各子目录
    hdf5_dir = os.path.join(base_dir, "hdf5")
    csv_dir = os.path.join(base_dir, "csv")
    temp_frames_base = os.path.join(base_dir, "temp_frames")
    videos_base = os.path.join(base_dir, "videos")
    for d in [hdf5_dir, csv_dir, temp_frames_base, videos_base]:
        os.makedirs(d, exist_ok=True)
    for sub in ["front", "left", "right"]:
        os.makedirs(os.path.join(videos_base, sub), exist_ok=True)

    # 当前 episode 的临时图片目录
    temp_image_dir = os.path.join(temp_frames_base, episode_name)
    os.makedirs(temp_image_dir, exist_ok=True)
    for cam_name in ["human_cam", "piper_x-0-hand_camera", "piper_x-1-hand_camera"]:
        os.makedirs(os.path.join(temp_image_dir, cam_name), exist_ok=True)
        os.makedirs(os.path.join(temp_image_dir, f"{cam_name}_depth"), exist_ok=True)

    # 如果指定了布局索引，使用 LAYOUT_8 初始化试管位置
    if tube_pos_idx is not None:
        # 验证 tube_pos_idx 范围
        if 0 <= tube_pos_idx < len(LAYOUT_8):
            layout_order = LAYOUT_8[tube_pos_idx]  # 例如 [0, 1, 2, 3, 4]
            print(f"  使用 LAYOUT_8[{tube_pos_idx}] = {layout_order} 初始化试管位置")

            from mani_skill.envs.tasks.custom_task.biolab.Scene_L5_02 import Scene_L5_02
            # 获取所有试管对象（Scene_L5_02 使用 OBJECTS 属性，格式为带 name/model_path 的 dict）
            tube_entries = [obj for obj in Scene_L5_02.OBJECTS
                            if isinstance(obj, dict) and "sample_tube" in obj.get("name", "")]
            # 确保有5个试管
            while len(tube_entries) > 5:
                obj_to_remove = tube_entries[-1]
                for j, existing in enumerate(Scene_L5_02.OBJECTS):
                    if existing is obj_to_remove:
                        del Scene_L5_02.OBJECTS[j]
                        break
                tube_entries.pop()
            while len(tube_entries) < 5:
                new_entry = {
                    "name": f"sample_tube_{len(tube_entries)}",
                    "type": "actor",
                    "model_path": os.path.join(Scene_L5_02.BIOLAB_CHEMISTRY_MODEL_DIR, "015_test_tube"),
                    "collision_file": "collision.ply",
                    "visual_file": "textured.obj",
                    "pos": [0, 0, 0], "rot_deg": [0, 0, 0], "scale": list(tube_scale),
                    "density": 1000, "static": control_cube_static
                }
                Scene_L5_02.OBJECTS.append(new_entry)
                tube_entries.append(new_entry)

            # 按 LAYOUT_8 映射设置试管位置
            # sample_tube_i 的初始位置 = cube_origin_place[LAYOUT_8[layout][i]]
            for i, entry in enumerate(tube_entries):
                if i < len(layout_order):
                    tube_slot_idx = layout_order[i]  # 从 LAYOUT_8 获取槽位编号
                    pos = cube_origin_place[tube_slot_idx]
                    entry["pos"] = list(pos)
                    # 使用默认旋转 [90, 90, 0]
                    entry["rot_deg"] = [0, 0, 0]
                    print(f"  sample_tube_{i} → 槽位{tube_slot_idx} → 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            # 更新 tube_origin_pose 为对应试管索引的初始位置
            if 0 <= tube_idx < len(layout_order):
                tube_origin_pose = cube_origin_place[layout_order[tube_idx]]
                print(f"  当前试管{tube_idx} (布局内编号{layout_order[tube_idx]})，原始槽位: [{tube_origin_pose[0]:.3f}, {tube_origin_pose[1]:.3f}, {tube_origin_pose[2]:.3f}]")
        else:
            print(f"⚠️ tube-pos-idx {tube_pos_idx} 超出范围 (0-{len(LAYOUT_8)-1})，使用默认位置")
            # 回退：直接使用 cube_origin_place[tube_idx]
            tube_origin_pose = cube_origin_place[tube_idx] if 0 <= tube_idx < len(cube_origin_place) else None
            if tube_origin_pose is not None:
                print(f"  选择试管{tube_idx}，原始槽位: [{tube_origin_pose[0]:.3f}, {tube_origin_pose[1]:.3f}, {tube_origin_pose[2]:.3f}]")

    # 创建并初始化环境
    print("=" * 60)
    print("创建 Scene_L5_02 环境...")
    print("=" * 60)

    env = gym.make(
        "Scene_L5_02",
        obs_mode="rgb+depth+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="auto",
        robot_uids=("piper_x", "piper_x"),
    )

    obs, info = env.reset()
    print("环境已重置")

    # 优化物理属性：提高摩擦力 + 降低试管质量
    improve_grip_physics(env)
 # 如果有 cover 序列步骤，在初始化时直接设置挡板位置
    if sequence is not None:
        for init_step in sequence:
            if init_step.get('type') == 'cover':
                object_name = init_step['object']
                init_tube_obj = env.unwrapped.sample_tube_1 if object_name == 'O1' else env.unwrapped.sample_tube_2
                init_obj_pose = init_tube_obj.pose.p
                if hasattr(init_obj_pose, 'cpu'):
                    init_obj_pose = init_obj_pose.cpu().numpy()
                init_obj_pose = np.array(init_obj_pose).flatten()
                init_obj_y = float(init_obj_pose[1])
                init_baffle_y = 0.2 if init_obj_y > 0 else -0.2
                print(f"  序列含 cover({object_name}): y={init_obj_y:.3f} → 挡板初始化 y={init_baffle_y:.2f}")
                init_baffle = env.unwrapped.baffle
                init_bp = init_baffle.pose
                init_baffle_pos = init_bp.p
                init_baffle_q = init_bp.q
                if len(init_baffle_pos.shape) == 2:
                    init_baffle_pos = init_baffle_pos[0]
                if len(init_baffle_q.shape) == 2:
                    init_baffle_q = init_baffle_q[0]
                init_bpn = np.array(init_baffle_pos.cpu().numpy() if hasattr(init_baffle_pos, 'cpu') else init_baffle_pos, dtype=np.float32)
                init_bqn = np.array(init_baffle_q.cpu().numpy() if hasattr(init_baffle_q, 'cpu') else init_baffle_q, dtype=np.float32)
                init_new_pos = init_bpn.copy()
                init_new_pos[1] = init_baffle_y
                init_new_pose = sapien.Pose(p=init_new_pos, q=init_bqn)
                init_baffle.set_pose(init_new_pose)
                break

    # 添加物理模拟步骤让场景稳定
    print("等待物理模拟稳定...")
    agent = env.unwrapped.agent
    current_qpos = {}
    for robot_idx, robot in enumerate(agent.agents):
        robot_uid = f"piper_x-{robot_idx}"
        current_qpos[robot_uid] = robot.robot.get_qpos()[0].cpu().numpy().copy()
        if len(current_qpos[robot_uid]) >= 7:
            current_qpos[robot_uid][6] = -1.0  # 夹爪闭合

    for _ in range(50):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        for robot_idx, robot in enumerate(agent.agents):
            robot_uid = f"piper_x-{robot_idx}"
            if robot_uid in action and robot_uid in current_qpos:
                action[robot_uid] = current_qpos[robot_uid][:len(action[robot_uid])]
        env.step(action)

    # ========== 读取初始末端姿态（用于调试） ==========
    print("\n初始末端姿态:")
    for ri in range(len(agent.agents)):
        _init_robot = agent.agents[ri]
        _init_ee = _init_robot.tcp_pose
        _init_p = _init_ee.p
        _init_q = _init_ee.q
        if len(_init_p.shape) == 2:
            _init_p = _init_p[0]
        if len(_init_q.shape) == 2:
            _init_q = _init_q[0]
        _init_p_np = np.array(_init_p.cpu().numpy() if hasattr(_init_p, 'cpu') else _init_p, dtype=np.float32)
        _init_q_np = np.array(_init_q.cpu().numpy() if hasattr(_init_q, 'cpu') else _init_q, dtype=np.float32)
        _init_euler = R.from_quat(_init_q_np).as_euler('xyz', degrees=True)
        _init_z_axis = R.from_quat(_init_q_np).as_matrix()[:, 2]
        print(f"  机械臂{ri} 初始位置: [{_init_p_np[0]:.3f}, {_init_p_np[1]:.3f}, {_init_p_np[2]:.3f}]")
        print(f"  机械臂{ri} 初始姿态 (欧拉角): [{_init_euler[0]:.1f}, {_init_euler[1]:.1f}, {_init_euler[2]:.1f}]")
        print(f"  机械臂{ri} 初始TCP Z轴方向: [{_init_z_axis[0]:.3f}, {_init_z_axis[1]:.3f}, {_init_z_axis[2]:.3f}]")

    # 设置相机名称
    global_cam_names = ["human_cam"]
    hand_cam_names = ["piper_x-0-hand_camera", "piper_x-1-hand_camera"]
    all_camera_names = global_cam_names + hand_cam_names

    recorded_data = {cam: [] for cam in all_camera_names}
    frame_counter = 0

    # 创建录制数据存储
    trajectory_data = {
        'joint_positions': [],
        'joint_velocities': [],
        'joint_torques': [],
        'end_effector_poses': [],
        'end_effector_forces': [],
        'contact_forces': [],
        'object_poses': [],
        'timestamps': [],
        'camera_rgb': {},
        'camera_depth': {}
    }
    for cam_name in all_camera_names:
        trajectory_data['camera_rgb'][cam_name] = []
        trajectory_data['camera_depth'][cam_name] = []

    print("\n创建数据记录器...")
    recorder = BioLabDataRecorder()
    # recorder.set_subtask("idle")

    # 定义录制函数
    def record_step(action, mock_obs=None):
        """录制一步数据（包括机械臂状态和相机图像）"""
        nonlocal frame_counter

        if mock_obs is not None:
            obs = mock_obs
            reward = 0
            terminated = False
            truncated = False
            info = {}
        else:
            try:
                step_result = env.step(action)
                if step_result is None:
                    return None
                obs, reward, terminated, truncated, info = step_result
            except Exception as e:
                print(f"  警告: env.step() 出错: {e}")
                return None

        # 记录机械臂状态和相机图像到数据记录器
        recorder.capture_robot_state(agent)
        recorder.capture_camera_images(env, camera_names=all_camera_names, obs=obs)

        # 边采集边保存图片（避免内存累积）
        if isinstance(obs, dict) and 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            if isinstance(sensor_data, dict):
                for cam_name in all_camera_names:
                    if cam_name in sensor_data:
                        cam_data = sensor_data[cam_name]
                        if isinstance(cam_data, dict) and 'rgb' in cam_data:
                            img = cam_data['rgb']
                            if isinstance(img, torch.Tensor):
                                img_np = img.cpu().numpy()
                            else:
                                img_np = np.array(img)
                            if len(img_np.shape) == 4:
                                img_np = img_np.squeeze()
                            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            else:
                                img_np = img_np.astype(np.uint8)

                            rgb_path = os.path.join(temp_image_dir, cam_name, f"frame_{frame_counter:06d}.jpg")
                            cv2.imwrite(rgb_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                            recorded_data[cam_name].append(rgb_path)

        frame_counter += 1

        # 定期内存清理
        if frame_counter % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [内存清理] 已录制 {frame_counter} 帧")

        return obs, reward, terminated, truncated, info

    # ========== 录制初始状态 ==========
    print("\n录制初始状态...")
    for i in range(10):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        for robot_idx, robot in enumerate(agent.agents):
            robot_uid = f"piper_x-{robot_idx}"
            if robot_uid in action:
                current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
                arm_qpos = current_qpos[:6]
                gripper_control = -1.0
                action[robot_uid] = np.concatenate([arm_qpos, [gripper_control]])
        record_step(action)
        if i % 5 == 0:
            print(f"  已录制 {len(recorded_data['human_cam'])} 帧画面")

    
    # ========== 根据 sequence 顺序执行，最后执行 target_object ==========
    if sequence is not None:
        handoff_state = {}
        right_grasp_quat_global = None
        for step_idx, step in enumerate(sequence):
            step_type = step['type']
            print(f"\n--- 序列步骤 {step_idx+1}/{len(sequence)}: {step_type} ---")
            if step_type == 'handoff':
                obj = int(step['object'][1]) - 1
                tube_pos = get_tube_position(env, obj)
                if tube_pos is not None:
                    print(f"  试管{obj} 位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}]")
                print_tube_pose(env, obj, prefix="  ")
                _, itq = get_tube_pose(env, obj)
                if itq is None: itq = lift_tube_quat
                z_axis = np.array([0, -1, 0]); x_axis = np.array([0, 0, -1])
                y_axis = np.cross(z_axis, x_axis)
                rot_mat = np.column_stack([x_axis, y_axis, z_axis])
                handover_quat = R.from_matrix(rot_mat).as_quat()
                print(f"\n交接姿态（已补偿试管pitch角）")
                success, robot_idx, ee_pos, ee_quat, qpos = pass_to(
                    env=env, tube_idx=obj, robot_idx=0,
                    target_pos=target_pos, target_quat=handover_quat,
                    record_step=record_step, recorder=recorder, grasp_rot_deg=grasp_rot_deg)
                if not success:
                    print("  ❌ pass_to 失败")
                    if calibrate: shutil.rmtree(temp_image_dir, ignore_errors=True)
                    env.close(); return
                tp_pos, tp_quat = get_tube_pose(env, obj)
                if tp_pos is not None:
                    # 修正格式转换（SAPIEN [w,x,y,z] → SciPy [x,y,z,w]）
                    tube_rot = R.from_quat([tp_quat[1], tp_quat[2], tp_quat[3], tp_quat[0]])
                    tube_z_world = tube_rot.apply([0, 0, 1])
                    
                    world_z = np.array([0, 0, 1])
                    cos_angle = np.dot(tube_z_world, world_z)
                    angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    
                    print(f"  试管z轴朝向: {tube_z_world}")
                    print(f"  与世界z轴的夹角: {angle_deg:.2f}°")
                    
                    # 用 align_vectors 自动计算最优旋转
                    R_map, _ = R.align_vectors([world_z], [tube_z_world])
                    
                    # 获取当前夹爪姿态
                    ee_pose = env.unwrapped.agent.agents[0].tcp_pose
                    ee_q = ee_pose.q
                    if len(ee_q.shape) == 2:
                        ee_q = ee_q[0]
                    gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
                    gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
                    
                    # 新夹爪姿态
                    gripper_rot_new = R_map * gripper_rot
                    adj_q_scipy = gripper_rot_new.as_quat()
                    adj_q = np.array([adj_q_scipy[3], adj_q_scipy[0], adj_q_scipy[1], adj_q_scipy[2]])
                    
                    print(f"    调整后四元数: [{adj_q[0]:.4f}, {adj_q[1]:.4f}, {adj_q[2]:.4f}, {adj_q[3]:.4f}]")
                    ee_p = env.unwrapped.agent.agents[0].tcp_pose.p
                    if len(ee_p.shape) == 2: ee_p = ee_p[0]
                    cp = np.array(ee_p.cpu().numpy() if hasattr(ee_p, 'cpu') else ee_p, dtype=np.float32)
                    move_to_position_only(env, robot_idx=0, target_pos=cp, target_quat=adj_q,
                        steps=60, record_step=record_step, gripper_open=False,
                        allow_adaptive_search=True, limit_search_range=True)
                    handover_quat = adj_q
                rp_s, rp_pos, rp_q, rp_qp, rh_q = right_pass(
                    env=env, tube_idx=obj, robot_idx=robot_idx,
                    initial_ee_pos=ee_pos, initial_ee_quat=ee_quat,
                    initial_qpos=qpos, recorder=recorder, record_step=record_step)
                if not rp_s:
                    print("  ❌ 右臂交接失败")
                    if calibrate: shutil.rmtree(temp_image_dir, ignore_errors=True)
                    env.close(); return
                la = robot_idx; ra = 1 - robot_idx; ag = env.unwrapped.agent
                handoff_state = {'la': la, 'ra': ra, 'rr': ag.agents[ra], 'lr': ag.agents[la],
                    'rh_q': rh_q, 'rp_pos': rp_pos, 'rp_quat': rp_q, 'rp_qpos': rp_qp, 'tube': obj}
            elif step_type == 'place':
                if not handoff_state: continue
                loc = int(step['location'][1]) - 1
                right_place_procedure(env=env, tube_idx=handoff_state['tube'],
                    right_arm_idx=handoff_state['ra'], left_arm_idx=handoff_state['la'],
                    right_robot=handoff_state['rr'], left_robot=handoff_state['lr'],
                    right_handover_quat=handoff_state['rh_q'],
                    right_initial_pos=handoff_state['rp_pos'],
                    right_initial_quat=handoff_state['rp_quat'],
                    right_initial_qpos=handoff_state['rp_qpos'],
                    right_place_idx=loc, recorder=recorder, record_step=record_step)
                print(f"  → 执行上升返回动作...")
                lift_and_return(env=env, robot_idx=handoff_state['ra'], lift_height=0.15, steps=100, record_step=record_step)
                handoff_state = {}
            elif step_type == 'distractor_place':
                obj = int(step['object'][1]) - 1; loc = int(step['location'][2]) - 1
                rg_q = R.from_euler('xyz', np.deg2rad([0, 60, 0])).as_quat().astype(np.float32)
                right_grasp_quat_global = rg_q
                print(f"\n{'='*60}\n  distractor_place: 右臂抓取试管{obj}->位置{loc}\n{'='*60}")
                grasp_and_place(env=env, tube_idx=obj if 0<=obj<=4 else None,
                    robot_idx=1, other_robot_idx=0, robot=agent.agents[1],
                    other_robot=agent.agents[0], start_idx=obj, end_idx=5+loc,
                    right_handover_quat=rg_q, recorder=recorder, record_step=record_step)
            elif step_type == 'fake_handoff':
                print(f"\n{'='*60}\n  fake_pass: 双臂空交接假动作\n{'='*60}")
                s, _, _, _, _, _, _, _, _ = fake_pass(env=env, robot_idx=0, record_step=record_step, recorder=recorder)
                if not s: print("  ❌ fake_pass 空交接失败")
            elif step_type == 'pass_over':
                obj = int(step['object'][1]) - 1
                print(f"\n{'='*60}\n  pass_over: 右臂经过试管{obj}上方返回\n{'='*60}")
                fake_place(env=env, tube_idx=obj, robot=agent.agents[1],
                    robot_idx=1, other_robot_idx=0, other_robot=agent.agents[0],
                    record_step=record_step)
        # ========== 最后执行 target_object 的抓取和放置 ==========
        if grasp_start02 is not None and grasp_end02 is not None:
            left_robot = agent.agents[0]; left_other = agent.agents[1]
            robot_base = left_robot.robot.pose.p
            if hasattr(robot_base, 'cpu'): robot_base = robot_base.cpu().numpy()
            robot_base = np.array(robot_base).flatten()
            if 0 <= grasp_start02 <= 4:
                tube_pos, _ = get_tube_pose(env, grasp_start02)
                if tube_pos is not None:
                    dx = tube_pos[0] - robot_base[0]; dy = tube_pos[1] - robot_base[1]
                    yaw_angle = np.rad2deg(np.arctan2(dy, dx))
                    print(f"  左臂基座: [{robot_base[0]:.3f},{robot_base[1]:.3f}] 试管: [{tube_pos[0]:.3f},{tube_pos[1]:.3f}]")
            if right_grasp_quat_global is None:
                right_grasp_quat_global = R.from_euler('xyz', np.deg2rad([0, 60, 0])).as_quat().astype(np.float32)
            print(f"\n{'='*60}\n  执行 target_object 抓取放置 (start={grasp_start02}, end={grasp_end02})\n{'='*60}")
            grasp_and_place_02(env=env, tube_idx=grasp_start02 if 0<=grasp_start02<=4 else None,
                robot_idx=0, other_robot_idx=1, robot=left_robot, other_robot=left_other,
                start_idx=grasp_start02, end_idx=grasp_end02,
                right_handover_quat=right_grasp_quat_global,
                recorder=recorder, record_step=record_step, repeat_id=repeat_id)
            print(f"\n{'='*60}\n✅ target_object 抓取放置完成！\n{'='*60}")
        if grasp_start02 is not None and grasp_end02 is not None:
            print(f"\n{'='*60}\n✅ 双臂夹取放置任务全部完成！\n{'='*60}")
    else:
        # ========== 旧参数模式（向后兼容） ==========
        if tube_idx is not None:
            # ========== 获取试管信息 ==========
            print(f"\n获取试管{tube_idx}信息...")
            tube_pos = get_tube_position(env, tube_idx)
            if tube_pos is not None:
                print(f"  试管{tube_idx} 位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}, {tube_pos[2]:.3f}]")
            else:
                print(f"  ❌ 未找到试管{tube_idx}")
                if hasattr(env.unwrapped, 'robocasa_objects'):
                    print(f"  robocasa_objects 数量: {len(env.unwrapped.robocasa_objects)}")
                    for i, obj in enumerate(env.unwrapped.robocasa_objects):
                        print(f"    [{i}] {obj.name}")
            print(f"\n========== 试管{tube_idx} 初始位姿 ==========")
            print_tube_pose(env, tube_idx, prefix="  ")
            print(f"========================================")
            _, itq = get_tube_pose(env, tube_idx)
            if itq is not None: print(f"\n将保持试管初始姿态进行交接")
            else: itq = lift_tube_quat
            if target_rot_deg is not None:
                rot_rad = [np.deg2rad(r) for r in target_rot_deg]
                r = R.from_euler('xyz', rot_rad)
                target_quat = r.as_quat()
                print(f"\n目标姿态 (欧拉角 {target_rot_deg}° -> 四元数): [{target_quat[0]:.4f}, {target_quat[1]:.4f}, {target_quat[2]:.4f}, {target_quat[3]:.4f}]")
            print(f"\n{'='*60}\n执行任务: 抓取试管{tube_idx} 并放到目标位置\n{'='*60}")
            determined_robot_idx = 0
            print(f"  任务要求: 由左臂（0号机械臂）抓取试管{tube_idx}到交接位置")
            z_axis = np.array([0, -1, 0]); x_axis = np.array([0, 0, -1])
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])
            r_curr = R.from_matrix(rot_mat); r_new = r_curr
            handover_quat = r_new.as_quat()
            print(f"\n交接姿态（已补偿试管pitch角）")
            success, robot_idx, initial_ee_pos, initial_ee_quat, initial_qpos = pass_to(
                env=env, tube_idx=tube_idx, robot_idx=0,
                target_pos=target_pos, target_quat=handover_quat,
                record_step=record_step, recorder=recorder, grasp_rot_deg=grasp_rot_deg)
            if tube_idx is not None:
                tube_place_pos, tube_place_quat = get_tube_pose(env, tube_idx)
            else: tube_place_pos = None
            if tube_place_pos is not None:
                # 修正格式转换（SAPIEN [w,x,y,z] → SciPy [x,y,z,w]）
                tube_rot = R.from_quat([tube_place_quat[1], tube_place_quat[2], tube_place_quat[3], tube_place_quat[0]])
                tube_z_world = tube_rot.apply([0, 0, 1])
                
                world_z = np.array([0, 0, 1])
                cos_angle = np.dot(tube_z_world, world_z)
                angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                
                print(f"  试管z轴朝向: {tube_z_world}")
                print(f"  与世界z轴的夹角: {angle_deg:.2f}°")
                
                # 用 align_vectors 自动计算最优旋转
                R_map, _ = R.align_vectors([world_z], [tube_z_world])
                
                # 获取当前夹爪姿态
                ee_pose = env.unwrapped.agent.agents[0].tcp_pose
                ee_q = ee_pose.q
                if len(ee_q.shape) == 2:
                    ee_q = ee_q[0]
                gripper_q = np.array(ee_q.cpu().numpy() if hasattr(ee_q, 'cpu') else ee_q, dtype=np.float32)
                gripper_rot = R.from_quat([gripper_q[1], gripper_q[2], gripper_q[3], gripper_q[0]])
                
                # 新夹爪姿态
                gripper_rot_new = R_map * gripper_rot
                adjusted_quat_scipy = gripper_rot_new.as_quat()
                adjusted_quat = np.array([adjusted_quat_scipy[3], adjusted_quat_scipy[0], adjusted_quat_scipy[1], adjusted_quat_scipy[2]])
                
                print(f"    调整后四元数: [{adjusted_quat[0]:.4f}, {adjusted_quat[1]:.4f}, {adjusted_quat[2]:.4f}, {adjusted_quat[3]:.4f}]")
                print(f"\n  切换至补偿后的矩阵基座姿态: {adjusted_quat}")
                ee_pose = env.unwrapped.agent.agents[0].tcp_pose
                p = ee_pose.p
                if len(p.shape) == 2: p = p[0]
                curr_pos = np.array(p.cpu().numpy() if hasattr(p, 'cpu') else p, dtype=np.float32)
                move_to_position_only(env, robot_idx=0, target_pos=curr_pos, target_quat=adjusted_quat,
                    steps=60, record_step=record_step, gripper_open=False,
                    allow_adaptive_search=True, limit_search_range=True)
                handover_quat = adjusted_quat
            else: handover_quat = None
            if success:
                rp_s, right_initial_pos, right_initial_quat, right_initial_qpos, right_handover_quat = right_pass(
                    env=env, tube_idx=tube_idx, robot_idx=robot_idx,
                    initial_ee_pos=initial_ee_pos, initial_ee_quat=initial_ee_quat,
                    initial_qpos=initial_qpos, recorder=recorder, record_step=record_step)
                if not rp_s:
                    print("  ❌ 右臂交接失败")
                    if calibrate: shutil.rmtree(temp_image_dir, ignore_errors=True)
                    env.close(); return
                left_arm_idx = robot_idx; right_arm_idx = 1 - robot_idx
                agent = env.unwrapped.agent
                right_robot = agent.agents[right_arm_idx]; left_robot = agent.agents[left_arm_idx]
                if right_place is not None:
                    right_place_procedure(env=env, tube_idx=tube_idx,
                        right_arm_idx=right_arm_idx, left_arm_idx=left_arm_idx,
                        right_robot=right_robot, left_robot=left_robot,
                        right_handover_quat=right_handover_quat,
                        right_initial_pos=right_initial_pos, right_initial_quat=right_initial_quat,
                        right_initial_qpos=right_initial_qpos, right_place_idx=right_place,
                        recorder=recorder, record_step=record_step)
                else:
                    print(f"\n  步骤7: 右臂返回初始位置...")
                    recorder.set_subtask("right arm retracts")
                    return_arm_to_initial_pose(env=env, robot_idx=right_arm_idx,
                        target_pos=right_initial_pos, target_quat=right_initial_quat,
                        initial_qpos=right_initial_qpos, record_step=record_step)
        # 确定 grasp_and_place 使用的机械臂
        if grasp_arm is not None:
            grasp_robot_idx = grasp_arm
            grasp_other_idx = 1 - grasp_arm
            grasp_robot = agent.agents[grasp_robot_idx]
            grasp_other = agent.agents[grasp_other_idx]
        else:
            grasp_robot_idx = 1
            grasp_other_idx = 0
            grasp_robot = agent.agents[1]
            grasp_other = agent.agents[0]
        # ========== 步骤8: grasp_and_place（如果指定了参数） ==========
        if grasp_start is not None and grasp_end is not None:
            robot_base = grasp_robot.robot.pose.p
            if hasattr(robot_base, 'cpu'):
                robot_base = robot_base.cpu().numpy()
            robot_base = np.array(robot_base).flatten()
            if 0 <= grasp_start <= 4:
                tube_pos, _ = get_tube_pose(env, grasp_start)
                if tube_pos is not None:
                    dx = tube_pos[0] - robot_base[0]
                    dy = tube_pos[1] - robot_base[1]
                    yaw_angle = np.rad2deg(np.arctan2(dy, dx))
                    print(f"  机械臂{grasp_robot_idx}基座位置: [{robot_base[0]:.3f}, {robot_base[1]:.3f}]")
                    print(f"  试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}]")
                    print(f"  yaw角补偿: {yaw_angle:.1f}°")
                else:
                    yaw_angle = 0
            else:
                yaw_angle = 0
            grasp_rot_deg = [0, 60, 0]
            right_grasp_quat = R.from_euler('xyz', np.deg2rad(grasp_rot_deg)).as_quat().astype(np.float32)
            print(f"\n{'=' * 60}")
            print(f"  步骤8: 右臂执行 grasp_and_place (start={grasp_start}, end={grasp_end})")
            print(f"{'=' * 60}")
            grasp_and_place(
                env=env, tube_idx=grasp_start if 0 <= grasp_start <= 4 else None,
                robot_idx=1, other_robot_idx=0,
                robot=agent.agents[1], other_robot=agent.agents[0],
                start_idx=grasp_start, end_idx=grasp_end,
                right_handover_quat=right_grasp_quat,
                recorder=recorder, record_step=record_step,
            )
            print(f"  → 执行上升返回动作...")
            lift_and_return(env=env, robot_idx=1, lift_height=0.15, steps=100, record_step=record_step)
            print(f"\n{'=' * 60}")
            print(f"✅ 右臂夹取放置完成！")
            print(f"{'=' * 60}")  
        # ========== fake_place: 右臂经过试管上方再返回 ==========
        if fake_place_idx is not None:
            print(f"\n{'=' * 60}")
            print(f"  fake_place: 右臂经过试管{fake_place_idx}上方再返回")
            print(f"{'=' * 60}")
            fake_place(
                env=env, tube_idx=fake_place_idx,
                robot=agent.agents[1],robot_idx=1, other_robot_idx=0,
                other_robot=agent.agents[0],
                record_step=record_step,
            )  
        if do_fake_pass:
            print(f"\n{'=' * 60}")
            print(f"  fake_pass: 双臂空交接假动作")
            print(f"{'=' * 60}")
            success, _, _, _, _, _, _, _, _ = fake_pass(
                env=env,
                robot_idx=0,  # 左臂固定为0
                record_step=record_step,
                recorder=recorder,
            )
            if not success:
                print(f"  ❌ fake_pass 空交接失败")
        # ========== 步骤9: 左臂 grasp_and_place_02（如果指定了参数） ==========
        if grasp_start02 is not None and grasp_end02 is not None:
            left_robot = agent.agents[0]
            left_other = agent.agents[1]
            robot_base = left_robot.robot.pose.p
            if hasattr(robot_base, 'cpu'):
                robot_base = robot_base.cpu().numpy()
            robot_base = np.array(robot_base).flatten()
            if 0 <= grasp_start02 <= 4:
                tube_pos, _ = get_tube_pose(env, grasp_start02)
                if tube_pos is not None:
                    dx = tube_pos[0] - robot_base[0]
                    dy = tube_pos[1] - robot_base[1]
                    yaw_angle = np.rad2deg(np.arctan2(dy, dx))
                    print(f"  左臂基座位置: [{robot_base[0]:.3f}, {robot_base[1]:.3f}]")
                    print(f"  试管位置: [{tube_pos[0]:.3f}, {tube_pos[1]:.3f}]")
                    print(f"  yaw角补偿: {yaw_angle:.1f}°")
            print(f"\n{'=' * 60}")
            print(f"  步骤9: 左臂执行 grasp_and_place_02 (start={grasp_start02}, end={grasp_end02})")
            print(f"{'=' * 60}")
            grasp_and_place_02(
                env=env, tube_idx=grasp_start02 if 0 <= grasp_start02 <= 4 else None,
                robot_idx=0, other_robot_idx=1,
                robot=left_robot, other_robot=left_other,
                start_idx=grasp_start02, end_idx=grasp_end02,
                recorder=recorder, record_step=record_step,
                repeat_id=repeat_id,
            )
            print(f"\n{'=' * 60}")
            print(f"✅ 左臂夹取放置完成！")
            print(f"{'=' * 60}")
        if grasp_start is not None and grasp_end is not None or grasp_start02 is not None and grasp_end02 is not None:
            print(f"\n{'=' * 60}")
            print(f"✅ 双臂夹取放置任务全部完成！")
            print(f"{'=' * 60}")

    # ========== 放置位置校验（仅 calibrate=True 且 grasp_start02/grasp_end02 有效时启用） ==========
    placement_ok = True
    if calibrate and grasp_start02 is not None and grasp_end02 is not None:
        tube_obj = get_tube_object(env, grasp_start02)
        if tube_obj is not None:
            tube_current = tube_obj.pose.p
            if len(tube_current.shape) == 2:
                tube_current = tube_current[0]
            tube_current = np.array(tube_current.cpu().numpy() if hasattr(tube_current, 'cpu') else tube_current, dtype=np.float32)

            # 解析 grasp_end02 对应的目标位置
            if 5 <= grasp_end02 <= 7:
                pos_idx = grasp_end02 - 5
                target_pos = np.array(cube_place02_pos[pos_idx], dtype=np.float32)
            elif 8 <= grasp_end02 <= 10:
                pos_idx = grasp_end02 - 8
                target_pos = np.array(cube_place03_pos[pos_idx], dtype=np.float32)
            else:
                target_pos = None
                print(f"  ⚠️ grasp_end02={grasp_end02} 超出有效范围 (5-10)，跳过校验")

            if target_pos is not None:
                distance = np.linalg.norm(tube_current - target_pos)
                print(f"  放置位置校验: 试管{grasp_start02} 距离目标位置 {distance:.3f}m (阈值 0.05m)")
                print(f"    试管当前位置: [{tube_current[0]:.4f}, {tube_current[1]:.4f}, {tube_current[2]:.4f}]")
                print(f"    目标位置:     [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                if distance > 0.15:
                    print(f"  ❌ 放置偏差: {distance:.3f}m > 0.15m")
                    placement_ok = False
                else:
                    print(f"  ✅ 放置成功: {distance:.3f}m <= 0.05m")
        else:
            print(f"  ⚠️ 无法获取试管{grasp_start02}对象，跳过校验")

    # ========== 重试机制（仅 calibrate=True 时启用） ==========
    MAX_RETRIES = 3
    if calibrate and not placement_ok:
        if retry_attempt < MAX_RETRIES - 1:
            print(f"\n{'=' * 60}")
            print(f"放置偏差 > 0.05m，准备重试 (第{retry_attempt+1}次/共{MAX_RETRIES}次)")
            print(f"{'=' * 60}")
            # 清理已保存的临时文件
            import shutil
            hdf5_path = os.path.join(hdf5_dir, f"{episode_name}.hdf5")
            csv_path = os.path.join(csv_dir, f"{episode_name}.csv")
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(temp_image_dir):
                shutil.rmtree(temp_image_dir)
            for sub in ["front", "left", "right"]:
                video_path = os.path.join(videos_base, sub, f"{episode_name}_{sub}.mp4")
                if os.path.exists(video_path):
                    os.remove(video_path)
            env.close()
            return False
        else:
            print(f"\n{'=' * 60}")
            print(f"已超过最大重试次数 ({MAX_RETRIES}次)，完全删除数据")
            print(f"{'=' * 60}")
            import shutil
            hdf5_path = os.path.join(hdf5_dir, f"{episode_name}.hdf5")
            csv_path = os.path.join(csv_dir, f"{episode_name}.csv")
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(temp_image_dir):
                shutil.rmtree(temp_image_dir)
            for sub in ["front", "left", "right"]:
                video_path = os.path.join(videos_base, sub, f"{episode_name}_{sub}.mp4")
                if os.path.exists(video_path):
                    os.remove(video_path)
            env.close()
            return False

    # ========== 录制结束状态 ==========
    print("\n录制结束状态...")
    # recorder.set_subtask("idle")

    agent = env.unwrapped.agent
    for i in range(20):
        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        for robot_idx, robot in enumerate(agent.agents):
            robot_uid = f"piper_x-{robot_idx}"
            if robot_uid in action:
                current_qpos = robot.robot.get_qpos()[0].cpu().numpy()
                arm_qpos = current_qpos[:6]
                gripper_control = -1.0
                action[robot_uid] = np.concatenate([arm_qpos, [gripper_control]])
        record_step(action)

    print(f"\n总共录制了 {frame_counter} 帧")

    # ========== 保存数据到 HDF5 ==========
    print(f"\n{'=' * 60}")
    print("保存数据到 HDF5...")
    print(f"{'=' * 60}")

    hdf5_path = os.path.join(hdf5_dir, f"{episode_name}.hdf5")

    instruction = f""
    # if target_pos is not None:
    #     instruction += f" at [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]"

    recorder.save_to_hdf5_direct(hdf5_path, instruction=instruction)
    print(f"  HDF5 已保存: {hdf5_path}")

    # ========== 导出子任务 CSV ==========
    csv_path = os.path.join(csv_dir, f"{episode_name}.csv")
    recorder.export_subtask_csv(csv_path)
    print(f"  子任务 CSV 已保存: {csv_path}")

    # ========== 生成视频 ==========
    print("\n生成视频...")
    cam_to_subdir = {
        "human_cam": "front",
        "piper_x-0-hand_camera": "left",
        "piper_x-1-hand_camera": "right",
    }
    for cam_name in all_camera_names:
        if cam_name in recorded_data and len(recorded_data[cam_name]) > 0:
            sub_dir = cam_to_subdir.get(cam_name, "other")
            video_name = f"{episode_name}_{sub_dir}.mp4"
            video_path = os.path.join(videos_base, sub_dir, video_name)
            img_paths = recorded_data[cam_name]

            if len(img_paths) > 0:
                first_img = cv2.imread(img_paths[0])
                if first_img is not None:
                    height, width = first_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

                    for img_path in img_paths:
                        img = cv2.imread(img_path)
                        if img is not None:
                            video_writer.write(img)

                    video_writer.release()
                    print(f"  视频已保存: {video_path} ({len(img_paths)} 帧)")

    print(f"\n所有数据已保存到: {base_dir}")
    print("完成!")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene_L5_02 试管抓取放置数据采集")
    parser.add_argument("--tube-idx", type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help="试管索引 (0: sample_tube_1, 1: sample_tube_2, 2: sample_tube_3)")
    parser.add_argument("--target-pos", type=float, nargs=3,
                        default=[-0.4,0, 0.25],
                        help="目标位置 x y z")
    parser.add_argument("--target-rot", type=float, nargs=3,
                        default=[0, 0, -60],
                        help="目标姿态欧拉角 roll pitch yaw (度)")
    parser.add_argument("--grasp-rot", type=float, nargs=3,
                        default=[0, 60, 0],
                        help="夹取姿态欧拉角 roll pitch yaw (度)，不传则使用默认 Ry90°")
    parser.add_argument("--calibrate", action="store_true",
                        help="开启自动微调模式：放置后检测精度，自动微调 target-pos y 值重试")
    parser.add_argument("--output-name", type=str, default=None,
                        help="输出名称，用于命名保存目录")
    parser.add_argument("--tube-pos-idx", type=int, default=None,
                        help="预定义试管位姿组索引 (0-5)，对应 PREDEFINED_TUBE_POSITIONS")
    parser.add_argument("--target-pos-idx", type=int, default=None,
                        help="预定义目标位置索引 (0-5)，对应 PREDEFINED_TARGET_POSITIONS")
    parser.add_argument("--task-idx", type=int, default=None,
                        help="任务序列索引，从 TASK_SEQUENCES 中读取配置")
    parser.add_argument("--task-idx-start", type=int, default=None,
                        help="任务序列起始索引（批量运行）")
    parser.add_argument("--task-idx-end", type=int, default=None,
                        help="任务序列结束索引，包含（批量运行）")
    # ===== 新 5 参数系统（layout_id, pass_id, place_01, grasp_id, mode） =====
    parser.add_argument("--layout-id", type=int, default=None, choices=list(range(9)),
                        help="布局索引 (0-8)，对应 PREDEFINED_TUBE_POSITIONS 的 9 组")
    parser.add_argument("--place-01", type=int, default=None, choices=[0, 1, 2],
                        help="右臂放置目标位置索引 (0-2)，对应 TUBE_PLACE_TARGETS_3")
    parser.add_argument("--right-place", type=int, default=None, choices=[0, 1, 2, 3],
                        help="右臂放置目标索引 (0-3)，对应 cube_place02_pos 的四组坐标")
    parser.add_argument("--grasp-start", type=int, default=None, choices=range(0, 5),
                        help="grasp_and_place 起始位置索引 (0-4): 直接读取试管位姿")
    parser.add_argument("--grasp-end", type=int, default=None, choices=range(5, 11),
                        help="grasp_and_place 目标位置索引 (5-7: cube_place02, 8-10: cube_place03)")
    parser.add_argument("--grasp-arm", type=int, default=None, choices=[0, 1],
                        help="grasp_and_place 使用的机械臂索引 (0=左臂, 1=右臂)")
    parser.add_argument("--grasp-start02", type=int, default=None, choices=range(0, 5),
                        help="左臂 grasp_and_place_02 起始位置索引 (0-4): 直接读取试管位姿")
    parser.add_argument("--grasp-end02", type=int, default=None, choices=range(5, 11),
                        help="左臂 grasp_and_place_02 目标位置索引 (5-7: cube_place02, 8-10: cube_place03)")
    parser.add_argument("--fake-place", type=int, default=None, choices=range(0, 5),
                        help="右臂在试管上方经过再返回 (0-4: 试管序号)")
    parser.add_argument("--fake-pass", action="store_true",
                        help="开启空交接模式（不夹试管，直接到交接点）")
    parser.add_argument("--grasp-id", type=int, default=None,
                        help="左臂抓取放置的试管编号（在 layout 内的索引）")
    parser.add_argument("--mode", type=int, default=None, choices=[0, 1],
                        help="0=放回原位(左臂), 1=放到shelf(就近臂)")
    parser.add_argument("--task-240-idx", type=int, default=None,
                        help="TASK_SEQUENCES_240 索引（新 5 参数系统）")
    parser.add_argument("--task-240-start", type=int, default=None,
                        help="TASK_SEQUENCES_240 起始索引（批量运行）")
    parser.add_argument("--task-240-end", type=int, default=None,
                        help="TASK_SEQUENCES_240 结束索引（批量运行）")

    args = parser.parse_args()

    def run_single_task(task_idx):
        """执行单个任务（旧 TASK_SEQUENCES 系统）"""
        if not (0 <= task_idx < len(TASK_SEQUENCES)):
            print(f"⚠️ task-idx {task_idx} 超出范围 (0-{len(TASK_SEQUENCES)-1})")
            return
        task = TASK_SEQUENCES[task_idx]
        tube_pos_idx = task['layout']
        tube_idx = task['target_object'] - 1
        repeat_id = task['repeat_id']-1
        sequence = task['sequence']
        # ========== 从 sequence 中解析参数 ==========
        right_place = None
        grasp_start = None
        grasp_end = None
        grasp_arm = None
        grasp_start02 = task['target_object'] - 1  # 左臂抓取目标试管
        grasp_end02 =  5 + 4  # 默认放到 D3，或根据需求调整
        fake_place_idx = None
        do_fake_pass = False
        for action in sequence:
            if action['type'] == 'place':
                # right_place: D1=5, D2=6, D3=7
                loc = action['location']
                loc_num = int(loc[1]) - 1
                right_place = loc_num  # D1=0, D2=1, D3=2
            elif action['type'] == 'distractor_place':
                # 干扰物放置: 从 layout 中取试管编号，放至 DD1/DD2/DD3
                obj = action['object']
                obj_num = int(obj[1]) - 1
                loc = action['location']
                loc_num = int(loc[2]) - 1
                grasp_start = obj_num
                grasp_end = 5 + loc_num  # DD1=8, DD2=9, DD3=10
                grasp_arm = 1  # 右臂
            elif action['type'] == 'pass_over':
                obj = action['object']
                obj_num = int(obj[1]) - 1
                fake_place_idx = obj_num
            elif action['type'] == 'fake_handoff':
                do_fake_pass = True
        print(f"\n{'=' * 60}")
        print(f"执行 TASK_SEQUENCES[{task_idx}]")
        print(f"  layout={tube_pos_idx}, target_object={task['target_object']} → tube_idx={tube_idx}")
        print(f"  repeat_id={repeat_id}, 序列步骤: {len(sequence)} 步")
        if right_place is not None:
            print(f"  右臂放置位置: idx={right_place}")
        print(f"{'=' * 60}\n")
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                print(f"\n--- 第 {attempt+1}/{MAX_RETRIES} 次尝试 ---")
            success = main(
                tube_idx=tube_idx,
                target_pos=np.array(args.target_pos),
                target_rot_deg=np.array(args.target_rot) if args.target_rot is not None else None,
                output_name=args.output_name,
                grasp_rot_deg=np.array(args.grasp_rot) if args.grasp_rot is not None else None,
                calibrate=args.calibrate,
                tube_pos_idx=tube_pos_idx,
                sequence=sequence,
                repeat_id=repeat_id,
                task_idx=task_idx,
                right_place=right_place,
                grasp_start=grasp_start,
                grasp_end=grasp_end,
                grasp_arm=grasp_arm,
                grasp_start02=grasp_start02,
                grasp_end02=grasp_end02,
                fake_place_idx=fake_place_idx,
                do_fake_pass=do_fake_pass,
                retry_attempt=attempt,
            )
            if success is not False:
                break
            if attempt == MAX_RETRIES - 1:
                print(f"  ✖ 任务 {task_idx} 超过 {MAX_RETRIES} 次重试，已跳过")

    def run_single_task_240(task_240_idx):
        """执行单个任务（使用 TASK_SEQUENCES 字典数据，通过 --task-240-* 参数调用）"""
        if not (0 <= task_240_idx < len(TASK_SEQUENCES)):
            print(f"⚠️ task-240-idx {task_240_idx} 超出范围 (0-{len(TASK_SEQUENCES)-1})")
            return
        task = TASK_SEQUENCES[task_240_idx]
        tube_pos_idx = task['layout']
        tube_idx = task['target_object'] - 1
        repeat_id_val = task['repeat_id'] - 1  # 减1适配数组索引
        sequence = task['sequence']
        print(f"\n{'=' * 60}")
        print(f"执行 TASK_SEQUENCES[{task_240_idx}] (通过 --task-240-*)")
        print(f"  layout={task['layout']}, target_object={task['target_object']} → tube_idx={tube_idx}")
        print(f"  repeat_id={repeat_id_val}, 序列步骤: {len(sequence)} 步")
        print(f"{'=' * 60}\n")
        # ========== 从 sequence 中解析参数 ==========
        right_place = None
        grasp_start = None
        grasp_end = None
        grasp_arm = None
        grasp_start02 = task['target_object'] - 1  # 左臂抓取目标试管
        grasp_end02 = 5 + 4  # 默认放到 D3
        fake_place_idx = None
        do_fake_pass = False
        for action in sequence:
            if action['type'] == 'place':
                loc = action['location']
                loc_num = int(loc[1]) - 1
                right_place = loc_num  # D1=0, D2=1, D3=2
            elif action['type'] == 'distractor_place':
                obj = action['object']
                obj_num = int(obj[1]) - 1
                loc = action['location']
                loc_num = int(loc[2]) - 1
                grasp_start = obj_num
                grasp_end = 5 + loc_num  # DD1=8, DD2=9, DD3=10
                grasp_arm = 1  # 右臂
            elif action['type'] == 'pass_over':
                obj = action['object']
                obj_num = int(obj[1]) - 1
                fake_place_idx = obj_num
            elif action['type'] == 'fake_handoff':
                do_fake_pass = True
        MAX_RETRIES = 4
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                print(f"\n--- 第 {attempt+1}/{MAX_RETRIES} 次尝试 ---")
            success = main(
                tube_idx=tube_idx,
                target_pos=np.array(args.target_pos),
                target_rot_deg=np.array(args.target_rot) if args.target_rot is not None else None,
                output_name=args.output_name,
                grasp_rot_deg=np.array(args.grasp_rot) if args.grasp_rot is not None else None,
                calibrate=args.calibrate,
                tube_pos_idx=tube_pos_idx,
                sequence=sequence,
                repeat_id=repeat_id_val,
                task_idx=task_240_idx,
                right_place=right_place,
                grasp_start=grasp_start,
                grasp_end=grasp_end,
                grasp_arm=grasp_arm,
                grasp_start02=grasp_start02,
                grasp_end02=grasp_end02,
                fake_place_idx=fake_place_idx,
                do_fake_pass=do_fake_pass,
                retry_attempt=attempt,
            )
            if success is not False:
                break
            if attempt == MAX_RETRIES - 1:
                print(f"  ✖ 任务 {task_240_idx} 超过 {MAX_RETRIES} 次重试，已跳过")

    # ===== 新 --task-240-* 批量运行（使用 TASK_SEQUENCES 数据） =====
    if args.task_240_start is not None and args.task_240_end is not None:
        if not TASK_SEQUENCES:
            print("⚠️ TASK_SEQUENCES 为空")
        else:
            total = len(TASK_SEQUENCES)
            start = max(0, args.task_240_start)
            end = min(total - 1, args.task_240_end)
            print(f"\n批量运行 TASK_SEQUENCES 任务 {start} ~ {end}，共 {end - start + 1} 个")
            for idx in range(start, end + 1):
                print(f"\n{'#' * 60}")
                print(f"#  任务 {idx} / {end}")
                print(f"{'#' * 60}")
                run_single_task_240(idx)
    elif args.task_240_idx is not None:
        if not TASK_SEQUENCES:
            print("⚠️ TASK_SEQUENCES 为空")
        else:
            run_single_task_240(args.task_240_idx)
    # ===== 旧系统批量运行 =====
    elif args.task_idx_start is not None and args.task_idx_end is not None:
        total = len(TASK_SEQUENCES)
        start = max(0, args.task_idx_start)
        end = min(total - 1, args.task_idx_end)
        print(f"\n批量运行任务 {start} ~ {end}，共 {end - start + 1} 个")
        for idx in range(start, end + 1):
            print(f"\n{'#' * 60}")
            print(f"#  任务 {idx} / {end}")
            print(f"{'#' * 60}")
            run_single_task(idx)
    elif args.task_idx is not None:
        run_single_task(args.task_idx)
    elif args.layout_id is not None and args.pass_id is not None and args.place_01 is not None and args.grasp_id is not None and args.mode is not None:
        # ===== 新 5 参数系统，直接执行 =====
        layout_id, pass_id, place_01, grasp_id, mode = (
            args.layout_id, args.pass_id, args.place_01, args.grasp_id, args.mode
        )
        final_target_pos = np.array(args.target_pos)
        print(f"\n{'=' * 60}")
        print(f"新 5 参数系统直接执行")
        print(f"  layout_id={layout_id}, pass_id={pass_id}, place_01={place_01}")
        print(f"  grasp_id={grasp_id}, mode={mode}")
        print(f"{'=' * 60}\n")

        MAX_RETRIES = 4
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                print(f"\n--- 第 {attempt+1}/{MAX_RETRIES} 次尝试 ---")
            success = main(
                output_name=args.output_name,
                calibrate=args.calibrate,
                grasp_rot_deg=np.array(args.grasp_rot) if args.grasp_rot is not None else None,
                layout_id=layout_id,
                pass_id=pass_id,
                place_01=place_01,
                grasp_id=grasp_id,
                mode=mode,
                retry_attempt=attempt,
                tube_idx=args.tube_idx,
                target_pos=final_target_pos,
                target_rot_deg=np.array(args.target_rot) if args.target_rot is not None else None,
                tube_pos_idx=args.tube_pos_idx,
                fake_place_idx=args.fake_place,
                do_fake_pass=args.fake_pass,

            )
            if success is not False:
                break
            if attempt == MAX_RETRIES - 1:
                print(f"  ✖ 超过 {MAX_RETRIES} 次重试，已跳过")
    else:
        # 原始逻辑：由 --target-pos / --target-pos-idx 决定目标位置
        print("\u26a0\ufe0f 未指定有效参数，使用默认值执行...")
        final_target_pos = np.array(args.target_pos)
        main(
            tube_idx=args.tube_idx,
            target_pos=final_target_pos,
            target_rot_deg=np.array(args.target_rot) if args.target_rot is not None else None,
            output_name=args.output_name,
            grasp_rot_deg=np.array(args.grasp_rot) if args.grasp_rot is not None else None,
            calibrate=args.calibrate,
            right_place=args.right_place,
            grasp_start=args.grasp_start,
            grasp_end=args.grasp_end,
            grasp_arm=args.grasp_arm,
            grasp_start02=args.grasp_start02,
            grasp_end02=args.grasp_end02,
            fake_place_idx=args.fake_place,
            do_fake_pass=args.fake_pass,
        )
