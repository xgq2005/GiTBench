"""
运动规划通用工具函数
供所有场景（biolab、household 等）的 run_scene 脚本共用

用法:
    from mani_skill.envs.tasks.custom_task.motion_planning_utils import (
        find_reachable_pose,
        combine_quaternions,
        move_to_position_only,
        compute_grasp_pose,
    )
"""

import numpy as np
import sapien
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class TrajectoryStepsExceededError(Exception):
    """轨迹步数超过限制时抛出的异常"""
    def __init__(self, n_step, max_steps):
        self.n_step = n_step
        self.max_steps = max_steps
        super().__init__(f"轨迹步数({n_step})超过限制({max_steps})")


def combine_quaternions(q1, q2):
    """组合两个四元数（q2 * q1）"""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    combined = r2 * r1
    return combined.as_quat()


def slerp_quat(q1, q2, t):
    """四元数球面线性插值（SLERP），兼容所有 SciPy 版本

    Args:
        q1: 起始四元数 [x, y, z, w] (scipy 格式)
        q2: 目标四元数 [x, y, z, w] (scipy 格式)
        t: 插值参数 (0~1)

    Returns:
        插值后的四元数 [x, y, z, w]
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    # 计算点积
    dot = np.dot(q1, q2)

    # 如果点积为负，反转 q2 以确保最短路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # 防止除零
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 角度很小，使用线性插值
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)  # 两四元数间的角度
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q1 + s1 * q2
    return result / np.linalg.norm(result)


def find_reachable_pose(
    env, robot_idx, planner, current_qpos,
    target_pos_np, target_q_np,
    max_attempts=20,
    allow_adaptive_search=True,
    limit_search_range=False,
    max_search_time=5.0,
    current_q_np=None,
):
    """
    自适应搜索可达的姿态

    Args:
        env: 环境
        robot_idx: 机械臂索引
        planner: 运动规划器
        current_qpos: 当前关节位置
        target_pos_np: 目标位置（numpy数组）
        target_q_np: 目标姿态四元数（numpy数组）
        max_attempts: 最大尝试次数（保留参数，兼容旧接口）
        allow_adaptive_search: 是否允许自适应姿态搜索
        limit_search_range: 是否限制搜索范围（±90度以内）
        max_search_time: 最大搜索时间（秒）
        current_q_np: 当前末端四元数（用于SLERP插值，可选）

    Returns:
        (success, final_pose_array)
    """
    search_radius = 0.03  # 改为0.03m，用户要求球面采样距离<0.03m
    start_time = time.time()

    def check_timeout():
        return time.time() - start_time > max_search_time

    # 策略1：直接尝试目标位置
    target_pose_array = np.concatenate([target_pos_np, target_q_np])
    try:
        if check_timeout():
            print(f"    搜索超时（{max_search_time}秒），停止搜索")
            return False, None
        result = planner.plan_screw(
            target_pose_array,
            current_qpos,
            time_step=env.unwrapped.control_timestep,
            use_point_cloud=False,
        )
        if result["status"] == "Success":
            print(f"    直接规划成功")
            return True, target_pose_array
    except Exception as e:
        print(f"    直接规划失败: {e}")

    if not allow_adaptive_search:
        print(f"    不允许自适应搜索，直接返回失败")
        return False, None

    # 策略2：SLERP 姿态插值（优先保持位置不变，逐步过渡姿态）
    if limit_search_range and current_q_np is not None:
        print(f"    开始受限姿态搜索（半径={search_radius}米，姿态调整范围±90度）...")
        print(f"    尝试姿态插值路径（位置固定，逐步过渡姿态）...")
        num_interpolation_steps = 10
        best_i = -1
        best_pose_array = None
        for i in range(1, num_interpolation_steps + 1):
            t = i / num_interpolation_steps
            interpolated_quat = slerp_quat(current_q_np, target_q_np, t)
            candidate_pose_array = np.concatenate([target_pos_np, interpolated_quat])
            try:
                result = planner.plan_screw(
                    candidate_pose_array,
                    current_qpos,
                    time_step=env.unwrapped.control_timestep,
                    use_point_cloud=False,
                )
                if result["status"] == "Success":
                    best_i = i
                    best_pose_array = candidate_pose_array
            except Exception:
                continue
        if best_i >= 0:
            dot_q = np.abs(np.dot(current_q_np, target_q_np))
            dot_q = np.clip(dot_q, -1.0, 1.0)
            angle_diff = 2.0 * np.arccos(dot_q)
            achieved_t = best_i / num_interpolation_steps
            print(f"    ✅ 找到可达姿态（插值步骤{best_i}/{num_interpolation_steps}，已达成{achieved_t*100:.0f}%目标姿态，全程角度差={angle_diff:.2f}弧度）")
            return True, best_pose_array
        print(f"    姿态插值路径失败，继续球面采样...")
    else:
        print(f"    开始自适应姿态搜索（半径={search_radius}米）...")

    # ========== 姿态候选集合：优先 roll/pitch，yaw 保持灵活 ==========
    # roll/pitch 有更多候选角度（优先保证这两个轴准确）
    # yaw 候选角度较少（可以更灵活调整以适应 IK）
    if limit_search_range:
        roll_candidates  = [0, np.pi/6, -np.pi/6, np.pi/4, -np.pi/4, np.pi/3, -np.pi/3]
        pitch_candidates = [0, np.pi/6, -np.pi/6, np.pi/4, -np.pi/4, np.pi/3, -np.pi/3]
        yaw_candidates   = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]
    else:
        roll_candidates  = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, 3*np.pi/4, -3*np.pi/4, np.pi]
        pitch_candidates = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, 3*np.pi/4, -3*np.pi/4, np.pi]
        yaw_candidates   = [0, np.pi/2, -np.pi/2, np.pi]

    from itertools import product
    rotation_offsets = []
    for roll_angle in roll_candidates:
        for pitch_angle in pitch_candidates:
            for yaw_angle in yaw_candidates:
                rotation = R.from_euler('xyz', [roll_angle, pitch_angle, yaw_angle])
                rotation_offsets.append(rotation.as_quat())

    # 位置候选（斐波那契球面采样，小半径内密集采样）
    # 球半径=search_radius，保证位置误差<3cm
    num_samples = 32
    phi = np.pi * (3. - np.sqrt(5.))

    print(f"    球面采样（半径={search_radius:.2f}米，{num_samples}个位置 × {len(rotation_offsets)}种姿态）...")
    for i in range(num_samples):
        if check_timeout():
            print(f"    搜索超时（{max_search_time}秒），停止搜索")
            return False, None
        y = 1 - (i / float(num_samples - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        candidate_pos = target_pos_np + np.array([x, y, z]) * search_radius
        for offset_quat in rotation_offsets:
            if check_timeout():
                print(f"    搜索超时（{max_search_time}秒），停止搜索")
                return False, None
            combined_quat = combine_quaternions(np.array(target_q_np), offset_quat)
            candidate_pose_array = np.concatenate([candidate_pos, combined_quat])
            try:
                result = planner.plan_screw(
                    candidate_pose_array,
                    current_qpos,
                    time_step=env.unwrapped.control_timestep,
                    use_point_cloud=False,
                )
                if result["status"] == "Success":
                    dist = np.linalg.norm(candidate_pos - target_pos_np)
                    print(f"    ✅ 球面采样找到可达姿态（距目标{dist:.3f}米）")
                    return True, candidate_pose_array
            except Exception:
                continue

    # 扩大搜索半径（放宽位置约束）
    print(f"    扩大搜索半径到{search_radius*3:.2f}米（放宽位置约束）...")
    num_samples_expanded = 20
    for i in range(num_samples_expanded):
        if check_timeout():
            print(f"    搜索超时（{max_search_time}秒），停止搜索")
            return False, None
        y = 1 - (i / float(num_samples_expanded - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        candidate_pos = target_pos_np + np.array([x, y, z]) * search_radius * 3
        candidate_pose_array = np.concatenate([candidate_pos, target_q_np])
        try:
            result = planner.plan_screw(
                candidate_pose_array,
                current_qpos,
                time_step=env.unwrapped.control_timestep,
                use_point_cloud=False,
            )
            if result["status"] == "Success":
                dist = np.linalg.norm(candidate_pos - target_pos_np)
                print(f"    ⚠️ 扩大半径找到可达姿态（距目标{dist:.3f}米，位置精度降低）")
                return True, candidate_pose_array
        except Exception:
            continue

    print(f"    自适应搜索失败，无法找到可达姿态")
    return False, None


def move_to_position_only(
    env, robot_idx, target_pos, steps=100, record_step=None,
    gripper_open=True, target_quat=None,
    allow_adaptive_search=True, limit_search_range=False,
    max_trajectory_steps=100,
    wrist_flip=False,
):
    """
    使用 mplib 直接规划移动到目标位置，支持自适应姿态搜索

    Args:
        env: 环境
        robot_idx: 机械臂索引
        target_pos: 目标位置 [x, y, z]
        steps: 移动步数（保留参数，兼容旧接口）
        record_step: 录制函数
        gripper_open: 是否保持夹爪张开
        target_quat: 目标姿态四元数 [x, y, z, w]（可选，默认使用当前姿态）
        allow_adaptive_search: 是否允许自适应姿态搜索
        limit_search_range: 是否限制搜索范围
        max_trajectory_steps: 最大轨迹步数，超过则判定不可达

    Returns:
        是否成功
    """
    import mplib

    agent = env.unwrapped.agent
    robot = agent.agents[robot_idx]

    print(f"  目标位置: {target_pos}")

    urdf_path = robot.urdf_path if hasattr(robot, 'urdf_path') else None
    if urdf_path is None:
        from mani_skill.utils import assets
        urdf_path = assets.get_asset_path("robots/piper_x/piper_x_description/piper_x_description_d435.urdf")

    srdf_path = urdf_path.replace(".urdf", ".srdf")
    urdf_dir = str(Path(urdf_path).parent)

    link_names = [link.get_name() for link in robot.robot.get_links()]
    joint_names = [joint.get_name() for joint in robot.robot.get_active_joints()]

    planner = mplib.Planner(
        urdf=urdf_path,
        srdf=srdf_path,
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="piper_x_hand_tcp",
        search_path=urdf_dir,
    )

    base_pose = robot.robot.pose
    base_pose_array = np.hstack([base_pose.p, base_pose.q])
    planner.set_base_pose(base_pose_array.reshape(-1, 1).astype(np.float64))

    current_qpos = robot.robot.get_qpos()[0].cpu().numpy()[:6]

    if wrist_flip:
        # 第5、6关节（index 4, 5）转180°，使IK求解器偏向腕部翻转解
        current_qpos[4] += np.pi
        current_qpos[5] += np.pi
        print(f"  启用腕部翻转: 种子关节角第5、6关节+180°")

    current_ee_pose = robot.tcp_pose
    current_p = current_ee_pose.p
    current_q = current_ee_pose.q
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    current_p_np = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q_np = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)

    print(f"  当前末端位置: {current_p_np}")
    print(f"  当前末端姿态 (四元数): {current_q_np}")

    target_pos_np = np.array(target_pos, dtype=np.float32)

    if target_quat is not None:
        target_q_np = np.array(target_quat, dtype=np.float32).flatten()
        print(f"  使用指定的目标姿态: {target_q_np}")
    else:
        target_q_np = current_q_np
        print(f"  使用当前末端姿态（默认行为）")

    target_pose = sapien.Pose(p=target_pos_np, q=target_q_np)
    target_pose_array = np.concatenate([target_pose.p, target_pose.q])
    print(f"  目标姿态: 位置={target_pose.p}, 四元数={target_pose.q}")

    success, final_pose_array = find_reachable_pose(
        env, robot_idx, planner, current_qpos, target_pos_np, target_q_np,
        allow_adaptive_search=allow_adaptive_search,
        limit_search_range=limit_search_range,
        current_q_np=current_q_np,
    )

    if not success:
        print(f"  运动规划失败: 无法找到可达姿态")
        return False

    try:
        result = planner.plan_screw(
            final_pose_array,
            current_qpos,
            time_step=env.unwrapped.control_timestep,
            use_point_cloud=False,
        )
    except Exception as e:
        print(f"  运动规划失败: {e}")
        return False

    if result["status"] != "Success":
        print(f"  运动规划失败: {result['status']}")
        return False

    n_step = result["position"].shape[0]

    if n_step > max_trajectory_steps:
        print(f"  轨迹长度({n_step}步)超过{max_trajectory_steps}步，判定为目标不可达，提前结束任务")
        raise TrajectoryStepsExceededError(n_step, max_trajectory_steps)

    print(f"  运动规划成功，轨迹长度: {n_step}")

    # 轨迹下采样
    if n_step > 1000:
        sample_interval = 3
    elif n_step > 500:
        sample_interval = 2
    else:
        sample_interval = 1


    print(f"  轨迹长度: {n_step}, 采样间隔: {sample_interval}")

    executed_count = 0
    for i in range(n_step):
        qpos = result["position"][i]

        if i == 0:
            print(f"  qpos长度: {len(qpos)}, qpos内容: {qpos}")

        if i % sample_interval != 0 and i != n_step - 1:
            continue

        action = {uid: np.zeros(space.shape) for uid, space in env.action_space.items()}
        target_robot_uid = f"piper_x-{robot_idx}"
        if target_robot_uid in action:
            if len(qpos) == 6:
                arm_qpos = qpos
                gripper_pos = 1.0 if gripper_open else -1.8
                action[target_robot_uid] = np.concatenate([arm_qpos, [gripper_pos]])
            elif len(qpos) >= 7:
                arm_qpos = qpos[:6]
                gripper_pos = 1.0 if gripper_open else -1.8
                action[target_robot_uid] = np.concatenate([arm_qpos, [gripper_pos]])
            else:
                continue

                # 被动机械臂保持当前姿态，夹具闭合
        for other_idx, other_robot in enumerate(agent.agents):
            if other_idx == robot_idx:
                continue
            other_uid = f"piper_x-{other_idx}"
            if other_uid in action:
                other_qpos = other_robot.robot.get_qpos()[0].cpu().numpy()
                other_arm_qpos = other_qpos[:6]
                other_gripper = -1.0  # 保持闭合
                action[other_uid] = np.concatenate([other_arm_qpos, [other_gripper]])
        # # 运动执行完成后清空，以便下次调用重新读取
        # if executed_count == steps:
        #     env._fixed_other_qpos = None
        executed_count += 1

        if executed_count % 10 == 0:
            current_ee_pose = robot.tcp_pose
            current_p = current_ee_pose.p
            if len(current_p.shape) == 2:
                current_p = current_p[0]
            current_p_np = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
            print(f"    步骤 {i}/{n_step}（已执行{executed_count}步）: 当前位置=[{current_p_np[0]:.6f}, {current_p_np[1]:.6f}, {current_p_np[2]:.6f}]")

        if record_step:
            record_step(action)
        else:
            env.step(action)

    print(f"  运动执行完成，共执行{executed_count}步")
    return True


def compute_grasp_pose(target_pos, approaching=None, closing=None):
    """计算抓取位姿

    Args:
        target_pos: 目标位置 [x, y, z]
        approaching: 接近方向向量，默认从上方接近 [0, 0, -1]
        closing: 夹爪关闭方向向量，默认沿x轴方向 [1, 0, 0]

    Returns:
        grasp_pose: 抓取位姿 (sapien.Pose)
    """
    if approaching is None:
        approaching = np.array([0, 0, -1])
    if closing is None:
        closing = np.array([1, 0, 0])

    approaching = approaching / np.linalg.norm(approaching)
    closing = closing / np.linalg.norm(closing)

    ortho = np.cross(closing, approaching)
    ortho = ortho / np.linalg.norm(ortho)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.stack([ortho, closing, approaching], axis=1)

    rotation = R.from_matrix(rotation_matrix[:3, :3])
    quaternion = rotation.as_quat()

    return sapien.Pose(p=target_pos, q=quaternion)