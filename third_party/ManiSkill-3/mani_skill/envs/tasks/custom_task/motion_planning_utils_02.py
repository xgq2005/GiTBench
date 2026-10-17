import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.envs.tasks.custom_task.motion_planning_utils import find_reachable_pose

class DataCollector:
    """数据收集器，用于存储每个episode的数据"""
    
    def __init__(self, episode_data):
        self.episode_data = episode_data
    
    def collect_step_data(self, action, obs, info, env=None):
        """收集单步数据"""
        # 处理动作数据
        if isinstance(action, dict):
            # 双臂动作
            left_action = action.get('piper_x-0', np.zeros(8))
            right_action = action.get('piper_x-1', np.zeros(8))
            full_action = np.concatenate([left_action, right_action])
        else:
            # 单臂动作
            full_action = np.zeros(16)  # 初始化16维动作
            if len(action) == 8:
                # 假设是左臂动作
                full_action[:8] = action
            else:
                # 其他情况
                pass
        self.episode_data["actions"].append(full_action)
        
        # 处理关节状态数据 - 从obs中获取
        if isinstance(obs, dict) and 'agent' in obs:
            agent_data = obs['agent']
            # 获取左臂状态
            if 'piper_x-0' in agent_data:
                left_qpos = agent_data['piper_x-0']['qpos']
                left_qvel = agent_data['piper_x-0']['qvel']
                if hasattr(left_qpos, 'cpu'):
                    left_qpos = left_qpos.cpu().numpy().squeeze()
                    left_qvel = left_qvel.cpu().numpy().squeeze()
            else:
                left_qpos = np.zeros(8)
                left_qvel = np.zeros(8)
            
            # 获取右臂状态
            if 'piper_x-1' in agent_data:
                right_qpos = agent_data['piper_x-1']['qpos']
                right_qvel = agent_data['piper_x-1']['qvel']
                if hasattr(right_qpos, 'cpu'):
                    right_qpos = right_qpos.cpu().numpy().squeeze()
                    right_qvel = right_qvel.cpu().numpy().squeeze()
            else:
                right_qpos = np.zeros(8)
                right_qvel = np.zeros(8)
            
            # 合并为16维数据
            qpos = np.concatenate([left_qpos, right_qpos])
            qvel = np.concatenate([left_qvel, right_qvel])
            
            self.episode_data["qpos"].append(qpos)
            self.episode_data["qvel"].append(qvel)
        
        # 处理相机数据 - 从sensor_data中获取
        if isinstance(obs, dict) and 'sensor_data' in obs:
            sensor_data = obs['sensor_data']
            
            # 映射相机名称
            camera_mapping = {
                'base_camera': 'cam_high_image',
                'piper_x-0-hand_camera': 'cam_left_wrist_image',
                'piper_x-1-hand_camera': 'cam_right_wrist_image'
            }
            
            for sensor_name, cam_name in camera_mapping.items():
                if sensor_name in sensor_data:
                    cam_data = sensor_data[sensor_name]
                    
                    # 处理RGB图像
                    if 'rgb' in cam_data:
                        rgb = cam_data['rgb']
                        if hasattr(rgb, 'cpu'):
                            rgb = rgb.cpu().numpy()
                        rgb = rgb.squeeze()  # 去除多余的维度
                        self.episode_data["rgb"][cam_name].append(rgb)
                    
                    # 处理深度图像
                    if 'depth' in cam_data:
                        depth = cam_data['depth']
                        if hasattr(depth, 'cpu'):
                            depth = depth.cpu().numpy()
                        depth = depth.squeeze()  # 去除多余的维度
                        self.episode_data["depth"][cam_name].append(depth)



def _recursive_unwrap(env):
    visited = set()
    current = env
    while True:
        if id(current) in visited:
            break
        visited.add(id(current))

        if hasattr(current, "env") and getattr(current, "env") is not current:
            current = current.env
            continue
        if hasattr(current, "unwrapped") and getattr(current, "unwrapped") is not current:
            current = current.unwrapped
            continue
        break
    return current


class _ArmEnvProxy:
    def __init__(self, env, agent):
        self._env = env
        self.agent = agent

    @property
    def control_mode(self):
        return self.agent.control_mode

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        if hasattr(self._env, name):
            return getattr(self._env, name)
        base = _recursive_unwrap(self._env)
        if hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")


class PiperXArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "piper_x_hand_tcp"

    def __init__(
            self,
            env: BaseEnv,
            debug: bool = False,
            vis: bool = True,
            base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
            left_base_pose: sapien.Pose = None,
            right_base_pose: sapien.Pose = None,
            visualize_target_grasp_pose: bool = True,
            print_env_info: bool = True,
            joint_vel_limits=0.9,
            joint_acc_limits=0.9,
            arm: str = "left",
            episode_data=None,
    ):
        self.left_arm_uid = None
        self.right_arm_uid = None
        self.active_arm_name = arm.lower()
        self.active_arm = None
        self.passive_arm = None
        self.active_arm_uid = None
        self.passive_arm_uid = None
        self.use_arm_proxy = False
        self.real_env = env
        self.left_solver = None
        self.right_solver = None
        self.data_collector = DataCollector(episode_data) if episode_data is not None else None  # 数据收集器
        self.gripper_state = self.OPEN  # 确保 gripper_state 始终存在（父类 __init__ 可能在某些路径下不被调用）

        base_env = _recursive_unwrap(env)
        if self.active_arm_name not in {"left", "right", "both"}:
            raise ValueError("arm must be 'left', 'right' or 'both'")

        if hasattr(base_env, "is_bimanual") and base_env.is_bimanual and hasattr(base_env, "left_arm") and hasattr(
                base_env, "right_arm"):
            self.left_arm = base_env.left_arm
            self.right_arm = base_env.right_arm
            self._init_arm_uids(base_env)
            if self.active_arm_name in {"left", "right"}:
                self.active_arm = self.left_arm if self.active_arm_name == "left" else self.right_arm
                self.passive_arm = self.right_arm if self.active_arm_name == "left" else self.left_arm
                self.use_arm_proxy = True
                if base_pose is None:
                    if self.active_arm_name == "left" and left_base_pose is not None:
                        base_pose = left_base_pose
                    elif self.active_arm_name == "right" and right_base_pose is not None:
                        base_pose = right_base_pose
                    elif hasattr(self.active_arm, "robot") and hasattr(self.active_arm.robot, "pose"):
                        base_pose = self.active_arm.robot.pose
                proxy_env = _ArmEnvProxy(env, self.active_arm)
                super().__init__(proxy_env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info,
                                 joint_vel_limits, joint_acc_limits)
                self.active_arm_uid = self.left_arm_uid if self.active_arm_name == "left" else self.right_arm_uid
                self.passive_arm_uid = self.right_arm_uid if self.active_arm_name == "left" else self.left_arm_uid
            else:
                self.active_arm = self.left_arm
                self.passive_arm = self.right_arm
                if left_base_pose is not None:
                    left_pose = left_base_pose
                elif base_pose is not None:
                    left_pose = base_pose
                elif hasattr(self.left_arm, "robot") and hasattr(self.left_arm.robot, "pose"):
                    left_pose = self.left_arm.robot.pose
                else:
                    left_pose = None
                if right_base_pose is not None:
                    right_pose = right_base_pose
                elif base_pose is not None:
                    right_pose = base_pose
                elif hasattr(self.right_arm, "robot") and hasattr(self.right_arm.robot, "pose"):
                    right_pose = self.right_arm.robot.pose
                else:
                    right_pose = None
                left_proxy_env = _ArmEnvProxy(env, self.left_arm)
                right_proxy_env = _ArmEnvProxy(env, self.right_arm)
                super().__init__(left_proxy_env, debug, vis, left_pose, visualize_target_grasp_pose, print_env_info,
                                 joint_vel_limits, joint_acc_limits)
                self.gripper_state = self.CLOSED
                self.gripper_state = self.CLOSED
                
                # 为右臂创建第二个可视化对象
                self.right_grasp_pose_visual = None
                if self.vis and self.visualize_target_grasp_pose:
                    if "right_grasp_pose_visual" not in self.base_env.scene.actors:
                        # 创建右臂可视化对象，使用唯一名称
                        self.right_grasp_pose_visual = self._build_unique_grasp_pose_visual(
                            self.base_env.scene, "right_grasp_pose_visual"
                        )
                    else:
                        self.right_grasp_pose_visual = self.base_env.scene.actors["right_grasp_pose_visual"]
                    # 设置右臂的初始位姿
                    if hasattr(self.right_arm, 'tcp_pose'):
                        self.right_grasp_pose_visual.set_pose(self.right_arm.tcp_pose)
                
                self.left_solver = PiperXArmMotionPlanningSolver(
                    left_proxy_env,
                    debug,
                    vis,
                    left_pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                    joint_vel_limits=joint_vel_limits,
                    joint_acc_limits=joint_acc_limits,
                    arm="left",
                    episode_data=episode_data,
                )
                self.right_solver = PiperXArmMotionPlanningSolver(
                    right_proxy_env,
                    debug,
                    vis,
                    right_pose,
                    visualize_target_grasp_pose=False,
                    print_env_info=False,
                    joint_vel_limits=joint_vel_limits,
                    joint_acc_limits=joint_acc_limits,
                    arm="right",
                    episode_data=episode_data,
                )
                self.active_arm_uid = self.left_arm_uid
                self.passive_arm_uid = self.right_arm_uid
        else:
            # 非标准双机械臂环境：处理 MultiAgent 或单臂情况
            _agent = base_env.agent
            if hasattr(_agent, 'agents') and hasattr(_agent, 'agents_dict'):
                # MultiAgent 环境（如 SceneH5_02，没有 is_bimanual 标记的双臂环境）
                agents_list = _agent.agents
                agents_dict = _agent.agents_dict
                uids = list(agents_dict.keys())
                if len(uids) >= 2:
                    self.left_arm_uid = uids[0]
                    self.right_arm_uid = uids[1]
                arm_idx = 0 if self.active_arm_name == "left" else 1
                if arm_idx < len(agents_list):
                    self.active_arm = agents_list[arm_idx]
                    self.passive_arm = agents_list[1 - arm_idx]
                    self.use_arm_proxy = True
                    if base_pose is None:
                        if hasattr(self.active_arm, "robot") and hasattr(self.active_arm.robot, "pose"):
                            base_pose = self.active_arm.robot.pose
                    proxy_env = _ArmEnvProxy(env, self.active_arm)
                    super().__init__(proxy_env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info,
                                     joint_vel_limits, joint_acc_limits)
                    self.active_arm_uid = uids[arm_idx]
                    self.passive_arm_uid = uids[1 - arm_idx]
                else:
                    super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info,
                                     joint_vel_limits, joint_acc_limits)
            else:
                # 单臂环境
                super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info,
                                 joint_vel_limits, joint_acc_limits)
    
    def _build_unique_grasp_pose_visual(self, scene, name):
        """创建具有唯一名称的可视化对象"""
        import numpy as np
        import sapien
        from transforms3d import quaternions
        
        # 复制build_two_finger_gripper_grasp_pose_visual的逻辑，但支持自定义名称
        builder = scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose())
        grasp_pose_visual_width = 0.01
        grasp_width = 0.05

        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0.0]),
            radius=grasp_pose_visual_width,
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
        )

        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.08]),
            half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, -0.05]),
            half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    grasp_width + grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[
                    0.03 - grasp_pose_visual_width * 3,
                    -grasp_width - grasp_pose_visual_width,
                    0.03 - 0.05,
                ],
                q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
            ),
            half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
        )
        grasp_pose_visual = builder.build_kinematic(name=name)
        return grasp_pose_visual
    
    def _update_dual_grasp_visual(self, left_target: sapien.Pose, right_target: sapien.Pose) -> None:
        """更新双臂抓取位姿可视化"""
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(left_target)
        if self.right_grasp_pose_visual is not None:
            self.right_grasp_pose_visual.set_pose(right_target)

    def _hide_both_arms_visual(self):
        """隐藏双臂的可视化"""
        if self.vis and self.visualize_target_grasp_pose:
            # 将两个可视化对象都移动到远处隐藏
            if self.grasp_pose_visual is not None:
                self.grasp_pose_visual.set_pose(sapien.Pose([100, 100, 100]))
            if self.right_grasp_pose_visual is not None:
                self.right_grasp_pose_visual.set_pose(sapien.Pose([100, 100, 100]))
    
    def move_to_pose_with_screw(self, target: sapien.Pose, right_target: sapien.Pose = None, **kwargs):
        """双臂模式下的运动规划方法"""
        if self.active_arm_name == "both" and right_target is not None:
            # 双臂模式：更新双目标可视化
            if self.vis and self.visualize_target_grasp_pose:
                self._update_dual_grasp_visual(target, right_target)
            
            # 调用左臂运动规划
            left_result = self.left_solver.move_to_pose_with_screw(target, **kwargs)
            
            # 调用右臂运动规划
            right_result = self.right_solver.move_to_pose_with_screw(right_target, **kwargs)
            
            # 合并结果
            return {
                "success": left_result["success"] and right_result["success"],
                "position": np.concatenate([left_result["position"], right_result["position"]], axis=1),
                "velocity": np.concatenate([left_result["velocity"], right_result["velocity"]], axis=1),
                "elapsed_steps": max(left_result["elapsed_steps"], right_result["elapsed_steps"])
            }
        else:
            # 单臂模式：隐藏另一个机械臂的可视化
            if self.active_arm_name == "both" and self.vis and self.visualize_target_grasp_pose:
                # 判断是左臂运动还是右臂运动
                if hasattr(self, 'left_solver') and hasattr(self, 'right_solver'):
                    # 如果是左臂运动，隐藏右臂可视化
                    if self.right_grasp_pose_visual is not None:
                        # 将右臂可视化移动到远处隐藏
                        self.right_grasp_pose_visual.set_pose(sapien.Pose([100, 100, 100]))
                    # 更新左臂可视化到目标位置
                    if self.grasp_pose_visual is not None:
                        self.grasp_pose_visual.set_pose(target)
            
            # 调用父类方法
            return super().move_to_pose_with_screw(target, **kwargs)

    def _init_arm_uids(self, base_env):
        if hasattr(base_env, "agent") and hasattr(base_env.agent, "agents_dict"):
            uids = list(base_env.agent.agents_dict.keys())
            if len(uids) >= 2:
                self.left_arm_uid = uids[0]
                self.right_arm_uid = uids[1]

        if self.left_arm_uid is None or self.right_arm_uid is None:
            if hasattr(base_env, "robot_uids") and isinstance(base_env.robot_uids, tuple) and len(
                    base_env.robot_uids) == 2:
                self.left_arm_uid = f"{base_env.robot_uids[0]}-0"
                self.right_arm_uid = f"{base_env.robot_uids[1]}-1"

    def _build_dual_arm_action(self, active_action: np.ndarray):
        if not self.use_arm_proxy or self.active_arm is None or self.passive_arm is None:
            return active_action

        active_action = np.asarray(active_action, dtype=np.float32)
        passive_qpos = self.passive_arm.robot.get_qpos()[0].cpu().numpy()
        if passive_qpos.ndim == 0:
            passive_qpos = np.asarray([passive_qpos], dtype=np.float32)

        joint_dim = len(self.planner.joint_vel_limits)
        if active_action.shape[-1] == joint_dim + 1:
            passive_action = np.concatenate(
                [passive_qpos[:joint_dim], np.array([self.OPEN], dtype=np.float32)]
            )
        elif active_action.shape[-1] == 2 * joint_dim + 1:
            passive_action = np.zeros_like(active_action, dtype=np.float32)
            passive_action[:joint_dim] = passive_qpos[:joint_dim]
            passive_action[joint_dim:2 * joint_dim] = 0.0
            passive_action[-1] = self.OPEN
        else:
            passive_action = passive_qpos[: active_action.shape[-1]]
            if passive_action.shape[-1] < active_action.shape[-1]:
                pad = np.zeros(active_action.shape[-1] - passive_action.shape[-1], dtype=np.float32)
                passive_action = np.concatenate([passive_action, pad])

        if self.active_arm_uid is None or self.passive_arm_uid is None:
            return active_action

        return {self.active_arm_uid: active_action, self.passive_arm_uid: passive_action}

    def _step_env(self, action):
        if hasattr(self, 'active_arm_name') and self.active_arm_name == 'both' and hasattr(self, 'real_env'):
            step_env = self.real_env
        else:
            step_env = self.env
        obs, reward, terminated, truncated, info = step_env.step(action)
        self.elapsed_steps += 1

        # 收集数据
        if self.data_collector is not None:
            # 提取关节状态
            robot_state = {}
            try:
                # 获取双臂状态
                if hasattr(self, 'left_arm') and hasattr(self.left_arm, 'robot') and hasattr(self.left_arm.robot,
                                                                                             'get_qpos'):
                    # 左臂状态
                    left_qpos = self.left_arm.robot.get_qpos()[0].cpu().numpy()
                    left_qvel = self.left_arm.robot.get_qvel()[0].cpu().numpy() if hasattr(self.left_arm.robot,
                                                                                           'get_qvel') else np.zeros_like(
                        left_qpos)

                    # 右臂状态
                    right_qpos = self.right_arm.robot.get_qpos()[0].cpu().numpy()
                    right_qvel = self.right_arm.robot.get_qvel()[0].cpu().numpy() if hasattr(self.right_arm.robot,
                                                                                             'get_qvel') else np.zeros_like(
                        right_qpos)

                    # 合并双臂状态（每个臂8维：6关节+2夹爪）
                    qpos = np.concatenate([left_qpos[:8], right_qpos[:8]])
                    qvel = np.concatenate([left_qvel[:8], right_qvel[:8]])

                    robot_state['qpos'] = qpos
                    robot_state['qvel'] = qvel
            except Exception as e:
                print(f"警告：获取机器人状态失败: {e}")

            # 收集数据
            self.data_collector.collect_step_data(action=action, obs=obs, info=robot_state, env=self.base_env)

        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if self.vis and hasattr(self.base_env, '_viewer') and self.base_env._viewer is not None:
            self.base_env.render_human()
        return obs, reward, terminated, truncated, info


def move_with_solver(
    solver: 'PiperXArmMotionPlanningSolver',
    target_pos,
    target_quat=None,
    gripper_state=None,
    allow_adaptive_search=True,
    limit_search_range=False,
    search_radius=0.01,
    max_search_time=10.0,
    plan_only=False,
    max_trajectory_steps=100,
):
    """
    使用 PiperXArmMotionPlanningSolver 进行运动规划，内置自适应姿态搜索。

    Args:
        solver: PiperXArmMotionPlanningSolver 实例
        target_pos: 目标位置 [x, y, z]
        target_quat: 目标姿态四元数 [x, y, z, w]（None=保持当前）
        gripper_state: 夹爪状态（None=保持当前）
        allow_adaptive_search: 是否允许自适应搜索
        limit_search_range: 是否限制搜索范围（SLERP + ±90°）
        search_radius: 球面采样搜索半径（米）
        max_search_time: 最大搜索时间（秒）
        plan_only: 若为 True，只规划不执行，返回 result dict（成功）或 None（失败）

    Returns:
        bool 或 dict: 默认返回 bool（是否成功）；
                      plan_only=True 时返回 result 对象（成功）或 None（失败）
    """
    import time
    from itertools import product
    from scipy.spatial.transform import Rotation as R
    from mani_skill.envs.tasks.custom_task.motion_planning_utils import (
        slerp_quat,
        combine_quaternions,
    )

    # === 获取当前状态 ===
    # 复用旧 move_to_position_only 的规划逻辑
    robot = solver.robot
    base_env = solver.base_env

    # 获取 URDF/SRDF 路径
    urdf_path = None
    if hasattr(solver, 'env_agent') and hasattr(solver.env_agent, 'urdf_path'):
        urdf_path = solver.env_agent.urdf_path
    from pathlib import Path
    if urdf_path is None:
        from mani_skill.utils import assets
        urdf_path = assets.get_asset_path("robots/piper_x/piper_x_description/piper_x_description_d435.urdf")
    srdf_path = urdf_path.replace(".urdf", ".srdf")
    urdf_dir = str(Path(urdf_path).parent)

    # 获取关节和连杆信息
    link_names = [link.get_name() for link in robot.get_links()]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]

    # 创建新 planner（与旧 move_to_position_only 完全一致）
    planner = mplib.Planner(
        urdf=urdf_path,
        srdf=srdf_path,
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="piper_x_hand_tcp",
        search_path=urdf_dir,
    )
    base_pose_array = np.hstack([robot.pose.p, robot.pose.q])
    planner.set_base_pose(base_pose_array.reshape(-1, 1).astype(np.float64))

    # 获取当前 qpos（只取前6个臂关节）
    current_qpos = robot.get_qpos()[0].cpu().numpy()[:6]

    ee_pose = solver.env_agent.tcp_pose
    current_p = ee_pose.p
    current_q = ee_pose.q
    if len(current_p.shape) == 2:
        current_p = current_p[0]
    if len(current_q.shape) == 2:
        current_q = current_q[0]
    current_p = np.array(current_p.cpu().numpy() if hasattr(current_p, 'cpu') else current_p, dtype=np.float32)
    current_q = np.array(current_q.cpu().numpy() if hasattr(current_q, 'cpu') else current_q, dtype=np.float32)

    target_p = np.array(target_pos, dtype=np.float32)
    target_q = np.array(target_quat, dtype=np.float32).flatten() if target_quat is not None else current_q

    print(f"  目标位置: {target_p}")
    print(f"  当前末端位置: {current_p}")
    print(f"  当前末端姿态: {current_q}")
    print(f"  目标姿态: {target_q}")

    # 夹爪状态
    old_gripper = solver.gripper_state
    if gripper_state is not None:
        solver.gripper_state = gripper_state

    # ──────────────────────────────────────────────
    # 复用旧 find_reachable_pose 进行自适应搜索
    # ──────────────────────────────────────────────
    # 直接构造 robot_idx 参数：从 solver 获取
    # 在 pass_to 中传入的 robot_idx 可以通过 solver.active_arm_uid 映射
    # 但这里我们直接使用 planner 和 current_qpos，不需要 robot_idx

    success, final_pose_array = find_reachable_pose(
        base_env, 0, planner, current_qpos,
        target_p, target_q,
        allow_adaptive_search=allow_adaptive_search,
        limit_search_range=limit_search_range,
        max_search_time=max_search_time,
        current_q_np=current_q,
    )

    if not success:
        print(f"  ❌ 自适应搜索均失败")
        solver.gripper_state = old_gripper
        return False

    # ──────────────────────────────────────────────
    # 执行规划
    # ──────────────────────────────────────────────
    try:
        result = planner.plan_screw(
            final_pose_array,
            current_qpos,
            time_step=base_env.control_timestep,
            use_point_cloud=False,
        )
    except Exception as e:
        print(f"  ❌ 运动规划执行失败: {e}")
        solver.gripper_state = old_gripper
        return False

    if result["status"] != "Success":
        print(f"  ❌ 运动规划执行失败: status={result['status']}")
        solver.gripper_state = old_gripper
        return False

    # ──────────────────────────────────────────────
    # plan_only 模式：返回 result 字典，由调用方执行
    # ──────────────────────────────────────────────
    if plan_only:
        print(f"  ✅ 规划成功，轨迹长度: {result['position'].shape[0]}步 (plan_only)")
        return result

    # ──────────────────────────────────────────────
    # 执行模式：执行轨迹
    # ──────────────────────────────────────────────
    n = result["position"].shape[0]
    print(f"  ✅ 规划成功，轨迹长度: {n}步")
    for i in range(n):
        qpos = result["position"][i]
        action = np.hstack([qpos, solver.gripper_state])
        # 如果是双臂环境，用 _build_dual_arm_action 包装成 dict
        if hasattr(solver, '_build_dual_arm_action'):
            action = solver._build_dual_arm_action(action)
        solver.base_env.step(action)
        solver.elapsed_steps += 1
        if solver.vis and hasattr(solver.base_env, '_viewer') and solver.base_env._viewer is not None:
            solver.base_env.render_human()
    return True

    # ──────────────────────────────────────────────
    # 策略1：直接尝试
    # ──────────────────────────────────────────────
    print(f"  尝试直接规划...")
    try:
        res = _plan_fn(np.concatenate([target_p, target_q]))
        if _is_ok(res):
            print(f"  ✅ 直接规划成功")
            return res if plan_only else True
    except Exception as e:
        print(f"    直接规划失败: {e}")

    if not allow_adaptive_search:
        solver.gripper_state = old_gripper
        return _ok

    # ──────────────────────────────────────────────
    # 策略2：SLERP 姿态插值（仅在 limit_search_range=True 时）
    # ──────────────────────────────────────────────
    if limit_search_range:
        print(f"  尝试姿态插值（10步SLERP）...")
        best_i, best_result = -1, None
        for i in range(1, 11):
            if timeout():
                break
            t = i / 10.0
            iquat = slerp_quat(current_q, target_q, t)
            candidate = np.concatenate([target_p, iquat])
            try:
                res = _plan_fn(candidate)
                if _is_ok(res):
                    n = res["position"].shape[0]
                    if plan_only and n > max_trajectory_steps:
                        print(f"    SLERP步{i}/10: 轨迹长度({n}步)超过限制({max_trajectory_steps}步)，跳过，继续搜索")
                        continue
                    best_i, best_result = i, res
            except Exception:
                continue
        if best_i > 0:
            dot = np.clip(np.abs(np.dot(current_q, target_q)), 0, 1)
            angle = 2.0 * np.arccos(dot)
            print(f"  ✅ 插值可达（步{best_i}/10，已达成{best_i*10}%目标姿态，全程{angle:.2f}弧度）")
            return best_result if plan_only else True
        print(f"    SLERP 插值均不可达")

    # ──────────────────────────────────────────────
    # 策略3：球面位置采样 + 姿态偏移
    # ──────────────────────────────────────────────
    # 姿态候选
    angles = [0, np.pi/6, -np.pi/6, np.pi/3, -np.pi/3, np.pi/2, -np.pi/2]
    rot_offsets = [R.from_euler('xyz', a).as_quat() for a in product(angles, repeat=3)]

    # 斐波那契球面采样点
    n_samples = 20
    phi = np.pi * (3. - np.sqrt(5.))
    fib_points = []
    for i in range(n_samples):
        y = 1 - i / (n_samples - 1) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        fib_points.append([np.cos(theta) * r, y, np.sin(theta) * r])

    for scale in [1, 2]:
        radius = search_radius * scale
        print(f"  球面采样（半径={radius:.2f}米，{len(rot_offsets)}种姿态）...")
        for fp in fib_points:
            if timeout():
                print(f"    搜索超时")
                solver.gripper_state = old_gripper
                return _ok
            pos = target_p + np.array(fp) * radius
            for roq in rot_offsets:
                if timeout():
                    break
                q = combine_quaternions(target_q, roq)
                try:
                    res = _plan_fn(np.concatenate([pos, q]))
                    if _is_ok(res):
                        n = res["position"].shape[0]
                        if plan_only and n > max_trajectory_steps:
                            continue
                        d = np.linalg.norm(pos - target_p)
                        print(f"  ✅ 球面采样找到可达姿态（距目标{d:.3f}米）")
                        return res if plan_only else True
                except Exception:
                    continue

    print(f"  ❌ 自适应搜索均失败")
    solver.gripper_state = old_gripper
    return _ok