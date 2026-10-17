import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver
from mani_skill.utils.structs.pose import to_sapien_pose


class DataCollector:
    """数据收集器，用于存储每个episode的数据"""
    
    def __init__(self, episode_data):
        self.episode_data = episode_data
    
    def collect_step_data(self, action, obs, info, env=None):
        if "base_time" not in self.episode_data:
            import time as _time
            self.episode_data["base_time"] = _time.time()
        frame_idx = self.episode_data.get('total_saved_frames', 0) + len(self.episode_data.get("timestamps", []))
        ts = self.episode_data["base_time"] + frame_idx / 30.0
        self.episode_data.setdefault("timestamps", []).append(f"{ts:.6f}")

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
        
        # 处理末端位姿数据
        left_pose_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        right_pose_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        try:
            if env is not None and hasattr(env, 'left_arm') and hasattr(env, 'right_arm'):
                from transforms3d.euler import quat2euler
                from transforms3d.quaternions import qmult
                left_base_pose = env.left_arm.robot.root.pose
                right_base_pose = env.right_arm.robot.root.pose
                left_tcp_world = env.left_arm.tcp.pose
                right_tcp_world = env.right_arm.tcp.pose
                left_tcp_local = left_base_pose.inv() * left_tcp_world
                right_tcp_local = right_base_pose.inv() * right_tcp_world
                left_pos = left_tcp_local.p
                left_quat = left_tcp_local.q
                right_pos = right_tcp_local.p
                right_quat = right_tcp_local.q
                if hasattr(left_pos, 'cpu'):
                    left_pos = left_pos.cpu().numpy().flatten()
                    left_quat = left_quat.cpu().numpy().flatten()
                else:
                    left_pos = np.asarray(left_pos).flatten()
                    left_quat = np.asarray(left_quat).flatten()
                if hasattr(right_pos, 'cpu'):
                    right_pos = right_pos.cpu().numpy().flatten()
                    right_quat = right_quat.cpu().numpy().flatten()
                else:
                    right_pos = np.asarray(right_pos).flatten()
                    right_quat = np.asarray(right_quat).flatten()
                q_z90 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
                left_quat = qmult(left_quat, q_z90)
                right_quat = qmult(right_quat, q_z90)
                left_roll, left_pitch, left_yaw = quat2euler(left_quat, axes='sxyz')
                right_roll, right_pitch, right_yaw = quat2euler(right_quat, axes='sxyz')
                left_pose_dict = {
                    "x": float(left_pos[0]), "y": float(left_pos[1]), "z": float(left_pos[2]),
                    "roll": float(left_roll), "pitch": float(left_pitch), "yaw": float(left_yaw),
                }
                right_pose_dict = {
                    "x": float(right_pos[0]), "y": float(right_pos[1]), "z": float(right_pos[2]),
                    "roll": float(right_roll), "pitch": float(right_pitch), "yaw": float(right_yaw),
                }
        except Exception as e:
            print(f"[endPose ERROR] {e}")
        self.episode_data.setdefault("endPose", {"puppetLeft": [], "puppetRight": []})
        self.episode_data["endPose"].setdefault("puppetLeft", []).append(left_pose_dict)
        self.episode_data["endPose"].setdefault("puppetRight", []).append(right_pose_dict)

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
        self._passive_arm_target_qpos = None
        self._passive_gripper_state = None
        self.real_env = env
        self.left_solver = None
        self.right_solver = None
        self.data_collector = DataCollector(episode_data) if episode_data is not None else None  # 数据收集器

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
                self.gripper_state = self.CLOSED
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

    def set_passive_arm_target_qpos(self, qpos=None):
        if qpos is not None:
            self._passive_arm_target_qpos = np.asarray(qpos, dtype=np.float32).copy()
        elif self.passive_arm is not None:
            self._passive_arm_target_qpos = self.passive_arm.robot.get_qpos()[0].cpu().numpy().copy()

    def clear_passive_arm_target_qpos(self):
        self._passive_arm_target_qpos = None

    def set_passive_gripper_state(self, state):
        self._passive_gripper_state = state

    def clear_passive_gripper_state(self):
        self._passive_gripper_state = None

    def _build_dual_arm_action(self, active_action: np.ndarray):
        if not self.use_arm_proxy or self.active_arm is None or self.passive_arm is None:
            return active_action

        active_action = np.asarray(active_action, dtype=np.float32)
        if self._passive_arm_target_qpos is not None:
            passive_qpos = self._passive_arm_target_qpos
        else:
            passive_qpos = self.passive_arm.robot.get_qpos()[0].cpu().numpy()
        if passive_qpos.ndim == 0:
            passive_qpos = np.asarray([passive_qpos], dtype=np.float32)

        passive_gripper = self._passive_gripper_state if self._passive_gripper_state is not None else self.CLOSED

        joint_dim = len(self.planner.joint_vel_limits)
        if active_action.shape[-1] == joint_dim + 1:
            passive_action = np.concatenate(
                [passive_qpos[:joint_dim], np.array([passive_gripper], dtype=np.float32)]
            )
        elif active_action.shape[-1] == 2 * joint_dim + 1:
            passive_action = np.zeros_like(active_action, dtype=np.float32)
            passive_action[:joint_dim] = passive_qpos[:joint_dim]
            passive_action[joint_dim:2 * joint_dim] = 0.0
            passive_action[-1] = passive_gripper
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

    def _sync_dual_action(self, left_action: np.ndarray, right_action: np.ndarray):
        return {self.left_arm_uid: left_action, self.right_arm_uid: right_action}

    def step_dual_action(self, left_action: np.ndarray, right_action: np.ndarray):
        """Execute one environment step with independent left/right arm actions.

        The actions must already include per-arm joint commands and gripper state.
        This is only supported when the solver is initialized with ``arm='both'``.
        """
        if self.active_arm_name != "both":
            raise RuntimeError("step_dual_action only supports arm='both'")
        if self.left_arm_uid is None or self.right_arm_uid is None:
            raise RuntimeError("Dual-arm UIDs are not initialized")

        left_action = np.asarray(left_action, dtype=np.float32)
        right_action = np.asarray(right_action, dtype=np.float32)
        action = self._sync_dual_action(left_action, right_action)
        return self._step_env(action)

    def follow_dual_path(self, left_result, right_result, refine_steps: int = 0):
        """Follow two independent planned paths for left and right arms."""
        return self._follow_dual_path(left_result, right_result, refine_steps)

    def _follow_dual_path(self, left_result, right_result, refine_steps: int = 0):
        n_left = left_result["position"].shape[0]
        n_right = right_result["position"].shape[0]
        max_step = max(n_left, n_right)

        left_velocity = left_result.get("velocity")
        right_velocity = right_result.get("velocity")
        for i in range(max_step):
            left_qpos = left_result["position"][min(i, n_left - 1)]
            right_qpos = right_result["position"][min(i, n_right - 1)]

            if self.control_mode == "pd_joint_pos_vel":
                left_qvel = (
                    left_velocity[min(i, n_left - 1)]
                    if left_velocity is not None
                    else np.zeros_like(left_qpos)
                )
                right_qvel = (
                    right_velocity[min(i, n_right - 1)]
                    if right_velocity is not None
                    else np.zeros_like(right_qpos)
                )
                left_action = np.hstack([left_qpos, left_qvel, self.gripper_state])
                right_action = np.hstack([right_qpos, right_qvel, self.gripper_state])
            else:
                left_action = np.hstack([left_qpos, self.gripper_state])
                right_action = np.hstack([right_qpos, self.gripper_state])

            action = self._sync_dual_action(left_action, right_action)
            obs, reward, terminated, truncated, info = self._step_env(action)

        for _ in range(refine_steps):
            obs, reward, terminated, truncated, info = self._step_env(action)

        # 运动完成后，隐藏双臂的可视化对象
        if self.vis and self.visualize_target_grasp_pose:
            self._hide_both_arms_visual()

        return obs, reward, terminated, truncated, info

    def move_to_pose_with_screw(
            self,
            left_pose,
            right_pose=None,
            dry_run: bool = False,
            refine_steps: int = 0,
    ):
        if right_pose is None:
            if self.active_arm_name == "both":
                raise ValueError(
                    "In arm='both' mode, call move_to_pose_with_screw(left_pose, right_pose, ...)"
                )
            return super().move_to_pose_with_screw(left_pose, dry_run=dry_run, refine_steps=refine_steps)
        if self.active_arm_name != "both":
            raise RuntimeError("move_to_pose_with_screw(left_pose, right_pose) only supports arm='both'")
        return self.move_to_pose_with_screw_both(left_pose, right_pose, dry_run=dry_run, refine_steps=refine_steps)

    def move_to_pose_with_screw_both(self, left_pose, right_pose, dry_run: bool = False, refine_steps: int = 0):
        if self.active_arm_name != "both":
            raise RuntimeError("move_to_pose_with_screw_both only supports arm='both'")
        if self.left_solver is None or self.right_solver is None:
            raise RuntimeError("Both-arm solvers are not initialized")

        left_pose = to_sapien_pose(left_pose)
        right_pose = to_sapien_pose(right_pose)
        
        # 在规划前立即更新双臂抓取位姿可视化
        if self.vis and self.visualize_target_grasp_pose:
            self._update_dual_grasp_visual(left_pose, right_pose)
        
        left_result = self.left_solver.planner.plan_screw(
            np.concatenate([left_pose.p, left_pose.q]),
            self.left_arm.robot.get_qpos()[0].cpu().numpy(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        right_result = self.right_solver.planner.plan_screw(
            np.concatenate([right_pose.p, right_pose.q]),
            self.right_arm.robot.get_qpos()[0].cpu().numpy(),
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )

        if left_result["status"] != "Success" or right_result["status"] != "Success":
            print(left_result["status"], right_result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return left_result, right_result
        return self._follow_dual_path(left_result, right_result, refine_steps)

    def follow_path(self, result, refine_steps: int = 0):
        if not self.use_arm_proxy:
            return super().follow_path(result, refine_steps)

        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            action = self._build_dual_arm_action(action)
            obs, reward, terminated, truncated, info = self._step_env(action)
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if not self.use_arm_proxy:
            return super().open_gripper(t, gripper_state)

        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state

        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                active_action = np.hstack([qpos, self.gripper_state])
            else:
                active_action = np.hstack([qpos, qpos * 0, self.gripper_state])
            action = self._build_dual_arm_action(active_action)
            obs, reward, terminated, truncated, info = self._step_env(action)
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if not self.use_arm_proxy:
            return super().close_gripper(t, gripper_state)

        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state

        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                active_action = np.hstack([qpos, self.gripper_state])
            else:
                active_action = np.hstack([qpos, qpos * 0, self.gripper_state])
            action = self._build_dual_arm_action(active_action)
            obs, reward, terminated, truncated, info = self._step_env(action)
        return obs, reward, terminated, truncated, info